"""TemporalTrainer: insertion of TemporalTransformer into the contrastive pipeline.

Projects precomputed CLIP image embeddings through a TemporalTransformer
before passing them to the contrastive loss.  BERT text encoder is frozen
so text embeddings are precomputed offline — no BERT forward pass during
training or evaluation.
"""

import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from .trainer import Trainer
from .loss import QueueInfoNCE, SymmetricInfoNCE


class TemporalTrainer(Trainer):
    """Trainer variant that routes image_emb through TemporalTransformer
    and receives precomputed text_emb directly (BERT frozen).

    :param temporal: The TemporalTransformer module (trainable).
    """

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn,
                 cfg, queue=None, temporal=None):
        super().__init__(model, train_loader, val_loader, optimizer, loss_fn, cfg, queue)
        self.temporal = temporal

    # ── helpers ────────────────────────────────────────────────────────────

    def _project_image(self, image_emb: torch.Tensor) -> torch.Tensor:
        """[B, 512] -> unsqueeze(1) -> temporal -> [B, 512] L2-normed."""
        B = image_emb.size(0)
        device = image_emb.device
        frame_embs = image_emb.unsqueeze(1)                     # [B, 1, 512]
        mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        positions = torch.tensor([[0.5, 0.5, 1.0]], device=device).expand(B, 1, 3)
        return self.temporal(frame_embs, mask, positions=positions)

    @property
    def _clip_params(self):
        """Parameters that need gradient clipping (only trainable ones)."""
        return list(self.temporal.parameters())

    # ── training step ──────────────────────────────────────────────────────

    def train_batch(self, image_emb, text_emb, image_ids=None) -> Dict[str, float]:
        """Override: receives precomputed text_emb directly (BERT frozen)."""
        self.model.train()
        self.temporal.train()
        image_emb = image_emb.to(self.device)
        text_emb = text_emb.to(self.device)
        image_emb = self._project_image(image_emb)

        if self.scaler:
            with torch.amp.autocast("cuda"):
                if isinstance(self.loss_fn, QueueInfoNCE):
                    loss = self.loss_fn(
                        image_emb, text_emb, image_ids,
                        logit_scale=getattr(self.model, "logit_scale", None),
                    )
                    with torch.no_grad():
                        loss_inbatch = self.eval_loss_fn(
                            image_emb, text_emb,
                            logit_scale=getattr(self.model, "logit_scale", None),
                        )
                else:
                    loss = self.loss_fn(
                        image_emb, text_emb,
                        logit_scale=getattr(self.model, "logit_scale", None),
                    )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._clip_params, self.cfg["training"]["max_grad_norm"],
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if isinstance(self.loss_fn, QueueInfoNCE):
                loss = self.loss_fn(
                    image_emb, text_emb, image_ids,
                    logit_scale=getattr(self.model, "logit_scale", None),
                )
                with torch.no_grad():
                    loss_inbatch = self.eval_loss_fn(
                        image_emb, text_emb,
                        logit_scale=getattr(self.model, "logit_scale", None),
                    )
            else:
                loss = self.loss_fn(
                    image_emb, text_emb,
                    logit_scale=getattr(self.model, "logit_scale", None),
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._clip_params, self.cfg["training"]["max_grad_norm"],
            )
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()
        self.global_step += 1

        if self.queue is not None and image_ids is not None:
            self.queue.enqueue(image_emb, text_emb, image_ids)

        from .trainer import _resolve
        result: Dict[str, float] = {"loss": loss.item()}
        if isinstance(self.loss_fn, QueueInfoNCE):
            result["loss_inbatch"] = loss_inbatch.item()
            result["uniformity"] = _resolve(getattr(self.loss_fn, "_last_uniformity", 0.0))
            result["i2t_q"] = _resolve(getattr(self.loss_fn, "_last_i2t", 0.0))
            result["t2i_q"] = _resolve(getattr(self.loss_fn, "_last_t2i", 0.0))
            result["i2t_ib"] = _resolve(getattr(self.eval_loss_fn, "_last_i2t", 0.0))
            result["t2i_ib"] = _resolve(getattr(self.eval_loss_fn, "_last_t2i", 0.0))
        else:
            result["i2t_q"] = _resolve(getattr(self.loss_fn, "_last_i2t", 0.0))
            result["t2i_q"] = _resolve(getattr(self.loss_fn, "_last_t2i", 0.0))
        return result

    # ── evaluation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Override: receives precomputed text_emb, projects images through temporal."""
        self.model.eval()
        self.temporal.eval()
        total_loss = 0.0
        all_image_embs: List[torch.Tensor] = []
        all_text_embs: List[torch.Tensor] = []
        all_image_ids: List[str] = []

        eval_loss_fn = SymmetricInfoNCE(temperature=self.cfg["training"]["temperature"])
        logit_scale = getattr(self.model, "logit_scale", None)

        pbar = tqdm(self.val_loader, desc="Val", leave=False)
        for batch in pbar:
            image_emb, text_emb, *rest = batch
            image_ids = rest[0] if rest else None
            image_emb = image_emb.to(self.device)
            text_emb = text_emb.to(self.device)

            image_emb = self._project_image(image_emb)
            loss = eval_loss_fn(image_emb, text_emb, logit_scale=logit_scale)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            all_image_embs.append(image_emb.cpu())
            all_text_embs.append(text_emb.cpu())
            if image_ids is not None:
                all_image_ids.extend(image_ids)

        avg_loss = total_loss / len(self.val_loader)
        results: Dict[str, float] = {"val_loss": avg_loss}

        if all_image_ids:
            from .evaluation import compute_recall_metrics
            recall = compute_recall_metrics(
                torch.cat(all_image_embs),
                torch.cat(all_text_embs),
                all_image_ids,
            )
            results.update(recall)

        return results

    # ── checkpointing ──────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, val_loss: float,
                        **extra_metrics) -> str:
        """Save temporal_state_dict (model is frozen, not saved)."""
        t2i_r1 = extra_metrics.get("t2i_R@1", None)
        best_t2i_r1 = extra_metrics.get("best_t2i_r1", t2i_r1)
        ckpt_dir = self.cfg["training"]["checkpoint_dir"]
        # Include global_step so every save gets a unique, sortable name
        path = os.path.join(
            ckpt_dir,
            f"temporal_e{epoch:02d}s{self.global_step:06d}_t2i{t2i_r1:.1f}.pt" if t2i_r1 is not None
            else f"temporal_e{epoch:02d}s{self.global_step:06d}_val{val_loss:.4f}.pt",
        )
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "temporal_state_dict": self.temporal.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "t2i_r1": t2i_r1,
            "best_t2i_r1": best_t2i_r1,
        }
        if self.scaler:
            state["scaler_state_dict"] = self.scaler.state_dict()
        if self.queue is not None:
            qstate = self.queue.state_dict()
            if qstate is not None:
                state["queue_state"] = qstate
        torch.save(state, path)
        return path

    def load_checkpoint(self, path: str):
        """Load temporal checkpoint (no model state to restore)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.temporal.load_state_dict(ckpt["temporal_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.scaler and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        if self.queue is not None and "queue_state" in ckpt:
            self.queue.load_state_dict(ckpt["queue_state"])
        return (ckpt["epoch"], ckpt["val_loss"],
                ckpt.get("t2i_r1", None),
                ckpt.get("best_t2i_r1", None))
