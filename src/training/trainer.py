import os
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from .loss import QueueInfoNCE, SymmetricInfoNCE
from .queue import ContrastiveQueue


def _get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup → cosine decay scheduler."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535)))

    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Training loop with AMP, TQDM logging, and checkpointing."""

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, cfg, queue=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.queue = queue
        self.device = next(model.parameters()).device
        # In-batch loss for monitoring (always comparable to val loss)
        self.eval_loss_fn = SymmetricInfoNCE(temperature=cfg["training"]["temperature"])

        total_steps = len(train_loader) * cfg["training"]["epochs"]
        self.scheduler = _get_cosine_schedule_with_warmup(
            optimizer, cfg["training"]["warmup_steps"], total_steps
        )
        self.scaler = torch.amp.GradScaler("cuda") if cfg["training"]["amp"] and torch.cuda.is_available() else None

        ckpt_dir = cfg["training"]["checkpoint_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)
        self.global_step = 0

    def train_batch(self, image_emb, input_ids, attention_mask, image_ids=None) -> Dict[str, float]:
        """Single training step. Returns dict with 'loss' (optimised) and
        optionally 'loss_inbatch' (monitor value, comparable to val loss)."""
        self.model.train()
        image_emb = image_emb.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        if self.scaler:
            with torch.amp.autocast("cuda"):
                text_emb = self.model(input_ids, attention_mask)
                if isinstance(self.loss_fn, QueueInfoNCE):
                    loss = self.loss_fn(image_emb, text_emb, image_ids,
                                        logit_scale=self.model.logit_scale)
                    with torch.no_grad():
                        loss_inbatch = self.eval_loss_fn(
                            image_emb, text_emb, logit_scale=self.model.logit_scale)
                else:
                    loss = self.loss_fn(image_emb, text_emb,
                                        logit_scale=self.model.logit_scale)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["training"]["max_grad_norm"]
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            text_emb = self.model(input_ids, attention_mask)
            if isinstance(self.loss_fn, QueueInfoNCE):
                loss = self.loss_fn(image_emb, text_emb, image_ids,
                                    logit_scale=self.model.logit_scale)
                with torch.no_grad():
                    loss_inbatch = self.eval_loss_fn(
                        image_emb, text_emb, logit_scale=self.model.logit_scale)
            else:
                loss = self.loss_fn(image_emb, text_emb,
                                    logit_scale=self.model.logit_scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["training"]["max_grad_norm"]
            )
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.scheduler.step()
        self.global_step += 1

        # Enqueue after loss computation
        if self.queue is not None and image_ids is not None:
            self.queue.enqueue(image_emb, text_emb, image_ids)

        result: Dict[str, float] = {"loss": loss.item()}
        if isinstance(self.loss_fn, QueueInfoNCE):
            result["loss_inbatch"] = loss_inbatch.item()
            result["uniformity"] = getattr(self.loss_fn, "_last_uniformity", 0.0)
        return result

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on val_loader. Returns dict with 'val_loss' and recall metrics."""
        self.model.eval()
        total_loss = 0.0
        all_image_embs: List[torch.Tensor] = []
        all_text_embs: List[torch.Tensor] = []
        all_image_ids: List[str] = []

        # Always use standard in-batch loss for evaluation (no queue)
        # Use model's learned logit_scale when available
        eval_loss_fn = SymmetricInfoNCE(temperature=self.cfg["training"]["temperature"])
        logit_scale = getattr(self.model, "logit_scale", None)

        pbar = tqdm(self.val_loader, desc="Val", leave=False)
        for batch in pbar:
            image_emb, input_ids, attention_mask, *rest = batch
            image_ids = rest[0] if rest else None
            image_emb = image_emb.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            text_emb = self.model(input_ids, attention_mask)
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

    def save_checkpoint(self, epoch: int, val_loss: float, **extra_metrics) -> str:
        path = os.path.join(
            self.cfg["training"]["checkpoint_dir"],
            f"bert_epoch{epoch:02d}_val{val_loss:.4f}.pt",
        )
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "lora_config": getattr(self.model, "lora_config", None),
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
        """Load checkpoint and restore model, optimizer, scheduler, scaler, and queue states."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if self.scaler and "scaler_state_dict" in ckpt:
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        if self.queue is not None and "queue_state" in ckpt:
            self.queue.load_state_dict(ckpt["queue_state"])
        return ckpt["epoch"], ckpt["val_loss"]
