import os
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR


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

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, cfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device = next(model.parameters()).device

        total_steps = len(train_loader) * cfg["training"]["epochs"]
        self.scheduler = _get_cosine_schedule_with_warmup(
            optimizer, cfg["training"]["warmup_steps"], total_steps
        )
        self.scaler = torch.amp.GradScaler("cuda") if cfg["training"]["amp"] and torch.cuda.is_available() else None

        ckpt_dir = cfg["training"]["checkpoint_dir"]
        os.makedirs(ckpt_dir, exist_ok=True)

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc="Train", leave=False)

        for image_emb, input_ids, attention_mask in pbar:
            image_emb = image_emb.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    text_emb = self.model(input_ids, attention_mask)
                    loss = self.loss_fn(image_emb, text_emb)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["training"]["max_grad_norm"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                text_emb = self.model(input_ids, attention_mask)
                loss = self.loss_fn(image_emb, text_emb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg["training"]["max_grad_norm"]
                )
                self.optimizer.step()

            self.optimizer.zero_grad()
            self.scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        pbar = tqdm(self.val_loader, desc="Val", leave=False)

        for image_emb, input_ids, attention_mask in pbar:
            image_emb = image_emb.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            text_emb = self.model(input_ids, attention_mask)
            loss = self.loss_fn(image_emb, text_emb)
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float) -> str:
        path = os.path.join(
            self.cfg["training"]["checkpoint_dir"],
            f"bert_epoch{epoch:02d}_val{val_loss:.4f}.pt",
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }, path)
        return path
