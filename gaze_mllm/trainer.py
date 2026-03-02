import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torchmetrics.functional.classification.auroc import binary_auroc
from tqdm.auto import tqdm

from .datasets.gazefollow_reason_dataset import GazeFollowReasonDataset, QwenVLBatchCollator
from .models.qwen_gaze_model import Qwen3VLGazeModel, compute_losses

try:
    import wandb
except Exception:
    wandb = None


@dataclass
class TrainArtifacts:
    model: Qwen3VLGazeModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler


def _numel(params):
    return sum(p.numel() for p in params)


def _human_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f} B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f} M"
    if n >= 1_000:
        return f"{n/1_000:.1f} K"
    return str(n)


def _print_model_summary(model: torch.nn.Module):
    trainable = _numel([p for p in model.parameters() if p.requires_grad])
    frozen = _numel([p for p in model.parameters() if not p.requires_grad])
    total = trainable + frozen
    est_mb = (total * 4) / (1024 ** 2)  # float32 estimate
    modules_train = sum(1 for m in model.modules() if m.training)
    modules_eval = sum(1 for m in model.modules() if not m.training)

    print("  | Name         | Type           | Params | Mode ")
    print("-------------------------------------------------------")
    print(f"0 | model        | {type(model).__name__:<14} | {_human_params(total):>6} | train")
    print("-------------------------------------------------------")
    print(f"{_human_params(trainable):<10} Trainable params")
    print(f"{_human_params(frozen):<10} Non-trainable params")
    print(f"{_human_params(total):<10} Total params")
    print(f"{est_mb:.3f}   Total estimated model params size (MB)")
    print(f"{modules_train:<10} Modules in train mode")
    print(f"{modules_eval:<10} Modules in eval mode")


def _init_wandb(cfg: Dict):
    wb_cfg = cfg.get("wandb", {})
    if not bool(wb_cfg.get("log", False)):
        return None
    if wandb is None:
        raise RuntimeError("wandb logging is enabled but `wandb` is not installed.")

    try:
        run = wandb.init(
            project=wb_cfg.get("project", "gaze_mllm"),
            entity=wb_cfg.get("entity", None),
            name=wb_cfg.get("name", None),
            group=wb_cfg.get("group", None),
            config=cfg,
        )
    except Exception as exc:
        print(f"[wandb] init failed ({exc}). Continuing without wandb logging.")
        return None

    # semgaze-like metric summaries
    wandb.define_metric("metric/test/dist_to_avg", summary="min")
    wandb.define_metric("metric/test/avg_dist", summary="min")
    wandb.define_metric("metric/test/min_dist", summary="min")
    wandb.define_metric("metric/test/auc", summary="max")
    wandb.define_metric("metric/test/multi_acc@1", summary="max")
    wandb.define_metric("metric/test/acc@1", summary="max")
    wandb.define_metric("metric/test/acc@3", summary="max")
    wandb.define_metric("metric/val/l2", summary="min")
    wandb.define_metric("metric/val/inout_acc", summary="max")
    wandb.define_metric("loss/train", summary="min")
    return run


def resolve_amp(train_cfg: Dict, device: torch.device):
    precision = str(train_cfg.get("precision", "bf16")).lower()
    if precision == "bf16":
        return True, torch.bfloat16, False
    if precision == "fp16":
        use_scaler = (device.type == "cuda")
        return True, torch.float16, use_scaler
    if precision == "fp32":
        return False, torch.float32, False
    raise ValueError(f"Unsupported precision: {precision}. Use one of: bf16, fp16, fp32")


def _default(path_base: str, filename: str) -> str:
    return os.path.join(path_base, filename)


def build_dataloaders(cfg: Dict, processor):
    data_cfg = cfg["data"]
    prompt_cfg = cfg["prompt"]

    label_root = data_cfg.get("label_root", os.path.dirname(data_cfg["train_annotation"]))
    label_embed_root = data_cfg.get("label_embed_root", _default(label_root, "label-embeds"))
    vocab2id_path = data_cfg.get("vocab2id_path", _default(label_root, "vocab2id.json"))

    train_label_csv = data_cfg.get("train_label_csv", _default(label_root, "gaze-labels-train.csv"))
    val_label_csv = data_cfg.get("val_label_csv", _default(label_root, "gaze-labels-val.csv"))
    test_label_csv = data_cfg.get("test_label_csv", _default(label_root, "gaze-labels-test.csv"))

    train_ds = GazeFollowReasonDataset(
        split="train",
        annotation_path=data_cfg["train_annotation"],
        image_root=data_cfg["image_root"],
        label_csv_path=train_label_csv,
        vocab2id_path=vocab2id_path,
        label_embed_root=label_embed_root,
        reason_output_root=data_cfg.get("reason_output_root"),
        reason_mark_root=data_cfg.get("reason_mark_root"),
        reason_prompt_root=data_cfg.get("reason_prompt_root"),
        reason_feature_root=data_cfg.get("reason_feature_root"),
        reason_feature_h5_path=data_cfg.get("reason_feature_h5_path"),
        reason_feature_dim=int(cfg["loss"].get("reason_dim", 1024)),
        use_precomputed_dino_features=bool(data_cfg.get("use_precomputed_dino_features", False)),
        dino_feature_h5_path=data_cfg.get("dino_feature_h5_train"),
        include_mark_image=data_cfg.get("include_mark_image", True),
        include_reason_text=data_cfg.get("include_reason_text", False),
    )
    val_ds = GazeFollowReasonDataset(
        split="val",
        annotation_path=data_cfg["val_annotation"],
        image_root=data_cfg["image_root"],
        label_csv_path=val_label_csv,
        vocab2id_path=vocab2id_path,
        label_embed_root=label_embed_root,
        reason_output_root=data_cfg.get("reason_output_root"),
        reason_mark_root=data_cfg.get("reason_mark_root"),
        reason_prompt_root=data_cfg.get("reason_prompt_root"),
        reason_feature_root=data_cfg.get("reason_feature_root"),
        reason_feature_h5_path=data_cfg.get("reason_feature_h5_path"),
        reason_feature_dim=int(cfg["loss"].get("reason_dim", 1024)),
        use_precomputed_dino_features=bool(data_cfg.get("use_precomputed_dino_features", False)),
        dino_feature_h5_path=data_cfg.get("dino_feature_h5_val"),
        include_mark_image=data_cfg.get("include_mark_image", True),
        include_reason_text=data_cfg.get("include_reason_text", False),
    )

    test_annotation = data_cfg.get("test_annotation")
    test_ds = None
    if test_annotation:
        test_ds = GazeFollowReasonDataset(
            split="test",
            annotation_path=test_annotation,
            image_root=data_cfg["image_root"],
            label_csv_path=test_label_csv,
            vocab2id_path=vocab2id_path,
            label_embed_root=label_embed_root,
            reason_output_root=data_cfg.get("reason_output_root"),
            reason_mark_root=data_cfg.get("reason_mark_root"),
            reason_prompt_root=data_cfg.get("reason_prompt_root"),
            reason_feature_root=data_cfg.get("reason_feature_root"),
            reason_feature_h5_path=data_cfg.get("reason_feature_h5_path"),
            reason_feature_dim=int(cfg["loss"].get("reason_dim", 1024)),
            use_precomputed_dino_features=bool(data_cfg.get("use_precomputed_dino_features", False)),
            dino_feature_h5_path=data_cfg.get("dino_feature_h5_test"),
            include_mark_image=data_cfg.get("include_mark_image", True),
            include_reason_text=data_cfg.get("include_reason_text", False),
        )

    collator = QwenVLBatchCollator(
        processor=processor,
        base_prompt=prompt_cfg["base"],
        include_mark_image=data_cfg.get("include_mark_image", True),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["eval"].get("num_workers", 2),
        pin_memory=True,
        collate_fn=collator,
    )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.get("test", {}).get("batch_size", cfg["eval"]["batch_size"]),
            shuffle=False,
            num_workers=cfg.get("test", {}).get("num_workers", cfg["eval"].get("num_workers", 2)),
            pin_memory=True,
            collate_fn=collator,
        )

    return train_loader, val_loader, test_loader


def build_train_artifacts(cfg: Dict, device: torch.device) -> TrainArtifacts:
    model_cfg = cfg["model"]
    model = Qwen3VLGazeModel(
        model_name=model_cfg["name"],
        torch_dtype=model_cfg.get("dtype", "bfloat16"),
        dino_name=model_cfg.get("dino_name", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
        train_dino=bool(model_cfg.get("train_dino", False)),
        use_gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
        train_mode=model_cfg.get("train_mode", "lora"),
        lora_r=model_cfg.get("lora_r", 16),
        lora_alpha=model_cfg.get("lora_alpha", 32),
        lora_dropout=model_cfg.get("lora_dropout", 0.05),
        reason_dim=cfg["loss"].get("reason_dim", 768),
        label_dim=cfg["loss"].get("label_dim", 512),
        angle_feature_dim=cfg["loss"].get("angle_feature_dim", 512),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"].get("weight_decay", 0.01),
    )

    total_steps = math.ceil(cfg["train"]["num_train_samples"] / cfg["train"]["batch_size"]) * cfg["train"]["epochs"]
    warmup_steps = int(total_steps * cfg["optim"].get("warmup_ratio", 0.03))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1.0, float(warmup_steps))
        progress = float(step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return TrainArtifacts(model=model, optimizer=optimizer, scheduler=scheduler)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def _gaussian_heatmap(pred_xy: torch.Tensor, size: int = 64, sigma: float = 3.0) -> torch.Tensor:
    y = torch.arange(size, device=pred_xy.device).float()
    x = torch.arange(size, device=pred_xy.device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cx = pred_xy[0].clamp(0, 1) * (size - 1)
    cy = pred_xy[1].clamp(0, 1) * (size - 1)
    heat = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return heat


def _heatmap_argmax_xy(heatmap: torch.Tensor) -> torch.Tensor:
    # heatmap: [B, H, W] -> normalized xy in [0,1]
    b, h, w = heatmap.shape
    flat = heatmap.view(b, -1)
    idx = flat.argmax(dim=1)
    y = idx // w
    x = idx % w
    x = x.to(torch.float32) / max(1, (w - 1))
    y = y.to(torch.float32) / max(1, (h - 1))
    return torch.stack([x, y], dim=1)


def _binary_gt_heatmap(gaze_points: torch.Tensor, size: int = 64) -> torch.Tensor:
    heat = torch.zeros((size, size), device=gaze_points.device, dtype=torch.float32)
    valid = gaze_points[gaze_points[:, 0] >= 0]
    if len(valid) == 0:
        return heat
    xs = (valid[:, 0].clamp(0, 1) * (size - 1)).long()
    ys = (valid[:, 1].clamp(0, 1) * (size - 1)).long()
    heat[ys, xs] = 1.0
    return heat


def _load_vocab_embeddings(cfg: Dict, device: torch.device) -> Optional[torch.Tensor]:
    data_cfg = cfg["data"]
    label_root = data_cfg.get("label_root", os.path.dirname(data_cfg["train_annotation"]))
    vocab2id_path = data_cfg.get("vocab2id_path", _default(label_root, "vocab2id.json"))
    label_embed_root = data_cfg.get("label_embed_root", _default(label_root, "label-embeds"))

    if (not os.path.exists(vocab2id_path)) or (not os.path.isdir(label_embed_root)):
        return None

    with open(vocab2id_path, "r", encoding="utf-8") as f:
        vocab2id = json.load(f)

    vocab_size = int(max(vocab2id.values())) + 1
    vocab_emb = torch.zeros((vocab_size, 512), dtype=torch.float32)
    for label, idx in vocab2id.items():
        p = os.path.join(label_embed_root, f"{label}-emb.pt")
        if not os.path.exists(p):
            continue
        emb = torch.load(p, map_location="cpu").to(torch.float32)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        vocab_emb[int(idx)] = emb
    return vocab_emb.to(device)


@torch.no_grad()
def evaluate(model: Qwen3VLGazeModel, loader: DataLoader, cfg: Dict, device: torch.device) -> Dict[str, float]:
    model.eval()
    amp_enabled, amp_dtype, _ = resolve_amp(cfg["train"], device)

    total_dist = 0.0
    total_in = 0
    total_inout_correct = 0
    total_n = 0

    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            preds = model(batch)
        pred_heat = torch.sigmoid(preds["gaze_heatmap_logit"])
        pred_xy = _heatmap_argmax_xy(pred_heat)

        in_mask = (batch["inout"] > 0.5)
        if in_mask.any():
            dist = torch.norm(pred_xy[in_mask] - batch["gaze_xy"][in_mask], dim=-1)
            total_dist += dist.sum().item()
            total_in += int(in_mask.sum().item())

        inout_pred = (torch.sigmoid(preds["inout_logit"]) > 0.5).float()
        total_inout_correct += int((inout_pred == batch["inout"]).sum().item())
        total_n += int(batch["inout"].numel())

    return {
        "val_l2": total_dist / max(1, total_in),
        "val_inout_acc": total_inout_correct / max(1, total_n),
    }


@torch.no_grad()
def evaluate_test_semgaze_metrics(model: Qwen3VLGazeModel, loader: DataLoader, cfg: Dict, device: torch.device) -> Dict[str, float]:
    model.eval()
    amp_enabled, amp_dtype, _ = resolve_amp(cfg["train"], device)
    vocab_emb = _load_vocab_embeddings(cfg, device)

    sum_dist_to_avg = 0.0
    sum_avg_dist = 0.0
    sum_min_dist = 0.0
    sum_auc = 0.0
    n_obs = 0

    acc1_correct = 0
    acc3_correct = 0
    acc_total = 0

    multi_acc1_correct = 0
    multi_total = 0

    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            preds = model(batch)

        pred_heat = torch.sigmoid(preds["gaze_heatmap_logit"])
        pred_xy = _heatmap_argmax_xy(pred_heat)

        if vocab_emb is not None:
            logits = preds["label_emb"] @ vocab_emb.T
            target = batch["gaze_label_id"]
            valid = target >= 0
            if valid.any():
                top1 = logits[valid].topk(k=1, dim=1).indices.squeeze(1)
                top3 = logits[valid].topk(k=3, dim=1).indices
                gt = target[valid]
                acc1_correct += int((top1 == gt).sum().item())
                acc3_correct += int((top3 == gt.unsqueeze(1)).any(dim=1).sum().item())
                acc_total += int(valid.sum().item())

            target_multi = batch["gaze_label_ids"]
            for i in range(logits.size(0)):
                valid_ids = target_multi[i][target_multi[i] >= 0]
                if valid_ids.numel() == 0:
                    continue
                pred1 = logits[i].argmax().item()
                multi_acc1_correct += int((valid_ids == pred1).any().item())
                multi_total += 1

        gaze_points = batch["gaze_points"]
        for i in range(pred_xy.size(0)):
            gp_gt = gaze_points[i]
            gp_gt = gp_gt[gp_gt[:, 0] >= 0]
            if gp_gt.numel() == 0:
                continue
            gp_pred = pred_xy[i]

            gp_avg = gp_gt.mean(dim=0)
            sum_dist_to_avg += torch.norm(gp_avg - gp_pred, p=2).item()
            dists = torch.norm(gp_gt - gp_pred.unsqueeze(0), p=2, dim=1)
            sum_avg_dist += dists.mean().item()
            sum_min_dist += dists.min().item()

            hm_gt = _binary_gt_heatmap(gp_gt, size=64)
            auc = binary_auroc(pred_heat[i].flatten(), hm_gt.flatten().long())
            sum_auc += float(auc.item())
            n_obs += 1

    metrics = {
        "metric/test/acc@1": (acc1_correct / max(1, acc_total)),
        "metric/test/acc@3": (acc3_correct / max(1, acc_total)),
        "metric/test/auc": (sum_auc / max(1, n_obs)),
        "metric/test/dist_to_avg": (sum_dist_to_avg / max(1, n_obs)),
        "metric/test/avg_dist": (sum_avg_dist / max(1, n_obs)),
        "metric/test/min_dist": (sum_min_dist / max(1, n_obs)),
        "metric/test/multi_acc@1": (multi_acc1_correct / max(1, multi_total)),
    }
    return metrics


def train_loop(cfg: Dict):
    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    artifacts = build_train_artifacts(cfg, device)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, artifacts.model.processor)

    model = artifacts.model
    optimizer = artifacts.optimizer
    scheduler = artifacts.scheduler
    if bool(cfg["train"].get("print_model_summary", True)):
        _print_model_summary(model)
    run = _init_wandb(cfg)
    wb_cfg = cfg.get("wandb", {})
    wb_watch = str(wb_cfg.get("watch", "none")).lower()
    if (run is not None) and (wb_watch in {"gradients", "parameters", "all"}):
        wandb.watch(model, log=wb_watch, log_freq=int(wb_cfg.get("watch_freq", 500)))

    print(f"[setup] train_dino={cfg['model'].get('train_dino', False)} dino={cfg['model'].get('dino_name')}")

    loss_cfg = {
        "heatmap": float(cfg["loss"].get("w_heatmap", 1.0)),
        "coord": float(cfg["loss"].get("w_coord", 1.0)),
        "vec": float(cfg["loss"].get("w_vec", 0.5)),
        "angle": float(cfg["loss"].get("w_angle", 1.0)),
        "inout": float(cfg["loss"].get("w_inout", 1.0)),
        "reason": float(cfg["loss"].get("w_reason", 0.3)),
        "label": float(cfg["loss"].get("w_label", 0.3)),
        "reason_loss_type": cfg["loss"].get("reason_loss_type", "cosine"),
        "heatmap_size": int(cfg["loss"].get("heatmap_size", 64)),
        "heatmap_sigma": float(cfg["loss"].get("heatmap_sigma", 3.0)),
    }

    amp_enabled, amp_dtype, use_scaler = resolve_amp(cfg["train"], device)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    best_val = float("inf")
    global_step = 0

    sanity_steps = int(cfg["train"].get("sanity_val_steps", 0))
    if sanity_steps > 0:
        model.eval()
        sanity_iter = tqdm(
            val_loader,
            total=min(len(val_loader), sanity_steps),
            desc="Sanity Checking DataLoader 0",
            dynamic_ncols=True,
            leave=True,
        )
        with torch.no_grad():
            for i, batch in enumerate(sanity_iter):
                if i >= sanity_steps:
                    break
                batch = to_device(batch, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    _ = model(batch)
        model.train()

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        use_pbar = bool(cfg["train"].get("progress_bar", True))
        iterator = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"train epoch {epoch+1}/{cfg['train']['epochs']}",
            dynamic_ncols=True,
            leave=True,
            disable=not use_pbar,
        )

        for i, batch in enumerate(iterator):
            batch = to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                preds = model(batch)
                losses = compute_losses(preds, batch, loss_cfg)
                loss = losses["total"] / grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % grad_accum == 0:
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += float(losses["total"].item())
            if use_pbar:
                avg = running / (i + 1)
                iterator.set_postfix(
                    loss=f"{avg:.4f}",
                    heatmap=f"{float(losses['heatmap'].item()):.4f}",
                    coord=f"{float(losses['coord'].item()):.4f}",
                    angle=f"{float(losses['angle'].item()):.4f}",
                    lr=f"{float(scheduler.get_last_lr()[0]):.2e}",
                )

            if (i + 1) % cfg["train"].get("log_every", 20) == 0:
                avg = running / (i + 1)
                # Avoid breaking tqdm rendering: when progress bar is active, keep metrics in postfix only.
                if not use_pbar:
                    print(
                        f"epoch={epoch+1} step={i+1}/{len(train_loader)} "
                        f"loss={avg:.4f} heatmap={losses['heatmap']:.4f} coord={losses['coord']:.4f} vec={losses['vec']:.4f} "
                        f"angle={losses['angle']:.4f} inout={losses['inout']:.4f} "
                        f"reason={losses['reason']:.4f} label={losses['label']:.4f}"
                    )
                if run is not None:
                    wandb.log(
                        {
                            "loss/train": avg,
                            "loss/train_step/heatmap": float(losses["heatmap"].item()),
                            "loss/train_step/coord": float(losses["coord"].item()),
                            "loss/train_step/vec": float(losses["vec"].item()),
                            "loss/train_step/angle": float(losses["angle"].item()),
                            "loss/train_step/inout": float(losses["inout"].item()),
                            "loss/train_step/reason": float(losses["reason"].item()),
                            "loss/train_step/label": float(losses["label"].item()),
                            "optim/lr": float(scheduler.get_last_lr()[0]),
                            "epoch": epoch + 1,
                        }
                    )

        metrics = evaluate(model, val_loader, cfg, device)
        tqdm.write(f"[val] epoch={epoch+1} val_l2={metrics['val_l2']:.4f} val_inout_acc={metrics['val_inout_acc']:.4f}")
        if run is not None:
            wandb.log(
                {
                    "metric/val/l2": metrics["val_l2"],
                    "metric/val/inout_acc": metrics["val_inout_acc"],
                    "epoch": epoch + 1,
                }
            )

        ckpt_last = os.path.join(cfg["train"]["output_dir"], "last.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}, ckpt_last)

        if metrics["val_l2"] < best_val:
            best_val = metrics["val_l2"]
            ckpt_best = os.path.join(cfg["train"]["output_dir"], "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}, ckpt_best)
            tqdm.write(
                f"Epoch {epoch+1}, global step {global_step}: "
                f"'metric/val/l2' reached {metrics['val_l2']:.5f} (best {best_val:.5f}), "
                f"saving model to '{ckpt_best}' as top 1"
            )

    run_test = bool(cfg.get("test", {}).get("run_after_train", True))
    if run_test and (test_loader is not None):
        test_metrics = evaluate_test_semgaze_metrics(model, test_loader, cfg, device)
        tqdm.write("[test]")
        for k, v in test_metrics.items():
            tqdm.write(f"  {k:<24} {v:.10f}")
        if run is not None:
            wandb.log(test_metrics)

    if run is not None:
        wandb.finish()
