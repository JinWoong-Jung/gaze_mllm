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
    wandb.define_metric("metric/val/dist", summary="min")
    wandb.define_metric("metric/val/l2", summary="min")
    wandb.define_metric("metric/val/inout_acc", summary="max")
    wandb.define_metric("loss/val", summary="min")
    wandb.define_metric("loss/val/heatmap", summary="min")
    wandb.define_metric("loss/val/coord", summary="min")
    wandb.define_metric("loss/val/angular", summary="min")
    wandb.define_metric("loss/val/label", summary="min")
    wandb.define_metric("loss/train", summary="min")
    return run


def _log_metric_bar_chart(run, metrics: Dict[str, float], chart_key: str, title: str, epoch: Optional[int] = None):
    if (run is None) or (wandb is None):
        return
    rows = []
    for k, v in metrics.items():
        if not str(k).startswith("metric/"):
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if not math.isfinite(fv):
            continue
        rows.append([str(k), fv])
    if len(rows) == 0:
        return
    table = wandb.Table(columns=["metric", "value"], data=rows)
    payload = {
        chart_key: wandb.plot.bar(table, "metric", "value", title=title),
    }
    if epoch is not None:
        payload["epoch"] = int(epoch)
    wandb.log(payload)


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


def _assert_head_only_eval_state(model: Qwen3VLGazeModel, cfg: Dict):
    train_mode = str(cfg.get("model", {}).get("train_mode", "")).lower()
    if train_mode != "head_only":
        return
    assert not model.backbone.training, "head_only mode requires backbone.eval() during training"
    if model.dino_encoder is not None:
        assert not model.dino_encoder.training, "head_only mode requires DINO eval() during training"


def _validate_data_config(cfg: Dict):
    data_cfg = cfg["data"]
    use_precomputed = bool(data_cfg.get("use_precomputed_dino_features", False))
    if use_precomputed:
        required = [("train", data_cfg.get("dino_feature_h5_train")), ("val", data_cfg.get("dino_feature_h5_val"))]
        if data_cfg.get("test_annotation"):
            required.append(("test", data_cfg.get("dino_feature_h5_test")))

        missing = []
        for split, path in required:
            if (path is None) or (str(path).strip() == "") or (not os.path.exists(path)):
                missing.append((split, path))
        if missing:
            details = ", ".join([f"{split}={path}" for split, path in missing])
            raise ValueError(
                "data.use_precomputed_dino_features=true 이지만 유효한 H5 경로가 없습니다. "
                f"누락/미존재: {details}"
            )

    use_cached_qwen = bool(data_cfg.get("use_cached_qwen_hidden", False))
    if use_cached_qwen:
        qwen_required = [("train", data_cfg.get("qwen_hidden_h5_train")), ("val", data_cfg.get("qwen_hidden_h5_val"))]
        if data_cfg.get("test_annotation"):
            qwen_required.append(("test", data_cfg.get("qwen_hidden_h5_test")))

        qwen_missing = []
        for split, path in qwen_required:
            if (path is None) or (str(path).strip() == "") or (not os.path.exists(path)):
                qwen_missing.append((split, path))
        if qwen_missing:
            details = ", ".join([f"{split}={path}" for split, path in qwen_missing])
            raise ValueError(
                "data.use_cached_qwen_hidden=true 이지만 유효한 Qwen hidden H5 경로가 없습니다. "
                f"누락/미존재: {details}"
            )


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
        use_cached_qwen_hidden=bool(data_cfg.get("use_cached_qwen_hidden", False)),
        qwen_hidden_h5_path=data_cfg.get("qwen_hidden_h5_train"),
        include_mark_image=data_cfg.get("include_mark_image", True),
        include_head_image=data_cfg.get("include_head_image", True),
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
        use_cached_qwen_hidden=bool(data_cfg.get("use_cached_qwen_hidden", False)),
        qwen_hidden_h5_path=data_cfg.get("qwen_hidden_h5_val"),
        include_mark_image=data_cfg.get("include_mark_image", True),
        include_head_image=data_cfg.get("include_head_image", True),
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
            use_cached_qwen_hidden=bool(data_cfg.get("use_cached_qwen_hidden", False)),
            qwen_hidden_h5_path=data_cfg.get("qwen_hidden_h5_test"),
            include_mark_image=data_cfg.get("include_mark_image", True),
            include_head_image=data_cfg.get("include_head_image", True),
            include_reason_text=data_cfg.get("include_reason_text", False),
        )

    collator = QwenVLBatchCollator(
        processor=processor,
        base_prompt=prompt_cfg["base"],
        include_mark_image=data_cfg.get("include_mark_image", True),
        include_head_image=data_cfg.get("include_head_image", True),
        use_cached_qwen_hidden=bool(data_cfg.get("use_cached_qwen_hidden", False)),
        cached_qwen_missing_policy=data_cfg.get("cached_qwen_missing_policy", "error"),
        qwen_image_size=int(data_cfg.get("qwen_image_size", 256)),
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
    data_cfg = cfg["data"]
    heads_cfg = cfg.get("heads", {})
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
        cache_dir=model_cfg.get("cache_dir", None),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        use_precomputed_dino_features=bool(data_cfg.get("use_precomputed_dino_features", False)),
        dino_hidden_size_override=int(model_cfg.get("dino_hidden_size", 768)),
        enabled_heads=heads_cfg.get("enabled", None),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"].get("weight_decay", 0.01),
    )
    return TrainArtifacts(model=model, optimizer=optimizer)


def to_device(batch: Optional[Dict[str, torch.Tensor]], device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
    if batch is None:
        return None
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
def evaluate(model: Qwen3VLGazeModel, loader: DataLoader, cfg: Dict, device: torch.device, loss_cfg: Dict) -> Dict[str, float]:
    model.eval()
    amp_enabled, amp_dtype, _ = resolve_amp(cfg["train"], device)
    use_pbar = bool(cfg["train"].get("progress_bar", True))

    total_dist = 0.0
    total_in = 0
    total_inout_correct = 0
    total_n = 0
    has_pred_xy = False
    has_pred_inout = False
    num_batches = 0
    loss_sums = {
        "total": 0.0,
        "heatmap": 0.0,
        "coord": 0.0,
        "vec": 0.0,
        "angle": 0.0,
        "inout": 0.0,
        "reason": 0.0,
        "label": 0.0,
    }

    iterator = tqdm(
        loader,
        total=len(loader),
        desc="val",
        dynamic_ncols=True,
        leave=False,
        disable=not use_pbar,
    )
    for batch in iterator:
        batch = to_device(batch, device)
        if batch is None:
            continue
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            preds = model(batch)
            losses = compute_losses(preds, batch, loss_cfg, enabled_heads=getattr(model, "enabled_heads", None))
        num_batches += 1
        for k in loss_sums.keys():
            loss_sums[k] += float(losses[k].item())
        pred_xy = None
        if "gaze_heatmap_logit" in preds:
            pred_heat = torch.sigmoid(preds["gaze_heatmap_logit"])
            pred_xy = _heatmap_argmax_xy(pred_heat)
            has_pred_xy = True
        elif "gaze_xy" in preds:
            pred_xy = preds["gaze_xy"]
            has_pred_xy = True

        in_mask = (batch["inout"] > 0.5)
        if (pred_xy is not None) and in_mask.any():
            dist = torch.norm(pred_xy[in_mask] - batch["gaze_xy"][in_mask], dim=-1)
            total_dist += dist.sum().item()
            total_in += int(in_mask.sum().item())

        if "inout_logit" in preds:
            has_pred_inout = True
            inout_pred = (torch.sigmoid(preds["inout_logit"]) > 0.5).float()
            total_inout_correct += int((inout_pred == batch["inout"]).sum().item())
            total_n += int(batch["inout"].numel())

        if use_pbar and has_pred_xy:
            iterator.set_postfix(dist=f"{(total_dist / max(1, total_in)):.4f}")

    val_dist = (total_dist / max(1, total_in)) if (has_pred_xy and total_in > 0) else float("nan")
    val_inout = (total_inout_correct / max(1, total_n)) if (has_pred_inout and total_n > 0) else float("nan")
    mean_losses = {k: (v / max(1, num_batches)) for k, v in loss_sums.items()}
    return {
        # semgaze-style keys
        "metric/val/dist": val_dist,
        "metric/val/inout_acc": val_inout,
        "loss/val": mean_losses["total"],
        "loss/val/heatmap": mean_losses["heatmap"],
        "loss/val/coord": mean_losses["coord"],
        "loss/val/vec": mean_losses["vec"],
        "loss/val/angular": mean_losses["angle"],
        "loss/val/inout": mean_losses["inout"],
        "loss/val/reason": mean_losses["reason"],
        "loss/val/label": mean_losses["label"],
        # backward-compatible aliases
        "metric/val/l2": val_dist,
        "metric/val_l2": val_dist,
        "metric/val_inout_acc": val_inout,
    }


@torch.no_grad()
def evaluate_test_semgaze_metrics(model: Qwen3VLGazeModel, loader: DataLoader, cfg: Dict, device: torch.device) -> Dict[str, float]:
    model.eval()
    amp_enabled, amp_dtype, _ = resolve_amp(cfg["train"], device)
    vocab_emb = _load_vocab_embeddings(cfg, device)
    use_pbar = bool(cfg["train"].get("progress_bar", True))

    sum_dist_to_avg = 0.0
    sum_avg_dist = 0.0
    sum_min_dist = 0.0
    sum_auc = 0.0
    n_dist_obs = 0
    n_auc_obs = 0

    acc1_correct = 0
    acc3_correct = 0
    acc_total = 0

    multi_acc1_correct = 0
    multi_total = 0

    iterator = tqdm(
        loader,
        total=len(loader),
        desc="test",
        dynamic_ncols=True,
        leave=False,
        disable=not use_pbar,
    )
    for batch in iterator:
        batch = to_device(batch, device)
        if batch is None:
            continue
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            preds = model(batch)

        pred_heat = None
        pred_xy = None
        if "gaze_heatmap_logit" in preds:
            pred_heat = torch.sigmoid(preds["gaze_heatmap_logit"])
            pred_xy = _heatmap_argmax_xy(pred_heat)
        elif "gaze_xy" in preds:
            pred_xy = preds["gaze_xy"]

        if (vocab_emb is not None) and ("label_emb" in preds):
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

        if pred_xy is None:
            continue
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
            n_dist_obs += 1

            if pred_heat is not None:
                hm_gt = _binary_gt_heatmap(gp_gt, size=64)
                auc = binary_auroc(pred_heat[i].flatten(), hm_gt.flatten().long())
                sum_auc += float(auc.item())
                n_auc_obs += 1

        if use_pbar:
            iterator.set_postfix(
                auc=f"{(sum_auc / max(1, n_auc_obs)):.4f}" if n_auc_obs > 0 else "nan",
                avg_dist=f"{(sum_avg_dist / max(1, n_dist_obs)):.4f}" if n_dist_obs > 0 else "nan",
            )

    metrics = {
        "metric/test/acc@1": (acc1_correct / max(1, acc_total)) if acc_total > 0 else float("nan"),
        "metric/test/acc@3": (acc3_correct / max(1, acc_total)) if acc_total > 0 else float("nan"),
        "metric/test/auc": (sum_auc / max(1, n_auc_obs)) if n_auc_obs > 0 else float("nan"),
        "metric/test/dist_to_avg": (sum_dist_to_avg / max(1, n_dist_obs)) if n_dist_obs > 0 else float("nan"),
        "metric/test/avg_dist": (sum_avg_dist / max(1, n_dist_obs)) if n_dist_obs > 0 else float("nan"),
        "metric/test/min_dist": (sum_min_dist / max(1, n_dist_obs)) if n_dist_obs > 0 else float("nan"),
        "metric/test/multi_acc@1": (multi_acc1_correct / max(1, multi_total)) if multi_total > 0 else float("nan"),
    }
    return metrics


def train_loop(cfg: Dict):
    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)
    device = torch.device(cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    _validate_data_config(cfg)

    artifacts = build_train_artifacts(cfg, device)
    train_loader, val_loader, test_loader = build_dataloaders(cfg, artifacts.model.processor)

    model = artifacts.model
    optimizer = artifacts.optimizer
    if bool(cfg["train"].get("print_model_summary", True)):
        _print_model_summary(model)
    run = _init_wandb(cfg)
    wb_cfg = cfg.get("wandb", {})
    wb_watch = str(wb_cfg.get("watch", "none")).lower()
    if (run is not None) and (wb_watch in {"gradients", "parameters", "all"}):
        wandb.watch(model, log=wb_watch, log_freq=int(wb_cfg.get("watch_freq", 500)))

    data_cfg = cfg["data"]
    use_precomputed = bool(data_cfg.get("use_precomputed_dino_features", False))
    use_cached_qwen = bool(data_cfg.get("use_cached_qwen_hidden", False))
    include_mark = bool(data_cfg.get("include_mark_image", True))
    include_head = bool(data_cfg.get("include_head_image", True))
    enabled_heads = sorted(list(getattr(model, "enabled_heads", set())))
    print(
        "[setup] "
        f"train_dino={cfg['model'].get('train_dino', False)} "
        f"dino={cfg['model'].get('dino_name')} "
        f"local_files_only={bool(cfg['model'].get('local_files_only', False))} "
        f"cache_dir={cfg['model'].get('cache_dir', None)} "
        f"head_image={include_head} "
        f"mark_image={include_mark} "
        f"precomputed_dino={use_precomputed} "
        f"cached_qwen_hidden={use_cached_qwen} "
        f"enabled_heads={enabled_heads}"
    )
    if use_precomputed:
        print(
            "[setup] dino_h5 "
            f"train={data_cfg.get('dino_feature_h5_train')} "
            f"val={data_cfg.get('dino_feature_h5_val')} "
            f"test={data_cfg.get('dino_feature_h5_test')}"
        )
    if use_cached_qwen:
        print(
            "[setup] qwen_hidden_h5 "
            f"train={data_cfg.get('qwen_hidden_h5_train')} "
            f"val={data_cfg.get('qwen_hidden_h5_val')} "
            f"test={data_cfg.get('qwen_hidden_h5_test')} "
            f"missing_policy={data_cfg.get('cached_qwen_missing_policy', 'error')}"
        )

    loss_cfg = {
        "heatmap": float(cfg["loss"].get("w_heatmap", 1.0)),
        "coord": float(cfg["loss"].get("w_coord", 1.0)),
        "vec": float(cfg["loss"].get("w_vec", 0.5)),
        "angle": float(cfg["loss"].get("w_angle", 1.0)),
        "inout": float(cfg["loss"].get("w_inout", 1.0)),
        "reason": float(cfg["loss"].get("w_reason", 0.3)),
        "label": float(cfg["loss"].get("w_label", 0.3)),
        "reason_loss_type": cfg["loss"].get("reason_loss_type", "cosine"),
        "reason_nce_temperature": float(cfg["loss"].get("reason_nce_temperature", 0.07)),
        "heatmap_size": int(cfg["loss"].get("heatmap_size", 64)),
        "heatmap_sigma": float(cfg["loss"].get("heatmap_sigma", 3.0)),
    }

    amp_enabled, amp_dtype, use_scaler = resolve_amp(cfg["train"], device)
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    update_steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum))
    total_steps = update_steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = int(total_steps * cfg["optim"].get("warmup_ratio", 0.03))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1.0, float(warmup_steps))
        progress = float(step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(
        "[setup] "
        f"grad_accum={grad_accum} updates/epoch={update_steps_per_epoch} "
        f"total_updates={total_steps} warmup_updates={warmup_steps}"
    )
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
                if batch is None:
                    continue
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    _ = model(batch)
        model.train()
        _assert_head_only_eval_state(model, cfg)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        _assert_head_only_eval_state(model, cfg)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        seen_batches = 0
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
            if batch is None:
                continue
            _assert_head_only_eval_state(model, cfg)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                preds = model(batch)
                losses = compute_losses(preds, batch, loss_cfg, enabled_heads=getattr(model, "enabled_heads", None))
                loss = losses["total"] / grad_accum
            seen_batches += 1

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (seen_batches % grad_accum) == 0:
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
                avg = running / max(1, seen_batches)
                iterator.set_postfix(
                    loss=f"{avg:.4f}",
                    heatmap=f"{float(losses['heatmap'].item()):.4f}",
                    coord=f"{float(losses['coord'].item()):.4f}",
                    angle=f"{float(losses['angle'].item()):.4f}",
                    lr=f"{float(scheduler.get_last_lr()[0]):.2e}",
                )

            if seen_batches > 0 and (seen_batches % cfg["train"].get("log_every", 20) == 0):
                avg = running / max(1, seen_batches)
                # Avoid breaking tqdm rendering: when progress bar is active, keep metrics in postfix only.
                if not use_pbar:
                    print(
                        f"epoch={epoch+1} step={seen_batches}/{len(train_loader)} "
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

        # Flush last partial accumulation so remainder micro-batches are not dropped.
        if (seen_batches % max(1, grad_accum)) != 0:
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

        metrics = evaluate(model, val_loader, cfg, device, loss_cfg)
        tqdm.write(
            f"[val] epoch={epoch+1} "
            f"dist={metrics['metric/val/dist']:.4f} "
            f"inout_acc={metrics['metric/val/inout_acc']:.4f} "
            f"loss={metrics['loss/val']:.4f}"
        )
        if run is not None:
            metrics_to_log = dict(metrics)
            metrics_to_log["epoch"] = epoch + 1
            wandb.log(metrics_to_log)
            _log_metric_bar_chart(
                run=run,
                metrics=metrics,
                chart_key="charts/val_metrics_bar",
                title=f"Validation Metrics (epoch {epoch+1})",
                epoch=epoch + 1,
            )

        ckpt_last = os.path.join(cfg["train"]["output_dir"], "last.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}, ckpt_last)

        if metrics["metric/val/dist"] < best_val:
            best_val = metrics["metric/val/dist"]
            ckpt_best = os.path.join(cfg["train"]["output_dir"], "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "metrics": metrics}, ckpt_best)
            tqdm.write(
                f"Epoch {epoch+1}, global step {global_step}: "
                f"'metric/val/dist' reached {metrics['metric/val/dist']:.5f} (best {best_val:.5f}), "
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
            _log_metric_bar_chart(
                run=run,
                metrics=test_metrics,
                chart_key="charts/test_metrics_bar",
                title="Test Metrics",
            )

    if run is not None:
        wandb.finish()
