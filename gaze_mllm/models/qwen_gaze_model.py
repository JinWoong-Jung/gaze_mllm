from typing import Dict, Optional, Sequence, Set
import os

import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, DINOv3ViTModel


ALL_HEADS = ("heatmap", "inout", "label", "coord", "reason", "angle")


def _normalize_enabled_heads(enabled_heads: Optional[Sequence[str]]) -> Set[str]:
    if enabled_heads is None:
        return set(ALL_HEADS)
    normalized = set()
    for head in enabled_heads:
        h = str(head).strip().lower()
        if h:
            normalized.add(h)
    unknown = sorted([h for h in normalized if h not in ALL_HEADS])
    if unknown:
        raise ValueError(f"Unsupported heads in heads.enabled: {unknown}. supported={list(ALL_HEADS)}")
    return normalized


def _resolve_qwen_hidden_size(cfg, default: int = 2048) -> int:
    # Qwen-VL configs may expose hidden size at top-level or under text_config.
    hs = getattr(cfg, "hidden_size", None)
    if hs is not None:
        return int(hs)
    text_cfg = getattr(cfg, "text_config", None)
    hs = getattr(text_cfg, "hidden_size", None) if text_cfg is not None else None
    if hs is not None:
        return int(hs)
    return int(default)


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def build_gaussian_heatmaps(
    gaze_xy: torch.Tensor,
    inout: torch.Tensor,
    size: int = 64,
    sigma: float = 3.0,
) -> torch.Tensor:
    b = gaze_xy.size(0)
    y = torch.arange(size, device=gaze_xy.device).float()
    x = torch.arange(size, device=gaze_xy.device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    out = torch.zeros((b, size, size), device=gaze_xy.device, dtype=torch.float32)
    for i in range(b):
        if inout[i] <= 0.5:
            continue
        cx = gaze_xy[i, 0].clamp(0, 1) * (size - 1)
        cy = gaze_xy[i, 1].clamp(0, 1) * (size - 1)
        out[i] = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return out


def _from_pretrained_with_dtype(cls, model_name: str, dtype: torch.dtype, **kwargs):
    # Newer transformers prefer `dtype`; older ones use `torch_dtype`.
    try:
        return cls.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_name, torch_dtype=dtype, **kwargs)


class Qwen3VLGazeModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        torch_dtype: str = "bfloat16",
        dino_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        train_dino: bool = False,
        use_gradient_checkpointing: bool = True,
        train_mode: str = "lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        reason_dim: int = 768,
        label_dim: int = 512,
        angle_feature_dim: int = 512,
        cache_dir: str = None,
        local_files_only: bool = False,
        use_precomputed_dino_features: bool = False,
        dino_hidden_size_override: int = 768,
        enabled_heads: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.train_mode = str(train_mode)
        self.enabled_heads = _normalize_enabled_heads(enabled_heads)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        if local_files_only and (not os.path.exists(model_name)):
            raise FileNotFoundError(
                f"local_files_only=True but model path does not exist: {model_name}. "
                "Use a local model directory path."
            )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.backbone = self._load_backbone(
            model_name,
            dtype,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        self.use_precomputed_dino_features = bool(use_precomputed_dino_features)
        # Shared DINOv3 encoder; scene/mark are forwarded separately.
        # If precomputed DINO features are always provided, skip loading the DINO model itself.
        self.dino_processor = None
        self.dino_encoder = None
        if not self.use_precomputed_dino_features:
            self.dino_processor = AutoImageProcessor.from_pretrained(
                dino_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            self.dino_encoder = _from_pretrained_with_dtype(
                DINOv3ViTModel,
                dino_name,
                dtype,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        if use_gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        # Only checkpoint DINO when it is trainable; otherwise this can emit warnings.
        if self.dino_encoder is not None and use_gradient_checkpointing and train_dino and hasattr(self.dino_encoder, "gradient_checkpointing_enable"):
            self.dino_encoder.gradient_checkpointing_enable()

        self.qwen_hidden_size = _resolve_qwen_hidden_size(self.backbone.config, default=2048)
        if self.dino_encoder is not None:
            self.dino_hidden_size = int(getattr(self.dino_encoder.config, "hidden_size", dino_hidden_size_override))
        else:
            self.dino_hidden_size = int(dino_hidden_size_override)

        # Inject DINO(scene, mark) tokens into MLLM sequence by cross-attention.
        self.dino_to_qwen = nn.Linear(self.dino_hidden_size, self.qwen_hidden_size)
        self.cond_attn = nn.MultiheadAttention(
            embed_dim=self.qwen_hidden_size,
            num_heads=8,
            batch_first=True,
        )
        self.cond_ln = nn.LayerNorm(self.qwen_hidden_size)

        self.hidden_size = self.qwen_hidden_size
        self.heatmap_size = 64
        self.coord_head = None
        self.inout_head = None
        self.reason_head = None
        self.label_head = None
        self.heatmap_head = None
        if "coord" in self.enabled_heads:
            self.coord_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.hidden_size // 2, 2),
            )
        if "inout" in self.enabled_heads:
            self.inout_head = nn.Linear(self.hidden_size, 1)
        if "reason" in self.enabled_heads:
            self.reason_head = nn.Linear(self.hidden_size, reason_dim)
        if "label" in self.enabled_heads:
            self.label_head = nn.Linear(self.hidden_size, label_dim)
        if "heatmap" in self.enabled_heads:
            self.heatmap_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.GELU(),
                nn.Linear(self.hidden_size // 2, self.heatmap_size * self.heatmap_size),
            )

        # semgaze style 2-MLP gaze angle predictor: Linear-ReLU-Linear-Tanh + normalize.
        self.gaze_predictor = None
        if "angle" in self.enabled_heads:
            self.gaze_predictor = nn.Sequential(
                nn.Linear(self.dino_hidden_size * 2, angle_feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(angle_feature_dim, 2),
                nn.Tanh(),
            )

        if self.train_mode == "lora":
            self._enable_lora(lora_r, lora_alpha, lora_dropout)
        elif self.train_mode == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True
        elif self.train_mode == "head_only":
            for p in self.backbone.parameters():
                p.requires_grad = False
        else:
            raise ValueError(f"Unsupported train_mode: {train_mode}")

        if self.dino_encoder is not None:
            for p in self.dino_encoder.parameters():
                p.requires_grad = bool(train_dino)

        self._logged_head_input = False
        # Enforce mode constraints immediately after init.
        self.train(self.training)

    def _enforce_head_only_eval_state(self):
        if self.train_mode != "head_only":
            return
        self.backbone.eval()
        if self.dino_encoder is not None:
            self.dino_encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep backbone/DINO frozen in eval mode during head-only optimization.
        if mode and (self.train_mode == "head_only"):
            self._enforce_head_only_eval_state()
        return self

    @staticmethod
    def _load_backbone(model_name: str, dtype: torch.dtype, cache_dir: str = None, local_files_only: bool = False):
        cfg = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        cfg_type = type(cfg)

        candidate_classes = [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModel",
        ]
        transformers_mod = importlib.import_module("transformers")
        attempts = []
        for cls_name in candidate_classes:
            cls = getattr(transformers_mod, cls_name, None)
            if cls is None:
                continue
            # Try only classes that explicitly support this config.
            try:
                if cfg_type not in cls._model_mapping.keys():
                    continue
            except Exception:
                # If mapping introspection fails, still try loading.
                pass
            try:
                return _from_pretrained_with_dtype(
                    cls,
                    model_name,
                    dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                )
            except Exception as exc:
                attempts.append((cls_name, exc))

        if attempts:
            cls_name, err = attempts[0]
            raise RuntimeError(
                f"Failed to load model={model_name} with {cls_name}. "
                f"first_error={type(err).__name__}: {err}"
            ) from err
        raise RuntimeError(
            f"Failed to load model={model_name}. "
            "No compatible AutoModel class found for its config."
        )

    def _enable_lora(self, lora_r: int, lora_alpha: int, lora_dropout: float):
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise RuntimeError("train_mode=lora requires `peft` to be installed") from exc

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _encode_dino(self, images, device: torch.device) -> torch.Tensor:
        if (self.dino_processor is None) or (self.dino_encoder is None):
            raise RuntimeError(
                "DINO encoder is not loaded. Enable online DINO or provide precomputed "
                "scene_dino_feat/mark_dino_feat for every sample."
            )
        dino_inputs = self.dino_processor(images=images, return_tensors="pt")
        dino_inputs = {k: v.to(device) for k, v in dino_inputs.items()}
        dino_outputs = self.dino_encoder(**dino_inputs, return_dict=True)
        return dino_outputs.last_hidden_state[:, 0]

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        use_cached_qwen_hidden = "qwen_pooled_hidden" in batch
        if use_cached_qwen_hidden:
            pooled = batch["qwen_pooled_hidden"]
            if pooled.dim() != 2:
                raise ValueError(f"qwen_pooled_hidden must be [B, H], got shape={tuple(pooled.shape)}")
            if pooled.size(-1) != self.hidden_size:
                raise ValueError(
                    f"qwen_pooled_hidden hidden mismatch: got={pooled.size(-1)} expected={self.hidden_size}"
                )
            query_hidden = pooled.unsqueeze(1)
        else:
            if ("head_in_qwen" in batch) and (not self._logged_head_input):
                head_tensor = batch["head_in_qwen"]
                head_count = int((head_tensor > 0.5).sum().item())
                total = int(head_tensor.numel())
                pixel_values = batch.get("pixel_values", None)
                pixel_shape = tuple(pixel_values.shape) if isinstance(pixel_values, torch.Tensor) else None
                print(
                    "[A1-check] "
                    f"head_in_qwen={head_count}/{total} "
                    f"pixel_values_shape={pixel_shape}"
                )
                self._logged_head_input = True

            outputs = self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask", None),
                pixel_values=batch.get("pixel_values", None),
                image_grid_thw=batch.get("image_grid_thw", None),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            attn = batch.get("attention_mask", torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long))
            query_hidden = hidden

        if "scene_dino_feat" in batch:
            scene_cls = batch["scene_dino_feat"].to(query_hidden.device)
            if "mark_dino_feat" in batch:
                mark_cls = batch["mark_dino_feat"].to(query_hidden.device)
            else:
                mark_cls = torch.zeros_like(scene_cls)
        else:
            scene_cls = self._encode_dino(batch["scene_images"], query_hidden.device)
            if "mark_images" in batch:
                mark_cls = self._encode_dino(batch["mark_images"], query_hidden.device)
            else:
                mark_cls = torch.zeros_like(scene_cls)

        dino_tokens = torch.stack([scene_cls, mark_cls], dim=1)
        dino_tokens = self.dino_to_qwen(dino_tokens)

        cond, _ = self.cond_attn(query=query_hidden, key=dino_tokens, value=dino_tokens)
        fused_hidden = self.cond_ln(query_hidden + cond)
        if use_cached_qwen_hidden:
            pooled = fused_hidden.squeeze(1)
        else:
            pooled = masked_mean(fused_hidden, attn)

        preds = {}
        if self.coord_head is not None:
            preds["gaze_xy"] = torch.sigmoid(self.coord_head(pooled))
        if self.inout_head is not None:
            preds["inout_logit"] = self.inout_head(pooled).squeeze(-1)
        if self.reason_head is not None:
            preds["reason_pred"] = F.normalize(self.reason_head(pooled), p=2, dim=-1)
        if self.label_head is not None:
            preds["label_emb"] = F.normalize(self.label_head(pooled), p=2, dim=-1)
        if self.heatmap_head is not None:
            preds["gaze_heatmap_logit"] = self.heatmap_head(pooled).view(-1, self.heatmap_size, self.heatmap_size)
        if self.gaze_predictor is not None:
            angle_in = torch.cat([scene_cls, mark_cls], dim=-1)
            preds["gaze_vec"] = F.normalize(self.gaze_predictor(angle_in), p=2, dim=-1)
        return preds


def compute_losses(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: Dict[str, float],
    enabled_heads: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    enabled = _normalize_enabled_heads(enabled_heads)
    in_mask = (targets["inout"] > 0.5)

    loss_coord = torch.tensor(0.0, device=targets["gaze_xy"].device)
    # vec loss intentionally disabled to avoid redundant direction supervision with angle head.
    loss_vec = torch.tensor(0.0, device=targets["gaze_xy"].device)
    if ("coord" in enabled) and ("gaze_xy" in preds) and in_mask.any():
        pred_xy = preds["gaze_xy"][in_mask]
        gt_xy = targets["gaze_xy"][in_mask]

        loss_coord = F.smooth_l1_loss(pred_xy, gt_xy)
        gt_vec = F.normalize(gt_xy - targets["eye_xy"][in_mask], p=2, dim=-1)
    else:
        gt_vec = F.normalize(targets["gaze_xy"] - targets["eye_xy"], p=2, dim=-1)

    loss_angle = torch.tensor(0.0, device=targets["gaze_xy"].device)
    if ("angle" in enabled) and ("gaze_vec" in preds) and in_mask.any():
        loss_angle = (1.0 - F.cosine_similarity(preds["gaze_vec"][in_mask], gt_vec, dim=-1)).mean()

    loss_inout = torch.tensor(0.0, device=targets["gaze_xy"].device)
    if ("inout" in enabled) and ("inout_logit" in preds):
        loss_inout = F.binary_cross_entropy_with_logits(preds["inout_logit"], targets["inout"])

    reason_valid = (targets["reason_valid"] > 0.5) & in_mask
    loss_reason = torch.tensor(0.0, device=targets["gaze_xy"].device)
    if ("reason" in enabled) and ("reason_pred" in preds) and reason_valid.any():
        pred = F.normalize(preds["reason_pred"][reason_valid], p=2, dim=-1)
        gt = F.normalize(targets["reason_feat"][reason_valid], p=2, dim=-1)
        reason_loss_type = str(weights.get("reason_loss_type", "cosine")).lower()
        if reason_loss_type == "mse":
            loss_reason = F.mse_loss(pred, gt)
        elif reason_loss_type == "infonce":
            temp = max(float(weights.get("reason_nce_temperature", 0.07)), 1e-6)
            logits = (pred @ gt.T) / temp
            labels = torch.arange(logits.size(0), device=logits.device)
            # Symmetric InfoNCE over in-batch reason embeddings.
            loss_reason = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        else:
            loss_reason = (1.0 - F.cosine_similarity(pred, gt, dim=-1)).mean()

    label_valid = in_mask & (targets["gaze_label_id"] >= 0)
    loss_label = torch.tensor(0.0, device=targets["gaze_xy"].device)
    if ("label" in enabled) and ("label_emb" in preds) and label_valid.any():
        pred = preds["label_emb"][label_valid]
        gt = F.normalize(targets["gaze_label_emb"][label_valid], p=2, dim=-1)
        loss_label = (1.0 - F.cosine_similarity(pred, gt, dim=-1)).mean()

    loss_heatmap = torch.tensor(0.0, device=targets["gaze_xy"].device)
    if ("heatmap" in enabled) and ("gaze_heatmap_logit" in preds):
        heatmap_size = int(weights.get("heatmap_size", 64))
        heatmap_sigma = float(weights.get("heatmap_sigma", 3.0))
        gt_heatmap = build_gaussian_heatmaps(targets["gaze_xy"], targets["inout"], size=heatmap_size, sigma=heatmap_sigma)
        loss_heatmap = F.binary_cross_entropy_with_logits(preds["gaze_heatmap_logit"], gt_heatmap)

    total = torch.tensor(0.0, device=targets["gaze_xy"].device)
    if "heatmap" in enabled:
        total = total + weights["heatmap"] * loss_heatmap
    if "coord" in enabled:
        total = total + weights["coord"] * loss_coord
    if "angle" in enabled:
        total = total + weights["angle"] * loss_angle
    if "inout" in enabled:
        total = total + weights["inout"] * loss_inout
    if "reason" in enabled:
        total = total + weights["reason"] * loss_reason
    if "label" in enabled:
        total = total + weights["label"] * loss_label
    if "vec" in weights:
        total = total + weights["vec"] * loss_vec

    return {
        "total": total,
        "coord": loss_coord.detach(),
        "vec": loss_vec.detach(),
        "angle": loss_angle.detach(),
        "heatmap": loss_heatmap.detach(),
        "inout": loss_inout.detach(),
        "reason": loss_reason.detach(),
        "label": loss_label.detach(),
    }
