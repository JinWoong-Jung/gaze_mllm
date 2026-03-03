import argparse
import os
from pathlib import Path
from typing import Dict
import sys

import h5py
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gaze_mllm.datasets.gazefollow_reason_dataset import GazeFollowReasonDataset, QwenVLBatchCollator
from gaze_mllm.models.qwen_gaze_model import Qwen3VLGazeModel, masked_mean


def _resolve_hidden_size(cfg) -> int:
    hs = getattr(cfg, "hidden_size", None)
    if hs is not None:
        return int(hs)
    text_cfg = getattr(cfg, "text_config", None)
    hs = getattr(text_cfg, "hidden_size", None) if text_cfg is not None else None
    if hs is not None:
        return int(hs)
    raise ValueError("Could not resolve Qwen hidden size from model config.")


def _pick_annotation_path(data_cfg: Dict, split: str) -> str:
    key = f"{split}_annotation"
    p = data_cfg.get(key)
    if not p:
        raise ValueError(f"Missing data.{key} in config")
    return p


def _to_device_inputs(batch: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        if k in batch and isinstance(batch[k], torch.Tensor):
            out[k] = batch[k].to(device, non_blocking=True)
    return out


@torch.no_grad()
def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    prompt_cfg = cfg["prompt"]
    split = args.split

    annotation_path = _pick_annotation_path(data_cfg, split)
    image_root = data_cfg["image_root"]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(str(model_cfg.get("dtype", "bfloat16")).lower(), torch.bfloat16)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    processor = AutoProcessor.from_pretrained(
        model_cfg["name"],
        trust_remote_code=True,
        cache_dir=model_cfg.get("cache_dir", None),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    )
    backbone = Qwen3VLGazeModel._load_backbone(
        model_name=model_cfg["name"],
        dtype=dtype,
        cache_dir=model_cfg.get("cache_dir", None),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
    ).to(device)
    backbone.eval()

    dataset = GazeFollowReasonDataset(
        split=split,
        annotation_path=annotation_path,
        image_root=image_root,
        label_csv_path=None,
        vocab2id_path=None,
        label_embed_root=None,
        reason_output_root=data_cfg.get("reason_output_root"),
        reason_mark_root=data_cfg.get("reason_mark_root"),
        reason_prompt_root=data_cfg.get("reason_prompt_root"),
        reason_feature_root=None,
        reason_feature_h5_path=None,
        reason_feature_dim=int(cfg["loss"].get("reason_dim", 1024)),
        use_precomputed_dino_features=False,
        dino_feature_h5_path=None,
        use_cached_qwen_hidden=False,
        qwen_hidden_h5_path=None,
        include_mark_image=bool(data_cfg.get("include_mark_image", True)),
        include_head_image=bool(data_cfg.get("include_head_image", True)),
        include_reason_text=bool(data_cfg.get("include_reason_text", False)),
    )

    collator = QwenVLBatchCollator(
        processor=processor,
        base_prompt=prompt_cfg["base"],
        include_mark_image=bool(data_cfg.get("include_mark_image", True)),
        include_head_image=bool(data_cfg.get("include_head_image", True)),
        use_cached_qwen_hidden=False,
        cached_qwen_missing_policy="error",
        qwen_image_size=int(data_cfg.get("qwen_image_size", 256)),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collator,
    )

    out_path = Path(args.output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if args.overwrite:
            out_path.unlink()
        else:
            raise FileExistsError(f"Output already exists: {out_path}. Use --overwrite.")

    hidden_size = _resolve_hidden_size(backbone.config)
    n = len(dataset)
    write_pos = 0

    with h5py.File(out_path, "w") as f:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        keys_ds = f.create_dataset("keys", shape=(n,), maxshape=(None,), dtype=str_dtype)
        sample_ids_ds = f.create_dataset("sample_ids", shape=(n,), maxshape=(None,), dtype="int64")
        emb_ds = f.create_dataset("embeddings", shape=(n, hidden_size), maxshape=(None, hidden_size), dtype="float32")

        for batch in tqdm(loader, total=len(loader), desc=f"cache-qwen-{split}", dynamic_ncols=True):
            if batch is None:
                continue
            model_inputs = _to_device_inputs(batch, device)
            outputs = backbone(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask", None),
                pixel_values=model_inputs.get("pixel_values", None),
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1]
            attn = model_inputs.get(
                "attention_mask",
                torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long),
            )
            pooled = masked_mean(hidden, attn).to(torch.float32).cpu()

            keys = [str(x) for x in batch["cache_key"]]
            sample_ids = batch["sample_id"].to(torch.int64).cpu()
            m = len(keys)
            keys_ds[write_pos:write_pos + m] = keys
            sample_ids_ds[write_pos:write_pos + m] = sample_ids.numpy()
            emb_ds[write_pos:write_pos + m] = pooled.numpy()
            write_pos += m

        keys_ds.resize((write_pos,))
        sample_ids_ds.resize((write_pos,))
        emb_ds.resize((write_pos, hidden_size))

        f.attrs["model_name"] = str(model_cfg["name"])
        f.attrs["split"] = split
        f.attrs["num_input_samples"] = int(n)
        f.attrs["num_written_samples"] = int(write_pos)
        f.attrs["qwen_image_size"] = int(data_cfg.get("qwen_image_size", 256))
        f.attrs["include_mark_image"] = int(bool(data_cfg.get("include_mark_image", True)))
        f.attrs["include_head_image"] = int(bool(data_cfg.get("include_head_image", True)))

    print(f"saved: {out_path}")
    print(f"written samples: {write_pos}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cache pooled Qwen hidden states into H5")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    ap.add_argument("--output_h5", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
