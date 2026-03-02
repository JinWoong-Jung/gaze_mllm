import argparse
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, DINOv3ViTModel


TRAIN_VAL_COLUMNS = [
    "path", "id", "body_x", "body_y", "body_w", "body_h",
    "eye_x", "eye_y", "gaze_x", "gaze_y",
    "head_xmin", "head_ymin", "head_xmax", "head_ymax",
    "inout", "origin", "meta",
]

TEST_COLUMNS = [
    "path", "id", "body_x", "body_y", "body_w", "body_h",
    "eye_x", "eye_y", "gaze_x", "gaze_y",
    "head_xmin", "head_ymin", "head_xmax", "head_ymax",
    "origin", "meta",
]


def key_from_path(split: str, rel_path: str, sample_id: int) -> str:
    rel_dir = os.path.dirname(rel_path)
    split_prefix = f"{split}/"
    if rel_dir.startswith(split_prefix):
        rel_dir = rel_dir[len(split_prefix):]
    base = os.path.splitext(os.path.basename(rel_path))[0]
    return os.path.join(rel_dir, f"{base}_{sample_id}")


def load_rows(annotation_path: str, split: str) -> pd.DataFrame:
    if split in {"train", "val"}:
        return pd.read_csv(annotation_path, sep=",", names=TRAIN_VAL_COLUMNS, encoding="utf-8-sig")
    if split == "test":
        return pd.read_csv(annotation_path, sep=",", names=TEST_COLUMNS, encoding="utf-8-sig")
    raise ValueError(f"Unsupported split: {split}")


def build_samples(df: pd.DataFrame, split: str, image_root: str, mark_root: str) -> List[Tuple[str, str, str]]:
    samples = []
    for _, row in df.iterrows():
        rel_path = str(row["path"])
        sample_id = int(row["id"])
        key = key_from_path(split, rel_path, sample_id)

        scene_path = os.path.join(image_root, rel_path)
        rel_dir = os.path.dirname(key)
        stem = os.path.basename(key)
        mark_path = os.path.join(mark_root, split, rel_dir, stem + ".jpg")

        if not os.path.exists(scene_path):
            continue
        if not os.path.exists(mark_path):
            mark_path = scene_path
        samples.append((key, scene_path, mark_path))
    return samples


def open_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def encode_batch(model, processor, paths: List[str], device: torch.device) -> torch.Tensor:
    images = [open_rgb(p) for p in paths]
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs, return_dict=True)
    cls = out.last_hidden_state[:, 0].to(torch.float32)
    return cls.cpu()


def main(args):
    df = load_rows(args.annotation, args.split)
    samples = build_samples(df, args.split, args.image_root, args.mark_root)
    print(f"Loaded samples: {len(samples)}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = AutoImageProcessor.from_pretrained(args.dino_name)
    model = DINOv3ViTModel.from_pretrained(args.dino_name, torch_dtype=getattr(torch, args.dtype)).to(device)
    model.eval()

    out_path = Path(args.output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and args.overwrite:
        out_path.unlink()

    batch_size = args.batch_size
    n = len(samples)
    hidden = int(model.config.hidden_size)

    with h5py.File(out_path, "w") as f:
        str_dtype = h5py.string_dtype(encoding="utf-8")
        keys_ds = f.create_dataset("keys", shape=(n,), dtype=str_dtype)
        scene_ds = f.create_dataset("scene_embeddings", shape=(n, hidden), dtype="float32")
        mark_ds = f.create_dataset("mark_embeddings", shape=(n, hidden), dtype="float32")

        for s in range(0, n, batch_size):
            e = min(n, s + batch_size)
            chunk = samples[s:e]
            keys = [x[0] for x in chunk]
            scene_paths = [x[1] for x in chunk]
            mark_paths = [x[2] for x in chunk]

            scene_emb = encode_batch(model, processor, scene_paths, device)
            mark_emb = encode_batch(model, processor, mark_paths, device)

            keys_ds[s:e] = keys
            scene_ds[s:e] = scene_emb.numpy()
            mark_ds[s:e] = mark_emb.numpy()

            if (s // batch_size) % 10 == 0:
                print(f"processed {e}/{n}")

        f.attrs["dino_name"] = args.dino_name
        f.attrs["split"] = args.split

    print(f"saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract DINOv3 scene/mark embeddings into h5")
    ap.add_argument("--annotation", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)
    ap.add_argument("--mark_root", type=str, required=True)
    ap.add_argument("--split", type=str, choices=["train", "val", "test"], required=True)
    ap.add_argument("--output_h5", type=str, required=True)
    ap.add_argument("--dino_name", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--overwrite", action="store_true")
    main(ap.parse_args())
