import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


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


@dataclass
class SamplePaths:
    scene: str
    mark: Optional[str]
    prompt: Optional[str]
    reason_txt: Optional[str]
    reason_feat: Optional[str]


class GazeFollowReasonDataset(Dataset):
    def __init__(
        self,
        split: str,
        annotation_path: str,
        image_root: str,
        label_csv_path: Optional[str] = None,
        vocab2id_path: Optional[str] = None,
        label_embed_root: Optional[str] = None,
        reason_output_root: Optional[str] = None,
        reason_mark_root: Optional[str] = None,
        reason_prompt_root: Optional[str] = None,
        reason_feature_root: Optional[str] = None,
        reason_feature_h5_path: Optional[str] = None,
        reason_feature_dim: int = 1024,
        use_precomputed_dino_features: bool = False,
        dino_feature_h5_path: Optional[str] = None,
        use_cached_qwen_hidden: bool = False,
        qwen_hidden_h5_path: Optional[str] = None,
        include_mark_image: bool = True,
        include_head_image: bool = True,
        include_reason_text: bool = False,
        max_test_gaze_points: int = 20,
        max_test_label_ids: int = 5,
    ):
        self.split = split
        self.image_root = image_root
        self.reason_output_root = reason_output_root
        self.reason_mark_root = reason_mark_root
        self.reason_prompt_root = reason_prompt_root
        self.reason_feature_root = reason_feature_root
        self.reason_feature_h5_path = reason_feature_h5_path
        self.reason_feature_dim = int(reason_feature_dim)
        self.use_precomputed_dino_features = use_precomputed_dino_features
        self.dino_feature_h5_path = dino_feature_h5_path
        self.use_cached_qwen_hidden = bool(use_cached_qwen_hidden)
        self.qwen_hidden_h5_path = qwen_hidden_h5_path
        self.include_mark_image = include_mark_image
        self.include_head_image = include_head_image
        self.include_reason_text = include_reason_text
        self.max_test_gaze_points = max_test_gaze_points
        self.max_test_label_ids = max_test_label_ids
        self.reason_feature_h5 = None
        self.reason_feature_index = None
        self.dino_feature_h5 = None
        self.dino_feature_index = None
        self.qwen_hidden_h5 = None
        self.qwen_hidden_index = None

        self.label_embed_root = label_embed_root
        self.label_emb_cache: Dict[str, torch.Tensor] = {}

        self.vocab2id = {}
        if vocab2id_path and os.path.exists(vocab2id_path):
            with open(vocab2id_path, "r", encoding="utf-8") as f:
                self.vocab2id = json.load(f)

        self.label_df = None
        if label_csv_path and os.path.exists(label_csv_path):
            self.label_df = pd.read_csv(label_csv_path)

        if split in {"train", "val"}:
            self.df = pd.read_csv(annotation_path, sep=",", names=TRAIN_VAL_COLUMNS, encoding="utf-8-sig")
            self.df = self.df[self.df["inout"] != -1].reset_index(drop=True)
            if self.label_df is not None:
                self.df = pd.merge(self.df, self.label_df, how="left", on=["path", "id"])
            self.image_paths = None
        elif split == "test":
            self.df = pd.read_csv(annotation_path, sep=",", names=TEST_COLUMNS, encoding="utf-8-sig")
            self.df["inout"] = 1
            if self.label_df is not None:
                self.df = pd.merge(self.df, self.label_df, how="left", on=["path"])
            self.image_paths = sorted(self.df["path"].unique().tolist())
        else:
            raise ValueError(f"Unsupported split: {split}")

    def _reason_key(self, rel_path: str, sample_id: int) -> str:
        kp = self._key_parts(rel_path, sample_id)
        return os.path.join(kp["rel_dir"], kp["stem"])

    def _ensure_reason_h5(self) -> bool:
        if self.reason_feature_h5 is not None:
            return True
        if (self.reason_feature_h5_path is None) or (not os.path.exists(self.reason_feature_h5_path)):
            return False
        try:
            self.reason_feature_h5 = h5py.File(self.reason_feature_h5_path, "r")
            return True
        except Exception:
            self.reason_feature_h5 = None
            return False

    def _build_reason_h5_index(self) -> Dict[str, int]:
        if not self._ensure_reason_h5():
            return {}
        keys_ds = self.reason_feature_h5.get("keys")
        if keys_ds is None:
            return {}
        index = {}
        for i, key in enumerate(keys_ds):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            index[str(key)] = i
        return index

    def _load_reason_from_h5(self, key: str) -> Optional[torch.Tensor]:
        if not self._ensure_reason_h5():
            return None
        if self.reason_feature_index is None:
            self.reason_feature_index = self._build_reason_h5_index()
        row_idx = self.reason_feature_index.get(key)
        if row_idx is None:
            return None
        emb_ds = self.reason_feature_h5.get("embeddings")
        if emb_ds is None:
            return None
        try:
            emb = torch.from_numpy(emb_ds[row_idx]).to(torch.float32)
            return emb
        except Exception:
            return None

    def _fit_reason_dim(self, emb: torch.Tensor) -> torch.Tensor:
        emb = emb.to(torch.float32).flatten()
        d = emb.numel()
        if d == self.reason_feature_dim:
            return emb
        if d > self.reason_feature_dim:
            return emb[: self.reason_feature_dim]
        pad = torch.zeros(self.reason_feature_dim - d, dtype=torch.float32)
        return torch.cat([emb, pad], dim=0)

    def __del__(self):
        if self.reason_feature_h5 is not None:
            try:
                self.reason_feature_h5.close()
            except Exception:
                pass
        if self.dino_feature_h5 is not None:
            try:
                self.dino_feature_h5.close()
            except Exception:
                pass
        if self.qwen_hidden_h5 is not None:
            try:
                self.qwen_hidden_h5.close()
            except Exception:
                pass

    def _ensure_dino_h5(self) -> bool:
        if self.dino_feature_h5 is not None:
            return True
        if (not self.use_precomputed_dino_features) or (self.dino_feature_h5_path is None) or (not os.path.exists(self.dino_feature_h5_path)):
            return False
        try:
            self.dino_feature_h5 = h5py.File(self.dino_feature_h5_path, "r")
            return True
        except Exception:
            self.dino_feature_h5 = None
            return False

    def _build_dino_h5_index(self) -> Dict[str, int]:
        if not self._ensure_dino_h5():
            return {}
        keys_ds = self.dino_feature_h5.get("keys")
        if keys_ds is None:
            return {}
        index = {}
        for i, key in enumerate(keys_ds):
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            index[str(key)] = i
        return index

    def _load_dino_from_h5(self, key: str):
        if not self._ensure_dino_h5():
            return None, None
        if self.dino_feature_index is None:
            self.dino_feature_index = self._build_dino_h5_index()
        row_idx = self.dino_feature_index.get(key)
        if row_idx is None:
            return None, None
        scene_ds = self.dino_feature_h5.get("scene_embeddings")
        mark_ds = self.dino_feature_h5.get("mark_embeddings")
        if (scene_ds is None) or (mark_ds is None):
            return None, None
        try:
            scene = torch.from_numpy(scene_ds[row_idx]).to(torch.float32)
            mark = torch.from_numpy(mark_ds[row_idx]).to(torch.float32)
            return scene, mark
        except Exception:
            return None, None

    def _ensure_qwen_hidden_h5(self) -> bool:
        if self.qwen_hidden_h5 is not None:
            return True
        if (not self.use_cached_qwen_hidden) or (self.qwen_hidden_h5_path is None) or (not os.path.exists(self.qwen_hidden_h5_path)):
            return False
        try:
            self.qwen_hidden_h5 = h5py.File(self.qwen_hidden_h5_path, "r")
            return True
        except Exception:
            self.qwen_hidden_h5 = None
            return False

    def _build_qwen_hidden_h5_index(self) -> Dict[str, int]:
        if not self._ensure_qwen_hidden_h5():
            return {}
        keys_ds = self.qwen_hidden_h5.get("keys")
        index = {}
        if keys_ds is not None:
            for i, key in enumerate(keys_ds):
                if isinstance(key, bytes):
                    key = key.decode("utf-8")
                index[str(key)] = i

        sample_ids_ds = self.qwen_hidden_h5.get("sample_ids")
        if sample_ids_ds is not None:
            for i, sid in enumerate(sample_ids_ds):
                index[f"id:{int(sid)}"] = i
        return index

    def _load_qwen_hidden_from_h5(self, key: str, sample_id: Optional[int] = None) -> Optional[torch.Tensor]:
        if not self._ensure_qwen_hidden_h5():
            return None
        if self.qwen_hidden_index is None:
            self.qwen_hidden_index = self._build_qwen_hidden_h5_index()
        row_idx = self.qwen_hidden_index.get(key)
        if (row_idx is None) and (sample_id is not None):
            row_idx = self.qwen_hidden_index.get(f"id:{int(sample_id)}")
        if row_idx is None:
            return None
        emb_ds = self.qwen_hidden_h5.get("embeddings")
        if emb_ds is None:
            return None
        try:
            emb = torch.from_numpy(emb_ds[row_idx]).to(torch.float32).flatten()
            return emb
        except Exception:
            return None

    def __len__(self) -> int:
        if self.split == "test":
            return len(self.image_paths)
        return len(self.df)

    def _key_parts(self, rel_path: str, sample_id: int) -> Dict[str, str]:
        rel_dir = os.path.dirname(rel_path)
        split_prefix = f"{self.split}/"
        if rel_dir.startswith(split_prefix):
            rel_dir = rel_dir[len(split_prefix):]
        base = os.path.splitext(os.path.basename(rel_path))[0]
        stem = f"{base}_{sample_id}"
        return {"rel_dir": rel_dir, "stem": stem}

    def _resolve_paths(self, rel_path: str, sample_id: int) -> SamplePaths:
        kp = self._key_parts(rel_path, sample_id)
        scene = os.path.join(self.image_root, rel_path)

        mark = None
        if self.reason_mark_root is not None:
            mark = os.path.join(self.reason_mark_root, self.split, kp["rel_dir"], kp["stem"] + ".jpg")

        prompt = None
        if self.reason_prompt_root is not None:
            prompt = os.path.join(self.reason_prompt_root, self.split, kp["rel_dir"], kp["stem"] + ".txt")

        reason_txt = None
        if self.reason_output_root is not None:
            reason_txt = os.path.join(self.reason_output_root, self.split, kp["rel_dir"], kp["stem"] + ".txt")

        reason_feat = None
        if self.reason_feature_root is not None:
            reason_feat = os.path.join(self.reason_feature_root, self.split, kp["rel_dir"], kp["stem"] + ".pt")

        return SamplePaths(scene=scene, mark=mark, prompt=prompt, reason_txt=reason_txt, reason_feat=reason_feat)

    @staticmethod
    def _read_text(path: Optional[str]) -> str:
        if not path or not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def _safe_open_image(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _maybe_to_pixel(v: float, size: int) -> float:
        # Support both normalized [0,1] and absolute pixel bbox annotations.
        if 0.0 <= v <= 1.0:
            return v * float(max(1, size - 1))
        return v

    def _extract_head_crop(self, scene_img: Image.Image, row: pd.Series) -> Image.Image:
        if not self.include_head_image:
            return scene_img

        width, height = scene_img.size
        try:
            xmin = float(row["head_xmin"])
            ymin = float(row["head_ymin"])
            xmax = float(row["head_xmax"])
            ymax = float(row["head_ymax"])
        except Exception:
            return scene_img

        if not all(math.isfinite(v) for v in [xmin, ymin, xmax, ymax]):
            return scene_img

        xmin = self._maybe_to_pixel(xmin, width)
        xmax = self._maybe_to_pixel(xmax, width)
        ymin = self._maybe_to_pixel(ymin, height)
        ymax = self._maybe_to_pixel(ymax, height)

        left_f, right_f = sorted([xmin, xmax])
        top_f, bottom_f = sorted([ymin, ymax])

        left = int(max(0, min(width - 1, math.floor(left_f))))
        top = int(max(0, min(height - 1, math.floor(top_f))))
        right = int(max(left + 1, min(width, math.ceil(right_f))))
        bottom = int(max(top + 1, min(height, math.ceil(bottom_f))))

        if (right <= left) or (bottom <= top):
            return scene_img

        return scene_img.crop((left, top, right, bottom))

    def _get_label_embedding(self, label_name: str) -> torch.Tensor:
        if (not label_name) or (self.label_embed_root is None):
            return torch.zeros(512, dtype=torch.float32)
        if label_name not in self.label_emb_cache:
            p = os.path.join(self.label_embed_root, f"{label_name}-emb.pt")
            if os.path.exists(p):
                emb = torch.load(p, map_location="cpu").to(torch.float32)
                emb = F.normalize(emb, p=2, dim=-1)
            else:
                emb = torch.zeros(512, dtype=torch.float32)
            self.label_emb_cache[label_name] = emb
        return self.label_emb_cache[label_name].clone()

    def _build_common(self, row: pd.Series, gaze_xy: torch.Tensor, gaze_points: Optional[torch.Tensor]) -> Dict:
        rel_path = row["path"]
        sample_id = int(row["id"])
        p = self._resolve_paths(rel_path, sample_id)
        reason_key = self._reason_key(rel_path, sample_id)

        scene_dino_feat, mark_dino_feat = self._load_dino_from_h5(reason_key)
        use_dino_precomputed = (scene_dino_feat is not None) and (mark_dino_feat is not None)

        cached_qwen_hidden = None
        qwen_hidden_valid = torch.tensor(0.0, dtype=torch.float32)
        if self.use_cached_qwen_hidden:
            cached_qwen_hidden = self._load_qwen_hidden_from_h5(reason_key, sample_id=sample_id)
            if cached_qwen_hidden is not None:
                qwen_hidden_valid = torch.tensor(1.0, dtype=torch.float32)

        need_qwen_images = not self.use_cached_qwen_hidden
        need_dino_images = not use_dino_precomputed
        need_scene_image = need_qwen_images or need_dino_images

        scene_img = None
        head_img = None
        mark_img = None
        if need_scene_image:
            scene_img = self._safe_open_image(p.scene)
            if need_qwen_images:
                head_img = self._extract_head_crop(scene_img, row)
            if self.include_mark_image and p.mark and os.path.exists(p.mark):
                mark_img = self._safe_open_image(p.mark)

        if need_qwen_images:
            prompt_text = self._read_text(p.prompt)
            reason_text = self._read_text(p.reason_txt) if self.include_reason_text else ""
        else:
            prompt_text = ""
            reason_text = ""

        reason_feat = torch.zeros(self.reason_feature_dim, dtype=torch.float32)
        reason_valid = torch.tensor(0.0, dtype=torch.float32)
        loaded_h5 = self._load_reason_from_h5(reason_key)
        if loaded_h5 is not None:
            reason_feat = self._fit_reason_dim(loaded_h5)
            reason_valid = torch.tensor(1.0, dtype=torch.float32)
        elif p.reason_feat and os.path.exists(p.reason_feat):
            loaded = torch.load(p.reason_feat, map_location="cpu")
            if isinstance(loaded, torch.Tensor):
                reason_feat = self._fit_reason_dim(loaded)
                reason_valid = torch.tensor(1.0, dtype=torch.float32)

        eye_xy = torch.tensor([float(row["eye_x"]), float(row["eye_y"])], dtype=torch.float32)
        inout = torch.tensor(float(row["inout"]), dtype=torch.float32)

        gaze_vec = F.normalize(gaze_xy - eye_xy, p=2, dim=-1)

        gaze_label_name = ""
        if "gaze_pseudo_label" in row and isinstance(row["gaze_pseudo_label"], str):
            gaze_label_name = row["gaze_pseudo_label"]
        elif "gaze_gt_label" in row and isinstance(row["gaze_gt_label"], str):
            gaze_label_name = row["gaze_gt_label"]

        gaze_label_emb = self._get_label_embedding(gaze_label_name)
        gaze_label_id = int(row["label_id"]) if "label_id" in row and pd.notna(row["label_id"]) else -1
        if "test_label_id" in row and pd.notna(row["test_label_id"]):
            gaze_label_id = int(row["test_label_id"])

        gaze_label_ids = torch.tensor([-1] * self.max_test_label_ids, dtype=torch.long)
        if "gaze_gt_labels" in row and isinstance(row["gaze_gt_labels"], str):
            labels = [x for x in row["gaze_gt_labels"].split("-") if x]
            mapped = [self.vocab2id.get(x, -1) for x in labels][: self.max_test_label_ids]
            for i, v in enumerate(mapped):
                gaze_label_ids[i] = int(v)
        elif gaze_label_id != -1:
            gaze_label_ids[0] = int(gaze_label_id)

        out = {
            "scene_image": scene_img,
            "head_image": head_img,
            "mark_image": mark_img,
            "prompt_text": prompt_text,
            "reason_text": reason_text,
            "gaze_xy": gaze_xy,
            "gaze_vec": gaze_vec,
            "eye_xy": eye_xy,
            "inout": inout,
            "reason_feat": reason_feat,
            "reason_valid": reason_valid,
            "gaze_label_emb": gaze_label_emb,
            "gaze_label_id": torch.tensor(gaze_label_id, dtype=torch.long),
            "gaze_label_ids": gaze_label_ids,
            "cache_key": reason_key,
            "qwen_pooled_hidden": cached_qwen_hidden,
            "qwen_hidden_valid": qwen_hidden_valid,
            "sample_id": sample_id,
            "path": rel_path,
        }
        if use_dino_precomputed:
            out["scene_dino_feat"] = scene_dino_feat
            out["mark_dino_feat"] = mark_dino_feat
        if gaze_points is not None:
            out["gaze_points"] = gaze_points
        return out

    def __getitem__(self, idx: int) -> Dict:
        if self.split in {"train", "val"}:
            row = self.df.iloc[idx]
            gaze_xy = torch.tensor([float(row["gaze_x"]), float(row["gaze_y"])], dtype=torch.float32)
            return self._build_common(row, gaze_xy=gaze_xy, gaze_points=None)

        image_path = self.image_paths[idx]
        rows = self.df[self.df["path"] == image_path]
        first = rows.iloc[0]
        points = torch.tensor(rows[["gaze_x", "gaze_y"]].values, dtype=torch.float32)
        n = min(len(points), self.max_test_gaze_points)
        gaze_points = torch.full((self.max_test_gaze_points, 2), -1.0, dtype=torch.float32)
        gaze_points[:n] = points[:n]
        gaze_xy = points[:n].mean(dim=0)
        return self._build_common(first, gaze_xy=gaze_xy, gaze_points=gaze_points)


class QwenVLBatchCollator:
    def __init__(
        self,
        processor,
        base_prompt: str,
        include_mark_image: bool = True,
        include_head_image: bool = True,
        use_cached_qwen_hidden: bool = False,
        cached_qwen_missing_policy: str = "error",
        qwen_image_size: int = 256,
    ):
        self.processor = processor
        self.base_prompt = base_prompt
        self.include_mark_image = include_mark_image
        self.include_head_image = include_head_image
        self.use_cached_qwen_hidden = bool(use_cached_qwen_hidden)
        self.cached_qwen_missing_policy = str(cached_qwen_missing_policy).lower().strip()
        if self.cached_qwen_missing_policy not in {"error", "skip"}:
            raise ValueError(f"Unsupported cached_qwen_missing_policy: {cached_qwen_missing_policy}")
        self.qwen_image_size = int(qwen_image_size)

    def _build_text(self, prompt_text: str, reason_text: str) -> str:
        pieces: List[str] = [self.base_prompt]
        if prompt_text:
            pieces.append("Prompt:\n" + prompt_text)
        if reason_text:
            pieces.append("Reference reasoning:\n" + reason_text)
        return "\n\n".join(pieces)

    @staticmethod
    def _resize_exact(img: Image.Image, width: int, height: int) -> Image.Image:
        return img.resize((int(width), int(height)))

    @staticmethod
    def _build_composite_image(
        scene: Image.Image,
        head: Optional[Image.Image],
        mark: Optional[Image.Image],
        size: int = 256,
    ) -> Image.Image:
        # Qwen-VL input image fixed to [size, size].
        if head is None and mark is None:
            return QwenVLBatchCollator._resize_exact(scene, size, size)

        if head is None:
            head = scene

        if mark is None:
            left_w = size // 2
            right_w = size - left_w
            scene_half = QwenVLBatchCollator._resize_exact(scene, left_w, size)
            head_half = QwenVLBatchCollator._resize_exact(head, right_w, size)
            canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
            canvas.paste(scene_half, (0, 0))
            canvas.paste(head_half, (left_w, 0))
            return canvas

        left_w = size // 3
        mid_w = size // 3
        right_w = size - left_w - mid_w
        scene_part = QwenVLBatchCollator._resize_exact(scene, left_w, size)
        head_part = QwenVLBatchCollator._resize_exact(head, mid_w, size)
        mark_part = QwenVLBatchCollator._resize_exact(mark, right_w, size)
        canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
        canvas.paste(scene_part, (0, 0))
        canvas.paste(head_part, (left_w, 0))
        canvas.paste(mark_part, (left_w + mid_w, 0))
        return canvas

    def __call__(self, batch: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        active_batch = batch
        if self.use_cached_qwen_hidden:
            missing = [item for item in batch if float(item.get("qwen_hidden_valid", torch.tensor(0.0)).item()) < 0.5]
            if missing:
                if self.cached_qwen_missing_policy == "error":
                    preview = ", ".join([str(m.get("cache_key", "")) for m in missing[:3]])
                    raise ValueError(
                        f"Missing cached Qwen hidden for {len(missing)}/{len(batch)} samples. "
                        f"policy=error preview_keys=[{preview}]"
                    )
                active_batch = [item for item in batch if float(item.get("qwen_hidden_valid", torch.tensor(0.0)).item()) > 0.5]
                if len(active_batch) == 0:
                    return None

        text_inputs: List[str] = []
        qwen_images: List[Image.Image] = []
        scene_images: List[Image.Image] = []
        mark_images: List[Image.Image] = []
        if self.include_mark_image:
            has_precomputed = [("scene_dino_feat" in item) and ("mark_dino_feat" in item) for item in active_batch]
        else:
            has_precomputed = [("scene_dino_feat" in item) for item in active_batch]

        for item in active_batch:
            scene = item.get("scene_image", None)
            head = item.get("head_image", None)
            mark = item.get("mark_image", None)
            if not self.use_cached_qwen_hidden:
                text_body = self._build_text(item["prompt_text"], item["reason_text"])
                if scene is None:
                    raise ValueError("Missing scene_image for Qwen-VL image input.")
                if self.include_head_image:
                    head = head if head is not None else scene
                else:
                    head = None
                if self.include_mark_image:
                    mark = mark if mark is not None else scene
                else:
                    mark = None

                # Ensure Qwen gets image tokens via chat template.
                if hasattr(self.processor, "apply_chat_template"):
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": text_body},
                            ],
                        }
                    ]
                    prompt = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                else:
                    prompt = text_body
                text_inputs.append(prompt)
                qwen_images.append(self._build_composite_image(scene, head, mark, size=self.qwen_image_size))

            if not all(has_precomputed):
                if scene is None:
                    raise ValueError("Missing scene_image for DINO online encoding.")
                scene_images.append(scene)
                if self.include_mark_image:
                    mark = mark if mark is not None else scene
                    mark_images.append(mark)

        if any(has_precomputed) and (not all(has_precomputed)):
            raise ValueError(
                "Mixed batch of precomputed/non-precomputed DINO features detected. "
                "Ensure dino_feature_h5 covers all samples or disable use_precomputed_dino_features."
            )

        if self.use_cached_qwen_hidden:
            model_inputs = {
                "qwen_pooled_hidden": torch.stack([item["qwen_pooled_hidden"] for item in active_batch], dim=0),
            }
        else:
            model_inputs = self.processor(
                text=text_inputs,
                images=qwen_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

        model_inputs["gaze_xy"] = torch.stack([item["gaze_xy"] for item in active_batch], dim=0)
        model_inputs["gaze_vec"] = torch.stack([item["gaze_vec"] for item in active_batch], dim=0)
        model_inputs["eye_xy"] = torch.stack([item["eye_xy"] for item in active_batch], dim=0)
        model_inputs["inout"] = torch.stack([item["inout"] for item in active_batch], dim=0)
        model_inputs["reason_feat"] = torch.stack([item["reason_feat"] for item in active_batch], dim=0)
        model_inputs["reason_valid"] = torch.stack([item["reason_valid"] for item in active_batch], dim=0)
        model_inputs["gaze_label_emb"] = torch.stack([item["gaze_label_emb"] for item in active_batch], dim=0)
        model_inputs["gaze_label_id"] = torch.stack([item["gaze_label_id"] for item in active_batch], dim=0)
        model_inputs["gaze_label_ids"] = torch.stack([item["gaze_label_ids"] for item in active_batch], dim=0)
        model_inputs["head_in_qwen"] = torch.full(
            (len(active_batch),),
            1.0 if (self.include_head_image and (not self.use_cached_qwen_hidden)) else 0.0,
            dtype=torch.float32,
        )
        model_inputs["sample_id"] = torch.tensor([int(item["sample_id"]) for item in active_batch], dtype=torch.long)
        model_inputs["cache_key"] = [str(item["cache_key"]) for item in active_batch]
        model_inputs["path"] = [str(item["path"]) for item in active_batch]
        if "gaze_points" in active_batch[0]:
            model_inputs["gaze_points"] = torch.stack([item["gaze_points"] for item in active_batch], dim=0)
        if "scene_dino_feat" in active_batch[0]:
            scene_feat = torch.stack([item["scene_dino_feat"] for item in active_batch], dim=0)
            model_inputs["scene_dino_feat"] = scene_feat
            if self.include_mark_image and ("mark_dino_feat" in active_batch[0]):
                model_inputs["mark_dino_feat"] = torch.stack([item["mark_dino_feat"] for item in active_batch], dim=0)
            else:
                # Hard-off mark branch when include_mark_image=False.
                model_inputs["mark_dino_feat"] = torch.zeros_like(scene_feat)
        else:
            model_inputs["scene_images"] = scene_images
            if self.include_mark_image:
                model_inputs["mark_images"] = mark_images
        return model_inputs
