# gaze_mllm

Qwen3-VL 기반 gaze estimation 학습 파이프라인입니다.  
생성 텍스트를 직접 평가하지 않고, Qwen hidden representation 위에 회귀/분류 head를 붙여 정량 지표로 학습합니다.

## 구성 요약
- Backbone: Qwen3-VL
- Visual branch: DINO scene/mark feature (online 또는 precomputed H5)
- Fusion: DINO token -> Qwen hidden cross-attention
- Heads:
  - `heatmap`
  - `coord`
  - `inout`
  - `reason`
  - `label`
  - `angle` (2-layer MLP)

## 현재 학습 경로(중요)
- 기본 권장 모드: `precomputed DINO + local model path`
- `data.use_precomputed_dino_features: true`이면 DINO H5를 사용합니다.
- 이 모드에서는 online DINO 인코더를 반드시 로드하지 않아도 학습 가능하도록 구현되어 있습니다.

## 입력
- scene image
- head crop image
- mark image (visual-prompted image)
- text prompt

Qwen 입력 이미지는 collator에서 고정 해상도로 구성됩니다.
- `data.qwen_image_size` (기본 256)
- mark/head 사용 시 scene/head/mark를 가로 분할한 단일 정사각 이미지로 합성

## Qwen hidden 캐시 (B1)
Qwen backbone이 frozen(`train_mode=head_only`)일 때는 pooled hidden을 H5로 미리 캐시해
학습 중 Qwen forward를 생략할 수 있습니다.

캐시 생성:
```bash
python tools/cache_qwen_hidden.py \
  --config configs/train_qwen3vl_gf.yaml \
  --split train \
  --output_h5 /home/elicer/gaze_mllm/data/qwen_hidden/train.h5 \
  --batch_size 64 --num_workers 12 --overwrite
```

출력 H5 포맷:
- `keys`
- `sample_ids`
- `embeddings` (pooled hidden)

학습 시 config:
- `data.use_cached_qwen_hidden: true`
- `data.qwen_hidden_h5_train/val/test`: split별 H5 경로
- `data.cached_qwen_missing_policy`: `error | skip`

## Loss
Global loss는 가중합입니다.
- `L_heatmap`
- `L_coord`
- `L_angle`
- `L_inout`
- `L_reason`
- `L_label`

참고:
- `vec` loss는 현재 `w_vec=0`으로 비활성화 사용을 권장합니다.

## Metrics
### Validation
- `metric/val/dist`
- `metric/val/inout_acc`
- `loss/val`
- `loss/val/heatmap`
- `loss/val/coord`
- `loss/val/vec`
- `loss/val/angular`
- `loss/val/inout`
- `loss/val/reason`
- `loss/val/label`

### Test (semgaze 호환)
- `metric/test/acc@1`
- `metric/test/acc@3`
- `metric/test/auc`
- `metric/test/avg_dist`
- `metric/test/dist_to_avg`
- `metric/test/min_dist`
- `metric/test/multi_acc@1`

## 진행바/로그
- train/val/test 모두 `tqdm` 진행바 지원
- 학습 시작 시 파라미터 요약 출력
- best checkpoint 기준: `metric/val/dist` 최소값

## DINO feature 사전추출
스크립트:
- `tools/extract_dino_features.py`

출력 H5 포맷:
- `keys`
- `scene_embeddings`
- `mark_embeddings`

train/val/test 각각 추출한 H5를 config에 지정해 사용합니다.

## Reason feature 사용
- 우선순위: `reason_feature_h5_path` -> `reason_feature_root`(`.pt`) fallback
- H5 포맷: `keys`, `embeddings`
- `loss.reason_loss_type`: `cosine | mse | infonce`
- `loss.reason_nce_temperature`: InfoNCE temperature (기본 `0.07`)

## 실행
```bash
python train.py --config configs/train_qwen3vl_gf.yaml
```

## 로컬 고정형(재다운로드 방지)
권장:
- `model.name`: 로컬 Qwen 스냅샷 경로
- `model.local_files_only: true`
- `model.cache_dir`: 로컬 캐시 경로

주의:
- `local_files_only: true`일 때는 모델 파일이 로컬에 모두 있어야 합니다.
- 누락 shard가 있으면 즉시 에러로 중단됩니다.

## 주요 설정 키
- `model.train_mode`: `head_only | lora | full`
- `heads.enabled`: 활성화할 head 리스트 (`heatmap,inout,label,coord,reason,angle`)
- `model.gradient_checkpointing`
- `train.precision`: `bf16 | fp16 | fp32`
- `train.batch_size`, `train.grad_accum`
- `train.progress_bar`
- `test.run_after_train`
- `wandb.project`, `wandb.entity`, `wandb.name`
