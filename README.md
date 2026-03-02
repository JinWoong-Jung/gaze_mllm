# gaze_mllm

Qwen3-VL 기반 gaze estimation 학습 파이프라인.

## 핵심 구성
- MLLM: `Qwen/Qwen3-VL-8B-Thinking` (pretrained 로드 후 trainable)
- 입력: `scene image + head crop + mark image + text prompt`
- DINOv3 분기: `scene image`, `visual-prompt(mark) image`를 각각 DINOv3에 넣어 feature 추출 후 MLLM hidden에 조건 주입
- supervision: `gaze_xy(in) + inout`
- 보조 학습: GPT reasoning feature(`.pt`, 768-dim) alignment loss
- 추가: semgaze style 2-MLP gaze angle head + angle loss

## 데이터 경로(기본값)
- 이미지: `/mnt/elice/datahub/pixel-bucket-jinwoong/data/gazefollow_extended`
- annotation: `/mnt/elice/datahub/pixel-bucket-jinwoong/data/gazefollow/{train,val}_annotations_new.txt`
- reasoning:
  - output txt: `/mnt/elice/datahub/pixel-bucket-jinwoong/data/gazefollow_reason/output`
  - mark image: `/mnt/elice/datahub/pixel-bucket-jinwoong/data/gazefollow_reason/mark`
  - prompt txt: `/mnt/elice/datahub/pixel-bucket-jinwoong/data/gazefollow_reason/prompt`
  - features pt: `/mnt/elice/datahub/pixel-bucket-jinwoong/data/gazefollow_reason/features`

## 실행
```bash
cd /home/elicer/gaze_mllm
python train.py --config configs/train_qwen3vl_gf.yaml
```

## DINO 사전추출 (선택)
```bash
python tools/extract_dino_features.py \
  --annotation /mnt/elice/datahub/pixel-bucket-jinwoong/data/gazefollow/train_annotations_new.txt \
  --image_root /mnt/elice/datahub/pixel-bucket-jinwoong/data/gazefollow_extended \
  --mark_root /mnt/elice/datahub/pixel-bucket-jinwoong/data/gazefollow_reason/mark \
  --split train \
  --output_h5 /home/elicer/gaze_mllm/data/dino_features/train.h5 \
  --batch_size 64 \
  --overwrite
```
YAML에서 아래를 켜면 학습 시 DINO online forward 대신 h5를 사용합니다.
- `data.use_precomputed_dino_features: true`
- `data.dino_feature_h5_train|val|test`

## 설정 포인트
- 학습 모드: `model.train_mode` in `{full,lora,head_only}`
- 이미지 인코더: `model.dino_name` (기본 `facebook/dinov3-vitb16-pretrain-lvd1689m`)
- DINO freeze: `model.train_dino=false` (pretrained frozen), `true`(finetune)
- mixed precision: `train.precision` in `{bf16,fp16,fp32}`
- loss 가중치: `loss.w_*`
- reasoning 정렬 방식: `loss.reason_loss_type` in `{cosine,mse}`
- mark image 비활성화: `data.include_mark_image=false`
- wandb: `wandb.log=true`로 활성화, 프로젝트/엔티티/run name은 `wandb.*`에서 설정

## Test 지표(semgaze 호환)
학습 종료 후 `test.run_after_train=true`이면 아래를 출력합니다.
- `metric/test/acc@1`
- `metric/test/acc@3`
- `metric/test/auc`
- `metric/test/dist_to_avg`
- `metric/test/avg_dist`
- `metric/test/min_dist`
- `metric/test/multi_acc@1`

## 주의
- 현재 `mark` 이미지는 생성 방식상 gaze point/arrow가 포함될 수 있어 label leakage 가능성이 있음.
- 공정한 비교 실험에서는 `include_mark_image: false` 또는 mark 생성 로직 수정 권장.
