# gaze_mllm

Qwen3-VL + DINOv3 기반 gaze estimation 학습 파이프라인.

## 개요
이 프로젝트는 생성형 텍스트를 직접 쓰는 방식이 아니라, Qwen의 마지막 hidden representation 위에 task-specific head를 붙여 gaze를 정량 예측합니다.

입력은 다음 3가지입니다.
- scene image
- visual-prompted image (mark)
- text prompt

구성은 다음과 같습니다.
- Qwen3-VL: image+text 멀티모달 입력
- DINOv3: scene/mark feature 추출 (online 또는 precomputed h5)
- Fusion: DINO feature를 Qwen hidden에 cross-attention으로 주입
- Heads: heatmap / coord / inout / reason / label / angle(2-MLP)

## 아키텍처 요약
1. Qwen 입력
- `apply_chat_template` + image placeholder로 image token 포함
- Qwen backbone의 `last_hidden_state [B,T,H]` 사용

2. DINO 분기
- scene, mark 각각 DINO 인코딩
- precomputed 모드에서는 h5에서 `scene_embeddings`, `mark_embeddings` 로드

3. 결합
- DINO feature stack -> `Linear(D->H)`
- Qwen hidden(query)와 DINO tokens(key/value)로 cross-attention
- conditioned hidden을 masked mean pooling

4. 출력 head
- Heatmap head: `H -> H/2 -> 64*64`
- Coord head: `H -> H/2 -> 2` (sigmoid)
- InOut head: `H -> 1`
- Reason head: `H -> reason_dim` + L2 norm
- Label head: `H -> 512` + L2 norm
- Angle head: `concat(scene_cls, mark_cls) -> 2-layer MLP -> 2D vector`

## Loss
Global loss는 아래 항의 가중합입니다.
- `L_heatmap`
- `L_coord`
- `L_angle`
- `L_inout`
- `L_reason`
- `L_label`

참고:
- `vec loss`는 중복 supervision 방지를 위해 현재 비활성화(`w_vec=0`) 상태

## 평가 지표
학습 종료 후(`test.run_after_train=true`) 아래 지표를 출력합니다.
- `metric/test/acc@1`
- `metric/test/acc@3`
- `metric/test/auc`
- `metric/test/dist_to_avg`
- `metric/test/avg_dist`
- `metric/test/min_dist`
- `metric/test/multi_acc@1`

추가로 val에서는
- `metric/val/l2`
- `metric/val/inout_acc`
를 사용합니다.

## DINO 사전추출
사전추출 스크립트:
- `tools/extract_dino_features.py`

출력 h5 포맷:
- `keys`
- `scene_embeddings`
- `mark_embeddings`

학습에서 사용하려면 config에서:
- `data.use_precomputed_dino_features: true`
- `data.dino_feature_h5_train|val|test` 지정

## Reasoning feature 사용
GPT reasoning feature는 보조 정렬 loss에 사용됩니다.
- 우선순위: `reason_feature_h5_path` -> `reason_feature_root(.pt)` fallback
- h5 포맷: `keys`, `embeddings`

## 실행
```bash
python train.py --config configs/train_qwen3vl_gf.yaml
```

## 주요 설정
- 모델 모드: `model.train_mode` in `{head_only,lora,full}`
- DINO 고정/학습: `model.train_dino`
- 혼합정밀도: `train.precision` in `{bf16,fp16,fp32}`
- 진행바/요약: `train.progress_bar`, `train.print_model_summary`, `train.sanity_val_steps`
- wandb: `wandb.log`, `wandb.project`, `wandb.entity`, `wandb.name`

## 로그 출력
학습 시작 시:
- trainable/non-trainable/total params
- estimated model size
- train/eval mode 모듈 수

학습 중:
- tqdm 진행바 + postfix(loss, heatmap, coord, angle, lr)

검증/테스트:
- epoch 단위 지표 출력
- best val metric 갱신 시 checkpoint 저장 로그 출력
