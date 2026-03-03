Codex 작업 지시서: gaze_mllm 파이프라인 개선 (속도/정합성/단순화)
목표

1 epoch 1시간 → 크게 단축 (성능 유지 우선)

파이프라인/구현의 명확한 정합성 문제 해결

불필요 head 축소 가능하도록 ablation-friendly 구조로 정리

레포: https://github.com/JinWoong-Jung/gaze_mllm

A. 즉시 수정해야 하는 “정합성/구현” 이슈
A1) Head crop이 현재 forward에 실사용되지 않는 문제 해결

현재 annotation에 head_xmin~head_ymax가 있음에도, dataset/collator/model forward 경로에 head crop branch가 없음.

구현 요구:

Dataset에서 head bbox로 head crop 이미지를 생성(또는 load/transform)해 batch에 포함.

Collator에서 Qwen input에 넣는 방식 2가지 중 1개 택:

3-way composite: [scene | head | mark] 를 하나의 composite image로 구성

multi-image 입력: Qwen processor가 다중 이미지를 받는 구조라면 images list로 제공

DINO도 head feature를 추가할지 선택:

최소 수정: head는 Qwen에만 넣고 DINO는 scene/mark만 유지

확장: DINO head feat 추가 후 cross-attn 토큰 3개로 확장

Acceptance check

forward에서 실제로 head crop이 모델 입력에 들어가는지(텐서 형태로) unit print/log로 확인.

A2) head_only 모드에서 backbone/DINO eval 고정

문제: trainer에서 model.train() 호출 시 freeze된 backbone이 train mode(드롭아웃 등)로 돌아갈 수 있음.

구현 요구:

train_mode == "head_only"일 때:

Qwen backbone, vision encoder, DINO 관련 모듈은 eval()로 강제

하지만 head/fusion 모듈은 train() 유지

optimizer param group은 기존처럼 requires_grad=True만 등록

Acceptance check

학습 중 backbone의 training flag가 False인지 assert.

A3) reason feature H5 split 경로 정리

현재 config가 reason_feature_h5_path: train.h5 단일 경로인데, trainer가 이를 train/val/test에 공통 전달.

구현 요구(둘 중 1개):

config에 reason_feature_h5_path_train/val/test로 분리해서 dataset에 맞게 전달

단일 H5를 유지하되, 내부 key가 split 포함(예: train/..., val/...)이 되도록 dataset key mapping 수정 + missing 시 경고/카운트

Acceptance check

val/test에서 reason feature hit rate(로드 성공 비율)를 로깅.

B. 속도 대폭 개선(성능 유지 우선)
B1) “Qwen frozen이면 Qwen hidden 캐시” 기능 추가 (가장 중요)

현재 기본 설정: train_mode=head_only, train_dino=false, use_precomputed_dino_features=true → 가장 큰 비용은 매 스텝 Qwen forward.

구현 요구:

새로운 전처리 스크립트 추가: tools/cache_qwen_hidden.py

입력: split(train/val/test), config, output path

출력: sample_id 기준으로 qwen_pooled_hidden(또는 final hidden + pooling 결과) 저장

추천: pooled vector만 저장(예: [B, H]) → 디스크/IO 최적

필요한 head가 token-level 정보(heatmap 등)에 의존한다면 final hidden도 고려하되, 우선 pooled부터

Dataset/Collator에 옵션 추가:

use_cached_qwen_hidden: true/false

true면 processor/Qwen forward를 생략하고 cached hidden을 batch로 제공

Model forward에 입력 분기:

cached hidden이 주어지면 Qwen forward 생략 → fusion/heads만 실행

Trainer에 “캐시가 없으면 에러/스킵” 동작 명확히

Acceptance check

동일 seed에서 cached on/off 시 metric 차이가 거의 없거나(헤드만 학습일 때) 수렴/성능이 유지되는지 확인.

epoch time이 크게 줄어드는지 확인.

B2) Collator/IO 병목 제거 (보조)

Dataset에서 매 샘플마다 txt 파일(prompt/reason)을 읽는다면:

init 시에 인덱싱/메모리 캐시(혹은 LMDB/JSONL로 통합)로 변경

composite image 생성/processor 토큰화 반복을 줄일 수 있으면 캐시:

최소: prompt 텍스트 및 paths 캐시

가능: processor output(이미지 텐서/텍스트 ids) 캐시

Acceptance check

DataLoader time vs GPU time 프로파일 간단 로그(배치 준비시간/forward 시간).

B3) 학습 반복 실험 중 test 자동 실행 끄기

config의 test.run_after_train=true면 실험 반복에서 wall time 증가.

구현 요구:

기본 config 또는 실험용 config에서 off 권장

혹은 CLI flag로 on/off 제어

C. Head 축소 전략 (ablation-friendly)
C1) “필수”와 “aux” 분리

현재 evaluator의 주 gaze 위치 metric은 heatmap argmax 좌표 기반이므로:

필수 유지: heatmap, inout

조건부: label(acc@1/3를 계속 볼 거면 유지)

제거 1순위: reason

제거 2순위: angle

제거 3순위 후보: coord(현재 metric 계산에는 직접 사용되지 않음)

C2) 구현 요구: head enable/disable을 config로 제어

config에 heads.enabled: [heatmap, inout, label, coord, reason, angle] 같은 리스트 도입

model에서 enabled head만 생성/forward/loss 계산

loss에서도 weight=0이 아니라 아예 head를 비활성화하도록 (불필요 compute 제거)

Acceptance check

enabled 리스트 변경 시 코드 수정 없이 학습 가능

reason/angle 제거해도 나머지 metric 정상 계산

D. 변경 후 최소 검증 체크리스트

head crop이 실제로 forward 입력으로 들어간다.

head_only에서 backbone/DINO는 eval 고정이다.

val/test reason feature hit rate가 기대 수준이다(또는 의도적으로 off).

cached qwen hidden 사용 시 epoch time이 크게 감소한다.

head enable/disable config로 ablation이 가능하다.

구현 우선순위(바로 이렇게 진행)

B1(Qwen hidden 캐시)

A2(backbone eval 고정)

C2(head enable/disable) + reason, angle 제거 실험

A1(head crop 실제 입력화)

A3(reason H5 split 정리)

B2/B3 최적화 마무리