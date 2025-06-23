# Car Image Classification Pipeline

이 프로젝트는 자동차 이미지를 분류하기 위한 **EfficientNet 기반의 이미지 분류 파이프라인**입니다.  
전체 과정은 데이터 준비부터 학습, 추론, 그리고 Knowledge Distillation 기반의 성능 개선 단계까지 포함되어 있습니다.

---

## 1. 프로젝트 목적

- 자동차 이미지 데이터를 분류하는 모델을 학습합니다.
- 이미지뿐 아니라 부가 정보를 활용해 정확도를 높입니다.
- EfficientNet-B5를 백본으로 사용하고, 이미지의 평균 색상(`color mean`)을 메타 정보로 추가합니다.
- 5-Fold 앙상블과 Knowledge Distillation 기반 Student 모델 학습까지 수행합니다.

---

## 2. 전체 구성 요약

| 단계       | 내용                                                             |
|------------|------------------------------------------------------------------|
| 데이터 준비 | Google Drive의 zip 파일 복사 및 압축 해제                        |
| 전처리     | 이미지에서 클래스명 추출, 파일 리스트 생성, transform 정의       |
| 모델 정의  | EfficientNet-B5 + Meta feature (`color_mean`) 처리 모델         |
| 학습       | 5-Fold Cross Validation 기반 학습                                 |
| 추론       | Fold별 모델 예측 결과 앙상블 (`softmax` 평균)                   |
| KD 학습    | Teacher 모델의 soft label을 사용하여 Student 모델 학습           |

---

## 3. 파일 구조 설명


---

## 4. 주요 코드 구성

### 데이터 준비

- `shutil.copy`와 `zipfile.ZipFile`을 이용해 Drive에서 압축 파일을 복사하고 해제합니다.
- `glob.glob`을 통해 모든 이미지 경로를 수집하고, 폴더명을 클래스명으로 추출합니다.

### 커스텀 Dataset

- `CarImageDataset`: 학습용 Dataset 클래스
- `CarJPGDataset`: 추론 및 KD용 Dataset 클래스
- 선택적으로 `aspect ratio`와 `color mean`을 추가 feature로 포함시킬 수 있습니다.

### 모델 정의

- `CustomModel` 클래스는 EfficientNet-B5 백본과 분류기를 포함합니다.
- 분류기는 이미지 feature와 meta feature를 입력으로 받아 최종 클래스를 예측합니다.

### 학습 (5-Fold)

- `StratifiedKFold`로 데이터를 5개 Fold로 나눕니다.
- 각 Fold마다 모델을 새로 생성하고, 최적 모델(`.pth`)을 저장합니다.
- Early Stopping 적용: validation loss가 개선되지 않으면 조기 종료합니다.

### 추론 및 앙상블

- 저장된 5개의 모델로 각각 테스트 이미지를 추론합니다.
- 각 모델의 `softmax` 확률을 평균내어 앙상블 결과를 도출합니다.
- 결과는 `submission_fold5_ensemble_C.csv`로 저장됩니다.

### Knowledge Distillation (Stage 2)

- **Teacher 모델**: 5-Fold 중 가장 성능 좋은 fold 모델 사용
- **Student 모델**: Teacher가 생성한 soft label을 학습 대상으로 사용
- Loss: `KLDivLoss` (temperature scaling 적용)
- AMP(Auto Mixed Precision), gradient clipping, cosine annealing scheduler 적용
- 최종 student 모델은 `/Student_stage2_fold1_best.pth`으로 저장됩니다.

---

## 5. 제출 파일 구조

| 파일명 | 설명 |
|--------|------|
| `submission_fold5_ensemble_C.csv` | 앙상블 결과 (각 클래스별 확률 포함) |
| `all_experiments_submission_compare.csv` | 다양한 실험 결과 비교용 통합 파일 |

---

## 6. 사용 방법 요약

1. Google Drive에 `open.zip`을 업로드합니다.(https://dacon.io/competitions/official/236493/data)
2. Colab 환경에서 전체 코드를 순차적으로 실행합니다.
3. 학습된 모델은 자동 저장되며, 제출 파일도 `/team_models/`에 생성됩니다.
4. 필요 시, Stage 2 학습으로 성능을 개선할 수 있습니다.

---
