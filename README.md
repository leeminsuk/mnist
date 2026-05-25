# CIFAR-10 — WideResNet-28-4 (Macro F1 최대화)

CIFAR-10 이미지 분류 프로젝트. **Macro F1 Score** 최대화를 목표로 WideResNet-28-4 아키텍처를 사용.

## 결과

| 지표 | 값 |
|------|-----|
| **Macro F1** | *훈련 후 업데이트 예정* |
| Accuracy | *훈련 후 업데이트 예정* |
| Weighted F1 | *훈련 후 업데이트 예정* |

## 모델 아키텍처

**WideResNet-28-4** (Wide Residual Network)

```
입력: 32×32×3 (RGB)
├── Conv 3×3 (16채널)
├── Group1: 4× BasicBlock (64채널, stride=1)
├── Group2: 4× BasicBlock (128채널, stride=2) → 16×16
├── Group3: 4× BasicBlock (256채널, stride=2) → 8×8
├── BatchNorm + ReLU
├── GlobalAveragePooling
└── Dense(10) + Softmax
```

- **파라미터**: 5,849,050
- **특징**: Residual 연결 + BatchNormalization + Dropout(0.3)

## 핵심 기법

| 기법 | 역할 |
|------|------|
| Random Crop (padding=4) | 위치 불변성 학습 |
| Random Horizontal Flip | 좌우 대칭 불변성 |
| ColorJitter | 밝기/대비/채도 다양성 |
| **Cutout (16×16)** | 특정 영역 의존성 방지 |
| SGD + Nesterov | CIFAR-10에 최적화된 옵티마이저 |
| **Cosine Annealing LR** | 학습률 부드러운 감소 |
| Weight Decay (5e-4) | L2 정규화 |

## 하이퍼파라미터

```python
batch_size   = 128
epochs       = 200
learning_rate = 0.1
momentum      = 0.9
weight_decay  = 5e-4
cutout_length = 16
```

## 환경

- **Framework**: PyTorch 2.12.0
- **Device**: Apple Silicon (MPS)
- **Python**: 3.14

## 파일 구조

```
mnist/
├── CIFAR10_WRN28.ipynb    # 메인 노트북 (실행 가능)
├── requirements.txt        # 의존 패키지
└── README.md
```

## 실행 방법

```bash
pip install torch torchvision scikit-learn matplotlib
jupyter notebook CIFAR10_WRN28.ipynb
```

## CIFAR-10 클래스

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
