# Sat-IR-to-Visible: 위성 적외선-가시광선 영상 변환기 [2023.11]

### 결과 예시 (Demo)

| 입력 (적외선 원본) | 생성된 이미지 (가시광선) |
| :---: | :---: |
| <img width="400" alt="Infrared" src="https://github.com/user-attachments/assets/44d6765d-f01a-46cc-b33c-be5af157174b"> | <img width="400" alt="Visible" src="https://github.com/user-attachments/assets/9b399749-b83e-4434-988d-c3582e203d79"> |


이 프로젝트는 지정된 기간의 천리안 2A호(GK2A) 위성 적외선(IR) 영상을 자동으로 다운로드하고, Pix2Pix 딥러닝 모델을 사용하여 가시광선(VI) 영상으로 변환하는 자동화 파이프라인입니다. 전공 수업에서 위성 영상을 수작업으로 수집하고 처리하는 불편함을 해결하고자 시작되었습니다.

## ✨ 주요 기능

  * **위성 영상 자동 다운로드**: 원하는 기간과 시간대의 GK2A 위성 적외선 영상을 자동으로 다운로드합니다.
  * **적외선 → 가시광선 영상 변환**: Pix2Pix (U-Net 기반) 딥러닝 모델을 통해 적외선 영상을 가시광선 영상으로 변환합니다.
  * **모델 자동 다운로드**: 실행 시 필요한 딥러닝 모델 파일(`.pt`)이 없으면 구글 드라이브에서 자동으로 다운로드하여 사용자 편의성을 높였습니다.

## 🛠️ 설치 방법

1.  **Git 저장소 복제:**

    ```bash
    git clone https://github.com/hyuniverse/Sat-IR-to-Visible.git
    cd Sat-IR-to-Visible
    ```

2.  **필요 라이브러리 설치:**
    가상환경 사용을 권장합니다.(필자의 경우 miniconda3 환경에서 진행)

    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ 기술 스택 (Tech Stack)
- Language : Python
- Deep Learning Framework: PyTorch
- Core Model: Pix2Pix (U-Net Generator, PatchGAN Discriminator)
- Libraries: `gdown`, `Pillow`, `numpy`

## 🚀 사용 방법

명령어 한 줄로 다운로드부터 변환까지 모든 과정을 실행할 수 있습니다.

```bash
python src/main.py --start_date "YYYY/MM/DD" --days <number_of_days>
```

**실행 예시:**
2024년 1월 1일부터 5일간의 영상을 다운로드하고 변환하려면 아래와 같이 실행합니다.

```bash
python src/main.py --start_date "2024/01/01" --days 5
```

  * **참고:** 처음 실행 시, 필요한 모델 파일 (`pix2pix_generator.pt`)이 `models/` 폴더에 없으면 자동으로 다운로드합니다.
  * 원본 적외선 영상은 `data/infrared/` 폴더에 저장됩니다.
  * 변환된 가시광선 영상은 `data/visible_generated/` 폴더에 저장됩니다.

## 📁 프로젝트 구조

```
.
├── data/                   # 다운로드 및 생성된 이미지 저장 (Git 추적 제외)
├── models/                 # 딥러닝 모델 가중치 저장 (Git 추적 제외)
├── results/                # README용 결과 이미지 저장
├── src/                    # 주요 소스 코드
│   ├── data_loader.py      # 위성 영상 다운로드 모듈
│   ├── inference.py        # Pix2Pix 추론(변환) 모듈
│   ├── main.py             # 메인 실행 파일
│   └── model.py            # Pix2Pix 모델 구조 정의
├── .gitignore              # Git 버전 관리 제외 목록
├── README.md               # 프로젝트 설명서
└── requirements.txt        # 필요 라이브러리 목록
```

## 🔧 개선 계획 (To-Do)
- [ ] 학습(Train) 모드 추가: 사용자가 직접 준비한 데이터셋으로 모델을 학습시킬 수 있는 기능 추가 예정

## 🧑‍💻 만든 사람

  * **박세현** - [hyuniverse](https://www.google.com/search?q=https://github.com/hyuniverse)
