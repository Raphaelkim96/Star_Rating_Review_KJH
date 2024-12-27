# ⭐ 리뷰 별점 분류기 (Review Star Rating Classifier)

리뷰 텍스트를 분석하여 별점을 예측하는 딥러닝 기반 분류 모델입니다.

## 🔍 프로젝트 소개

이 프로젝트는 텍스트 리뷰를 입력받아 1-5점 사이의 별점을 예측하는 분류기를 구현했습니다. LSTM 네트워크를 활용하여 자연어 처리를 수행합니다.

## 🛠 개발 환경

- Python 3.10
- TensorFlow 2.18.0


## 📁 프로젝트 구조

```
review-star-classifier/
├── data/
│   ├── raw/                # 원본 데이터
│   └── processed/          # 전처리된 데이터
├── src/
│   ├── preprocessing.py    # 데이터 전처리 
│   ├── model.py           # 모델 아키텍처
│   ├── train.py           # 학습 실행
│   └── predict.py         # 예측 실행
├── notebooks/             # 분석 노트북
├── config/               
│   └── config.yaml        # 설정 파일
├── requirements.txt       # 패키지 의존성
└── README.md
```

## ⚙️ 설치 방법

1. 저장소 클론
```bash
[git clone https://github.com/username/review-star-classifier.git](https://github.com/Raphaelkim96/Star_rating_review_KJH.git)
cd review-star-classifier
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 🚀 사용 방법

1. 데이터 전처리
```bash
python src/preprocessing.py
```

2. 모델 학습
```bash
python src/train.py
```

3. 예측 실행
```bash
python src/predict.py --text "리뷰 텍스트를 입력하세요"
```

## 📊 성능 지표

- 정확도: 85%
- F1 Score: 0.83
- Precision: 0.82
- Recall: 0.84

## 📝 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.

## 👥 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ✨ 참고자료

- [TensorFlow 공식 문서](https://www.tensorflow.org/)
- [Keras 문서](https://keras.io/)
