import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtGui import QPixmap
from keras.models import load_model
import numpy as np

import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# GUI 파일 정보를 class로 변환하여 가져옴
form_window = uic.loadUiType('C:/workspace/Star_rating_review/review_ui.ui')[0]


# 저장된 토크나이저 불러오기
with open('C:/workspace/Star_rating_review/Star_rating_review/models/review_token_MAX_129.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)



# 상속(위젯이면서 UI를 같이 가지고 있음)
class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()  # 조상 class 생성자 호출
        self.setupUi(self)  # UI 초기화

        #학습한 모델 가져오기
        self.model = load_model('C:/workspace/Star_rating_review/Star_rating_review/models/review_data_classfication_model_0.57.h5')


        # 버튼 동작
        self.review_btn.clicked.connect(self.btn_clicked_slot)

        # 리뷰 입력창 초기화
        self.text_edit.clear()
        self.stars_progressBar.setValue(0)



    def preprocess_text(self, text):
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=129)
        return padded[0]


    def btn_clicked_slot(self):

        #입력된 리뷰와 별점 가져오기
        review_text = self.text_edit.toPlainText()
        print(review_text)
        review_star = self.my_star_spinBox.value()
        print(review_star)

        # 리뷰 전처리
        review_text_vector = self.preprocess_text(review_text)

        # AI 예측
        pred = self.model.predict(np.array([review_text_vector]))
        pred_probs = pred[0]


        for i in range(5):
            print(f"별점 {i + 1}: {pred_probs[i]}")


        predicted_star = np.argmax(pred[0]) + 1  # 클래스 인덱스를 별점으로 변환
        print("AI 예측 별점:", predicted_star)

        # AI가 예측한 별점 출력
        ai_stars = "★" * predicted_star
        self.AI_star_print_label.setText(ai_stars)

        # AI 예측
        # 정확도 출력
        accuracy = np.max(pred[0]) * 100
        self.stars_progressBar.setValue(int(accuracy))


        #내가 준 별점 출력
        my_stars = "★" * review_star
        self.my_star_label.setText(my_stars)

        #AI 리뷰 예측비교





if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        mainWindow = Exam()
        mainWindow.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        print("프로그램 종료")
