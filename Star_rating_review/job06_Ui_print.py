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

form_window = uic.loadUiType('C:/workspace/Star_rating_review/review_ui.ui')[0]

with open('C:/workspace/Star_rating_review/Star_rating_review/models/review_token_MAX_129.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

class Exam(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 학습한 이진 분류 모델 가져오기
        self.model = load_model(
            'C:/workspace/Star_rating_review/Star_rating_review/models/review_data_classfication_model_0.9109051823616028.h5')

        self.review_btn.clicked.connect(self.btn_clicked_slot)
        self.text_edit.clear()
        self.stars_progressBar.setValue(0)
        self.AI_star_print_label.setText('')
        self.my_star_label.setText('')

    def preprocess_text(self, text):
        """리뷰 텍스트를 전처리하여 모델 입력 형태로 변환"""
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=129)
        return padded

    def btn_clicked_slot(self):
        try:
            review_text = self.text_edit.toPlainText()
            review_star = self.my_star_spinBox.value()
            print("입력된 리뷰:", review_text)
            print("내가 준 별점:", review_star)

            # 리뷰 전처리
            review_text_vector = self.preprocess_text(review_text)
            print("리뷰 벡터화 결과:", review_text_vector)

            # AI 예측
            pred = self.model.predict(review_text_vector).item()
            print("AI 예측 확률:", pred)

            # 예측 확률을 5단계로 변환
            if pred <= 0.2:
                predicted_star = 1
                label = "매우 부정적"
            elif pred <= 0.4:
                predicted_star = 2
                label = "부정적"
            elif pred <= 0.6:
                predicted_star = 3
                label = "중립적"
            elif pred <= 0.8:
                predicted_star = 4
                label = "긍정적"
            else:
                predicted_star = 5
                label = "매우 긍정적"

            print("AI 예측 별점:", predicted_star)

            # AI가 예측한 별점 출력
            ai_stars = "★" * predicted_star + "☆" * (5 - predicted_star)  # 빈 별표로 5개 채움
            self.AI_star_print_label.setText(ai_stars)

            # AI 예측 확률 출력
            accuracy = pred * 100 if pred > 0.5 else (1 - pred) * 100
            self.stars_progressBar.setValue(int(accuracy))

            # 내가 준 별점 출력
            my_stars = "★" * review_star + "☆" * (5 - review_star)  # 빈 별표로 5개 채움
            self.my_star_label.setText(my_stars)

            # AI 예측 결과 텍스트 출력
            msg = f"AI 예측: {label} ({ai_stars})\n정확도: {accuracy:.2f}%"
            QMessageBox.information(self, "AI 예측 결과", msg)

        except Exception as e:
            print(f"예측 중 오류 발생: {e}")
            QMessageBox.warning(self, "오류", "예측 중 오류가 발생했습니다.")

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