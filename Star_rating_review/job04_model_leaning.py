import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPool1D, LSTM, Dropout, Dense, GRU
import numpy as np
from tensorflow.python.keras.saving.saved_model.load import metrics
from keras.callbacks import EarlyStopping

# 데이터 로드
X_train = np.load('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_train_max_129_wordsize_15845.npy', allow_pickle=True)
X_test = np.load('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_test_max_129_wordsize_15845.npy', allow_pickle=True)
Y_train = np.load('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_train_max_129_wordsize_15845.npy', allow_pickle=True)
Y_test = np.load('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_test_max_129_wordsize_15845.npy', allow_pickle=True)

# 원-핫 인코딩된 Y 데이터를 단일 열로 변환
Y_train = Y_train[:, 1]  # 두 번째 열만 사용 (1에 대한 확률)
Y_test = Y_test[:, 1]    # 두 번째 열만 사용 (1에 대한 확률)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 모델 정의
model = Sequential()

# Embedding 층
model.add(Embedding(input_dim=15845, output_dim=300))

# Convolution 층
model.add(Conv1D(64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=1))

# GRU 층들
model.add(GRU(512, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(256, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(GRU(64, activation='tanh'))
model.add(Dropout(0.3))

# 출력층
model.add(Dense(1, activation='sigmoid'))

# 모델 빌드
model.build(input_shape=(None, 129))
model.summary()

# Early Stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=2)

# 모델 컴파일
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 모델 학습
fit_hist = model.fit(X_train, Y_train,
                    batch_size=128,
                    epochs=30,
                    validation_data=(X_test, Y_test),
                    callbacks=[early_stopping])

# 평가
score = model.evaluate(X_test, Y_test, verbose=0)
print('학습결과: ', score[1])

# 모델 저장
model.save('./models/review_data_classfication_model_{}.h5'.format(
    fit_hist.history['val_accuracy'][-1]))

# 학습 결과 시각화
plt.plot(fit_hist.history['val_accuracy'], label='val_accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()