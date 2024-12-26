# 필요한 라이브러리 및 모듈 불러오기
import pickle
from operator import index
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from konlpy.tag import Okt, Kkma
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 데이터 전처리 과정
# CSV 파일 불러오기 및 중복 제거
df = pd.read_csv('C:/workspace/Star_rating_review/Star_rating_review/Star_All_Datas/All_Data.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

# 데이터 프레임 정보 확인
print(df.head())
df.info()
print(df.category.value_counts())

# X: 뉴스 제목, Y: 뉴스 카테고리로 분리
X = df['titles'].tolist()  # Series를 리스트로 변환
Y = df['category'].values  # numpy array로 변환

print()
print('Y values:', np.unique(Y))

# 원-핫 인코딩
onehot_Y = to_categorical(Y)

# 형태소 분석
okt = Okt()
okt_x = okt.morphs(X[0], stem=True)
print('Okt: ', okt_x)

# 형태소 분석을 전체 데이터에 적용
for i in range(len(X)):
    if (i%1000)==0:
        print(i)
    X[i] = okt.morphs(X[i], stem=True)

print('X: ',X)

# 불용어 처리
stopwords = pd.read_csv('C:/workspace/Star_rating_review/Star_rating_review/stopwords_data/stopwords.csv', index_col=0)
print(stopwords)

# 불용어 및 한 글자 단어 제거
for sentence in range(len(X)):
    words = []
    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:
            if X[sentence][word] not in list(stopwords['stopword']):
                words.append(X[sentence][word])
    X[sentence] = ' '.join(words)

print(X[:5])

# 텍스트 데이터 숫자 라벨링
token = Tokenizer()
token.fit_on_texts(X)
tokened_X = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1
print(wordsize)

# max 길이 조정
for i in range(len(tokened_X)):
    if len(tokened_X[i])>129:
        tokened_X[i] = tokened_X[i][:129]

X_pad = pad_sequences(tokened_X,129)

print(tokened_X[:5])

# 최대 길이 확인
max = 0
for i in range(len(tokened_X)):
    if max < len(tokened_X[i]):
        max = len(tokened_X[i])
print(max)

# 토큰 저장
with open('./models/review_token_MAX_{}.pickle'.format(max),'wb') as f:
    pickle.dump(token, f)

X_pad = pad_sequences(tokened_X, max)
print(X_pad)
print(len(X_pad[0]))

# 학습 및 테스트 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(X_pad, onehot_Y, test_size=0.1)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# 데이터 저장
np.save('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_train_max_{}_wordsize_{}'.format(max, wordsize), X_train)
np.save('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_train_max_{}_wordsize_{}'.format(max, wordsize), Y_train)
np.save('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_X_test_max_{}_wordsize_{}'.format(max, wordsize), X_test)
np.save('C:/workspace/Star_rating_review/Star_rating_review/crawling_data/review_data_Y_test_max_{}_wordsize_{}'.format(max, wordsize), Y_test)