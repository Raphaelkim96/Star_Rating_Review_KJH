import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


#오늘 날짜의 뉴스를 가져와 얼마나 잘 맞추는지 예측해보기

# CSV 파일 불러오기 및 중복 제거
df = pd.read_csv('C:/workspace/Star_rating_review/Star_rating_review/Star_All_Datas/All_Data.csv')
df.drop_duplicates(inplace=True)  # 중복 데이터 제거
df.reset_index(drop=True, inplace=True)  # 인덱스 초기화

# 데이터 프레임 정보 확인
print(df.head())  # 상위 5개 데이터 출력
df.info()  # 데이터 구조 확인
print(df.category.value_counts())  # 카테고리별 데이터 개수 확인

# X: 리뷰 제목, Y: 리뷰 카테고리로 분리
X = df['titles']
Y = df['category']


# 라벨 인코더 읽어오기
with open('C:/workspace/Star_rating_review/Star_rating_review/models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)


label = encoder.classes_  # 클래스 확인
print(label)

#이미 라벨을 가지고 있는걸 라벨링 할때 는 transform
#빈 엔코더를 사용하여 라벨링 할때는 fit_transform
labeled_y = encoder.transform(Y)


# 카테고리 데이터를 원-핫 인코딩
onehot_Y = to_categorical(labeled_y)
print(onehot_Y[0])


okt = Okt()
# 형태소 분석을 전체 데이터에 적용
for i in range(len(X)):
    if (i%100)==0:
        print('형태소:', i)
    X[i] = okt.morphs(X[i], stem=True)
print(X)  # 분석 결과 확인

# 불용어 처리
stopwords = pd.read_csv('C:/workspace/Star_rating_review/Star_rating_review/stopwords_data/stopwords.csv', index_col=0)
print(stopwords)

# 불용어 및 한 글자 단어 제거
for sentence in range(len(X)):
    words = []

    if (sentence%100)==0:
        print('불용어 처리:', sentence)


    for word in range(len(X[sentence])):
        if len(X[sentence][word]) > 1:  # 한 글자 제거
            if X[sentence][word] not in list(stopwords['stopword']):  # 불용어 제거
                words.append(X[sentence][word])
    X[sentence] = ' '.join(words)  # 단어들을 공백으로 연결

# 전처리 결과 확인
print(X[:5])



# 텍스트 데이터 숫자 라벨링 (단어 인덱싱)
# 기존에 모델에서 썼던 라벨링에 맞춰서 같게 해줘야 된다
# 그래서 기존 토큰을 저장함
#저장한 토큰 읽어오기
with open('C:/workspace/Star_rating_review/Star_rating_review/models/review_token_MAX_129.pickle','rb') as f: #새로 만들면 쓸 코드
    token = pickle.load(f)
tokened_X = token.texts_to_sequences(X)
print(tokened_X[:5])

#오늘 뉴스에서 긁어온 데이터가 어제 긁은 것보다 길이(MAX)가 클수도 있음
#그래서 자름
#x_pad = pad_sequences(tokened_X, maxlen=16, padding='post', truncating='post')
for i in range(len(tokened_X)):
    if len(tokened_X[i])>100:
        tokened_X[i] = tokened_X[i][:100]

#어제 보다 작은건 0으로 채움
X_pad = pad_sequences(tokened_X,100)

print(X_pad[:5])  # 첫 5개의 샘플 출력



#어제 모델과 비교하기

model = load_model('C:/workspace/Star_rating_review/Star_rating_review/models/review_data_classfication_model_0.9052031636238098.h5')
preds = model.predict(X_pad)

num_classes = len(label)  # 클래스 수 확인
onehot_Y = to_categorical(labeled_y, num_classes=num_classes)  # 클래스 수 명시
print('클래스 수: ',onehot_Y)

predicts = []
for pred in preds:
    most = label[np.argmax(pred)]#최대값
    pred[np.argmax(pred)] = 0 #제일 큰값을 0으로 해서 두번째 값 출력
    second = label[np.argmax(pred)]
    predicts.append([most,second])
df['predict'] = predicts

print(df.head(30))


#얼마나 맞췄는지 확인
score = model.evaluate(X_pad, onehot_Y)
print(score[1])

df['OX'] = 0
for i in range(len(df)):
    #맞으면 1로 덮어씀
    if df.loc[i,'category'] in df.loc[i,'predict']:
        df.loc[i, 'OX'] = 1
print(df.OX.mean())
