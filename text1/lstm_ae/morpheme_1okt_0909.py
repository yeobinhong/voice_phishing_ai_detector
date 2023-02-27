# -*- coding: utf-8 -*- #맥북에서 한글 깨짐 방지 코드
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('preprocessed_done_0816.csv') #okt 전처리 되어 있는 파일로 돌리기
# print(df.head())
# print('중복 제거 전:', df.shape)
df.drop_duplicates(subset=['text'], inplace=True) #중복 제거
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# print('중복 제거 후:', df.shape)

# df의 text에 글자수를 계산하여 word_length column에 글자수 계산하여 넣음
df['length'] = df['text'].apply(lambda x:len(str(x).split(",")))
# print(df.head())
# length = df.groupby('label').mean()
# print(length)
# print(df.head())
x_data = df['text']
y_data = df['label']
x_len = len(x_data)
y_len = len(y_data)
# print('전체 텍스트 파일 개수: %d'%x_len)
# print("x_data:", x_data.head()) # 문장으로 돼있는 데이터

label0 = df[df.label == 0].describe()
label1 = df[df.label == 1].describe()
# print(label0, label1, sep="\n")

#train, test 분리 비율 8:2
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=0, stratify=y_data)
# print("split done")

# 토크나이징: 문장 데이터였던 x_train 대신 리스트 데이터 X_train 사용
# 사용한 토크나이저: 형태소 단위 토크나이저 okt(전처리 단계에서 완료)
X_train = []
for stc in x_train:
    stc = str(stc)
    token = []
    words = stc.split(",")
    for word in words:
        token.append(word.lower())
    X_train.append(token)

X_test = []
for stc in x_test:
    stc = str(stc)
    token = []
    words = stc.split(",")
    for word in words:
        token.append(word.lower())
    X_test.append(token)
# print("tokenization done")
# print("token :", X_train[:1])
# 인덱싱: 단어에 인덱스 부여
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_encoded = tokenizer.texts_to_sequences(X_train) #sequence로 나타내기; 인코딩
tokenizer.fit_on_texts(X_test)
X_test_encoded = tokenizer.texts_to_sequences(X_test)
word_index = tokenizer.word_index
# print("index :", X_train_encoded[:1])

# saving
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 데이터 padding: 벡터의 크기를 통일시켜준다
max_len = 85

X_train_padded = pad_sequences(X_train_encoded, maxlen=int(max_len), padding = 'post', truncating='post') #평균 벡터 길이에 맞춰서 행렬로 데이터를 정리
X_test_padded = pad_sequences(X_test_encoded, maxlen=int(max_len), padding = 'post', truncating='post') #padding = 'pre'를 선택하면 앞에 0을 채우고 'post'를 선택하면 뒤에 0을 채움
# print(X_train_padded.shape)
# print(X_test_padded.shape)

#X: 임베딩된 sequence, Y: label
X_train = X_train_padded
X_test = X_test_padded
Y_train = y_train
Y_test = y_test
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

Y_train_bool = Y_train.astype(bool)
Y_valid_bool = Y_valid.astype(bool)
normal_X_train = X_train[Y_train_bool==False] #autoencoder는 label 0만 사용(training, validation)
normal_X_valid = X_valid[Y_valid_bool==False]
anomal_X_train = X_train[Y_train_bool]
anomal_X_valid = X_valid[Y_valid_bool]

import numpy as np
np.save('normal_X_train_post.npy', normal_X_train)
np.save('normal_X_valid_post.npy', normal_X_valid)
np.save('anomal_X_train_post.npy', anomal_X_train)
np.save('anomal_X_valid_post.npy', anomal_X_valid)
np.save('X_train_post.npy', X_train)
np.save('X_valid_post.npy', X_valid)
np.save('X_test_post.npy', X_test)
np.save('Y_train_post.npy', Y_train)
np.save('Y_valid_post.npy', Y_valid)
np.save('Y_test_post.npy', Y_test)
