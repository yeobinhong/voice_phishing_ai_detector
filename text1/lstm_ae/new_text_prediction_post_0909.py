import test_preprocessing_0909
import numpy as np
import pandas as pd

# 모델 불러오기
from tensorflow.keras.models import load_model
autoencoder = load_model('lstm_post_0907.h5')

print("입력하세요 :")
x_input = input()

# 데이터 전처리
x_input_preprocessed = test_preprocessing_0909.okt_test(x_input)
# print("x_input_preprocessed :", x_input_preprocessed)
# 토크나이징
X_input = []
x_input_preprocessed = str(x_input_preprocessed)
token = []
words = x_input_preprocessed.split(",")
for word in words:
    token.append(word.lower())
X_input.append(token)
# print("token :", X_input)
#인덱싱
import pickle
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

tokenizer.fit_on_texts(X_input)
X_input_encoded = tokenizer.texts_to_sequences(X_input) #sequence로 나타내기; 인코딩
# print("index :", X_input_encoded)
#패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 85
X_input_padded = pad_sequences(X_input_encoded, maxlen=int(max_len), padding = 'post', truncating='post')
# print("padded :", X_input_padded)
X_train = np.load('X_train_post.npy')

from sklearn.preprocessing import StandardScaler
def scale(X, scaler):
    for i in range(X.shape[0]):
        scaled_X = scaler.transform(X[ :, :])
    return scaled_X
scaler = StandardScaler().fit(X_train)
X_input_scaled = scale(X_input_padded, scaler)
# print(X_input_scaled)
input_X_predictions = autoencoder.predict(X_input_scaled)

# threshold를 기준으로 보이스피싱 판단하는 코드 짜기
mse = np.mean(np.power(X_input_scaled - input_X_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse})
# import model_prediction_0907
# threshold_fixed = model_prediction_0907.threshold_fixed # test file 바꿀 경우
threshold_fixed = 3.6004519942431243 #현재 데이터 그대로 사용할 경우
y_pred = [1 if e > threshold_fixed else 0 for e in error_df['Reconstruction_error'].values]
# print(error_df['Reconstruction_error'].values)
if y_pred == [1]:
    print("보이스피싱입니다")
else:
    print("보이스피싱이 아닙니다")
