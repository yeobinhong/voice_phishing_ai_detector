#keras 참고문헌: https://wikidocs.net/32105
#autoencoder 참고문헌: https://velog.io/@jaehyeong/LSTM-Autoencoder-for-Anomaly-Detection#references
from tensorflow.keras import Model,models, layers, optimizers, regularizers
from tensorflow.keras.layers import LSTM, Embedding, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle
# config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,
#                         inter_op_parallelism_threads=8,
#                         allow_soft_placement=True,
#                         device_count = {'CPU': 2})
# #Tune using inter_op_parallelism_threads for best performance에러 뜬다면 intra~, inter~ 변수에 컴퓨터 성능에 맞는 숫자 넣어주세요
# # 해당 에러 참고문헌: https://stackoverflow.com/questions/41233635/meaning-of-inter-op-parallelism-threads-and-intra-op-parallelism-threads
# #cpu 코어, 스레드 수 확인 방법: https://joytk.tistory.com/44?category=908904
# session = tf.compat.v1.Session()(config=config)
tf.config.threading.set_inter_op_parallelism_threads(4)

# 1. 전처리 완료된 파일 불러오기
# import morpheme_1okt #사용한 토크나이저에 맞춰서 임포트
# print("import done")

#X: 임베딩된 sequence, Y: label
normal_X_train = np.load('normal_X_train_post.npy')
normal_X_valid = np.load('normal_X_valid_post.npy')
anomal_X_train = np.load('anomal_X_train_post.npy')
anomal_X_valid = np.load('anomal_X_valid_post.npy')
X_train = np.load('X_train_post.npy')
X_valid = np.load('X_valid_post.npy')
X_test = np.load('X_test_post.npy')
Y_train = np.load('Y_train_post.npy')
Y_valid = np.load('Y_valid_post.npy')
Y_test = np.load('Y_test_post.npy')

# 정규화
# from sklearn.preprocessing import StandardScaler
def scale(X, scaler):
    for i in range(X.shape[0]):
        scaled_X = scaler.transform(X[ :, :])
    return scaled_X
scaler = StandardScaler().fit(X_train)
# X_valid_scaled = scale(X_valid, scaler)
# X_test_scaled = scale(X_test, scaler)
# normal_X_train_scaled = scale(normal_X_train, scaler)
# normal_X_valid_scaled = scale(normal_X_valid, scaler)
# np.save('normal_X_train_scaled_post.npy', normal_X_train_scaled)
# np.save('normal_X_valid_scaled_post.npy', normal_X_valid_scaled)
# np.save('X_valid_scaled_post.npy', X_valid_scaled)
# np.save('X_test_scaled_post.npy', X_test_scaled)

normal_X_train_scaled = np.load('normal_X_train_scaled_post.npy')
normal_X_valid_scaled = np.load('normal_X_valid_scaled_post.npy')
X_valid_scaled = np.load('X_valid_scaled_post.npy')
X_test_scaled = np.load('X_test_scaled_post.npy')
print("scaling done")

# 2. 워드 임베딩으로 2차원 > 3차원 만들어서 lstm에 넣어줍니다
# vocab_size = len(morpheme_1okt.word_index)
vocab_size = 66600
encoding_dim = 256 #임베딩하고 싶은 차원으로 설정
input_dim = normal_X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,)) # input data: (number of samples , input_length) = (batch size = none, 2844); 2d
embedding = Embedding(vocab_size, encoding_dim)(input_layer) # embedding done: (number of samples, input_length, embedding word dimensionality) = (batch size = none, 2844, 256); 3d > lstm input

# 3. 모델링: lstm autoencoder training
# Encoder
encoder = LSTM(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5), return_sequences=True)(embedding)
encoder = LSTM(int(encoding_dim /2), activation="tanh", return_sequences=True)(encoder)
encoder = LSTM(int(encoding_dim / 4), activation="tanh", return_sequences=True)(encoder)
# Decoder
decoder = LSTM(int(encoding_dim / 4), activation='tanh', return_sequences=True)(encoder)
decoder = LSTM(int(encoding_dim / 2), activation='tanh', return_sequences=True)(decoder) #256으로 할거면 바꾸기
decoder = LSTM(int(encoding_dim), activation='tanh', return_sequences=True)(decoder)
decoder = LSTM(input_dim, activation='tanh', return_sequences=False)(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

# 4. compile, training
autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error', metrics=['accuracy'])
# print(autoencoder.summary())
early_stop = EarlyStopping(monitor = 'val_loss', min_delta = 0, mode = 'min', verbose =1 , patience = 5)
model_checkpoint = ModelCheckpoint(filepath='./{epoch}-{val_loss:.2f}-{val_accuracy:.2f}.h5', monitor='val_loss', save_best_only=True, verbose=1)
hist = autoencoder.fit(normal_X_train_scaled, normal_X_train_scaled, batch_size = 64, epochs=200, callbacks = [early_stop], validation_data=(normal_X_valid_scaled, normal_X_valid_scaled)) # batch size: 몇 개의 샘플로 가중치를 갱신할 것인지 지정
#미니 배치 경사 하강법(Mini-Batch Gradient Descent)

#모델 저장
autoencoder.save('lstm_post_0907.h5')
#history 저장
with open('hist_post_0907', 'wb') as file_pi:
    pickle.dump(hist.history, file_pi)


