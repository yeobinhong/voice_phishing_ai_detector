import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics

print("import done")

normal_X_train_scaled = np.load('normal_X_train_scaled_post.npy')
normal_X_valid_scaled = np.load('normal_X_valid_scaled_post.npy')
X_valid_scaled = np.load('X_valid_scaled_post.npy')
X_test_scaled = np.load('X_test_scaled_post.npy')
Y_train = np.load('Y_train_post.npy')
Y_valid = np.load('Y_valid_post.npy')
Y_test = np.load('Y_test_post.npy')

# 모델 불러오기
from tensorflow.keras.models import load_model
autoencoder = load_model('lstm_post_0907.h5')
#history 불러오기
from pickle import load
with open('hist_post_0907', 'rb') as handle: # loading old history
    hist = load(handle)

# 5. threshold 설정
def flatten(X):
    flattened_X = np.empty((X.shape[0], X.shape[2]))  # sample x features array.
    for i in range(X.shape[0]):
        flattened_X[i] = X[i, (X.shape[1] - 1), :]
    return (flattened_X)

# X_valid_predictions = autoencoder.predict(X_valid_scaled)
# np.save('X_valid_predictions_post.npy', X_valid_predictions)
# 오래걸리니까 넘파이 파일 불러오기
X_valid_predictions = np.load('X_valid_predictions_post.npy')
mse = np.mean(np.power((X_valid_scaled) - (X_valid_predictions), 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error':mse,'True_class':list(Y_valid)})
precision_rt, recall_rt, threshold_rt = metrics.precision_recall_curve(error_df['True_class'], error_df['Reconstruction_error'])

# 그래프 그리기
# plt.figure(figsize=(8,5))
# plt.plot(threshold_rt, precision_rt[1:], label='Precision')
# plt.plot(threshold_rt, recall_rt[1:], label='Recall')
# plt.xlabel('Threshold'); plt.ylabel('Precision/Recall')
# plt.legend()
# print("threshold :")
# plt.savefig('threshold_post.png')

# best position of threshold
index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precision_rt, recall_rt)) if p==r][0]
# print('precision :',precision_rt[index_cnt],', recall: ',recall_rt[index_cnt])
# fixed Threshold
threshold_fixed = threshold_rt[index_cnt]
print('threshold :',threshold_fixed)

# 6. evaluation, prediction using threshold
# prediction
test_x_predictions = autoencoder.predict(X_test_scaled)
mse = np.mean(np.power(X_test_scaled - test_x_predictions, 2), axis=1)

error_df = pd.DataFrame({'Reconstruction_error': mse,
                         'True_class': Y_test.tolist()})
# 그래프 그리기
# groups = error_df.groupby('True_class')
# fig, ax = plt.subplots()
#
# for name, group in groups:
#     ax.plot(group.index, group.Reconstruction_error, marker='x', ms=3.5, linestyle='',
#             label= "phishing" if name == 1 else "Normal")
# ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
# ax.legend()
# plt.title("Reconstruction error for different classes")
# plt.ylabel("Reconstruction error")
# plt.xlabel("Data point index")
# print("anomaly detection predicted plot :")
# plt.savefig('recondstruction_error_marker_x.png')

# evaluation
y_pred = [1 if e > threshold_fixed else 0 for e in error_df['Reconstruction_error'].values]

cf_matrix = metrics.confusion_matrix(error_df['True_class'], y_pred)
# 그래프 그리기
# plt.figure(figsize=(7, 7))
#
# group_names = ['True Neg','False Pos','False Neg','True Pos']
# group_counts = ["{0:0.0f}".format(value) for value in
#                 cf_matrix.flatten()]
# group_percentages = ["{0:.2%}".format(value) for value in
#                      cf_matrix.flatten()/np.sum(cf_matrix)]
# labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
#           zip(group_names,group_counts,group_percentages)]
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
# plt.savefig('cf_matrix_post.png')
