import pandas as pd
import re

data = pd.read_csv('080822.csv', encoding = 'utf-8')
# text 열에서 중복인 내용이 있다면 중복 제거
data.drop_duplicates(subset=['text'], inplace=True)
# 정규표현식을 써서 전처리
p = re.compile(r' \(.*?\)/') #(문자)/(문자) 구조에서 뒤에거만 남김
data['text']= data['text'].str.replace(p, " ")
q = re.compile(r"[$&+:;~=@#|'<>^*()%!/(a-z)*(A-Z)*]") #특수기호,알파벳 남은거 있으면 지우기
data['text'] = data['text'].str.replace(q, '')
data['text'] = data['text'].replace(r'\s+',' ', regex=True) #remove 2 or more spaces
data['text'] = data['text'].replace(r"^\s+", '',regex=True) #remove space from start
data['text'] = data['text'].replace(r'\s+$', '', regex=True) #remove space from the end
# 개행문자 제거
data['text'] = data['text'].replace(r'\n','', regex=True)
print(data.head())
print('총 샘플의 수 :',len(data))
print('1/0의 개수')
print(data.groupby('label').size().reset_index(name='count'))
print("regex preprocessing done")
processed_data = data

#피싱과 정상으로 분리
cond1 = data['label'] == 1
cond2 = data['label'] == 0
df_phishing = data.loc[cond1]
df_non_phishing = data.loc[cond2]
# print(df_non_phishing)

#결측값 제거
df_phishing = df_phishing.dropna(subset = ['text'], axis=0)
df_phishing = df_phishing.reset_index()
df_phishing = df_phishing.drop(['index'], axis = 1)
df_non_phishing = df_non_phishing.dropna(subset = ['text'], axis=0)
df_non_phishing = df_non_phishing.reset_index()
df_non_phishing.drop(['index'], axis = 1, inplace =True)
# print(df_non_phishing)
print('결측값 여부 :',df_phishing['text'].isnull().values.any())
print('결측값 여부 :',df_non_phishing['text'].isnull().values.any())
#print("phishing df:", df_phishing.head(12))
#print("non phishing df:", df_non_phishing.head(12))
print("split done")