import pandas as pd
import preprocessing

print("n_rows_phishing:", preprocessing.n_rows_phishing)
print("n_rows_non_phishing:", preprocessing.n_rows_non_phishing)

print(preprocessing.df_phishing)
preprocessing.spell_stopwords(preprocessing.df_phishing, preprocessing.n_rows_phishing)
#preprocessing.spell_stopwords(preprocessing.df_non_phishing, preprocessing.n_rows_non_phishing)
preprocessing.df_phishing.to_csv('step1_phishing.csv', encoding = 'utf-8')
#preprocessing.df_non_phishing.to_csv('step1_non_phishing.csv', encoding='utf-8')

df_phishing = pd.read_csv('step1_phishing.csv', encoding = 'utf-8')
#df_non_phishing = pd.read_csv('step1_non_phishing.csv', encoding = 'utf-8')
print(df_phishing)
#print(df_non_phishing)


import morpheme
morpheme.mor(df_phishing)
# morpheme.mor(df_non_phishing)
df_phishing.to_csv('step2_phishing.csv', encoding = 'utf-8')
# df_non_phishing.to_csv('step2_non_phishing.csv', encoding = 'utf-8')
#
# #df 합치기
# df_concat = pd.concat([df_phishing, df_non_phishing], axis=0)
# # 파일 내보내기
# df_concat.to_csv('preprocessed_220810.csv', encoding = 'utf-8')