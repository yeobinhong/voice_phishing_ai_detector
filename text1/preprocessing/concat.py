import pandas as pd
#df 합치기
df_concat = pd.concat([df_phishing, df_non_phishing], axis=0)
# 파일 내보내기
df_concat.to_csv('preprocessed_220810.csv', encoding = 'utf-8')