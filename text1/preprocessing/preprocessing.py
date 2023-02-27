import pandas as pd
import split

df_phishing = split.df_phishing
df_non_phishing = split.df_non_phishing

#맞춤법 고치기, 불용어 지우기
import stopwords
import spelling
n_rows_phishing = df_phishing.shape[0]
n_rows_non_phishing = df_non_phishing.shape[0]

def spell_stopwords(data, n):
  for i in range(0, n):
    input = data.at[i,'text']
    #print("input:",i, input)
    hanspell_text = spelling.spell_check(input)
    #print("hanspell_text:",i, hanspell_text)
    new_text = stopwords.remove_stopword_text(hanspell_text)
    #print("new_text", i, new_text)
    data.loc[i,'text'] = new_text
  return data

