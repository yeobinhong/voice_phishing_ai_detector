#형태소 단위로 쪼개기
from konlpy.tag import Okt
import preprocessing
import stopwords
okt = Okt()
df_phishing = preprocessing.df_phishing
df_non_phishing = preprocessing.df_non_phishing

my_weights = [
    ('num_nouns', -0.1),
    ('num_words', -0.2),
    ('no_noun', -1),
    ('len_sum_of_nouns', 0.2)
]

def my_evaluate_function(candidate):
    num_nouns = len([word for word, pos, begin, e in candidate if pos == 'Noun'])
    num_words = len(candidate)
    has_no_nouns = (num_nouns == 0)
    len_sum_of_nouns = 0 if has_no_nouns else sum(
        (len(word) for word, pos, _, _ in candidate if pos == 'Noun'))

    scores = (num_nouns, num_words, has_no_nouns, len_sum_of_nouns)
    score = sum((score * weight for score, (_, weight) in zip(scores, my_weights)))
    return score

#konlpy로 형태소 분석
def mor(df_phishing):
    for i in range(0, preprocessing.n_rows_phishing):
      text = preprocessing.df_phishing.at[i, 'text']
      list = okt.morphs(text)
      a =''
      for element in list:
        a = a+element+','
        a = a[:-1]
      text = a
      text = stopwords.josa_eomi(text) #조사, 어미 제거
      df_phishing.loc[i, 'text'] = text
    return df_phishing

def mor(df_non_phishing):
    for i in range(0, preprocessing.n_rows_non_phishing):
      text = df_non_phishing.at[i, 'text']
      list = okt.morphs(text)
      a =''
      for element in list:
        a = a+element+','
        a = a[:-1]
      text = a
      text = stopwords.josa_eomi(text)
      df_non_phishing.loc[i, 'text'] = text
    return df_non_phishing

#morpheme.twitter.set_evaluator(morpheme.my_weights, morpheme.my_evaluate_function)
