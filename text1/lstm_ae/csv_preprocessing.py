# csv 전처리 코드 오류나서 다시 작업
# 정규표현식으로 특수문자 제거까지 된 파일입니다

# from google.colab import drive #구글드라이브 마운트 시 주석 해제하고 사용하세요
# drive.mount('/content/drive')
# %cd /content/drive/My\ Drive/preprocessing/csv
import pandas as pd
# 사용하는 파일만 주석 해제해서 사용하시면 돼요
# phishing_1 = pd.read_csv('split_phishing.csv',encoding = 'utf-8') #step 1에 쓸 데이터
# 1_non0 = pd.read_csv('non0.csv', encoding = 'utf-8')
# 1_non1 = pd.read_csv('non1.csv', encoding = 'utf-8')
# 1_non2 = pd.read_csv('non2.csv', encoding = 'utf-8')
# 1_non3 = pd.read_csv('non3.csv', encoding = 'utf-8')
# 1_non4 = pd.read_csv('non4.csv', encoding = 'utf-8')
# phishing_1 = phishing_1.loc[:, ['label', 'text']]
# 1_non0 = 1_non0.loc[:, ['label', 'text']]
# 1_non1 = 1_non1.loc[:, ['label', 'text']]
# 1_non2 = 1_non2.loc[:, ['label', 'text']]
# 1_non3 = 1_non3.loc[:, ['label', 'text']]
# 1_non3 = 1_non4.loc[:, ['label', 'text']]
# n_rows_phishing_1 = phishing_1.shape[0]
# 1_n_rows_non0 = 1_non0.shape[0]
# 1_n_rows_non1 = 1_non1.shape[0]
# 1_n_rows_non2 = 1_non2.shape[0]
# 1_n_rows_non3 = 1_non3.shape[0]
# 1_n_rows_non4 = 1_non4.shape[0]
# phishing_2 = pd.read_csv('phishing_done_1.csv', encoding = 'utf-8') #step 2에 쓸 데이터
# 2_non0 = pd.read_csv('non0.csv', encoding = 'utf-8')
# 2_non1 = pd.read_csv('non1.csv', encoding = 'utf-8')
# 2_non2 = pd.read_csv('non2.csv', encoding = 'utf-8')
# 2_non3 = pd.read_csv('non3.csv', encoding = 'utf-8')
# 2_non4 = pd.read_csv('non4.csv', encoding = 'utf-8')
# phishing_2 = phishing_2.loc[:, ['label', 'text']]
# 2_non0 = 2_non0.loc[:, ['label', 'text']]
# 2_non1 = 2_non1.loc[:, ['label', 'text']]
# 2_non2 = 2_non2.loc[:, ['label', 'text']]
# 2_non3 = 2_non3.loc[:, ['label', 'text']]
# 2_non3 = 2_non4.loc[:, ['label', 'text']]
# n_rows_phishing_2 = phishing_2.shape[0]
# 2_n_rows_non0 = 2_non0.shape[0]
# 2_n_rows_non1 = 2_non1.shape[0]
# 2_n_rows_non2 = 2_non2.shape[0]
# 2_n_rows_non3 = 2_non3.shape[0]
# 2_n_rows_non4 = 2_non4.shape[0]

def spell_check(input): # input은 대화 1개
    from hanspell import spell_checker
    input_convert = input.replace('.', '.#').split('#') #문장 단위로 분리
    input_list = [""]
    for i in input_convert:
        if (len(input_list[-1]) + len(i)) < 500:
            input_list[-1] += i
        else:
            input_list.append(i)
    result = spell_checker.check(input_list)
    result = ''
    for j, k in enumerate(input_list):
        a = spell_checker.check([input_list[j]])
        a = a[0].checked
        result = result + a
    return result

def spell_stopwords_phishing(data, n): #1단계 공통 전처리
  # %cd /content/drive/My\ Drive/preprocessing/step1
  for i in range(0, n):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  print("done")
  return data

def spell_stopwords_nonphishing(data, n): #1단계 공통 전처리
  # %cd /content/drive/My\ Drive/preprocessing/step1
  for i in range(0, 1000):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_%d.csv' %(data,1000), encoding='utf-8')
  print("1000 done")
  for i in range(1000,2000):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_$d.csv' %(data, 2000), encoding='utf-8')
  print("2000 done")
  for i in range(2000,3000):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_$d.csv' %(data, 3000), encoding='utf-8')
  print("3000 done")
  for i in range(3000,4000):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_$d.csv' %(data, 4000), encoding='utf-8')
  print("4000 done")
  for i in range(4000,5000):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_$d.csv' %(data, 5000), encoding='utf-8')
  print("5000 done")
  for i in range(5000,6000):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_$d.csv' %(data, 6000), encoding='utf-8')
  print("6000 done")
  for i in range(6000,7000):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_$d.csv' %(data, 7000), encoding='utf-8')
  print("7000 done")
  for i in range(7000,8000):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_$d.csv' %(data, 8000), encoding='utf-8')
  print("8000 done")
  for i in range(8000,9000):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_$d.csv' %(data, 9000), encoding='utf-8')
  print("9000 done")
  for i in range(9000,n):
    input = data.at[i,'text']
    print(i)
    hanspell_text = spell_check(input)
    data.loc[i,'text'] = hanspell_text
  data.to_csv('%s_done.csv' %(data), encoding='utf-8')
  print("done")
  return data

## 여기까지 기본 전처리 끝 ##
## 이후로는 토크나이저에 따라 다른 전처리 진행

def josa_eomi(text):
    import re
    # % cd / content / drive / My\ Drive / ai_project / 데이터\ 전처리 / 취합본
    df_stopwords = pd.read_csv('한국어_불용어.csv')
    Josa = ['가', '같이', '같이나', '같이는', '같이는야', '같이는커녕', '같이도', '같이만', '같인', '고', '과', '과는', '과는커녕', '과도', '과를', '과만',
            '과만은', '과의', '까지', '까지가', '까지나', '까지나마', '까지는', '까지는야', '까지는커녕', '까지도', '까지든지', '까지라고', '까지라고는', '까지라고만은',
            '까지라도', '까지로', '까지로나', '까지로나마', '까지로는', '까지로는야', '까지로는커녕', '까지로도', '까지로든', '까지로든지', '까지로라서', '까지로라야',
            '까지로만', '까지로만은', '까지로서', '까지로써', '까지를', '까지만', '까지만은', '까지만이라도', '까지야', '까지야말로', '까지에', '까지와', '까지의',
            '까지조차', '까지조차도', '까진', '께옵서', '께옵서는', '께옵서는야', '께옵서는커녕', '께옵서도', '께옵서만', '께옵서만은', '께옵서만이', '께옵선', '나', '나마',
            '는', '는야', '는커녕', '니', '다', '다가', '다가는', '다가도', '다간', '대로', '대로가', '대로는', '대로의', '더러', '더러는', '더러만은', '도',
            '든', '든지', '라', '라고', '라고까지', '라고까지는', '라고는', '라고만은', '라곤', '라도', '라든지', '라서', '라야', '라야만', '라오', '라지',
            '라지요', '랑', '랑은', '로고', '로구나', '로구려', '로구먼', '로군', '로군요', '로다', '로되', '로세', '를', '마다', '마다라도', '마다를',
            '마다에게', '마다의', '마따나', '마저', '마저나마라도', '마저도', '마저라도', '마저야', '만', '만도', '만에', '만으로', '만으로는', '만으로도', '만으로라도',
            '만으로써', '만으론', '만은', '만을', '만의', '만이', '만이라도', '만치', '만큼', '만큼도', '만큼만', '만큼씩', '만큼은', '만큼의', '만큼이나',
            '만큼이라도', '만큼이야', '말고', '말고는', '말고도', '며', '밖에', '밖에는', '밖에도', '밖엔', '보고', '보고는', '보고도', '보고만', '보고만은',
            '보고만이라도', '보곤', '보다', '보다는', '보다는야', '보다도', '보다만', '보다야', '보단', '부터', '부터가', '부터나마', '부터는', '부터도', '부터라도',
            '부터를', '부터만', '부터만은', '부터서는', '부터야말로', '부터의', '부턴', '아', '야', '야말로', '에', '에게', '에게가', '에게까지', '에게까지는',
            '에게까지는커녕', '에게까지도', '에게까지만', '에게까지만은', '에게나', '에게는', '에게는커녕', '에게다', '에게도', '에게든', '에게든지', '에게라도', '에게로',
            '에게로는', '에게마다', '에게만', '에게며', '에게보다', '에게보다는', '에게부터', '에게서', '에게서가', '에게서까지', '에게서나', '에게서는', '에게서도',
            '에게서든지', '에게서라도', '에게서만', '에게서보다', '에게서부터', '에게서야', '에게서와', '에게서의', '에게서처럼', '에게선', '에게야', '에게와', '에게의',
            '에게처럼', '에게하고', '에게하며', '에겐', '에까지', '에까지는', '에까지도', '에까지든지', '에까지라도', '에까지만', '에까지만은', '에까진', '에나', '에는',
            '에다', '에다가', '에다가는', '에다간', '에도', '에든', '에든지', '에라도', '에로', '에로의', '에를', '에만', '에만은', '에부터', '에서', '에서가',
            '에서까지', '에서까지도', '에서나', '에서나마', '에서는', '에서도', '에서든지', '에서라도', '에서만', '에서만도', '에서만이', '에서만큼', '에서만큼은',
            '에서보다', '에서부터', '에서부터는', '에서부터도', '에서부터라도', '에서부터만', '에서부터만은', '에서야', '에서와', '에서와는', '에서와의', '에서의', '에서조차',
            '에서처럼', '에선', '에야', '에의', '에조차도', '에하며', '엔', '엔들', '엘', '엘랑', '여', '와', '와는', '와도', '와라도', '와를', '와만',
            '와만은', '와에만', '와의', '와처럼', '와한테', '요', '으로', '으로가', '으로까지', '으로까지만은', '으로나', '으로나든지', '으로는', '으로도', '으로든지',
            '으로라도', '으로랑', '으로만', '으로만은', '으로부터', '으로부터는', '으로부터는커녕', '으로부터도', '으로부터만', '으로부터만은', '으로부터서는', '으로부터서도',
            '으로부터서만', '으로부터의', '으로서', '으로서가', '으로서나', '으로서는', '으로서도', '으로서든지', '으로서라도', '으로서만', '으로서만도', '으로서만은',
            '으로서야', '으로서의', '으로선', '으로써', '으로써나', '으로써는', '으로써라도', '으로써만', '으로써야', '으로야', '으로의', '으론', '은', '은커녕', '을',
            '의', '이', '이고', '이나', '이나마', '이니', '이다', '이든', '이든지', '이라', '이라고', '이라고는', '이라고도', '이라고만은', '이라곤', '이라는',
            '이라도', '이라든지', '이라서', '이라야', '이라야만', '이랑', '이랑은', '이며', '이며에게', '이며조차도', '이야', '이야말로', '이여', '인들', '인즉',
            '인즉슨', '일랑', '일랑은', '조차', '조차가', '조차도', '조차를', '조차의', '처럼', '처럼과', '처럼도', '처럼만', '처럼만은', '처럼은', '처럼이라도',
            '처럼이야', '치고', '치고는', '커녕', '커녕은', '커니와', '토록', '하고', '하고가', '하고는', '하고는커녕', '하고도', '하고라도', '하고마저', '하고만',
            '하고만은', '하고야', '하고에게', '하고의', '하고조차', '하고조차도', '하곤']
    Eomi = ['거나', '거늘', '거니', '거니와', '거드면', '거드면은', '거든', '거들랑', '거들랑은', '건', '건대', '건댄', '건마는', '건만', '것다', '게', '게끔',
            '게나', '게나마', '게는', '게도', '게라도', '게만', '게만은', '게시리', '게요', '고', '고는', '고도', '고만', '고말고', '고서', '고서는', '고서도',
            '고선', '고야', '고요', '고자', '곤', '관데', '구나', '구려', '구료', '구먼', '군', '군요', '기', '기까지', '기까지는', '기까지도', '기까지만',
            '기까지만은', '기로', '기로서', '기로서니', '기로선들', '기에', '긴', '길', '나', '나니', '나마', '나요', '나이까', '나이다', '냐', '냐고', '냐는',
            '냐라고', '냐라고도', '냐라고만', '냐에', '네', '네만', '네요', '노', '노라', '노라고', '노라니', '노라면', '느냐', '느냐고', '느냐는', '느냐라고',
            '느냐라고는', '느냐라고도', '느냐라고만', '느냐라고만은', '느냐에', '느뇨', '느니', '느니라', '느니만', '느라', '느라고', '는', '는가', '는가라고',
            '는가라는', '는가를', '는가에', '는걸', '는고', '는구나', '는구려', '는구료', '는구먼', '는군', '는다', '는다거나', '는다고', '는다고는', '는다는',
            '는다는데', '는다니', '는다니까', '는다든지', '는다마는', '는다만', '는다만은', '는다며', '는다며는', '는다면', '는다면서', '는다면은', '는단다', '는담',
            '는답니까', '는답니다', '는답디까', '는답디다', '는답시고', '는대', '는대로', '는대서', '는대서야', '는대야', '는대요', '는데', '는데는', '는데다', '는데도',
            '는데서', '는만큼', '는만큼만', '는바', '는지', '는지가', '는지고', '는지는', '는지도', '는지라', '는지를', '는지만', '는지에', '는지요', '는지의', '니',
            '니까', '니까는', '니깐', '니라', '니만치', '니만큼', '다', '다가', '다가는', '다가도', '다간', '다거나', '다고', '다고까지', '다고까지는', '다고까지도',
            '다고까지라도', '다고까지만', '다고까지만은', '다고는', '다고도', '다고만', '다고만은', '다고요', '다곤', '다느냐', '다느니', '다는', '다는데', '다니',
            '다마는', '다마다', '다만', '다만은', '다며', '다며는', '다면', '다면서', '다면서도', '다면야', '다면은', '다시피', '다오', '단', '단다', '담',
            '답시고', '더구나', '더구려', '더구먼', '더군', '더군요', '더냐', '더니', '더니라', '더니마는', '더니만', '더라', '더라도', '더라며는', '더라면', '더란',
            '더면', '던', '던가', '던가요', '던걸', '던걸요', '던고', '던데', '던데다', '던데요', '던들', '던지', '데', '데도', '데요', '도록', '도록까지',
            '도록까지도', '도록까지만', '도록까지만요', '도록까지만은', '되', '든', '든지', '듯', '듯이', '디', '라', '라고', '라고까지', '라고까지는', '라고까지도',
            '라고까지만', '라고까지만은', '라고는', '라고도', '라고만', '라고만은', '라곤', '라느니', '라는', '라는데', '라는데도', '라는데요', '라니', '라니까',
            '라니까요', '라도', '라든지', '라며', '라면', '라면서', '라면서까지', '라면서까지도', '라면서도', '라면서요', '란', '란다', '란다고', '람', '랍니까',
            '랍니다', '랍디까', '랍디다', '랍시고', '래', '래도', '랴', '랴마는', '러', '러니', '러니라', '러니이까', '러니이다', '러만', '러만은', '러이까',
            '러이다', '런가', '런들', '려', '려거든', '려고', '려고까지', '려고까지도', '려고까지만', '려고까지만은', '려고는', '려고도', '려고만', '려고만은', '려고요',
            '려기에', '려나', '려네', '려느냐', '려는', '려는가', '려는데', '려는데요', '려는지', '려니', '려니까', '려니와', '려다', '려다가', '려다가는',
            '려다가도', '려다가요', '려더니', '려더니만', '려던', '려면', '려면요', '려면은', '려무나', '련', '련마는', '련만', '렴', '렷다', '리', '리까',
            '리니', '리니라', '리다', '리라', '리라는', '리란', '리로다', '리만치', '리만큼', '리요', '리요마는', '마', '매', '며', '며는', '면', '면서',
            '면서까지', '면서까지도', '면서까지만은', '면서도', '면서부터', '면서부터는', '면요', '면은', '므로', '사', '사오이다', '사옵니까', '사옵니다', '사옵디까',
            '사옵디다', '사외다', '세', '세요', '소', '소서', '소이다', '쇠다', '습니까', '습니다', '습니다마는', '습니다만', '습디까', '습디다', '습디다마는',
            '습디다만', '아', '아다', '아다가', '아도', '아라', '아서', '아서까지', '아서는', '아서도', '아서만', '아서요', '아선', '아야', '아야만', '아요',
            '어', '어다', '어다가', '어도', '어라', '어서', '어서까지', '어서는', '어서도', '어서만', '어서만은', '어선', '어야', '어야만', '어야지', '어야지만',
            '어요', '어지이다', '언정', '엇다', '오', '오리까', '오리까마는', '오리까만', '오리다', '오이다', '올습니다', '올습니다마는', '올습니다만', '올시다',
            '옵나이까', '옵나이다', '옵니까', '옵니다', '옵니다만', '옵디까', '옵디다', '외다', '요', '으나', '으나마', '으냐', '으냐고', '으니', '으니까',
            '으니까는', '으니깐', '으니라', '으니만치', '으니만큼', '으라', '으라고', '으라고까지', '으라고까지는', '으라고까지도', '으라고까지만은', '으라고는', '으라고도',
            '으라고만', '으라고만은', '으라고요', '으라느니', '으라는', '으라니', '으라니까', '으라든지', '으라며', '으라면', '으라면서', '으라면은', '으란', '으람',
            '으랍니까', '으랍니다', '으래', '으래서', '으래서야', '으래야', '으래요', '으랴', '으랴마는', '으러', '으러까지', '으러까지도', '으려', '으려거든', '으려고',
            '으려고까지', '으려고까지는', '으려고까지도', '으려고까지만', '으려고까지만은', '으려고는', '으려고도', '으려고만', '으려고만은', '으려고요', '으려기에', '으려나',
            '으려느냐', '으려느냐는', '으려는', '으려는가', '으려는데', '으려는데도', '으려는데요', '으려는지', '으려니', '으려니까', '으려니와', '으려다', '으려다가',
            '으려다가는', '으려다가요', '으려다간', '으려더니', '으려면', '으려면야', '으려면은', '으려무나', '으려서야', '으려오', '으련', '으련다', '으련마는', '으련만',
            '으련만은', '으렴', '으렵니까', '으렵니다', '으렷다', '으리', '으리까', '으리니', '으리니라', '으리다', '으리라', '으리로다', '으리만치', '으리만큼',
            '으리요', '으마', '으매', '으며', '으면', '으면서', '으면서까지', '으면서까지도', '으면서까지만', '으면서까지만은', '으면서는', '으면서도', '으면서부터',
            '으면서부터까지', '으면서부터까지도', '으면서부터는', '으면서요', '으면요', '으면은', '으므로', '으세요', '으셔요', '으소서', '으시어요', '으오', '으오리까',
            '으오리다', '으오이다', '으옵니까', '으옵니다', '으옵니다만', '으옵디까', '으옵디다', '으외다', '으이', '은', '은가', '은가를', '은가에', '은가에도',
            '은가에만', '은가요', '은걸', '은걸요', '은고', '은다고', '은다고까지', '은다고까지도', '은다고는', '은다는', '은다는데', '은다니', '은다니까', '은다든지',
            '은다마는', '은다면', '은다면서', '은다면서도', '은다면요', '은다면은', '은단다', '은담', '은답니까', '은답니다', '은답디까', '은답디다', '은답시고', '은대',
            '은대서', '은대서야', '은대야', '은대요', '은데', '은데는', '은데다', '은데도', '은데도요', '은데서', '은들', '은만큼', '은만큼도', '은만큼만은', '은만큼은',
            '은바', '은즉', '은즉슨', '은지', '은지가', '은지고', '은지는', '은지도', '은지라', '은지라도', '은지를', '은지만', '은지만은', '은지요', '을', '을거나',
            '을거냐', '을거다', '을거야', '을거지요', '을걸', '을까', '을까마는', '을까봐', '을까요', '을께', '을께요', '을꼬', '을는지', '을는지요', '을라',
            '을라고', '을라고까지', '을라고까지도', '을라고까지만', '을라고는', '을라고도', '을라고만', '을라고만은', '을라고요', '을라요', '을라치면', '을락', '을래',
            '을래도', '을래요', '을러니', '을러라', '을런가', '을런고', '을레', '을레라', '을만한', '을망정', '을밖에', '을밖에요', '을뿐더러', '을새', '을세라',
            '을세말이지', '을소냐', '을수록', '을쏘냐', '을이만큼', '을작이면', '을지', '을지가', '을지나', '을지니', '을지니라', '을지도', '을지라', '을지라도',
            '을지어다', '을지언정', '을지요', '을진대', '을진댄', '을진저', '을테다', '을텐데', '음', '음세', '음에도', '음에랴', '읍쇼', '읍시다', '읍시다요',
            '읍시오', '자', '자고', '자고까지', '자고까지는', '자고까지라도', '자고는', '자고도', '자고만', '자고만은', '자꾸나', '자는', '자마자', '자면', '자면요',
            '잔', '잘', '지', '지는', '지도', '지를', '지마는', '지만', '지요', '진', '질']
    Else = df_stopwords['stop_word'].values.tolist()
    pattern0 = re.compile(r'\b(' + r'|'.join(Josa) + r')\b\s*')
    pattern1 = re.compile(r'\b(' + r'|'.join(Eomi) + r')\b\s*')
    pattern = re.compile(r'\b(' + r'|'.join(Else) + r')\b\s*')
    text = pattern0.sub('', text)
    text = pattern1.sub('', text)
    text = pattern.sub('', text)
    pattern2 = re.compile('\s+')
    text = pattern2.sub(' ', text)
    pattern3 = re.compile('^\s+')
    text = pattern3.sub('', text)
    pattern4 = re.compile('\s+$')
    text = pattern4.sub('', text)
    pattern5 = re.compile(',{2,}')
    text = pattern5.sub(',', text)
    pattern6 = re.compile(',,+')
    text = pattern6.sub(',', text)
    return text

def okt_csv(data, n):
    # % cd / content / drive / My\ Drive / Colab_Notebooks
    from konlpy.tag import Okt
    import re
    okt = Okt()
    # %cd /content/drive/My\ Drive/preprocessing/final
    for i in range(0, n):
      text = data.at[i, 'text']
      text = str(text)
      list = okt.morphs(text)
      a =''
      for element in list:
        a = a+element+','
      a = a[:-1]
      text = a
      text = josa_eomi(text) #조사, 어미 제거
      data.loc[i, 'text'] = text
      print(i)
    data['text'] = data['text'].str.replace('.', '')  # 온점 지우기
    p = re.compile(r"^,")
    data['text'] = data['text'].str.replace(p, '')  # 쉼표로 시작하는 것 지우기
    q = re.compile(r"$,")
    data['text'] = data['text'].str.replace(q, '')  # 쉼표로 끝나는 것 지우기
    r = re.compile(r"[-]+")  # 특수기호 `도 지우기
    data['text'] = data['text'].str.replace(r, '')
    s = re.compile(r"[`]+")  # 특수기호 `도 지우기
    data['text'] = data['text'].str.replace(s, '')
    data['text'] = data['text'].str.replace(',,', ',')  # 쉼표 2개면 지우기
    data = data.loc[:, ['label', 'text']]
    data = data.dropna(subset=['text'], axis=0) # ,,/ ,로 시작하거나 끝남/ -` 등 남은 특수기호 지우고 결측값 제거하는 최종 전처리
    return data

# main에 이렇게 놓고 전처리 다시 할 수 있다
# import pandas as pd
# import csv_preprocessing
#
# #step 1
# phishing_1 = pd.read_csv('split_phishing.csv',encoding = 'utf-8') #step 1에 쓸 데이터
# phishing_1 = phishing_1.loc[:, ['label', 'text']]
# n_rows_phishing_1 = phishing_1.shape[0]
# print("start step 1")
# phishing_1 = csv_preprocessing.spell_stopwords_phishing(phishing_1, n_rows_phishing_1)
# phishing_1.to_csv('%s_done.csv' % phishing_1, encoding='utf-8')
# print("step 1 done")
#
# #step 2
# phishing_2 = pd.read_csv('phishing_1_done.csv', encoding = 'utf-8') #step 2에 쓸 데이터
# phishing_2 = phishing_2.loc[:, ['label', 'text']]
# n_rows_phishing_2 = phishing_2.shape[0]
# print("start step 2")
# phishing_2 = csv_preprocessing.okt_csv(phishing_2, n_rows_phishing_2)
# phishing_2.to_csv('final_phishing.csv', encoding='utf-8')
# print("step 2 done")