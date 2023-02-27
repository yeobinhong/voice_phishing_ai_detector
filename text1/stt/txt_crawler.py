import glob
import os
import pandas as pd
from pathlib import Path
import sys 

root_dir = '/Users/wychoi/Desktop/seocho_ai/voice_dataset/Training'
pointer = ""
# root_dir에서 항상 시작하도록 정의
os.chdir(root_dir) 
# 컬럼만 있는 df 정의
df = pd.DataFrame(columns=['label','text','category', 'id'])
grand_dir_list = []
mother_dir_list = []
son_dir_list = []

def cd(path):
  os.chdir(path)

def grand_ls_to_list():
  pre_grand_dir_list = os.listdir('./')
  filtered_grand_dir_list = []
  for f_name in pre_grand_dir_list:
    if "D6" in f_name:
      filtered_grand_dir_list.append(f_name)
  return sorted(filtered_grand_dir_list)

def mother_ls_to_list():
  pre_mother_dir_list = os.listdir('./')
  filtered_mother_dir_list = []
  for f_name in pre_mother_dir_list:
    if "J" in f_name:
      filtered_mother_dir_list.append(f_name)
  return sorted(filtered_mother_dir_list)

def son_ls_to_list():
  pre_son_dir_list = os.listdir('./')
  filtered_son_dir_list = []
  for f_name in pre_son_dir_list:
    if "S0" in f_name:
      filtered_son_dir_list.append(f_name)
  return sorted(filtered_son_dir_list)

def set_pointer(g,m=None,s=None):
  if m is None:
    return f"{g}"
  elif  m is not None and s is None:
    return f"{g}/{m}"
  elif  m is not None and s is not None:
    return f"{g}/{m}/{s}"

# grand dir list 선언  ex) [D60_1, D60_2]
grand_dir_list = grand_ls_to_list()
print(grand_dir_list)

for cur_g in grand_dir_list:
  cd(root_dir)
  # cd g_dir & set pointer
  cd(cur_g)
  pointer = set_pointer(cur_g)
  # mother dir 필터링 받아서 리스트로 전환
  mother_dir_list = mother_ls_to_list()
  # cd m_dir & set pointer
  cd(mother_dir_list[0])
  pointer = set_pointer(cur_g,mother_dir_list[0])
  # son dir 필터링 받아서 리스트로 전환
  son_dir_list = son_ls_to_list()

  # S0000000x 디렉토리를 loop로 돈다
  for cur_s in son_dir_list:
    file_list = sorted(glob.glob(os.path.join(os.getcwd(), cur_s, "*.txt")))
    corpus = ""
    for file_path in file_list:
      with open(file_path) as f_input:
        if len(corpus) > 0:
          corpus = corpus + " " + f_input.read()
        else:
          corpus += f_input.read()

    pointer = set_pointer(cur_g,mother_dir_list[0],cur_s)
    df.loc[len(df.index)] = [0, corpus, None, pointer]    
    print(str(len(df.index))+"개 변환 완료")
    print(pointer)

    if pointer == "D62_2/J93/S00018624":
      os.chdir(root_dir) 
      filepath = Path('./results/training_result.csv') 
      filepath.parent.mkdir(parents=True, exist_ok=True) 
      df.to_csv(filepath)
      print("finished")
      sys.exit()
