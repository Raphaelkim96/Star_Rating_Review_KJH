import pandas as pd

import os
from pandas.compat.numpy import np_long

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC



# 병합할 CSV 파일이 있는 폴더 경로 설정
folder_path = 'C:/PyCharm_workspace/Star_rating_review/test'

# 폴더 내 파일 확인
try:
    files_in_folder = os.listdir(folder_path)
    print(f"폴더 내 파일: {files_in_folder}")
except FileNotFoundError:  # 폴더가 없을 경우 예외 처리
    print(f"폴더 경로가 올바르지 않습니다: {folder_path}")

# CSV 파일 필터링
csv_files = [file for file in files_in_folder if file.endswith('.csv')]
print(f"CSV 파일 목록: {csv_files}")

# CSV 파일 병합
if csv_files:
    dataframes = [pd.read_csv(os.path.join(folder_path, file)) for file in csv_files]
    merged_df = pd.concat(dataframes, ignore_index=True)

    # 병합된 결과를 저장
    output_path = os.path.join(folder_path, "all_Coupang_Review.csv")
    merged_df.to_csv(output_path, index=False)
    print(f"병합된 파일이 저장되었습니다: {output_path}")
else:
    print("CSV 파일이 폴더에 없습니다.")

# 병합된 CSV 파일 확인
df = pd.read_csv('C:/PyCharm_workspace/Star_rating_review/test/all_Coupang_Review.csv')

# 중복 데이터 제거 및 정보 출력
df.drop_duplicates(inplace=True)
print(df.head())
df.info()
print(df.category.value_counts())