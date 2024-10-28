import pandas as pd
import yaml
import os
from pathlib import Path

# ID, target 열만 가진 데이터 미리 호출
data_path = Path("results")
submission_df: pd.DataFrame = pd.read_csv(data_path / "lgbm.csv") 
submission_df.drop(columns=['deposit'], inplace=True)

# config 파일을 읽어서 config 변수에 저장
with open('config/ensemble.yaml', 'r') as file:
    config = yaml.safe_load(file)

# CSV 파일 경로와 가중치 가져오기
csv_files = [(item['path'], item['weight']) for item in config['csv_files']]

# 가중치를 곱한 DataFrame 저장 변수 초기화
weighted_sum_df = None

# 각 csv 파일을 읽고 가중치 적용하여 더하기
for csv_path, weight in csv_files:

    # csv 파일 읽기
    df = pd.read_csv(csv_path)

    # 가중치 적용
    weighted_df = df['deposit'] * weight

    # DataFrame을 가중치가 적용된 결과에 더하기
    if weighted_sum_df is None:
        weighted_sum_df = weighted_df
    else:
        weighted_sum_df += weighted_df

# output file 할당후 save 
submission_df = submission_df.assign(deposit = weighted_sum_df)

# 결과를 CSV 파일로 저장
output_filename = "Ensemble.csv"
output_path = os.path.join(data_path / output_filename)
submission_df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")