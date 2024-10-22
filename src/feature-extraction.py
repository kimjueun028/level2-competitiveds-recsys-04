import sys
import pandas as pd
import numpy as np
from pathlib import Path
from utils.compute_place_metrics import calculate_nearby_stats
from utils.config_loader import load_config

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data(data_path):
    # 주요 데이터 호출
    train_df = pd.read_csv(data_path / "train.csv")
    test_df = pd.read_csv(data_path / "test.csv")
    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    
    # sub data 호출
    interestrate_df = pd.read_csv(data_path / "interestRate.csv")
    park_df = pd.read_csv(data_path / "parkInfo.csv")
    school_df = pd.read_csv(data_path / "schoolinfo.csv")
    subway_df = pd.read_csv(data_path / "subwayInfo.csv")
    
    return train_df, test_df, sample_submission, interestrate_df, park_df, school_df, subway_df

def merge_interest_rate(df, interestrate_df):
    """금리 데이터를 병합하는 함수"""
    df = df.merge(interestrate_df[['year_month', 'interest_rate']], 
                  left_on='contract_year_month', right_on='year_month', how='left')
    df = df.drop(columns=['year_month'])
    return df

def save_data(df, file_path):
    """결과를 CSV 파일로 저장하는 함수"""
    df.to_csv(file_path, index=False)

def main(config_path):
    # 설정 파일 로드
    config = load_config(config_path)
    radii = config['radii_values']  # YAML 파일에서 radii 값 불러오기

    # 데이터 경로 설정
    data_path = Path("../data")

    # 주요 데이터 및 부가 데이터 로드
    train_df, test_df, _, interestrate_df, park_df, school_df, subway_df = load_data(data_path)

    # interest rate 병합
    train_df = merge_interest_rate(train_df, interestrate_df)
    test_df = merge_interest_rate(test_df, interestrate_df)

    # 장소 통계 계산
    places_dict = {'park': park_df, 'school': school_df, 'subway': subway_df}
    radii_dict = {'park': radii['park'], 'school': radii['school'], 'subway': radii['subway']}

    train_df = calculate_nearby_stats(train_df, places_dict, radii_dict)
    test_df = calculate_nearby_stats(test_df, places_dict, radii_dict)

    # 결과 저장
    save_data(train_df, data_path / "train_aftercountplace.csv")
    save_data(test_df, data_path / "test_aftercountplace.csv")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python script.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    main(config_path)

