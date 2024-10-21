import os
import pandas as pd
import numpy as np
from pathlib import Path
from utils.compute_place_metrics import calculate_nearby_stats


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 파일 호출
data_path = Path("../data")
train_df = pd.read_csv(data_path / "train.csv")
test_df = pd.read_csv(data_path / "test.csv")
sample_submission = pd.read_csv(data_path / "sample_submission.csv")

# sub data
interestrate_df = pd.read_csv(data_path / "interestRate.csv")
park_df = pd.read_csv(data_path / "parkInfo.csv")
school_df = pd.read_csv(data_path / "schoolinfo.csv")
subway_df = pd.read_csv(data_path / "subwayInfo.csv")

# merge interest rate
train_df = train_df.merge(interestrate_df[['year_month', 'interest_rate']], left_on='contract_year_month', right_on='year_month', how='left')
train_df = train_df.drop(columns=['year_month'])

test_df = test_df.merge(interestrate_df[['year_month', 'interest_rate']], left_on='contract_year_month', right_on='year_month', how='left')
test_df = test_df.drop(columns=['year_month'])

places_dict = {'park':park_df, 'school':school_df, 'subway':subway_df}
radii = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000]


train_df = calculate_nearby_stats(train_df, places_dict, radii)
test_df = calculate_nearby_stats(test_df, places_dict, radii)

train_df.to_csv("../data/train_aftercountplace.csv", index=False)
test_df.to_csv("../data/test_aftercountplace.csv", index=False)
