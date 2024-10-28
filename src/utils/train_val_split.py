def train_val_Xy_split(holdout_start, holdout_end, train_df, test_df):
    valid_df = train_df[(train_df['contract_year_month'] >= holdout_start) & (train_df['contract_year_month'] <= holdout_end)]
    final_train_df = train_df[~((train_df['contract_year_month'] >= holdout_start) & (train_df['contract_year_month'] <= holdout_end))]

    X_train = final_train_df.drop(columns=['deposit_per_area'])
    y_train = final_train_df['deposit_per_area']
    X_valid = valid_df.drop(columns=['deposit_per_area'])
    y_valid = valid_df['deposit_per_area']
    X_test = test_df.copy()

    X_total = train_df.drop(columns=['deposit_per_area'])
    y_total = train_df['deposit_per_area']

    return X_train, y_train, X_valid, y_valid, X_test, X_total, y_total