
import pandas as pd

age_mapping = {'0-10': 0, '10-20': 10, '20-30': 20, '30+': 30}

def age_grouping(main_df):

    main_df['age'] = main_df['age'].clip(lower=0)
    main_df['age_group'] = pd.cut(main_df['age'], bins=[0, 10, 20, 30, 300], labels=[0, 10, 20, 30], right=False)
    main_df['age_group'] = main_df['age_group'].astype(int)

    return main_df