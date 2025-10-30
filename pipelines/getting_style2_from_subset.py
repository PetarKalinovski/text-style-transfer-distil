import pandas as pd
import json

def get_style2_from_subset(subset, full_data):
    subset_df = pd.read_csv(subset, header=None)
    full_df = pd.read_csv(full_data, sep='\t', header=None, names=['style1', 'style2'])

    merged_df = pd.merge(subset_df, full_df, left_on=subset_df.columns[0], right_on='style1', how='left')



    return merged_df[['style1', 'style2']]


if __name__ == "__main__":
    df=get_style2_from_subset('data/testing_subset.csv','data/val_en.txt')

    df.to_csv('data/training_subset_with_style2.csv', index=False)