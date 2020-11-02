import numpy as np
import pandas as pd

def extract_title_from_name(df):
    
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    df['Title'] = df['Title'].replace(['Lady', 'Countless', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Ms', 'Mlle'], 'Miss')
    df['Title'] = df['Title'].replace(['Mme'], 'Mrs')
    
    return df


def add_is_alone_column(df):
    family_size = df['SibSp'] + df['Parch'] + 1
    
    df['IsAlone'] = 0
    df.loc[family_size == 1, 'IsAlone'] = 1
    
    return df

def continous_category(series, n_bias):
    bins = pd.cut(series, n_bias, retbins=True)[1]
    
    return pd.Series(np.digitize(series, bins, right=True))

def add_categorical_columns(df):
    df['Title'] = df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)
    
    return df

def fill_nans(df, median_columns):
    df["Title"] = df["Title"].fillna(0)
    df["Embarked"] = df["Embarked"].fillna(df.Embarked.dropna().mode()[0])
    
    df = fill_nan_with_median(df, median_columns)
    
    return df
    
def fill_nan_with_median(df, columns):
    for col in columns:
        df[col] = df[col].fillna(df[col].dropna().median())

    return df    