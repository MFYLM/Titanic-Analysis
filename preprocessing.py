from ast import Pass
import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import resample


def column_selection(filename: str) -> pd.DataFrame:            
    df = pd.read_csv(filename)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    #df["Survived"].value_counts().plot.bar()
    df["Fare"].fillna(df["Fare"].mean(), inplace=True)
    # mapping features to numerical values
    d1 = {"female": 0, "male": 1}
    d2 = {"S": 0, "C": 1, "Q": 2}

    df["Sex"] = df["Sex"].map(d1)
    df["Embarked"] = df["Embarked"].map(d2)


    mapping = {"Mr": 1, "Miss": 0, "Mrs": 1}                 # extracting married column from their names. Thanks to Jicheng Li
    df["Married"] = df["Name"].str.extract("([A-Za-z]+)\.")
    df["Married"] = df["Married"].map(mapping)
    
    df["Married"] = df["Married"].fillna(value=1)            # replacing missing categorical data with the most frequent label
         
    df = df.drop(columns="PassengerId", axis=1).drop(columns="Name", axis=1).drop(columns="Cabin", axis=1).drop(columns="Ticket")
    return df.dropna()



def dataResampling(df, n_sample, r_state) -> pd.DataFrame:
    df["Survived"].value_counts().plot.bar()
    temp_df = df[df["Survived"] == 1]               # get 'dead' samples
    other_df = df[df["Survived"] != 1]
    temp_df_upsampled = resample(temp_df, n_samples=n_sample, random_state=r_state, replace=True)
    df = pd.concat([temp_df_upsampled, other_df])
    return df
