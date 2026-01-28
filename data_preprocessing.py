import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_and_explore_data(filepath):
    # load the dataset using pandas
    df = pd.read_csv(filepath)
    print("data loaded.")

    # print basic info about the data
    print("shape of data:", df.shape)
    print("first 5 rows:")
    print(df.head())
    print("data types:")
    print(df.dtypes)

    # remove the target column because this is unsupervised learning
    # usually the target is called 'target' or 'output'
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
        print("removed target column.")
    elif 'output' in df.columns:
        df = df.drop(columns=['output'])
        print("removed output column.")
            
    # calculate basic stats
    print("basic statistics:")
    print(df.describe())

    # visualization part
    print("making plots...")
    
    # histogram
    df.hist(figsize=(10, 8), bins=15)
    plt.title("feature distributions")
    plt.savefig('1_histograms.png')
    plt.close()
    print("saved histogram.")

    # box plots for outliers
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title("box plots")
    plt.savefig('2_boxplots.png')
    plt.close()
    print("saved boxplots.")

    # scatter plot
    # plotting age vs max heart rate
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='age', y='thalach', data=df)
    plt.title("age vs max heart rate")
    plt.savefig('3_scatter.png')
    plt.close()
    print("saved scatter plot.")
    
    return df

def preprocess_data(df):
    print("starting preprocessing...")
    
    # remove rows with missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print("dropped missing values.")
        
    # remove id column if it exists
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # remove duplicates
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        print("dropped duplicates.")

    # one hot encoding for categorical columns
    # finding columns with less than 5 unique values to encode
    cat_cols = []
    num_cols = []
    
    for col in df.columns:
        if df[col].nunique() < 5:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    
    print("categorical columns:", cat_cols)
    print("numerical columns:", num_cols)
    
    df_encoded = pd.get_dummies(df, columns=cat_cols)
    
    # scaling the numerical features
    scaler = StandardScaler()
    df_final = df_encoded.copy()
    df_final[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    
    print("preprocessing done.")
    return df_final