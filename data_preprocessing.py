import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_and_explore_data(filepath):
    # load the dataset
    df = pd.read_csv(filepath)
    print("data loaded.")

    # basic info
    print("shape:", df.shape)
    print("first 5 rows:")
    print(df.head())
    print("types:")
    print(df.dtypes)

    # remove target/output column (step 3)
    # checking for common target names
    if 'target' in df.columns:
        df = df.drop(columns=['target'])
        print("removed target column.")
    elif 'output' in df.columns:
        df = df.drop(columns=['output'])
        print("removed output column.")
            
    # stats (step 4)
    # showing only mean, min, std, max
    print("stats:")
    print(df.describe().loc[['mean', 'min', 'std', 'max']])

    # visualization (step 5)
    print("making plots...")
    
    # histograms
    df.hist(figsize=(10, 8), bins=15)
    plt.title("feature distributions")
    plt.savefig('1_histograms.png')
    plt.close()
    
    # box plots
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=df)
    plt.xticks(rotation=45)
    plt.title("box plots")
    plt.savefig('2_boxplots.png')
    plt.close()

    # scatter plot
    plt.figure(figsize=(8, 5))
    if 'age' in df.columns and 'thalach' in df.columns:
        sns.scatterplot(x='age', y='thalach', data=df)
        plt.title("age vs max heart rate")
        plt.savefig('3_scatter.png')
        plt.close()
    
    print("saved all plots.")
    return df

def preprocess_data(df):
    print("starting preprocessing...")
    
    # STEP 6: Identify and remove missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"found {missing_count} missing values. removing them...")
        df = df.dropna()
    else:
        print("step 6: no missing values found.")
        
    # STEP 7: Remove irrelevant columns
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        print("step 7: removed 'id' column.")
    else:
        print("step 7: no irrelevant 'id' column found.")
        
    # STEP 8: Remove duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"found {dupes} duplicates. removing...")
        df = df.drop_duplicates()
    else:
        print("step 8: no duplicates found.")

    # STEP 9: One Hot Encoding
    # MANUALLY DEFINING LISTS TO BE EXACT
    # categorical: cp, restecg, slope, thal, ca (as you requested)
    # plus sex, fbs, exang because they are binary categories
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'ca']
    
    # numerical: the rest (continuous numbers)
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    print("categorical columns:", cat_cols)
    print("numerical columns:", num_cols)
    
    # perform one hot encoding
    df_encoded = pd.get_dummies(df, columns=cat_cols)
    
    # STEP 11: Normalization
    scaler = StandardScaler()
    df_final = df_encoded.copy()
    
    # scale only the numerical columns
    df_final[num_cols] = scaler.fit_transform(df_encoded[num_cols])
    
    print("preprocessing done.")
    return df_final