import pandas as pd
from sklearn.metrics import roc_auc_score

df = pd.read_csv("D:\Project\BTL-PTDL\src\data\hospital_readmissions.csv")

df.info()
print(df[df.duplicated()])
print(df[df.isna()].sum())
print(df[df.isnull()].sum())

df["age"] = df["age"].str.replace(r"[\[\]\(\)]", "", regex=True)
df['age'] = df['age'].str.split('-', expand=True).astype(float).mean(axis=1)
pd.to_numeric(df['age'], errors='coerce')
print(df["age"])