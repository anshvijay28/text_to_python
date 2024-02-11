import pandas as pd 
import json

df1 = pd.read_csv("data.csv")
df1 = df1.rename(columns={"question": "Problem", "code":"Python Code"})
df2 = pd.read_csv("ProblemSolutionPythonV3.csv")
df2 = df2.iloc[:, 1:3]

df_combine = pd.concat([df1, df2], ignore_index=True, axis=0)
df_combine = df_combine.rename(columns={"Problem": "input", "Python Code": "output"})
df_beg = df_combine.iloc[:100, :]

df_train = df_beg.iloc[:80, :]
df_val = df_beg.iloc[80:, :]

train_json = df_train.to_dict(orient='records')
val_json = df_val.to_dict(orient='records')

with open('train.json', 'w') as f:
    json.dump(train_json, f, indent=4)

with open('val.json', 'w') as f:
    json.dump(val_json, f, indent=4)

