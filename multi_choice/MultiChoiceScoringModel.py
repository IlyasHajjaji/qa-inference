from rouge import Rouge
import pandas as pd
import numpy as np
import re

#First Iteration Score = 0.36 Model = potsawee/longformer-large-4096-answering-race

def multichoice_score(reference, completion) :
    """Compute difference scores given a completion and reference pair."""
    classes = ("A", "B", "C", "D")

    matches = [
        word
        for word in re.sub(r"\W", " ", completion).split()
        if word in classes
    ]

    # Take the last match as the answer
    if matches:
        output = matches[-1] == reference
    else:
        output = 0

    return output


#*******************************************************************************************************************


df_score = pd.read_csv("empty_data1.csv")
df_score = df_score[:200]
# Define new column names
new_columns = ['date', 'task', 'challenge', 'reference', 'result_model', 'Time_elapsed', 'Titles']

# Assign new column names
df_score.columns = new_columns

print(len(df_score))
print(df_score.columns)  # Check available column names
print(df_score.head())   # Check the first few rows of the DataFrame


df_score["MULTICHOICE_SCORE"] = None

rouge = Rouge()

for i in range(len(df_score)) :
    reference = str(df_score.loc[i, "reference"])
    completion = str(df_score.loc[i, "result_model"])
    df_score.loc[i, "MULTICHOICE_SCORE"] = multichoice_score(reference, completion)

df_score = df_score.dropna(subset=["result_model"]).reset_index(drop=True)
print("SUCCESS RATIO :",(sum(df_score.loc[:, "MULTICHOICE_SCORE"])/len(df_score))*100, "%")
df_score.to_csv("model_score_multichoice.csv", index=False)

#SUCCESS RATIO : 42.857142857142854 % potsawee
