import pandas as pd
df = pd.read_csv("../model_sample_qa.csv")[['date','task', 'challenge', 'reference']]
multichoice_df = df[df['task'] == 'date_qa'].reset_index(drop=True)

question = multichoice_df.loc[198, "challenge"]
print(question)