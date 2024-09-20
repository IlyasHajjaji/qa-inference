import pandas as pd
import numpy as np
from angle_emb import AnglE
import torch
from scipy import spatial

#First Iteration Score = 0.45 Model = valhalla/longformer-base-4096-finetuned-squadv1

def relevance_score(reference, completion, model) :
        """Calculate the cosine similarity between sentence embeddings of the reference and completions.

        We subtract a baseline score which is what an empty string would get (a failed completion).
        This is usually around 0.35. We also clip the rewards between 0 and 1.
        The maximum effective score is around 0.65.
        """
        print('************************************************')
        print('REFERENCE :', reference)
        print('COMPLETION :', completion)

        reference_embedding = model.encode(reference, to_numpy=True)
        reference_emb_flatten = reference_embedding.flatten()

        # baseline is the cosine similarity between the reference and an empty string
        baseline = 1 - float(
            spatial.distance.cosine(
                reference_emb_flatten, model.encode("", to_numpy=True).flatten()
            )
        )

        if len(completion) == 0:
            score = 0
        else :
            emb = model.encode(completion, to_numpy=True)
            # Calculate cosine similarity between reference and completion embeddings, and subtract baseline
            score = 1 - float(spatial.distance.cosine(reference_emb_flatten, emb.flatten() - baseline))

        print("SCORE :", score)
        return score


#*******************************************************************************************************************



# Load your CSV
df_score = pd.read_csv("empty_data1.csv")
df_score = df_score[:200]

df_score["QA_SCORE"] = None  # Initialize QA_SCORE as None

model = AnglE.from_pretrained("WhereIsAI/UAE-Large-V1", pooling_strategy="cls", device="cuda:0")
model = model.cuda()

# Debug: Print initial data
print("Initial Data Sample:\n", df_score.head())

for i in range(len(df_score)):
    reference = str(df_score.iloc[i]['reference'])
    completion = str(df_score.iloc[i]['result_model'])
    
    # Check if the strings are properly assigned
    print(f"Processing Row {i+1}:")
    
    # Call relevance_score and debug print the result
    score = relevance_score(reference, completion, model)
    
    # Ensure that None values are converted to np.nan
    df_score.loc[i, "QA_SCORE"] = score

# Alternatively, you can also replace 'None' with 'NaN' before dropping
df_score["QA_SCORE"].replace({None: np.nan}, inplace=True)
df_score = df_score.dropna(subset=["QA_SCORE"]).reset_index(drop=True)

print(df_score.loc[:, "QA_SCORE"])


print("SUCCESS RATIO :",(sum(df_score.loc[:, "QA_SCORE"])/len(df_score)))
# Save the result to a CSV file
df_score.to_csv("model_score_qa_ha.csv", index=False)
