from rouge import Rouge
import pandas as pd
import numpy as np
import re

#First Iteration Score = 0.54 Model = valhalla/longformer-base-4096-finetuned-squadv1
# DATE SCORE : 0.5997401300927698
# DATE SCORE : 0.6497283177901685

def date_diff(ref_date, comp_date):
    """
    Calculates the absolute difference in days between two dates.
    """
    DATE_NOT_FOUND_CODE = 9999
    if not comp_date:
        return DATE_NOT_FOUND_CODE
    # Check if ref date is just a year
    if ref_date.isdigit():
        # Extract the last 3-4 digits from the completion date using a regex pattern that would detect 3 or 4 digit years
        comp_year = re.findall(r"\b\d{3,4}\b", comp_date)
        # Extract the last 3-4 digits from the completion date using a regex pattern that would detect 3 or 4 digit years
        comp_year = re.findall(r"\b\d{3,4}\b", comp_date)
        if comp_year:
            return abs(int(ref_date) - int(comp_year[0])) * 365
            return abs(int(ref_date) - int(comp_year[0])) * 365
        else:
            return DATE_NOT_FOUND_CODE
    # If the reference date is not only a year, take the difference between the two dates
    try:
        ref_date = pd.to_datetime(ref_date)
        comp_date = pd.to_datetime(comp_date)
        return abs((ref_date - comp_date).days)
    except Exception as _:
        if ref_date == comp_date:
            return 0
        else:
            return DATE_NOT_FOUND_CODE

def parse_dates_from_text(text):
    # Regular expression to find dates in various formats
    date_pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember))\s+\d{4}\b|\b\d{4}\b"
    date_pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{4}\b|\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember))\s+\d{4}\b|\b\d{4}\b"

    # Compile the regex pattern
    date_regex = re.compile(date_pattern)

    # Split text into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", text)

    # Initialize dictionary to store results

    # Iterate through sentences and find dates
    for sentence in sentences:
        # Find all dates in the sentence
        dates = date_regex.findall(sentence)
        # If dates are found, add them to the result dictionary with the corresponding sentence
        if dates:
            return dates[0]
    return None

def date_score(reference, completion):
    """Assign a score based on the difference between two dates using a negative exponential function.

    Args:
        reference (str): The reference date.
        completion (str): The completion date.

    Returns:
        float: The score."""
    score = 0
    if not completion:
        return score
    ref_date = parse_dates_from_text(reference)
    comp_date = parse_dates_from_text(completion)
    score = np.exp(-(date_diff(ref_date, comp_date) ** 2 / 1000))
    # Clip any very small scores
    if score < 0.001:
        score = 0
    return score

def rouge_score(reference, completion, rouge):
    if not completion or not reference:
        return 0.0
    return rouge.get_scores(reference, completion, avg=False)[0]["rouge-l"]["f"]


#*******************************************************************************************************************

df_score = pd.read_csv("empty_data1.csv")
#df_score = df_score[:200]

df_score["DATE_SCORE"] = None

rouge = Rouge()

for i in range(len(df_score)) :
    reference = str(df_score.loc[i, "reference"])
    completion = str(df_score.loc[i, "result_model"])
    date_scoring = date_score(reference, completion)
    rouge_scoring = rouge_score(reference, completion, rouge)
    df_score.loc[i, "DATE_SCORE"] = 0.7*date_scoring + 0.3*rouge_scoring

df_score = df_score.dropna(subset=["result_model"]).reset_index(drop=True)
print("DATE SCORE :",sum(df_score.loc[:, "DATE_SCORE"])/len(df_score))
df_score.to_csv("model_score_date_qa_vvalhalle11111.csv", index=False)

