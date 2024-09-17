import pandas as pd
import numpy as np
import requests
import json
import re
import wikipedia
import spacy
from transformers import pipeline
from bs4 import BeautifulSoup
import time

# Load the NER pipeline
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

def extract_query_from_question_transformers(question):
    # Get the named entities using the NER model
    ner_results = ner(question)
    
    # Extract relevant entities (ORG, LOC, DATE, MISC) and numbers (for population, etc.)
    query_terms = []
    for entity in ner_results:
        if entity['entity_group'] in ["PER", "LOC", "ORG", "MISC", "DATE", "CARDINAL"]:
            query_terms.append(entity['word'])
    print("query_terms transformers :",query_terms)

    return query_terms



def search_wikipedia_title_from_question(question):
    
    query = extract_query_from_question_transformers(question)
    
    results = []
    for item in query:
        results.extend(search_wikipedia_titles(item, 3))
         # Search Wikipedia based on the automatically extracted query
        #results.extend(wikipedia.search(item))
    results.extend(query)
    return results

def search_wikipedia_titles(keywords, num_results = 10):
    """Search Wikipedia for page titles related to the given keywords."""
    search_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": keywords,
        "format": "json",
        "srlimit": num_results  # Number of results to return
    }
    response = requests.get(search_url, params=params)
    data = response.json()
    search_results = data.get("query", {}).get("search", [])
    titles = [result["title"] for result in search_results]
    return titles

def get_article_sections(title) :
    # Fetch the HTML content of the Wikipedia article
    url = f"https://en.wikipedia.org/wiki/{title}"
    response = requests.get(url)
    html_content = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    sections = {}
    for section in soup.find_all("h2"):
        if (p_tag := section.find_next("p")) is not None:
            sections[section.text] = p_tag.text

    return sections
    
def combine_wikipedia_titles_and_sections(question):
    # Search Wikipedia titles for both models
    keywords_transformers = search_wikipedia_title_from_question(question)
    
    # Convert lists to sets to remove duplicates
    set_transformers = set(keywords_transformers)

    # Combine the sets to get unique items from both lists
    combined_set = set_transformers

    # Convert the set back to a list (if needed)
    combined_list = list(combined_set)

    SYSTEM_PROMPT_MESSAGE = "Please answer the following question. To do this, you need to look for the answer in the set of Widipedia content sections I will provide you below.\n\n Question : "+question+"\n\n"

    INPUT_MODEL_PROMPT = SYSTEM_PROMPT_MESSAGE

    # Iterate through the combined list of titles, print title and section titles
    for title in combined_list:
        #all_pages[title] = {}
        section_title = get_article_sections(title)
        INPUT_MODEL_PROMPT += title+" :\n" 
        for key in section_title :
            INPUT_MODEL_PROMPT += key+"\n"+section_title[key]+"\n\n"
    return INPUT_MODEL_PROMPT

def call_open_source_model(messages):

    open_source_model_url = "http://129.146.101.114:16985/v1/completions"

    data = {
        "prompt": messages[0]['content'],
        "model": "microsoft/Phi-3-medium-4k-instruct",
        "max_tokens" : 64,
        "n": 1
    }

    headers = {"Content-Type": "application/json"}

    try:
        # Send the POST request
        resp = requests.post(open_source_model_url, data=json.dumps(data), headers=headers, timeout=30)

        # Check for successful response
        if resp.status_code == 200:
            response_json = resp.json()
            print("response_json: ",response_json)

            if 'choices' in response_json:
                completion_text = response_json['choices'][0]['text']
                print("choices:", completion_text)
                return completion_text
            else:
                print("Unexpected response format:", response_json)
                return None
        else:
            print(f"Failed to fetch response: {resp.status_code}, {resp.text}")
            return None

    except requests.exceptions.RequestException as e:
        # Handle any exceptions (e.g., timeout, connection issues)
        print(f"Request failed: {e}")
        return None

model_data_input = pd.read_csv("C:/Users/hp/Desktop/Document/wiki_qa_sn1/model_sample_qa.csv")

model_data_input["result_model"] = None
model_data_input["Time_elapsed"] = None
for i in range(4):
    start_date = time.time()
    question = model_data_input.loc[i, "challenge"]
    INPUT_MODEL_PROMPT = combine_wikipedia_titles_and_sections(question)
    response = call_open_source_model([{"role":"user", "content":INPUT_MODEL_PROMPT}])
    end_date = time.time()
    model_data_input.loc[i, "Time_elapsed"] = start_date - end_date
    model_data_input.loc[i, "result_model"] = response


model_data_input.to_csv("model_sample_qa_result.csv", index=False)
