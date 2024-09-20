import pandas as pd
import numpy as np
import requests
import json
import re
import wikipedia
from transformers import pipeline
from bs4 import BeautifulSoup
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
import spacy
import nltk
from nltk.corpus import stopwords
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import csv
from unidecode import unidecode
import torch


# Function to merge named entities with commas (e.g., "Kenton, Ohio")
def merge_named_entities(doc):
    new_sentence = []
    skip_count = 0

    for i, token in enumerate(doc):
        if skip_count > 0:
            skip_count -= 1
            continue

        # Check if the current token is a named entity followed by a comma and another named entity
        if token.ent_type_ and i < len(doc) - 2 and doc[i + 1].text == ',' and doc[i + 2].ent_type_ == token.ent_type_:
            # Merge current entity and the one after the comma
            new_sentence.append(f"{token.text} {doc[i + 2].text}")
            skip_count = 2  # Skip the next two tokens (comma and next entity)
        else:
            new_sentence.append(token.text)

    return " ".join(new_sentence)
    
def keyword_nabil(sentence):
    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Remove stopwords from the sentence
    filtered_sentence = []
    for token in doc:
        if token.text.lower() not in stop_words:
            filtered_sentence.append(token.text)

    # Join the filtered tokens back into a string
    sentence = " ".join(filtered_sentence)

    # Merge named entities connected by commas
    processed_sentence = merge_named_entities(doc)

    # Process the new sentence with merged entities
    doc = nlp(processed_sentence)

    # Extract noun phrases
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]

    # Remove stopwords from noun phrases and filter by first uppercase character or numbers
    filtered_noun_phrases = []
    for np in noun_phrases:
        filtered_words = [word for word in np.split() if word.lower() not in stop_words]
        filtered_np = ' '.join(filtered_words)
    
        # Check if the first character is uppercase or a digit
        if filtered_np and (filtered_np[0].isupper() or filtered_np[0].isdigit()):
            filtered_noun_phrases.append(filtered_np)

    return ', '.join([x for x in filtered_noun_phrases])

def extract_query_from_question_transformers(question):

    # Get the named entities using the NER model
    ner_results = ner(question)
    
    # Extract relevant entities (ORG, LOC, DATE, MISC) and numbers (for population, etc.)
    query_terms = []
    for entity in ner_results:
        if entity['entity_group'] in ["PER", "LOC", "ORG", "MISC", "DATE", "CARDINAL"]:
            query_terms.append(entity['word'])
    #print("query_terms transformers :",query_terms)

    return ', '.join([x for x in query_terms]) 

def search_google(query):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': API_KEY,
        'cx': CX,
        'q': query,
        'num': 2  # Adjust number of results as needed
    }
    response = requests.get(url, params=params)

    response.raise_for_status()  # Will raise an HTTPError for bad responses
    return response.json()


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
    CONTEXT = question
    combined_set = []
    
    keywords_transformers = unidecode(extract_query_from_question_transformers(question))
    print("keywords_transformers*************",keywords_transformers)
    wiki_items = []
    if keywords_transformers:
        print("Methode : extract_query_from_question_transformers")
        search_results = search_google(keywords_transformers)
        if "items" in search_results:
            wiki_items = search_results["items"]
        else :
            print("Methode : keyword_nabil")
            keywords_transformers = unidecode(keyword_nabil(question))
            search_results = search_google(keywords_transformers)
            if "items" in search_results :
                wiki_items = search_results["items"]  
            else :
                print("Methode : Google API")
                search_results = search_google(question)
                if "items" in search_results :
                    wiki_items = search_results["items"]     
                else : 
                    print("Methode : Nan")
                    return CONTEXT,combined_set

    for element in wiki_items :
        wiki_url = element['link']
        title = wiki_url.replace('https://en.wikipedia.org/wiki/', '')
        combined_set.append(title)
        sections_title = get_article_sections(title)
        CONTEXT += title+" :\n" 
        for h2 in sections_title :
            CONTEXT += h2+"\n"+sections_title[h2]+"\n\n"
    
    return CONTEXT, combined_set


def generate_answer(question, content):
    prompt = f"question: {question} context: {content}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=30, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def qa_model_hugging_face(question, CONTEXT,model):
    max_length = 4096  # Longformer's maximum length
    encoding = tokenizer(question, CONTEXT, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    output = model(input_ids, attention_mask=attention_mask)
    start_scores = output.start_logits
    end_scores = output.end_logits
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    # Convert tokens to the answer
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_scores):torch.argmax(end_scores) + 1]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens), clean_up_tokenization_spaces=False)

    return answer

def generat_empty_backup(generate):
        if generate == True:
            # Define the columns for the DataFrame
            columns = ['date', 'task', 'challenge', 'reference', 'result_model', 'Time_elapsed', 'Titles']
            # Create an empty DataFrame with the specified columns
            df = pd.DataFrame(columns=columns)
            # Save the empty DataFrame to a CSV file
            df.to_csv('empty_data1.csv', index=False)

def write_to_csv(list_input):

    # Extract data without index and convert to list
    row_data_no_index = list_input.values.tolist()

    with open('empty_data1.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data_no_index)

def clean_query(query) : 
    query = unidecode(query)
    query = re.sub(r'(.*)Here is a question(.*):(\n)*', '', query)
    query = query.strip().replace("\n", "")  # Clean the query
    return query





# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Download stopwords from nltk
nltk.download('stopwords')

# Get the list of stopwords from nltk
stop_words = set(stopwords.words('english'))

# Define the model and task
#model_name = "deepset/roberta-base-squad2"
#nlp_hugg = pipeline('question-answering', model=model_name, tokenizer=model_name)

# Load the NER pipeline
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Your Google Custom Search Engine credentials
API_KEY = 'AIzaSyD_tquylmyhRpoGqBHMFh3HeMC-kLy1Z1U'
CX = '5607d294a06a04e49'



ckpt = "valhalla/longformer-base-4096-finetuned-squadv1"
tokenizer = AutoTokenizer.from_pretrained(ckpt)
model = AutoModelForQuestionAnswering.from_pretrained(ckpt)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


df = pd.read_csv("../model_sample_qa.csv")[['date','task', 'challenge', 'reference']]
date_df = df[df['task'] == 'date_qa'].reset_index(drop=True)

# Generate emptybackup
generat_empty_backup(True)

model_data_input = date_df.copy()
model_data_input["result_model"] = None
model_data_input["Time_elapsed"] = None
model_data_input['Titles'] = None

start_date_globale = time.time()
for i in range(198,200):
    print("Steps :",str(i))
    start_date = time.time()
   
    question = model_data_input.loc[i, "challenge"]
    question = clean_query(question)
    CONTEXT, combined_set = combine_wikipedia_titles_and_sections(question)
    answer = qa_model_hugging_face(question, CONTEXT,model)

    end_date = time.time()
    model_data_input.loc[i, "Time_elapsed"] = end_date - start_date
    model_data_input.loc[i, "result_model"] = CONTEXT
    model_data_input.loc[i, "Titles"] = ', '.join([x for x in combined_set])
    
    list_input = model_data_input.iloc[i]
    write_to_csv(list_input)
    
model_data_input.to_csv("model_predict_date_qa_valhalle_Ilyas.csv", index=False)
print("Timing globale for running  is :", time.time() - start_date_globale)



# DATE SCORE : 0.5997401300927698 (valhalla/longformer-base-4096-finetuned-squadv1) avec script modelDateQA plus temps execu long enr in model_predict_date_qa
# DATE SCORE : 0.6352899107281648 (valhalla/longformer-base-4096-finetuned-squadv1) enr in model_predict_date_qa_1
# DATE SCORE : 0.5622530140640416 "mrm8488/longformer-base-4096-finetuned-squadv2" enr in model_predict_date_qa_2
# DATE SCORE : 0.7753635866466402 nouveau
# DATE SCORE : 0.6368442373650793 ancien