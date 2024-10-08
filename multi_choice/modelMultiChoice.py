import pandas as pd
import numpy as np
import requests
import re
from bs4 import BeautifulSoup
import time
import spacy
import nltk
from nltk.corpus import stopwords
from transformers import LongformerTokenizer, LongformerForMultipleChoice, pipeline
import torch
from unidecode import unidecode
import csv

#********************************* METHODS ***********************************************************

def get_options_from_multichoice_question(question) :

    if "[Input Question]" in question :
        question = question.split("[Input Question]")[1]
    
    # Define the regex patterns
    regex_A = r"A\.\s*(.*?)\s*B\."
    regex_B = r"B\.\s*(.*?)\s*C\."
    regex_C = r"C\.\s*(.*?)\s*D\."
    regex_D = r"D\.\s*(.*?)\s*Answer"

    # List of regex patterns for matching
    regex_patterns = {
        'A': regex_A,
        'B': regex_B,
        'C': regex_C,
        'D': regex_D
    }

    # Dictionary to store the matches
    matches = {}

    # Iterate through the regex patterns and find matches
    for key, regex in regex_patterns.items():
        match = re.findall(regex, question, re.DOTALL)
        matches[key] = match

    # Print all matches
    all_matches = []
    for key, match_list in matches.items():
        matches = []
        for i, match in enumerate(match_list, 1):
            matches.append(match.strip())

        if len(matches) > 0 :
            all_matches.append(matches[-1])
            
    if "A." in question :
        question = question.split("A.")[0]
        
    return question, all_matches

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
    
def keyword_generator(sentence):

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


def search_google(query, API_KEY, CX):

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

def clean_query(query) : 
    query = unidecode(query)
    query = query.strip().replace("\n", "")  # Clean the query
    return query

def combine_wikipedia_titles_and_sections(question, API_KEY, CX):
    CONTEXT = question
    combined_set = []
    
    keywords_transformers = unidecode(extract_query_from_question_transformers(question))
    print("keywords_transformers*************",keywords_transformers)
    wiki_items = []
    if keywords_transformers:
        print("Methode : extract_query_from_question_transformers")
        search_results = search_google(keywords_transformers, API_KEY, CX)
        if "items" in search_results:
            wiki_items = search_results["items"]
        else :
            print("Methode : keyword_nabil")
            keywords_transformers = unidecode(keyword_generator(question))
            search_results = search_google(keywords_transformers, API_KEY, CX)
            if "items" in search_results :
                wiki_items = search_results["items"]
            else :
                print("Methode : Google API")
                search_results = search_google(question, API_KEY, CX)
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



def prepare_multiple_choice_inputs(question, context, options, tokenizer, device, max_length=512):
    choices_inputs = []
    for option in options:
        try:
            inputs = tokenizer(
                question,
                option + " " + context,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            # Move tensors to the device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            choices_inputs.append(inputs)
        
        except Exception as e:
            print(f"Exception occurred while tokenizing option '{option}': {e}")

    # If choices_inputs is empty, raise an error
    if len(choices_inputs) == 0:
        raise ValueError("No valid tokenized inputs found. Please check the question, context, and options.")

    # Stack the input tensors to match the required shape
    input_ids = torch.stack([choice['input_ids'] for choice in choices_inputs], dim=1)
    attention_mask = torch.stack([choice['attention_mask'] for choice in choices_inputs], dim=1)
    token_type_ids = torch.stack([choice['token_type_ids'] for choice in choices_inputs], dim=1) if 'token_type_ids' in choices_inputs[0] else None

    return input_ids, attention_mask, token_type_ids

def multichoice_model_hugging_face(question, context ,options, model, tokenizer):

    match_index_letter = {0:"A", 1 : "B", 2: "C", 3 :"D"}
    # Prepare the input
    context = context.replace('\n', ' ')
    selected_answer = None
    try:
        input_ids, attention_mask, token_type_ids = prepare_multiple_choice_inputs(question, context, options, tokenizer, device)
        # Now pass these inputs to the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits
        prob = torch.softmax(logits, dim=-1)[0].tolist()

        # Select the best option based on the probability
        selected_answer = options[np.argmax(prob)]
        
        selected_answer = match_index_letter[options.index(selected_answer)]
            
    except Exception as e:
        print(f"An error occurred: {e}")

    return selected_answer


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


# ***********************************************************************************************************************

df = pd.read_csv("../model_sample_qa.csv")[['date','task', 'challenge', 'reference']]
multichoice_df = df[df['task'] == 'multi_choice'].reset_index(drop=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Download stopwords from nltk
nltk.download('stopwords')

# Get the list of stopwords from nltk
stop_words = set(stopwords.words('english'))

# Load the NER pipeline
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

# Your Google Custom Search Engine credentials
API_KEY = 'AIzaSyD_tquylmyhRpoGqBHMFh3HeMC-kLy1Z1U'
CX='5607d294a06a04e49'

#Multichoice
tokenizer = LongformerTokenizer.from_pretrained("potsawee/longformer-large-4096-answering-race")
model = LongformerForMultipleChoice.from_pretrained("potsawee/longformer-large-4096-answering-race")

# Generate emptybackup
generat_empty_backup(True)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

model_data_input = multichoice_df.copy()
model_data_input["result_model"] = None
model_data_input["Time_elapsed"] = None
model_data_input['Titles'] = None

start_date_globale = time.time()
for i in range(200):
    print("Steps :",str(i))
    start_date = time.time()
    question = model_data_input.loc[i, "challenge"]
    question = clean_query(question)

    question, options = get_options_from_multichoice_question(question)

    CONTEXT, combined_set = combine_wikipedia_titles_and_sections(question, API_KEY, CX)

    response = multichoice_model_hugging_face(question, CONTEXT, options, model, tokenizer)
    
    end_date = time.time()
    model_data_input.loc[i, "Time_elapsed"] = end_date - start_date
    model_data_input.loc[i, "result_model"] = response
    model_data_input.loc[i, "Titles"] = ', '.join([x for x in combined_set])

    list_input = model_data_input.iloc[i]
    write_to_csv(list_input)


model_data_input.to_csv("model_predict_multichoice_potsawee2.csv", index=False)
print("Timing globale for running  is :", time.time() - start_date_globale)


# Avant Google API puis Trans puis Spacy  2  SUCCESS RATIO : 64.82412060301507 %
# Mtn avec Trans puis Spacy puis Google API 1 SUCCESS RATIO : SUCCESS RATIO : 64.0 %
# Mtn avec Trans puis Google API puis Spacy 3 SUCCESS RATIO : 62.5 %
