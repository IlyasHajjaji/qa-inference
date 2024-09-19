import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import spacy
import nltk
from nltk.corpus import stopwords
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, LongformerTokenizer, LongformerForMultipleChoice
import torch
from unidecode import unidecode
import warnings
import re
# Suppress FutureWarning (like the ones from tokenization changes)
warnings.filterwarnings("ignore", category=FutureWarning)
# Suppress UserWarning (like the model initialization warnings)
warnings.filterwarnings("ignore", category=UserWarning)

#********************************* METHODS ***********************************************************

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


def search_google(query):

    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'key': API_KEY,
        'cx': CX,
        'q': query,
        'num': 2
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Will raise HTTPError for bad responses
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")  # Print HTTP error details
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")  # Print request error details
    except Exception as err:
        print(f"An error occurred: {err}")  # Print any other error details
    
    return {}  # Return an empty dictionary if an error occurs


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

    search_results = search_google(question)
    wiki_items = []
    if "items" in search_results :
        wiki_items = search_results["items"]
    else :  
        keywords_transformers = keyword_generator(question)
        search_results = search_google(keywords_transformers)
        if "items" in search_results :
            wiki_items = search_results["items"]

    CONTEXT = ""
    combined_set = []
    
    for element in wiki_items :
        wiki_url = element['link']
        title = wiki_url.replace('https://en.wikipedia.org/wiki/', '')
        combined_set.append(title)
        sections_title = get_article_sections(title)
        CONTEXT += title+" :\n" 
        for h2 in sections_title :
            CONTEXT += h2+"\n"+sections_title[h2]+"\n\n"

    return CONTEXT, combined_set


def qa_model_hugging_face(question, context):
    # Load model and tokenizer
    model_name = "valhalla/longformer-base-4096-finetuned-squadv1"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add separator tokens explicitly to ensure the correct format
    sep_token = tokenizer.sep_token

    # Join the question and context with the appropriate separator tokens
    input_text = f"{question} {sep_token} {context} {sep_token}"

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=4096, padding='max_length')

    # Perform inference (get the start and end logits)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract start and end logits
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Convert the input_ids to tokens
    input_ids = inputs['input_ids']
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Get the most probable start and end positions
    start_index = torch.argmax(start_scores, dim=1).item()
    end_index = torch.argmax(end_scores, dim=1).item() + 1

    # Convert tokens to answer
    answer_tokens = all_tokens[start_index:end_index]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))

    return answer

def clean_query(query) : 
    query = unidecode(query)
    query = re.sub(r'(.*)Here is a question(.*):(\n)*', '', query)
    query = query.strip().replace("\n", "")  # Clean the query
    return query

# ***********************************************************************************************************************

df = pd.read_csv("../model_sample_qa.csv")[['date','task', 'challenge', 'reference']]
date_df = df[df['task'] == 'date_qa'].reset_index(drop=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Download stopwords from nltk
nltk.download('stopwords')

# Get the list of stopwords from nltk
stop_words = set(stopwords.words('english'))

# Your Google Custom Search Engine credentials
API_KEY = 'AIzaSyCVot1ZUjAS-kVXc5HgDeQFJbaokaJxozE'
CX='53282e4da3bc047e2'

tokenizer = AutoTokenizer.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")
model = AutoModelForQuestionAnswering.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model_data_input = date_df.copy()
model_data_input["result_model"] = None
model_data_input["Time_elapsed"] = None
model_data_input['Titles'] = None

for i in range(50):
    start_date = time.time()
    question = model_data_input.loc[i, "challenge"]
    question = clean_query(question)
    print("STEP :", str(i))
    print("Question :", question)
    print('')
    task = model_data_input.loc[i, "task"]
    CONTEXT, combined_set = combine_wikipedia_titles_and_sections(question)

    response = qa_model_hugging_face(question, CONTEXT)
    
    end_date = time.time()
    model_data_input.loc[i, "Time_elapsed"] = end_date - start_date
    model_data_input.loc[i, "result_model"] = response
    model_data_input.loc[i, "Titles"] = ', '.join([x for x in combined_set])


model_data_input.to_csv("model_predict_date_qa.csv", index=False)

