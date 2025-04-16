import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import spacy
from wordsegment import load, segment
from itertools import permutations 
from autocorrect import Speller
import os

import re

# nltk.download_shell()
nlp = spacy.load('en_core_web_trf')


def convert_series_to_dict(df, col_name):
    labels_set = {}
    for row in list(df[col_name]):
        if row != "":
            split = row.split(",")
            if len(split)>1:
                for elem in split:
                    if elem.strip():
                        if elem not in labels_set:
                            labels_set[elem] = 1
                        else:
                            labels_set[elem] += 1
            else:
                if split[0].strip():
                    if split[0] not in labels_set:
                            labels_set[split[0]] = 1
                    else:
                        labels_set[split[0]] += 1
    return labels_set

def concatenateLabels(row):
    a = ', '.join(row)

    print("row: ",row)
    print("a: ",a)
    return a

def makeLowerCase(row):
    return row.lower() if isinstance(row, str) else row

def removeStopWords(row):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(row)
    statement_no_stop = [word for word in word_tokens if word not in stop_words]

    return " ".join(statement_no_stop)

def removeNonEnglishWordsTypos(row):
    words = set(nltk.corpus.words.words())

    return " ".join(w for w in word_tokenize(row) \
            if w.lower() in words)


def lemmatize(row):
    doc = nlp(row)
 
    # Extract lemmatized tokens
    lemmatized_tokens = [token.lemma_ for token in doc]
    
    # Join the lemmatized tokens into a sentence
    lemmatized_text = " ".join(lemmatized_tokens)

    return lemmatized_text

def stem(row):
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(token) for token in word_tokenize(row)]

    return " ".join(stemmed)

def removeNoise(row):

    return re.sub(r'[\.\?\!\,\:\;\"]', '', row)

# Source: https://www.geeksforgeeks.org/python-program-to-sort-words-in-alphabetical-order/
def sortWords(S):
    W = S.split(" ")
    
    W.sort()
 
    # return the sorted words
    return ' '.join(W)

def segmentWords(row):
    segmented_words = segment(row)
    return ' '.join(segmented_words)

def autocorrect(row):
    spell = Speller(lang='en')

    return spell(row)

def preprocessLabels(row):
    # remove noise
    row = removeNoise(row)
    
    # Make lower case
    row = makeLowerCase(row)

    row = autocorrect(row)

    row = segmentWords(row)

    return row

# PREPROCESS LABELS
def preprocessing_human(df, file_name, columns):
    df = df.fillna('')

    # Remove stopwords, make lowercase, etc.
    # for i in columns:
    load() 
    df[columns] = df[columns].map(lambda x: preprocessLabels(str(x)))
    print(f"Words in all columns have been preprocessed!")

    df["labels"] = df[columns].apply(lambda y: ','.join(y.values.astype(str)), axis=1)  
    # print(df)
    print(f"Words in all columns have been concatenated!")
    

    df_name = f'{file_name}.csv'
    print(df_name)
    # df.to_csv(df_name, index=False)
    return df

# PREPROCESS LABELS
def preprocessing_ml_labels(ml_dict):

    for key, val in ml_dict.items():
        temp_list = []
        for elem in val.split(","):
            new_item = preprocessLabels(elem)
            temp_list.append(new_item)

        ml_dict.update({key: ','.join(temp_list)})
        
    print(f"Words in all keys have been preprocessed!")

    return ml_dict


def add_labels_to_list(row, cols):
    final_list = []
    for col in cols:
        final_list.append(row[col])
    
    return final_list

def create_permutations_from_list(row, cols, permutation):
    str_list = add_labels_to_list(row, cols)
    pmt_list = list(permutations(str_list, permutation))

    return pmt_list



def remove_extra_commas(row):
    # Replace multiple commas with a single comma
    cleaned_string = re.sub(r",+", ",", row)
    
    # Remove leading or trailing commas (optional)
    cleaned_string = cleaned_string.strip(",")
    
    return cleaned_string

def upload_directory(local_dir, bucket_name, s3_path, s3_client):
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_dir)
            s3_file_path = os.path.join(s3_path, relative_path).replace("\\", "/")  # Replace for S3 compatibility
            
            print(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_file_path}")
            s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
