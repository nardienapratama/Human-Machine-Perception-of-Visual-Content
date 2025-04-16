import pandas as pd
import numpy as np
import ast

import math
import pickle, os
import boto3
from io import StringIO, BytesIO


def convert_str_to_array(row):
    cleaned_string = row.strip('[]').strip()
    array_1d = np.fromstring(cleaned_string, sep=' ')
    array_2d = array_1d[np.newaxis, :]
    return array_2d
    
def make_embedding(row, model):
    embeddings = model.encode([row])
    return embeddings

def compare_embedding_similarity(data, col_name_1, col_name_2, model):
    embedding1 = model.encode([data[col_name_1]])
    embedding2 = model.encode([data[col_name_2]])

    test_emb_1 = convert_str_to_array(data[col_name_1.replace("labels", "embed")])
    test_emb_2 = convert_str_to_array(data[col_name_2.replace("labels", "embed")])
    print(embedding1.reshape(-1))
    print("=========")
    print(test_emb_1)

    assert np.array(embedding1.reshape(-1)).all() == np.array(test_emb_1).all(), f"{col_name_1} did not generate the same kind of embeddings!"
    assert embedding2.reshape(-1).all() == test_emb_2.all(), f"{col_name_2} did not generate the same kind of embeddings!"
    similarities = model.similarity(embedding1, embedding2)

    return similarities.item()


# https://stackoverflow.com/a/52617883
def normal_round(n, decimals=0):
    expoN = n * 10 ** decimals
    if abs(expoN) - abs(math.floor(expoN)) < 0.5:
        return math.floor(expoN) / 10 ** decimals
    return math.ceil(expoN) / 10 ** decimals

# Function to read a pickle file from S3
def read_pickle_from_s3(s3, bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    pickle_data = response['Body'].read()
    return pickle.loads(pickle_data)

def download_model_from_s3(bucket_name, s3_model_path, local_dir):
    """
    Download the contents of an S3 directory (model) to a local directory.
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_model_path):
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_key = obj['Key']
                local_path = os.path.join(local_dir, os.path.relpath(s3_key, s3_model_path))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                print(f"Downloading {s3_key} to {local_path}")
                s3.download_file(bucket_name, s3_key, local_path)

def check_s3_path_exists(bucket, path):
    response = s3.list_objects_v2(Bucket=bucket, Prefix=path)
    return 'Contents' in response

def upload_image_to_s3(bucket, s3_path, image):
    # Convert the image to a byte stream
    buffer = BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)
    
    # Upload the image to S3
    s3.put_object(Bucket=bucket, Key=s3_path, Body=buffer, ContentType='image/jpeg')


def upload_pickle_to_s3(s3, bucket, s3_path, data):
    # Serialize the data to a byte stream
    buffer = BytesIO()
    pickle.dump(data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
    buffer.seek(0)
    
    # Upload the pickle file to S3
    s3.put_object(Bucket=bucket, Key=s3_path, Body=buffer.getvalue(), ContentType='application/octet-stream')