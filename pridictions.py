import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re
import string
import warnings
import os
import tempfile
import boto3
import os

warnings.filterwarnings('ignore')

# Initialize the S3 client
s3 = boto3.client('s3',
aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION'))

# Set the S3 bucket name and model path
bucket_name = 'cardscanml'
model_s3_path = 'model-best/'

# Create a temporary directory to download the model
with tempfile.TemporaryDirectory() as tmpdirname:
    # List objects in the model directory on S3
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix=model_s3_path)
    
    # Download each file in the model directory
    for obj in result.get('Contents', []):
        file_path = obj['Key']
        local_path = os.path.join(tmpdirname, os.path.relpath(file_path, model_s3_path))
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download the file
        s3.download_file(bucket_name, file_path, local_path)
        
    # Load the model from the temporary directory
    model_ner = spacy.load(tmpdirname)



def clean_text(txt):
    whitespace = string.whitespace
    punctuation = '!#$%&\'()*+:;<=>?[\\]^`{|}~'
    tableWhiteSpace = str.maketrans("", "", whitespace)
    tablePunctuation = str.maketrans("", "", punctuation)
    text = str(txt)
    #text = text.lower()
    removewhitespace = text.translate(tableWhiteSpace)
    removepunctuation = removewhitespace.translate(tablePunctuation)  
    return str(removepunctuation)
# group the label
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = ''
        
    def getgroup(self,text):
        if self.text == text:
            return self.id
        else:
            self.id +=1
            self.text = text
            return self.id
        


def parser(text,label):
    if label == "PHONE":
        text = text.lower()
        phone_pattern = re.compile(r"""
        (?:
            (\+?\d{1,2}[-\s]?)?      # Optional country code
            (\d{1,4}[-\s]?)?         # Optional area code
            (\d{1,4}[-\s]?)?         # First part of the number
            (\d{1,4}[-\s]?)?         # Second part of the number
            (\d{1,4}[-\s]?)+         # Remaining part of the number (repeating pattern to handle varied lengths)
        ) | (
            (\+?\d{1,2}[-\s]?)?      # Optional country code
            (\d{1,4}[-\s]?)?         # Optional area code
            (\d{3,4}[-\s]?)          # First part of the number
            (\d{4})                  # Second part of the number
        )
        """, re.VERBOSE)
    
        matches = phone_pattern.findall(text)
        formatted_numbers = [''.join(match).strip() for match in matches if any(match)]
        #raise Exception(formatted_numbers)
        # Regex pattern to match phone numbers with various formats
        # phone_pattern = re.compile(r"""
        #     (\+?\d{1,2}[-\s]?)?           # Optional country code
        #     (\d{2,5}[-\s]?)?              # Optional area code
        #     (\d{3,5}[-\s]?)               # First part of the number
        #     (\d{4,7})                     # Second part of the number
        # """, re.VERBOSE)
        
        # matches = phone_pattern.findall(text)
        # formatted_numbers = [''.join(match).strip() for match in matches]
        # return formatted_numbers[0] if formatted_numbers else text
        
        
    elif label == "EMAIL":
        text = text.lower()
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        # Find all email addresses in the text, preserving the case
        emails = re.findall(email_pattern, text)
        # Return the first found email or an empty string if none found
        text = emails[0] if emails else ""
        
    elif label == "WEB":
        text = text.lower()
        allow_special_char = ":/.%#\\-"
        text = re.sub(r"[^A-Za-z0-9{} ]".format(allow_special_char),"",text)
        
    elif label in ("NAME", "DES"):
        text = text.lower()
        allow_special_char = ":/.%#-"
        text = re.sub(r"[^a-z ]","",text)
        text = re.sub(r"[^a-z\s{}]".format(re.escape(allow_special_char)), "", text)
        #text = title_case(text)
        text = re.sub(r"\s+", " ", text).strip()
        
    elif label == "ORG":
        text = text.lower()
    # Define allowed special characters
        allowed_special_char = ":/.%#-"
    # Remove unwanted characters
        text = re.sub(r"[^a-z\s{}]".format(re.escape(allowed_special_char)), "", text)
    # Clean up extra spaces
        text = re.sub(r"\s+", " ", text).strip()
    # Capitalize each word (title case)
        text = text.title()
        
    elif label == 'GEO':
        text = text.lower()
        allow_special_char = ':/.%#-_'  # Include underscore (_) if necessary
        text = re.sub(r'[^A-Za-z0-9{} ]'.format(re.escape(allow_special_char)), '', text)

        
    return text


grp_gen = groupgen()

def getPredictions(image):
    # extract data using Pytesseract 
    tessData = pytesseract.image_to_data(image)
    # convert into dataframe
    tessList = list(map(lambda x:x.split('\t'), tessData.split('\n')))
    df = pd.DataFrame(tessList[1:],columns=tessList[0])
    df.dropna(inplace=True) # drop missing values
    df['text'] = df['text'].apply(clean_text)

    # convet data into content
    df_clean = df.query('text != "" ')
    content = " ".join([w for w in df_clean['text']])
    print(content)
    # get prediction from NER model
    doc = model_ner(content)

    # converting doc in json
    docjson = doc.to_json()
    doc_text = docjson['text']

    # creating tokens
    datafram_tokens = pd.DataFrame(docjson['tokens'])
    datafram_tokens['token'] = datafram_tokens[['start','end']].apply(
        lambda x:doc_text[x[0]:x[1]] , axis = 1)

    right_table = pd.DataFrame(docjson['ents'])[['start','label']]
    datafram_tokens = pd.merge(datafram_tokens,right_table,how='left',on='start')
    datafram_tokens.fillna('O',inplace=True)

    # join lable to df_clean dataframe
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1 
    df_clean['start'] = df_clean[['text','end']].apply(lambda x: x[1] - len(x[0]),axis=1)

    # inner join with start 
    dataframe_info = pd.merge(df_clean,datafram_tokens[['start','token','label']],how='inner',on='start')

    # Bounding Box

    bb_df = dataframe_info.query("label != 'O' ")

    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df['group'] = bb_df['label'].apply(grp_gen.getgroup)

    # right and bottom of bounding box
    bb_df[['left','top','width','height']] = bb_df[['left','top','width','height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']

    # tagging: groupby group
    col_group = ['left','top','right','bottom','label','token','group']
    group_tag_img = bb_df[col_group].groupby(by='group')
    img_tagging = group_tag_img.agg({

        'left':min,
        'right':max,
        'top':min,
        'bottom':max,
        'label':np.unique,
        'token':lambda x: " ".join(x)

    })
    
    img_tagging["label"] = img_tagging["label"].apply(clean_text)
    
    img_bb = image.copy()
    for l,r,t,b,label,token in img_tagging.values:
        cv2.rectangle(img_bb, (l,t), (r,b), (0,255,0),2)
        cv2.putText(img_bb, str(label), (l,t), cv2.FONT_HERSHEY_PLAIN, 0, (255,0,0), 2)


    # Entities

    info_array = dataframe_info[['token','label']].values
    entities = dict(GEO=[],NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[],SOCIAL=[],PIN=[])

    previous = 'O'

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]

        # step -1 parse the token
        text = parser(token,label_tag)

        if bio_tag in ('B','I'):

            if previous != label_tag:
                entities[label_tag].append(text)

            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)

                else:
                    if label_tag in ("NAME", "ORG", "DES","PHONE","EMAIL","WEB","SOCIAL","PIN","GEO"):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text

                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text



        previous = label_tag
        
    return img_bb, entities





