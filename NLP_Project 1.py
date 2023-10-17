#!/usr/bin/env python
# coding: utf-8

# ### The steps are broadly divided into:
# 1. [Reading and Understanding the Data](#1)
# 1. [Web Scraping and Text Extraction from URLs](#2)
# 1. [Text Preprocessing: Sentence Splitting and Dataframe Creation.](#3)
# 1. [Importing Stopwords and Creating Dataframes](#4)
# 1. [Text Processing and Stopwords Removal](#5)
# 1. [Importing Master Dictionary and Cleaning Words.](#6)
# 1. [Tokenization of Sentences](#7)
# 1. [Text Analysis](#8)
# 1. [Output Data Structure](#9)

# In[2]:


import pandas as pd
import numpy as np


# <a id="1"></a> <br>
# ### Reading and Understanding the Data

# In[3]:


data = pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\Input.csv")
data.head()


# In[4]:


data = pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\Input.csv")[['URL_ID','URL']]
data


# In[5]:


data.info()


# In[6]:


data = data.iloc[0:150]
data


# In[7]:


data.drop('URL_ID', axis=1, inplace= True)
data


# <a id="2"></a> <br>
# ### Web Scraping and Text Extraction from URLs

# In[8]:


import requests
from bs4 import BeautifulSoup
import nltk
import string
nltk.download('punkt')


# In[9]:


extracted_text = []

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
}

for url_id, url in enumerate(data['URL'], start=1):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        content = soup.find(attrs={'class': 'td-post-content'}).text

        title = soup.find(attrs={'class': 'entry-title'}).text

        full_text = title + '.' + content

        with open(f'{url_id}.txt', 'w', encoding='utf-8') as text_file:
            text_file.write(full_text)

        extracted_text.append(full_text)
    except Exception as e:
        print(f"Error while processing URL: {url}")
        print(e)

extracted_data = pd.DataFrame({'Text': extracted_text})


# In[10]:


extracted_data.info()


# In[11]:


extracted_data


# In[12]:


extracted_data['Text'] = extracted_data['Text'].astype(str)


# <a id="3"></a> <br>
# ### Text Preprocessing: Sentence Splitting and Dataframe Creation.

# In[13]:


import re

sentences = re.split(r'\.\s', extracted_data['Text'].str.cat(sep=' '))
sentences


# In[14]:


sentences = [re.sub(r'\.', '', sentence) if sentence else np.nan for sentence in sentences]

result_df = pd.DataFrame({'sentences': sentences})
result_df


# In[15]:


result_df.dropna(inplace=True) 
result_df.reset_index(drop=True, inplace=True)
result_df


# <a id="4"></a> <br>
# ### Importing Stopwords and Creating Dataframes

# In[16]:


import nltk
import string


# In[17]:


punc=[punc for punc in string.punctuation]
punc


# In[18]:


StopWords_Auditor=pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\StopWords-20231010T124812Z-001\StopWords\StopWords_Auditor.txt",header=None)

StopWords_DatesandNumbers=pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\StopWords-20231010T124812Z-001\StopWords\StopWords_DatesandNumbers.txt",header=None)
StopWords_Generic=pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\StopWords-20231010T124812Z-001\StopWords\StopWords_Generic.txt",header=None)
StopWords_GenericLong=pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\StopWords-20231010T124812Z-001\StopWords\StopWords_GenericLong.txt",header=None)
StopWords_Geographic=pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\StopWords-20231010T124812Z-001\StopWords\StopWords_Geographic.txt",header=None)
StopWords_Names=pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\StopWords-20231010T124812Z-001\StopWords\StopWords_Names.txt",header=None)


# In[19]:


with open(r"C:\Users\zerin\Downloads\NLP Blackoffer project\StopWords-20231010T124812Z-001\StopWords\StopWords_Currencies.txt", 'r', encoding='ISO-8859-1') as file:
    lines = file.readlines()

StopWords_Currencies = pd.DataFrame({'Column Name': lines})
StopWords_Currencies


# <a id="5"></a> <br>
# ### Text Processing and Stopwords Removal

# In[20]:


def text_process(text):
    nopunc =[char for char in text if char not in punc or char not in [':',',','(',')','’','?']]
    nopunc=''.join(nopunc)
    txt=' '.join([word for word in nopunc.split() if word.lower() not in StopWords_Auditor])
    txt1=' '.join([word for word in txt.split() if word.lower() not in StopWords_Currencies])
    txt2=' '.join([word for word in txt1.split() if word.lower() not in StopWords_DatesandNumbers])
    txt3=' '.join([word for word in txt2.split() if word.lower() not in StopWords_Generic])
    txt4=' '.join([word for word in txt3.split() if word.lower() not in StopWords_GenericLong])
    txt5=' '.join([word for word in txt4.split() if word.lower() not in StopWords_Geographic])
    return ' '.join([word for word in txt5.split() if word.lower() not in StopWords_Names])


# In[21]:


result_df['sentences'] = result_df['sentences'].apply(text_process)
result_df


# <a id="6"></a> <br>
# ### Importing Master Dictionary and Cleaning Words.

# In[22]:


positive=pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\MasterDictionary-20231010T124711Z-001\MasterDictionary\positive-words.txt",header=None)
negative=pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\MasterDictionary-20231010T124711Z-001\MasterDictionary\negative-words.txt",header=None,encoding="ISO-8859-1")


# In[23]:


print(negative)
print(positive)


# In[24]:


positive.columns=['+ve words']
negative.columns=['-ve words']
positive['+ve words']=positive['+ve words'].astype(str)
negative['-ve words']=negative['-ve words'].astype(str)


# In[25]:


positive['+ve words']=positive['+ve words'].apply(text_process)
negative['-ve words']=negative['-ve words'].apply(text_process)


# In[26]:


def clean_word(word):
    return ''.join(char for char in word if char not in string.punctuation and char != '+')

positive_list = [clean_word(word) for word in positive['+ve words']]
negative_list = [clean_word(word) for word in negative['-ve words']]


# In[27]:


len(positive_list)


# <a id="7"></a> <br>
# ### Tokenization of Sentences

# In[28]:


from nltk.tokenize import word_tokenize


# In[29]:


text_list = []

for sentence in result_df['sentences']:
    text_list.append(sentence)

tokenized_words = []

for text in text_list:
    words = word_tokenize(text)
    tokenized_words.extend(words)


# In[30]:


len(tokenized_words)


# In[31]:


print(tokenized_words)


# <a id="8"></a> <br>
# ### Text Analysis

# #### 1] POSITIVE SCORE

# In[32]:


positive_score = 0

for word in tokenized_words:
    if word.lower() in positive_list:
        positive_score += 1

print('Positive score =', positive_score)


# #### 2] NEGATIVE SCORE

# In[33]:


negative_score = 0

for word in tokenized_words:
    if word.lower() in negative_list:
        negative_score += 1

print('Negative score =', negative_score)


# #### 3] POLARITY SCORE

# In[34]:


Polarity_Score=(positive_score-negative_score)/((positive_score+negative_score)+0.000001)

print('polarity_score=', Polarity_Score)


# #### 4] SUBJECTIVITY SCORE

# In[35]:


subjectivity_score=(positive_score-negative_score)/((len(tokenized_words))+ 0.000001)

print('subjectivity_score',subjectivity_score)


# #### 5] AVG SENTENCE LENGTH

# In[36]:


total_words = 0
total_sentences = result_df.shape[0]

for sentence in result_df['sentences']:
    words = sentence.split()
    total_words += len(words)

average_sentence_length = total_words / total_sentences
print('Average Sentence Length:', average_sentence_length)


# #### 6] PERCENTAGE OF COMPLEX WORDS¶

# In[37]:


vowels = ['a', 'e', 'i', 'o', 'u']
complex_word_count = 0

def count_syllables(word):
    count = 0
    word = word.lower()

    if word.endswith('es') or word.endswith('ed'):
        return count

    for char in word:
        if char in vowels:
            count += 1

    return count

for word in tokenized_words:
    if count_syllables(word) > 2:
        complex_word_count += 1

total_words = len(tokenized_words)
percentage_complex_words = (complex_word_count / total_words) * 100

print('Percentage of Complex Words:', percentage_complex_words)


# #### 7] FOG INDEX

# In[38]:


Fog_Index = 0.4 * (average_sentence_length + percentage_complex_words)
print('fog index= ',Fog_Index )


# #### 8] AVG NUMBER OF WORDS PER SENTENCE

# In[39]:


total_words = 0
total_sentences = result_df.shape[0]

for sentence in result_df['sentences']:
    words = sentence.split()
    total_words += len(words)

average_words_per_sentence = total_words / total_sentences
print('Average Number of Words Per Sentence:', average_words_per_sentence)


# #### 9] COMPLEX WORD COUNT

# In[40]:


vowels = ['a', 'e', 'i', 'o', 'u']

def count_syllables(word):
    count = 0
    word = word.lower()
    
    if word.endswith('es') or word.endswith('ed'):
        return count

    for char in word:
        if char in vowels:
            count += 1

    return count

complex_word_count = 0

for word in tokenized_words:
    if count_syllables(word) > 2:
        complex_word_count += 1

print('Complex Word Count:', complex_word_count)


# #### 10] WORD COUNT

# In[41]:


word_count=len(tokenized_words)
print('word count = ', word_count)


# #### 11] SYLLABLE PER WORD

# In[42]:


vowels = ['a', 'e', 'i', 'o', 'u']

def count_syllables(word):
    count = 0
    word = word.lower()

    if word.endswith('es') or word.endswith('ed'):
        return count

    for char in word:
        if char in vowels:
            count += 1

    return count

syllable_count_per_word = [count_syllables(word) for word in tokenized_words]
print(syllable_count_per_word)


# #### 12] PERSONAL PRONOUNS

# In[44]:


personal_pronouns = ['i', 'we', 'my', 'ours', 'us']

personal_pronoun_count = 0

for word in tokenized_words:
    if word.lower() in personal_pronouns:
        personal_pronoun_count += 1

print('Personal Pronoun Count:', personal_pronoun_count)


# #### 13] AVG WORD LENGTH

# In[45]:


total_characters = sum(len(word) for word in tokenized_words) 
total_words = len(tokenized_words)  
average_word_length = total_characters / total_words 

print('Average Word Length:', average_word_length)


# <a id="9"></a> <br>
# ### Output Data Structure

# In[47]:


output_df = pd.read_csv(r"C:\Users\zerin\Downloads\NLP Blackoffer project\Input.csv")[['URL_ID','URL']]

output_df = output_df.iloc[0:150]
indices_to_drop = [23, 43, 56, 143]

output_df.drop(indices_to_drop, axis=0, inplace=True)
output_df


# In[62]:


# Add the calculated parameters to the DataFrame
output_df['POSITIVE SCORE'] = positive_score
output_df['NEGATIVE SCORE'] = negative_score
output_df['POLARITY SCORE'] = Polarity_Score
output_df['SUBJECTIVITY SCORE'] = subjectivity_score
output_df['AVG SENTENCE LENGTH'] = average_sentence_length
output_df['PERCENTAGE OF COMPLEX WORDS'] = percentage_complex_words
output_df['FOG INDEX'] = Fog_Index
output_df['AVG NUMBER OF WORDS PER SENTENCE'] = average_words_per_sentence
output_df['COMPLEX WORD COUNT'] = complex_word_count
output_df['WORD COUNT'] = word_count
output_df['SYLLABLE PER WORD'] = syllable_count_per_word 
output_df['PERSONAL PRONOUNS'] = personal_pronouns
output_df['AVG WORD LENGTH'] = average_word_length


# In[79]:


output_df.to_csv('Output_Data.csv', index=False)


# In[64]:


import os

current_directory = os.getcwd()
print("Current working directory:", current_directory)

