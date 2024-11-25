'''Load data from respective corpus in experiment_data folder, 
perform appropriate text cleaning, suffling, feature extraction etc as per the chosen experiment options,
prepare them as dataframes or other required formats for training,
Perform train-test split split etc.'''
import regex
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from language_detect.script_detector import detect_script
# from transformers import BertTokenizer

def process_text(text):
    try:
        script_name = detect_script(text)
        text = regex.sub(fr'[^\w\s\p{{{script_name}}}]', ' ', text)
        text = regex.sub(fr'\s+', ' ', text)
    except Exception as e:
        print(f"Error processing text: {e}")

    return text

def freq_words(df, script_name):
    all_text = ' '.join(df['Text'])
    words = regex.findall(fr'\p{{{script_name}}}+', all_text)
    word_freq = Counter(words)
    threshold = 50
    frequent_words = {word for word, freq in word_freq.items() if freq > threshold}
    return frequent_words

def retain_freq_words(text, script_name, frequent_words):
    words_in_text = regex.findall(fr'\p{{{script_name}}}+', text)
    filtered_words = [word for word in words_in_text if word in frequent_words]

    return ' '.join(filtered_words)

def get_ngrams(data_file, num, ns, test_set_ratio):
    df = pd.read_csv(data_file, delimiter = ',')
    df['Text'] = df['Text'].apply(process_text)

    X = df['Text']
    y = df['Language']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_ratio, random_state=42)

    if not ns:
        raise ValueError("Parameter 'ns' should be a list of n-gram values.")

    count_vectorizer = CountVectorizer(max_features=num, ngram_range=(min(ns), max(ns)), analyzer='char')

    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)

    train_data = (X_train_count, y_train)
    test_data = (X_test_count, y_test)

    return (train_data, test_data, count_vectorizer)

def get_freq_words(data_file, num, test_set_ratio):
    df = pd.read_csv(data_file, delimiter = ',')
    df['Text'] = df['Text'].apply(process_text)

    script_name = detect_script(' '.join(df['Text']))

    frequent_words = freq_words(df, script_name) 

    df['Text'] = df['Text'].apply(lambda text: retain_freq_words(text, script_name, frequent_words))

    X = df['Text']
    y = df['Language']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_ratio, random_state=42)

    count_vectorizer = CountVectorizer(max_features=num, analyzer='word')

    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)

    train_data = (X_train_count, y_train)
    test_data = (X_test_count, y_test)

    return (train_data, test_data, count_vectorizer)

# def get_wordpiece(data_file, test_set_ratio):
#     df = pd.read_csv(data_file, delimiter = ',')
#     df['Text'] = df['Text'].apply(process_text)
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     df['Text'] = df['Text'].apply(lambda text: ' '.join(tokenizer.tokenize(text)))

#     X = df['Text']
#     y = df['Language']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_ratio, random_state=42)

#     count_vectorizer = CountVectorizer()

#     X_train_count = count_vectorizer.fit_transform(X_train)
#     X_test_count = count_vectorizer.transform(X_test)

#     train_data = (X_train_count, y_train)
#     test_data = (X_test_count, y_test)

#     return (train_data, test_data, count_vectorizer)
