# language-detect
This is a language detection library using ML models

## System Description
Given a text, examine the script and word formations to determine the language. 
The aim is to correctly distinguish those languages that follow same script.
Meant to be used as the fist step for text processing pipelines that use language dependant next steps like spell checks or rendering.

## Installation
To install language_detect, you can use pip:
```python
    pip install language-detect
```

## Usage
```python
import language_detect as ld

res1 = ld.recognize_language("Hello there, how are you? Hope you are doing well.") #("English", "Latin")

res2 = ld.detect_script("नमस्ते, आप कैसे हैं? मुझे आशा है कि आप अच्छा कार्य कर रहे हैं।") #Devanagari

ld.list_languages() #[("English", "Latin"), ("Hindi", "Devanagari"), ...]

ld.list_scripts() #["Latin", "Devanagari", "Cyrillic", ...]

models = ld.list_models(script_name="Devanagari", lang_name="Hindi", downloaded=True) #[{"script_name": "Devanagari","languages": ["Marathi", "Nepali (individual language)", "Sanskrit", "Urdu", "Hindi",...], "model_name": "Devanagari_model", "downloaded": True, "model_type": "Multinomial Naive Bayes", "vectorizer_model_name": "Devanagari_vectorizer", "vectorizer_type": "CountVectorizer", "vectorizer_params": {"ngram_range": "(3, 3)", "max_features": 2000, "analyzer": "char"}, ...]

model = ld.get_model(script_name="Devanagari", lang_name="Hindi") #{"script_name": "Devanagari","languages": ["Marathi", "Nepali (individual language)", "Sanskrit", "Urdu", "Hindi",...], "model_name": "Devanagari_model", "downloaded": True, "model_type": "Multinomial Naive Bayes", "vectorizer_model_name": "Devanagari_vectorizer", "vectorizer_type": "CountVectorizer", "vectorizer_params": {"ngram_range": "(3, 3)", "max_features": 2000, "analyzer": "char"}
```

## Functions
recognize_language(text) : Takes a string as input and returns a tuple containing the detected language name and script name.

detect_script(text) : Takes a string as input and returns the name of the script used in the given text.

list_languages() : Returns a list of all languages available in the database, each paired with its associated script.

list_scripts() : Returns a list of all script names available in the database.

list_models(script_name=None, lang_name=None, downloaded=None) : Filters and returns models as a list of dictionaries based on the provided script name, language name or download status. If no arguments are provided, it returns all available models.

get_model(script_name, lang_name=None) : Fetches and returns a specific model from the database as a dictionary, based on the provided script name and optional language name. 

## Experiments
We conducted experiments to identify the most effective methods for script-wise language detection. Below is a summary of our approach:

> Datasets:
We utilized datasets from the eBible Corpus and Vachan Data. These datasets were categorized based on their scripts.

> Models and Algorithms:
Each script-wise dataset was used to train multiple machine learning models, including:
- Multinomial Naive Bayes
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)

> Feature Extraction Techniques:
To represent text data effectively, we tried different feature extraction techniques:
- Character n-gram Approach: Explored ranges like [3, 3], [2, 4], and [2, 3].
- Frequent Word Selection: Retained frequent words from the dataset as features.

> Feature Limitations:
To optimize performance and reduce overfitting, we experimented with limiting the maximum number of features:
- Used all available features
- 5000 features
- 2000 features.

> Results:
After testing various combinations of algorithms, feature extraction methods, and feature limits, we found that the most accurate results were achieved using:
1. Multinomial Naive Bayes as the algorithm
2. Character n-gram Approach with ranges [3, 3] or [2, 3]
3. Maximum Features set to 2000
Using this combination, we trained and finalized 15 models, each corresponding to a specific script.

For detailed experiment findings, please refer to the [Experiment notes](https://github.com/Bridgeconn/LanguageRecognizer/blob/main/docs/Experiment_notes.md)