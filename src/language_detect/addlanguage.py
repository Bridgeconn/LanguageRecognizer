import os
import joblib
import pandas as pd
from sklearn import metrics
from .train import train_model_MNNB
from .data_prep import get_ngrams
from .script_detector import detect_script
from .ld_dir import create_ld_dir
import importlib.util

home_dir = os.path.expanduser("~")
ld_dir = os.path.join(home_dir, ".ld_data")
ld_models_dir = os.path.join(ld_dir, "models")
ld_data_dir = os.path.join(ld_dir, "data")

def load_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def update_model_list(model_details):
    modellist_path = os.path.join(ld_data_dir, "modellist.py")
    try:
        with open(modellist_path, 'r+', encoding='utf-8') as file:
            file.seek(0, os.SEEK_END)
            pos = file.tell()
            while pos > 0:
                pos -= 1
                file.seek(pos, os.SEEK_SET)
                if file.read(1) == ']':
                    break
            file.seek(pos - 1, os.SEEK_SET)
            formatted_entry = "\n    {\n"
            for key, value in model_details.items():
                if isinstance(value, (list, dict)):
                    formatted_entry += f"        \"{key}\": {value},\n"
                else:
                    formatted_entry += f"        \"{key}\": \"{value}\",\n"
            formatted_entry += "    },\n]"
            file.write(formatted_entry)
    except Exception as e:
        raise RuntimeError(f"Error appending to model list: {e}")



def update_lang_list(language, script):
    langlist_path = os.path.join(ld_data_dir, "langlist.py")
    entry = (language, script)
    try:
        with open(langlist_path, 'r+', encoding='utf-8') as file:
            file.seek(0, os.SEEK_END)
            pos = file.tell()
            while pos > 0:
                pos -= 1
                file.seek(pos, os.SEEK_SET)
                if file.read(1) == ']':
                    break
            lang_list = load_module(langlist_path).lang_list
            if entry in lang_list:
                return
            file.seek(pos - 1, os.SEEK_SET)
            formatted_entry = f"{entry}, \n]"
            file.write(formatted_entry)
    except Exception as e:
        raise RuntimeError(f"Error appending to language list: {e}")


def update_script_list(script):
    scriptlist_path = os.path.join(ld_data_dir, "scriptlist.py")
    try:
        with open(scriptlist_path, 'r+', encoding='utf-8') as file:
            file.seek(0, os.SEEK_END)
            pos = file.tell()
            while pos > 0:
                pos -= 1
                file.seek(pos, os.SEEK_SET)
                if file.read(1) == ']':
                    break
            script_list = load_module(scriptlist_path).script_list
            if script in script_list:
                return
            file.seek(pos - 1, os.SEEK_SET)
            formatted_entry = f"'{script}', \n]"
            file.write(formatted_entry)
    except Exception as e:
        raise RuntimeError(f"Error appending to script list: {e}")


def add_language(data_file):
    try:
        create_ld_dir()
        if not os.path.isfile(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found.")

        df = pd.read_csv(data_file)
        if 'Text' not in df.columns or 'Language' not in df.columns:
            raise ValueError("CSV file must contain `Text` and `Language` columns.")

        df['Script'] = df['Text'].apply(detect_script)

        detected_scripts = df['Script'].unique()
        if len(detected_scripts) > 1:
            raise ValueError(
                f"Dataset contains multiple scripts: {detected_scripts}. "
                "Please provide a dataset with languages in a single script."
            )

        script_name = detected_scripts[0]

        train_data, test_data, count_vectorizer = get_ngrams(data_file=data_file, num=2000, ns=[3, 3], test_set_ratio=0.9)

        model = train_model_MNNB(train_data)

        X_test, y_test = test_data
        y_pred = model.predict(X_test)

        accuracy = metrics.accuracy_score(y_test, y_pred)
        classification_report = metrics.classification_report(y_test, y_pred, labels=list(set(y_test)), zero_division=1).replace('\n\n', '').replace('\n', '')
        precision = float(metrics.precision_score(y_test, y_pred, average='weighted', zero_division=1))
        recall = float(metrics.recall_score(y_test, y_pred, average='weighted', zero_division=1))
        f1_score = float(metrics.f1_score(y_test, y_pred, average='weighted', zero_division=1))

        vectorizer_dir = os.path.join(ld_models_dir, f"{script_name}-vectorizer")
        model_dir = os.path.join(ld_models_dir, f"{script_name}-model")
        os.makedirs(vectorizer_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        vectorizer_count = len([f for f in os.listdir(vectorizer_dir) if f.endswith('.joblib')])
        model_count = len([f for f in os.listdir(model_dir) if f.endswith('.joblib')])

        vectorizer_filename = f"{script_name}_vectorizer_{vectorizer_count}.joblib"
        model_filename = f"{script_name}_model_{model_count}.joblib"

        vectorizer_path = os.path.join(vectorizer_dir, vectorizer_filename)
        model_path = os.path.join(model_dir, model_filename)

        model_name = (os.path.splitext(os.path.basename(model_filename)))[0]
        vectorizer_name = (os.path.splitext(os.path.basename(vectorizer_filename)))[0]

        joblib.dump(count_vectorizer, vectorizer_path)
        joblib.dump(model, model_path)

        update_model_list({
            "script_name": script_name,
            "languages": df['Language'].unique().tolist(),
            "model_name": model_name,
            "model_type": "Multinomial Naive Bayes",
            "vectorizer_model_name": vectorizer_name,
            "vectorizer_type": "CountVectorizer",
            "vectorizer_params": {
                "ngram_range": "(3, 3)",
                "max_features": 2000,
                "analyzer": "char"
            },
            "accuracy": accuracy,
        })

        for language in df['Language'].unique():
            update_lang_list(language, script_name)

        update_script_list(script_name)

        return {
            "script_name": script_name,
            "model_name": model_name,
            "vectorizer_name": vectorizer_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "classification_report": classification_report,
        }
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e
