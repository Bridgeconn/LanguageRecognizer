from huggingface_hub import hf_hub_download
import joblib
from sklearn import metrics
import pandas as pd
import os
import shutil
from .script_detector import detect_script
from .ld_dir import create_ld_dir

def evaluate_models(labelled_test_data):
    create_ld_dir()
    if not labelled_test_data:
        raise ValueError("Empty test data provided")
        
    if not all(isinstance(item, tuple) and len(item) == 2 for item in labelled_test_data):
        raise ValueError("Test data must be list of (Text, Language) tuples")

    test_df = pd.DataFrame(labelled_test_data, columns=['Text', 'Language'])
    results = {}

    test_df['Script'] = test_df['Text'].apply(detect_script)
      
    script_groups = test_df.groupby('Script')
    
    home_dir = os.path.expanduser("~")
    models_dir = os.path.join(home_dir, ".ld_data", "models")
    repo_id = "Gladys-Ann-Varughese/multi-script-language-identifier"
    
    for script_name, script_data in script_groups:
        try:
            vectorizer_filename = f"{script_name}_vectorizer.joblib"
            model_filename = f"{script_name}_model.joblib"
            vectorizer_path = os.path.join(models_dir, f"{script_name}-vectorizer", vectorizer_filename)
            model_path = os.path.join(models_dir, f"{script_name}-model", model_filename)

            try:
                vectorizer_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{script_name}-vectorizer/{vectorizer_filename}",
                    local_dir=models_dir,
                )
    
            except:
                partial_dir = os.path.join(models_dir, f"{script_name}-vectorizer")
                if os.path.exists(partial_dir):
                    shutil.rmtree(partial_dir)
                raise FileNotFoundError(f"Vectorizer not found for {script_name}")
        
            try:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=f"{script_name}-model/{model_filename}",
                    local_dir=models_dir,
                )
            except:
                partial_dir = os.path.join(models_dir, f"{script_name}-model")
                if os.path.exists(partial_dir):
                    shutil.rmtree(partial_dir)
                raise FileNotFoundError(f"Model not found for {script_name}")

            vectorizer = joblib.load(vectorizer_path)
            model = joblib.load(model_path)

            X_test = vectorizer.transform(script_data['Text'])
            y_test = script_data['Language'].values
            y_pred = model.predict(X_test)

            clean_report = metrics.classification_report(
            y_test, 
            y_pred, 
            labels=list(set(y_test)), 
            zero_division=0
            ).replace('\n\n', '').replace('\n', '')

            results[script_name] = {
            'accuracy': float(metrics.accuracy_score(y_test, y_pred)),
            'precision': float(metrics.precision_score(y_test, y_pred, average='weighted', zero_division=1)),
            'recall': float(metrics.recall_score(y_test, y_pred, average='weighted', zero_division=1)),
            'f1_score': float(metrics.f1_score(y_test, y_pred, average='weighted', zero_division=1)),
            'classification_report': clean_report,
            'languages_tested': list(set(y_test)),
            }
                
        except Exception as e:
            results[script_name] = {'Error': (e)}
            continue

    return results
