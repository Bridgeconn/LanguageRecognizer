from .ld_dir import create_ld_dir
from fuzzywuzzy import fuzz
import importlib.util
import glob
import os

create_ld_dir()
home_dir = os.path.expanduser("~")
models_dir = os.path.join(home_dir, ".ld_data", "models")
data_dir = os.path.join(home_dir, ".ld_data", "data")
modellist_path = os.path.join(data_dir, "modellist.py")

def load_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def is_model_downloaded(script_name):
    
    model_files = glob.glob(os.path.join(models_dir, f"{script_name}-model", f"{script_name}_model*.joblib"))
    vectorizer_files = glob.glob(os.path.join(models_dir, f"{script_name}-vectorizer", f"{script_name}_vectorizer*.joblib"))

    return bool(model_files) and bool(vectorizer_files)


def list_models(script_name=None, lang_name=None, downloaded=None):
    create_ld_dir()
    model_list = load_module(modellist_path).model_list
    filtered_models = model_list[:]
    
    for model in model_list:
        score = 0
        
        if script_name:
            script_score = fuzz.partial_ratio(script_name, model["script_name"])
            if script_score >= 65:
                score += script_score
            else:
                filtered_models.remove(model)
                continue
                
        if lang_name:
            lang_scores = [fuzz.partial_ratio(lang_name, lang) for lang in model["languages"]]
            max_lang_score = max(lang_scores, default=0)
            if max_lang_score >= 65:
                score += max_lang_score
            else:
                filtered_models.remove(model)
                continue
                
        if downloaded is not None:
            if is_model_downloaded(model["script_name"]) != downloaded:
                filtered_models.remove(model)
                continue
                
        model["similarity"] = score
        
    sorted_models = sorted(filtered_models, key=lambda x: x.get("similarity", 0), reverse=True)
    for model in sorted_models:
        model.pop("similarity", None)
        
    return sorted_models


def get_model(script_name, lang_name=None):
    create_ld_dir()
    model_list = load_module(modellist_path).model_list
    if not script_name:
        raise ValueError("Script name must be provided")  
    script_found = False
    lang_found = False
    for model in model_list:
        if model["script_name"] == script_name:
            script_found = True
            if lang_name:
                for language in model["languages"]:
                    if language == lang_name:
                        lang_found = True
                        return model

                if not lang_found:
                    raise ValueError(f"Language '{lang_name}' not available in script '{script_name}'")
            else:
                return model
    if not script_found:
        raise ValueError(f"Script '{script_name}' not available")
