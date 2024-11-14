from modellist import model_list
from fuzzywuzzy import fuzz

def list_models(script_name=None, lang_name=None, downloaded=None):
    filtered_models = []
    
    for model in model_list:
        if script_name:
            script_score = fuzz.partial_ratio(script_name, model["script_name"])
            if script_score >= 10:
                model["similarity"] = script_score
                filtered_models.append(model)
 
        if lang_name:
            for lang in model["languages"]:
                lang_score = fuzz.partial_ratio(lang_name, lang)
                if lang_score >= 10:
                    model["similarity"] = lang_score
                    filtered_models.append(model)

        if downloaded is not None:
            if model["downloaded"] == downloaded:
                model["similarity"] = 100
                filtered_models.append(model)

    sorted_models = sorted(filtered_models, key=lambda x: x["similarity"], reverse=True)

    for model in sorted_models:
        model.pop("similarity", None)
        
    return sorted_models

def get_model(script_name, lang_name):
    if not script_name or not lang_name:
        raise ValueError("Both script_name and lang_name must be provided")
    script_found = False
    lang_found = False
    for model in model_list:
        if model["script_name"] == script_name:
            script_found = True
            for language in model["languages"]:
                if language == lang_name:
                    lang_found = True
                    return model
    if not script_found:
        raise ValueError(f"Invalid entry or script not available: {script_name}")
    if not lang_found:
        raise ValueError(f"Invalid entry or language not available: {lang_name}")
