from modellist import model_list
from fuzzywuzzy import fuzz
 
def list_models(script_name=None, lang_name=None, downloaded=None):
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
            if model["downloaded"] != downloaded:
                filtered_models.remove(model)
                continue

        model["similarity"] = score

    sorted_models = sorted(filtered_models, key=lambda x: x.get("similarity", 0), reverse=True)

    for model in sorted_models:
        model.pop("similarity", None)
        
    return sorted_models


def get_model(script_name, lang_name=None):
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
