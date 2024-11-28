'''Take input text and output its language and script'''
from huggingface_hub.utils import enable_progress_bars
from huggingface_hub import hf_hub_download
from script_detector import detect_script
from ld_dir import create_ld_dir
import shutil
import joblib
import os

def recognize_language(text):
    """
    Input: The text in the unknown language
    Output: The identified script name and language as a tuple
    """
    create_ld_dir()
    enable_progress_bars()
    repo_id="Gladys-Ann-Varughese/multi-script-language-identifier"
    try:
        script_name = detect_script(text)

        home_dir = os.path.expanduser("~")
        models_dir = os.path.join(home_dir, ".ld_data", "models")

        os.makedirs(models_dir, exist_ok=True)

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
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Vectorizer not found for {script_name}") from e
        
        finally:
            partial_dir = os.path.join(models_dir, f"{script_name}-vectorizer")
            if os.path.exists(partial_dir):
                shutil.rmtree(partial_dir)
            
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=f"{script_name}-model/{model_filename}",
                local_dir=models_dir,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error: Model not found for {script_name}") from e
            
        finally:  
            partial_dir = os.path.join(models_dir, f"{script_name}-model")
            if os.path.exists(partial_dir):
                shutil.rmtree(partial_dir)
            

        vectorizer = joblib.load(vectorizer_path)
        vectorized_text = vectorizer.transform([text])

        model = joblib.load(model_path)
        language_name = str(model.predict(vectorized_text)[0])

        return language_name, script_name
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}") from e
