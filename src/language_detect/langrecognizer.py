'''Take input text and output its language and script'''
from huggingface_hub.utils import enable_progress_bars
from huggingface_hub import hf_hub_download
from script_detector import detect_script
from ld_dir import create_ld_dir
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

        if not script_name:
            raise ValueError("Script detection failed.")

        home_dir = os.path.expanduser("~")
        models_dir = os.path.join(home_dir, ".ld_data", "models")

        os.makedirs(models_dir, exist_ok=True)

        vectorizer_filename = f"{script_name}_vectorizer.joblib"
        model_filename = f"{script_name}_model.joblib"

        vectorizer_path = os.path.join(models_dir, f"{script_name}-vectorizer", vectorizer_filename)
        model_path = os.path.join(models_dir, f"{script_name}-model", model_filename)

        if not os.path.exists(vectorizer_path):
            print(f"{vectorizer_filename} not found locally. Downloading...")
            filename=f"{script_name}-vectorizer/{vectorizer_filename}"
            vectorizer_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            )

        if not os.path.exists(model_path):
            print(f"{model_filename} not found locally. Downloading...")
            filename = f"{script_name}-model/{model_filename}"
            model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            )

        vectorizer = joblib.load(vectorizer_path)
        vectorized_text = vectorizer.transform([text])

        model = joblib.load(model_path)
        language_name = str(model.predict(vectorized_text)[0])

        return language_name, script_name
    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
