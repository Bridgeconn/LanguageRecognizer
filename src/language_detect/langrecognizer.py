'''Take input text and output its language and script'''
from .script_detector import detect_script
import joblib
import os

def recognize_language(text):
    '''
    Input: the text in the unknown language
    Output: The identified script name and language as a tuple
    '''
    try:
        script_name = detect_script(text)

        if not script_name:
            raise ValueError("Script detection failed.")

        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        vectorizer_path = os.path.join(model_dir, f"{script_name}_vectorizer.joblib")
        model_path = os.path.join(model_dir, f"{script_name}_model.joblib")

        vectorizer = joblib.load(vectorizer_path)
        vectorized_text = vectorizer.transform([text])

        model = joblib.load(model_path)
        language_name = str(model.predict(vectorized_text)[0])
        
        return (language_name, script_name)
    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")

if __name__ == "__main__":
    text = "यहोवा का भय मानना, जीवन का सोता है, और उसके द्वारा लोग मृत्यु के फंदों से बच जाते हैं।"
    print(recognize_language(text))   