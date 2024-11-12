'''Take input text and output its language and script'''
from script_detector import detect_script
import joblib

def recognize_language(text):
    '''
    Input: the text in the unknown language
    Output: The identified script name and language as a tuple
    '''
    try:
        script_name = detect_script(text)
        vectorizer_path = f"../models/{script_name}_model_vectorizer.joblib"
        vectorizer = joblib.load(vectorizer_path)
        vectorized_text = vectorizer.transform([text])
        model_path = f"../models/{script_name}_model.joblib"
        model = joblib.load(model_path)
        language_name = str(model.predict(vectorized_text)[0])
        return (language_name, script_name)
    except FileNotFoundError as fnf_error:
        print(f"Error: Language model not available for the script {script_name}")

if __name__ == "__main__":
    text = "यहोवा का भय मानना, जीवन का सोता है, और उसके द्वारा लोग मृत्यु के फंदों से बच जाते हैं।"
    print(recognize_language(text))