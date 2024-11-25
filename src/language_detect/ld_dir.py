import os
import shutil

ld_dir = os.path.join(os.path.expanduser("~"), ".ld_models")
pkg_ld_dir = os.path.join(os.path.dirname(__file__), 'models')

def check_for_ld_dir():
    if not os.path.exists(ld_dir):
        os.makedirs(ld_dir, exist_ok=True)

    model_path = os.path.join(pkg_ld_dir, 'Devanagari-model/Devanagari_model.joblib')
    vectorizer_path = os.path.join(pkg_ld_dir, 'Devanagari-vectorizer/Devanagari_vectorizer.joblib')
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model_dir = os.path.join(ld_dir, 'Devanagari-model')
        vectorizer_dir = os.path.join(ld_dir, 'Devanagari-vectorizer')

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vectorizer_dir, exist_ok=True)

        if not os.path.exists(os.path.join(model_dir, 'Devanagari_model.joblib')):
            shutil.copy(model_path, model_dir)
        if not os.path.exists(os.path.join(vectorizer_dir, 'Devanagari_vectorizer.joblib')):
            shutil.copy(vectorizer_path, vectorizer_dir)

    else:
        raise FileNotFoundError()
