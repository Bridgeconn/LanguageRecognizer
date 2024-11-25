import os
import shutil

home_dir = os.path.expanduser("~")
ld_dir = os.path.join(home_dir, ".ld_data")
ld_models_dir = os.path.join(ld_dir, "models")
ld_data_dir = os.path.join(ld_dir, "data")

pkg_dir = os.path.dirname(__file__)
pkg_models_dir = os.path.join(pkg_dir, 'models')
pkg_data_dir = os.path.join(pkg_dir, 'data')

def create_ld_dir():
    os.makedirs(ld_models_dir, exist_ok=True)
    os.makedirs(ld_data_dir, exist_ok=True)

    model_path = os.path.join(pkg_models_dir, 'Devanagari-model/Devanagari_model.joblib')
    vectorizer_path = os.path.join(pkg_models_dir, 'Devanagari-vectorizer/Devanagari_vectorizer.joblib')
    
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model_dir = os.path.join(ld_models_dir, 'Devanagari-model')
        vectorizer_dir = os.path.join(ld_models_dir, 'Devanagari-vectorizer')

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vectorizer_dir, exist_ok=True)

        if not os.path.exists(os.path.join(model_dir, 'Devanagari_model.joblib')):
            shutil.copy(model_path, model_dir)
        if not os.path.exists(os.path.join(vectorizer_dir, 'Devanagari_vectorizer.joblib')):
            shutil.copy(vectorizer_path, vectorizer_dir)
    else:
        raise FileNotFoundError("Default model files not found")

    for data_file in os.listdir(pkg_data_dir):
        if data_file.endswith('.py'):
            src_path = os.path.join(pkg_data_dir, data_file)
            dst_path = os.path.join(ld_data_dir, data_file)
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
