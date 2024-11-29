import importlib.util
import os

home_dir = os.path.expanduser("~")
ld_dir = os.path.join(home_dir, ".ld_data")
ld_data_dir = os.path.join(ld_dir, "data")

langlist_path = os.path.join(ld_data_dir, "langlist.py")
scriptlist_path = os.path.join(ld_data_dir, "scriptlist.py")

def load_module(path):
    spec = importlib.util.spec_from_file_location("module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def list_languages():
    lang_list = load_module(langlist_path).lang_list
    return lang_list

def list_scripts():
    script_list = load_module(scriptlist_path).script_list
    return script_list
