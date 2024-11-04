import os
import pandas as pd
import glob
import argparse
import unicodedataplus as ud
from collections import Counter
import pycountry

def get_script(text):
    script_counts = Counter()
    limit = 50

    for char in text[:limit]:
        if char.strip():
            try:
                script = ud.script(char)
                script_counts[script] += 1
            except Exception as e:
                print(f"Error getting script for character {char}: {e}")

    if not script_counts:
        return None

    most_common_script = script_counts.most_common(1)[0][0]

    return most_common_script


def get_language_name(language_code):
    try:
        lang = pycountry.languages.get(alpha_3=language_code)

        if lang:
            return lang.name
        else:
            print(f"Could not find language name for code {language_code}")
            return None
    except Exception as e:
        print(f"Error getting language name for {language_code}: {e}")
        return None

def preprocess_ebible_data(inputpath, outputpath):
    script_dataframes = {}

    corpus_folder = os.path.join(inputpath, 'corpus')

    txt_files = glob.glob(os.path.join(corpus_folder, '*.txt'))

    output_folder = os.path.join(outputpath, 'ebible_corpus_data')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    for file_path in txt_files:
        file_name = os.path.basename(file_path)

        language_code = file_name.split('-')[0]

        language_name = get_language_name(language_code)
        if not language_name:
            print(f"Skipping {file_name} due to unknown language code.")
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_lines = [line.strip() for line in lines if line.strip() and '<range>' not in line]

        limited_lines = cleaned_lines[:1000]

        if not limited_lines:
            print(f"No valid text found in {file_name}, skipping.")
            continue

        script = get_script(''.join(limited_lines))
        if not script:
            print(f"Skipping {file_name} due to unknown script.")
            continue

        temp_df = pd.DataFrame({'Text': limited_lines, 'Language': [language_name] * len(limited_lines)})

        if script not in script_dataframes:
            script_dataframes[script] = pd.DataFrame(columns=['Text', 'Language'])
        
        script_dataframes[script] = pd.concat([script_dataframes[script], temp_df], ignore_index=True)

    for script, df in script_dataframes.items():
        output_file = os.path.join(output_folder, f'{script}_data.csv')
        df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ebible data and generate CSV files by script")
    parser.add_argument('--inputpath', type=str, required=True, help='Path to the input folder containing the corpus of text files')
    parser.add_argument('--outputpath', type=str, required=True, help='Path to the output folder where the CSV files will be saved')

    args = parser.parse_args()

    preprocess_ebible_data(args.inputpath, args.outputpath)
