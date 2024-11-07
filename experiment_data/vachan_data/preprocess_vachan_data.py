import os
import pandas as pd
import glob
import argparse

def preprocess_vachan_data(inputpath, outputpath):
    combined_df = pd.DataFrame(columns=['Text', 'Language'])
    combined_csv_folder = os.path.join(inputpath, 'data/combined_csv')
    csv_files = glob.glob(os.path.join(combined_csv_folder, 'combine_*_*.csv'))

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        name_parts = file_name.replace('combine_', '').split('_')

        language_name = '_'.join(name_parts[:-1])
        language_name = language_name.replace('_', ' ')

        df = pd.read_csv(file_path, header=None)
        df = df.iloc[:1000]
        text_column = df.iloc[:, 3]

        temp_df = pd.DataFrame({'Text': text_column, 'Language': language_name})
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    output_file = os.path.join(outputpath, 'vachandata.csv')
    combined_df.to_csv(output_file, index=False)

    print(f"Data combined and saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Vachan data and combine into a single CSV file")
    parser.add_argument('--inputpath', type=str, required=True, help='Path to the input folder containing the combined CSV files')
    parser.add_argument('--outputpath', type=str, required=True, help='Path to the output folder where the final CSV file will be saved')

    args = parser.parse_args()

    preprocess_vachan_data(args.inputpath, args.outputpath)
