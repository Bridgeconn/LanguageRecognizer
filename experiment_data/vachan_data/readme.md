## Vachan Data Preparation

12 languages that follow the same Devangiri script.
Source: https://github.com/Bridgeconn/vachan-data/tree/master


### How to setup the data for our work

1. Clone the above repo to this folder
2. Run the following script to format the data for our usage:
	```bash
	python preprocess_vachan_data.py --inputpath ./vachan-data --outputpath ./
	```
	This creates one csv/tsv file having columns with input text and language label.
3. Make sure not to commit the whole vachan-data repo or processed data to this repo. Only keep necessary scripts to pull and prepare the data as required.