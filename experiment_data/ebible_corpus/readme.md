## Ebible Corpus Preparation

One of the largest curated corpora with parallel bible data in 1000+ languages. 
Source: https://github.com/BibleNLP/ebible


### How to setup the data for our work

1. Clone the above repo to this folder
2. Run the following script to format the data for our usage:
	```bash
	python preprocess_ebible.py --inputpath ./ebible --outputpath ./
	```
	This creates one csv/tsv file per script, that needs a model. Each file would be having columns with input text and language label.
3. Make sure not to commit the whole ebible or processed data to this repo. Only keep necessary scripts to pull and prepare the data as required.
