'''Configure and run an end to end experiment pipeline'''

from enum import Enum

from data_prep import get_train_test_data
from train import train_model_svm, train_model_decisiontree, train_model_MNNB, train_model_LR
from eval import run_eval

class Algorithm(Enum):
    LOGISTIC_REGRESSION = 1
    DECISION_TREE = 2
    MNNB = 3
    SVM = 4

class FeatureExtraction(Enum):
	FREQ_WORDS_ALL = 1
	FREQ_WORD_DIM2000 = 2
	FREQ_WORD_DIM5000 = 3
	FREQ_WORD_DIM10000 = 4
	NGRAMS_3_ALL = 5
	NGRAMS_2_3_ALL = 6
	NGRAMS_2_3_4_ALL = 7
	NGRAMS_2_3_4_5_ALL = 8
	NGRAMS_3_DIM5000 = 9
	NGRAMS_2_3_DIM5000 = 10
	NGRAMS_2_3_4_DIM5000 = 11
	NGRAMS_2_3_4_5_DIM5000 = 12
	NGRAMS_3_2000 = 13
	NGRAMS_2_3_2000 = 14
	NGRAMS_2_3_4_2000 = 15
	NGRAMS_2_3_4_5_2000 = 16
	WORDPIECE = 17


def experiment_run(exp_no, script_name, data_file, algo, feat, test_set_ratio, model_path):
	train_data, test_data = get_train_test_data(data_file, feat, test_set_ratio)

	model = None
	if algo == Algorithm.MNNB:
		model = train_model_MNNB(train_data)
	elif algo == Algorithm.SVM:
		model = train_model_svm(train_data)
	elif algo == Algorithm.DECISION_TREE:
		model = train_model_decisiontree(train_data)
	elif algo == Algorithm.LOGISTIC_REGRESSION:
		model = train_model_LR(train_data)
	
	model.save(model_path)

	results = run_eval(model, test_data, log_file=f"../logs/{exp_no}_{scipt_name}")


if "__name__" == "__main__":
	'''Configure the required experiment setup here for each run'''
	exp_no = 1
	script_name = 'latin'
	data_file = f"../experiment_data/ebible_corpus/processed/{script_name}.txt"
	model_path = f"../models/run_{exp_no}_{script_name}"
	algo = Algorithm.MNNB
	feat = FeatureExtraction.NGRAMS_2_3_DIM5000
	test_set_ratio = 0.2
