'''Methods to generate and report evaluation metrics like accuracy, confusion matrix etc'''
from sklearn import metrics
import logging

def run_eval(model, test_data, log_file):
    logging.basicConfig(filename=f'{log_file}.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    X_test, y_test = test_data
    y_pred = model.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1_score = metrics.f1_score(y_test, y_pred, average='weighted', zero_division=1)
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy}")
    logging.info(f"Precision: {precision}")
    logging.info(f"Recall: {recall}")
    logging.info(f"F1 Score: {f1_score}")
    logging.info(f"Confusion Matrix: \n{confusion_matrix}")

    unique_classes = len(set(y_test))
    
    classification_report = None
    if unique_classes > 1:
        classification_report = metrics.classification_report(y_test, y_pred)
        logging.info(f"Classification Report:\n{classification_report}")