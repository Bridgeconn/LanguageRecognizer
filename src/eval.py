'''Methods to generate and report evaluation metrics like accuracy, confusion matrix etc'''
from sklearn import metrics
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

def run_eval(model, test_data, log_file=None):
    X_test, y_test = test_data
    y_pred = model.predict(X_test)

    results = {
        'accuracy': metrics.accuracy_score(y_test, y_pred),
        'classification_report': metrics.classification_report(y_test, y_pred),
        'confusion_matrix': metrics.confusion_matrix(y_test, y_pred)
    }

    if log_file:
        with open(f"{log_file}.txt", 'w') as f:
            f.write(f"Accuracy: {results['accuracy']*100:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
            f.write("\nConfusion Matrix:\n")
            f.write(str(results['confusion_matrix']))

        # plt.figure(figsize=(10,8))
        # sns.heatmap(results['confusion_matrix'], annot=True, fmt='d')
        # plt.title('Confusion Matrix')
        # plt.savefig(f"{log_file}_confusion_matrix.png")
        # plt.close()
    
    return results
