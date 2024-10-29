'''Implements fdifferent function that use multiple algorithms to train models as selected in each experiment run'''
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
import joblib

def train_model_MNNB(train_data):
    """Train a Multinomial Naive Bayes model"""
    X_train, y_train = train_data
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_model_LR(train_data):
    """Train a Logistic Regression model"""
    X_train, y_train = train_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_model_decisiontree(train_data):
    """Train a Decision Tree model"""
    X_train, y_train = train_data
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def train_model_svm(train_data):
    """Train a Linear SVM model"""
    X_train, y_train = train_data
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model