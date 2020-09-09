import sys

import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import pickle

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM cleaned_data", engine)
    X = df['message']
    Y = df.iloc[:,4:]
    categories = Y.columns
    return X, Y, categories


def tokenize(text):
    stop_words = stopwords.words('english')
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize andremove stop words
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def custom_f1_score(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    diffs = []
    for col in range(y_true.shape[1]):
        err = f1_score(y_true[:,col], y_pred[:,col], average='weighted')
        diffs.append(err)
    return np.mean(diffs)

def build_model():
    pipeline = Pipeline([
        ('tokenizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = [
        {
            'clf__estimator': [RandomForestClassifier()],
            'clf__estimator__n_estimators': [1],
            'clf__estimator__max_depth': [1, 5],
            'clf__estimator__class_weight': [None, 'balanced']
        },
        {
            'clf__estimator': [AdaBoostClassifier()],
            'clf__estimator__n_estimators': [10, 20],
        }

    ]

    scoring = make_scorer(custom_f1_score, greater_is_better=True)

    model = GridSearchCV(pipeline, parameters, scoring=scoring)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = pd.DataFrame(model.predict(X_test), columns=Y_test.columns, index=X_test.index)
    results_fmt = "{:<22} Precision: {:.3f}   Recall: {:.3f}   F1-score: {:.3f}"
    for col in Y_test.columns:
        res = [ col.upper(),
               precision_score(Y_test[col], y_pred[col], average='weighted'),
               recall_score(Y_test[col], y_pred[col], average='weighted'),
               f1_score(Y_test[col], y_pred[col], average='weighted')]
        print(results_fmt.format(*res))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()