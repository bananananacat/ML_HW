import click
import numpy as np
import pandas as pd
import nltk
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
import warnings
from sklearn.exceptions import DataConversionWarning
import pickle
import os

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    lst = text.split()
    fixed_text = []
    for word in lst:
        for mark in string.punctuation:
            word = word.replace(mark, "")
        fixed_text.append(word.lower())
    s = " ".join(fixed_text)
    return s

def lemmatization_and_del_stopw(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    preprocessed_text = preprocess_text(text)
    lst = preprocessed_text.split()
    filtered_and_lemmatized_lst = [lemmatizer.lemmatize(word) for word in lst if word not in stop_words]
    filtered_text = " ".join(filtered_and_lemmatized_lst)
    return filtered_text

@click.group()
def main():
    pass

@main.command()
@click.option('--data', type=str, help='Путь к обучающим данным', required=True)
@click.option('--test', type=str, help='Путь к тестовым данным')
@click.option('--split', type=float, help='Доля тестовой выборки')
@click.option('--model', type=str, help='Путь для сохранения модели', required=True)
def train(data, test, split, model):
    df = pd.read_csv(data)
    new_text = df["text"].apply(lemmatization_and_del_stopw)
    x_train, x_test, y_train, y_test = train_test_split(new_text, df["rating"], train_size=1-split)
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    
    lrmodel = LogisticRegression()
    lrmodel.fit(x_train, y_train)
    
    with open(model, 'wb') as f:
        pickle.dump((lrmodel, vectorizer), f)
    
    prediction = lrmodel.predict(x_test)
    f1 = f1_score(prediction, y_test, average="weighted")
    click.echo(f"F1-мера на тестовой выборке: {f1}")

@main.command()
@click.option('--model', type=str, help='Путь к обученной модели', required=True)
@click.option('--data', type=str, help='Текст для предсказания', required=True)
def predict(model, data):
    if ".csv" in data:
        with open(model, 'rb') as f:
            lrmodel, vectorizer = pickle.load(f)

        df = pd.read_csv(data)
        text = df["text"].apply(lemmatization_and_del_stopw)
        new_test = vectorizer.transform(text)
        prediction = lrmodel.predict(new_test)
        for p in prediction:
            click.echo(p)
    else:
        with open(model, 'rb') as f:
            lrmodel, vectorizer = pickle.load(f)

        text = pd.Series(data).apply(lemmatization_and_del_stopw)
        new_test = vectorizer.transform(text)

        prediction = lrmodel.predict(new_test)
        click.echo(prediction)

if __name__ == '__main__':
    main()
