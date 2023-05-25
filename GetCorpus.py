#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pprint

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation


# In[2]:


# Set random seed for reproducibility
np.random.seed(42)

def load_corpus(corpus_path):
    """Load and preprocess the data from a JSON file."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    return corpus


# In[3]:


def convert_annotations(corpus):
    """Convert annotations in the corpus to a standard format."""
    annotation_convert = {
        'none': 'none',
        'applause': 'applause',
        'laughter': 'laughter',
        'laughing': 'laughter',
        'laughs': 'laughter',
        'laughter applause': 'laughter applause',
        'laughter) (applause': 'laughter applause',
        'audience gasps': 'gasp',
        'audio': 'audio',
        'gasping': 'gasp',
        'mock sob': 'gasp',
    }
    for item in corpus:
        for sentence in item['transcript']:
            annotation = sentence['annotation']
            if annotation in annotation_convert:
                sentence['annotation'] = annotation_convert[annotation]
            else:
                sentence['annotation'] = 'none'


# In[4]:


def extract_features_labels(corpus):
    """Extract features and labels from the corpus."""
    X = []
    y = []
    for data in corpus:
        features = [
            data["FKRE_score"],
            data["NAWL"],
            data["NGSL"],
            data["WPM"],
            score(data)
        ]
        X.append(features)
        y.append(data["like_count"] / data["view_count"])
    X = np.array(X)
    y = np.array(y)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1))

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    y = np.reshape(y, (y.shape[0], y.shape[1], 1))

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


# In[5]:


def score(item):
    """Calculate the score based on annotations."""
    annotations = {}
    for sentence in item['transcript']:
        annots = sentence['annotation'].split(" ")
        for annot in annots:
            annotations[annot] = annotations.get(annot, 0) + 1

    reactions = {'applause': 3, 'laughter': 2, 'none': 0, 'gasp': 1, 'audio': 0}
    total_score = 0
    for key in annotations:
        total_score += annotations[key] * reactions[key]

    return total_score


# In[6]:


def train_model(X_train, y_train):
    """Build and train the LSTM model."""
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, epochs=30, batch_size=32)

    return regressor


# In[7]:


def predict_and_evaluate(regressor, X_test, y_test):
    """Predict and evaluate the model."""
    y_pred = regressor.predict(X_test)
    y_pred_normal = np.squeeze(y_pred)
    y_test_normal = np.squeeze(y_test)

    for i, y_pred_val in enumerate(y_pred_normal):
        print("Predicted:", y_pred_val, "Actual:", y_test_normal[i])

    mse = mean_squared_error(y_test_normal, y_pred_normal)
    print("MSE:", mse)

# Load and preprocess data
corpus_path = "data.json"
corpus = load_corpus(corpus_path)
convert_annotations(corpus)
X, y = extract_features_labels(corpus)

# Perform KMeans clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(np.reshape(y, (y.shape[0], y.shape[1])))

# Get the good and bad examples
good_examples_X = X[kmeans.labels_ == 0]
bad_examples_X = X[kmeans.labels_ == 1]
good_examples_y = y[kmeans.labels_ == 0]
bad_examples_y = y[kmeans.labels_ == 1]

print("Good examples:", good_examples_X.shape, good_examples_y.shape)
print("Bad examples:", bad_examples_X.shape, bad_examples_y.shape)

# Split the corpus into train and test sets
X_train_good, X_test_good, y_train_good, y_test_good = train_test_split(good_examples_X, good_examples_y, test_size=0.2, shuffle=False)
X_train_bad, X_test_bad, y_train_bad, y_test_bad = train_test_split(bad_examples_X, bad_examples_y, test_size=0.2, shuffle=False)

# Concatenate the good and bad examples
X_train = np.concatenate((X_train_good, X_train_bad))
X_test = np.concatenate((X_test_good, X_test_bad))
y_train = np.concatenate((y_train_good, y_train_bad))
y_test = np.concatenate((y_test_good, y_test_bad))

print("Train corpus size:", len(X_train))
print("Test corpus size:", len(X_test))

# Reshape the target data
y_train = np.squeeze(y_train, axis=2)

# Build and train the LSTM model
regressor = train_model(X_train, y_train)

# Predict and evaluate the model
predict_and_evaluate(regressor, X_test, y_test)

