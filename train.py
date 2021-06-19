import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import argparse
import joblib
import os
import utils

def create_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    pos_df = df[df['label'] == 1].sample(460, random_state=2021)
    neg_df = df[df['label'] == 0]

    sampled_train_df = pd.concat([pos_df, neg_df])
    dataset_text = sampled_train_df['message'].values
    dataset_label = sampled_train_df['label'].values

    return dataset_text, dataset_label

def save_model(model, model_path='model/', model_name='model'):
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(model_path + model_name + '.joblib', 'wb') as model_file:
        joblib.dump(model, model_file)

    print('Model Saved at: ' + model_path + model_name + '.joblib')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Train data here', default='data/train.csv')
    args = parser.parse_args()

    train_text, train_label = create_dataset(args.data)
    norm_train_data = utils.normalize_data(train_text)

    vectorizer = TfidfVectorizer(sublinear_tf=True, norm='l2', ngram_range=(1, 2), stop_words='english')
    vectorizer.fit(norm_train_data)
    vectorizer_train_text = vectorizer.transform(norm_train_data)

    # Train model
    model = DecisionTreeClassifier()
    model.fit(vectorizer_train_text, train_label)

    # Create pipeline
    pipeline = Pipeline([('vect', vectorizer), ('model', model)])

    # Save trained model
    save_model(pipeline)
    