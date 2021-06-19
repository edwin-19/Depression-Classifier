import joblib
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import utils

import argparse
import pandas as pd
# import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='Loading test data', default='data/test.csv')
    parser.add_argument('-m', '--model', help='Model Path', default='model/model.joblib')
    args = parser.parse_args()

    # Load model
    with open(args.model, 'rb') as f:
        model = joblib.load(f)

    # Load and prepare data
    test_df = pd.read_csv(args.data)
    y_test = test_df['label'].tolist()
    test_text = test_df['message'].tolist()
    norm_test_text = utils.normalize_data(test_text)
    
    # Run data
    y_pred = model.predict(norm_test_text)

    # Get evaluation score
    print(classification_report(y_test, y_pred))
    print('MCC Accuracy: {}'.format(round(matthews_corrcoef(y_test, y_pred) * 100, 2)))

    # sns.heatmap(confusion_matrix(y_test, y_pred))