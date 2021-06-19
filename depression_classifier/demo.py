import argparse
import joblib
import utils

labels = {
    0 : "Not Depresed",
    1 : 'Depressed'
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Model Path', default='model/model.joblib')
    args = parser.parse_args()

    sentiment = input("Enter sentiment here: ")
    with open(args.model, 'rb') as f:
        model = joblib.load(f)

    norm_sentiment = utils.text_preproc(sentiment)
    results = model.predict([norm_sentiment])[0]
    probs = model.predict_proba([norm_sentiment])[0].max()
    print('Label: {}'.format(labels[results]))
    print('Score: {} '.format(round(probs * 100, 2)))