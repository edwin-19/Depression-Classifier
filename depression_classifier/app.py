from flask import Flask, render_template
from flask import jsonify, request
import joblib
import utils
import numpy as np

app  = Flask(__name__)
with open('model/model.joblib', 'rb') as f:
        model = joblib.load(f)

labels = {
    0 : 'Not depressed',
    1: 'Depressed'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pred_form')
def get_pred_form():
    return render_template('pred_form.html')

@app.route('/pred', methods=['POST', 'GET'])
def get_depression_index():
    sentiment = request.form['sentiment']
    norm_text = utils.text_preproc(sentiment)
    results = model.predict([norm_text])
    prob = model.predict_proba([norm_text])
    classification = labels[results[0]]

    return jsonify({
        'score': round(prob[0].max() * 100, 2),
        'classification': classification
    })

if __name__ == '__main__':
    app.run(port=8000, host='0.0.0.0', debug=True)