# Depression Classifier
The following stuff contains the code to perform sentiment analysis for depression

The file structure is as follows, look at how to use for more instructions
```bash
├── data
│   ├── test.csv
│   └── train.csv
├── model
│   └── model.joblib
├── static
│   └── index.js
├── templates
│   ├── index.html # Login page
│   └── pred_form.html # Page to do sentiment analysis
├── app.py # To run flask app
├── demo.py # Run for single sentiment
├── test.py # Run evaluation code
├── train.py # Run trainining code
└── utils.py # Utilities to run preprocessing
```

## How to use 

Training the model with a decision tree classifier
```bash
python train.py 

## Or you can run if you want to train from a specific folder

python train.py --data data/train.csv
```

Test the model using the following metrics:
- Accuracy
- Confusion Matrix
- F1 Score
- Mathhews Corref

To run the code to get the score:
```bash
python test.py

## Or you can run if you want to train from a specific folder

python test.py --data data/train.csv
```

Score:
```bash
              precision    recall  f1-score   support

           0       0.97      0.98      0.97        94
           1       0.98      0.97      0.97        89

   micro avg       0.97      0.97      0.97       183
   macro avg       0.97      0.97      0.97       183
weighted avg       0.97      0.97      0.97       183

MCC Accuracy: 94.5
```

## Demo Run
To run a simple inference or check if the text is depressed or not run the following comamand
```bash
python demo.py # you will be prompt to enter a text 
```

To start the web app, run the following command
```bash
python app.py

# Visit the url 
# http://localhost:8000/ - for login page
# http://localhost:8000/pred_form - for sentiment analysis page
```

## ScreenShot of web app
![Test](assets/pred_form.png?raw=True)