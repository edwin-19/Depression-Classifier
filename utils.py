# from nltk import word_tokenize
# from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
import re

# porter = PorterStemmer()

def text_preproc(data):
    data = re.sub(r"http\S+", "", data)
    data = re.sub(r'pic.twitter.com/[\w]*',"", data)

    data = re.sub(r"(#[\d\w\.]+)", 'happiness', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    data = re.sub('\W+',' ', data)

    return data.strip().lower()


def normalize_data(train_data):
    normalized_data = []
    for data in train_data:
        normalized_data.append(text_preproc(data))

    return normalized_data