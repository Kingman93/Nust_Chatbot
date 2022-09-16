# import random
import json
import pickle
import random
import numpy as np

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import nltk

lemmatizer = WordNetLemmatizer()
communications = json.loads(open('Bot_data.json').read())

words = pickle.load(open("words.pkl", 'rb'))
classes = pickle.load(open("classes.pkl", 'rb'))

model = load_model('NUST_BOT_model.model')


def sentence_cleaner(sentence):
    words_in_sentence = nltk.word_tokenize(sentence)
    words_in_sentence = [lemmatizer.lemmatize(word) for word in words_in_sentence]
    return words_in_sentence


def word_collection(sentence):
    words_in_sentence = sentence_cleaner(sentence)
    collection = [0] * len(words)
    for word in words_in_sentence:
        for i, wrd in enumerate(words):
            if wrd == word:
                collection[i] = 1
    return np.array(collection)


def predict_class(sentence):
    bow = word_collection(sentence)
    container = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(container) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'communication': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(communication_list, bot_data_json):
    tag = communication_list[0]['communication']
    list_of_communication = bot_data_json['communications']
    for i in list_of_communication:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print('NUST BOT Running')

while True:
    message = input('')
    com = predict_class(message)
    res = get_response(com, communications)
    print(res)
