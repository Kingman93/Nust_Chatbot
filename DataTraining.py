# Data training MODEL

import json
import pickle
import random
import numpy as np

# import self
from nltk.stem import WordNetLemmatizer
import nltk
from keras.optimizers import SGD
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential

lemmatizer = WordNetLemmatizer()
communications = json.loads(open('Bot_data.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', ',', '.', '!']

for communication in communications['communications']:
    for pattern in communication['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, communication['tag']))
        if communication['tag'] not in classes:
            classes.append(communication['tag'])

print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

print(words)

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# training the data with machine learning

training = []
output_empty = [0] * len(classes)

try:
    for document in documents:
        carrier = []
        word_linkage = document[0]
        word_linkage = [lemmatizer.lemmatize(word.lower()) for word in word_linkage]
        for word in words:
            carrier.append(1) if word in word_linkage else carrier.append(0)

        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([carrier, output_row])

    random.shuffle(training)
    training = np.array(training)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    # Building the neural network model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics='accuracy')
    hold = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save('NUST_BOT_model.model', hold)
    print("Successful")

except:
    print("An error occurred during processing")
