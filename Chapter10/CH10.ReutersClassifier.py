from keras.datasets import reuters

(XTrain, YTrain),(XTest, YTest) = reuters.load_data(num_words=None, test_split=0.3)

print('XTrain class = ',type(XTrain))
print('YTrain class = ',type(YTrain))
print('XTest shape = ',type(XTest))
print('YTest shape = ',type(YTest))

print('XTrain shape = ',XTrain.shape)
print('XTest shape = ',XTest.shape)
print('YTrain shape = ',YTrain.shape)
print('YTest shape = ',YTest.shape)

import numpy as np
print('YTrain values = ',np.unique(YTrain))
print('YTest values = ',np.unique(YTest))

unique, counts = np.unique(YTrain, return_counts=True)
print('YTrain distribution = ',dict(zip(unique, counts)))
unique, counts = np.unique(YTest, return_counts=True)
print('YTrain distribution = ',dict(zip(unique, counts)))

import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(121)
plt.hist(YTrain, bins='auto')
plt.xlabel("Classes")
plt.ylabel("Number of occurrences")
plt.title("YTrain data")

plt.subplot(122)
plt.hist(YTest, bins='auto')
plt.xlabel("Classes")
plt.ylabel("Number of occurrences")
plt.title("YTest data")
plt.show()

print(XTrain[1])
len(XTrain[1])

#The dataset_reuters_word_index() function returns a list where the names are words and the values are integer
WordIndex = reuters.get_word_index(path="reuters_word_index.json")

print(len(WordIndex))

IndexToWord = {}
for key, value in WordIndex.items():
    IndexToWord[value] = key

print(' '.join([IndexToWord[x] for x in XTrain[1]]))
print(YTrain[1])

from keras.preprocessing.text import Tokenizer

MaxWords = 10000

Tok = Tokenizer(num_words=MaxWords)
XTrain = Tok.sequences_to_matrix(XTrain, mode='binary')
XTest = Tok.sequences_to_matrix(XTest, mode='binary')



NumClasses = max(YTrain) + 1

from keras.utils import to_categorical
YTrain = to_categorical(YTrain, NumClasses)
YTest = to_categorical(YTest, NumClasses)

print(XTrain[1])
print(len(XTrain[1]))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation

DNNmodel = Sequential()
DNNmodel.add(Dense(512, input_shape=(MaxWords,)))
DNNmodel.add(Activation('relu'))
DNNmodel.add(Dropout(0.5))
DNNmodel.add(Dense(NumClasses))
DNNmodel.add(Activation('softmax'))
DNNmodel.summary()

DNNmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


DNNmodel.fit(XTrain, YTrain, validation_data=(XTest, YTest), epochs=10, batch_size=64, verbose=1)

Scores = DNNmodel.evaluate(XTest, YTest, verbose=1)
print('Test loss:', Scores[0])
print('Test accuracy:', Scores[1])

