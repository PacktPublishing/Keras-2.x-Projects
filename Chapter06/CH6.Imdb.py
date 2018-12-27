from keras.datasets import imdb

(XTrain, YTrain),(XTest, YTest) = imdb.load_data(path="imdb.npz",
                    num_words=None,
                    skip_top=0,
                    maxlen=None,
                    seed=113,
                    start_char=1,
                    oov_char=2,
                    index_from=3)


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


print(XTrain[0])
len(XTrain[0])

for i in XTrain[0:10]:
    print(len(i))


WordIndex = imdb.get_word_index()
print(type(WordIndex))
print(len(WordIndex))

for keys, values in WordIndex.items():
    if values == 88283:
        print(keys)
        
print(WordIndex.items())

for keys, values in WordIndex.items():
    if values == 16115:
        print(keys)

ReverseIndex = dict([(value, key) for (key, value) in WordIndex.items()]) 
DecodedReview = " ".join( [ReverseIndex.get(i - 3, "!") for i in XTrain[0]] )
print(DecodedReview) 

(XTrain, YTrain),(XTest, YTest) = imdb.load_data(num_words=10000)

print(max([max(sequence) for sequence in XTrain]))


from keras.preprocessing.sequence import pad_sequences

XTrain= pad_sequences(XTrain, maxlen=100)
XTest = pad_sequences(XTest, maxlen=100)

print(XTrain[5])

for i in XTrain[0:10]:
    print(len(i))

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import SimpleRNN
from keras.layers import Dense
from keras.layers import Activation

RNNModel=Sequential()
RNNModel.add(Embedding(10000, 32, input_length=100))
RNNModel.add(SimpleRNN(32, input_shape=(10000, 100), return_sequences=False))
RNNModel.add(Dense(1))
RNNModel.add(Activation('sigmoid'))

print(RNNModel.summary())

RNNModel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


RNNModel.fit(XTrain, YTrain, validation_data=(XTest, YTest), epochs=3, batch_size=64, verbose=1)

scores = RNNModel.evaluate(XTest, YTest, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

