from keras.layers import Input, Dense
from keras.models import Model

InputTensor = Input(shape=(100,))
H1 = Dense(10, activation='relu')( InputTensor)
H2 = Dense(20, activation='relu')(H1)
Output = Dense(1, activation='softmax')(H2)

model = Model(inputs=InputTensor, outputs=Output)

model.summary()
