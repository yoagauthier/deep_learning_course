from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

NUM_DIGITS = 10
NUM_HIDDEN = 100
NB_CLASSES = 4
NB_EPOCHS = 10000
BATCH_SIZE = 128


# binary encoding of a digit (max NUM_DIGITS bits)
def binary_encode(i):
    return [i >> d & 1 for d in range(NUM_DIGITS)]


# creation of ground truth : number, "fizz", "buzz", "fizzbuzz"
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


# creating dataset
trX = np.array([binary_encode(i) for i in range(101, 2 ** NUM_DIGITS)])
trY = to_categorical(
    np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)]),
    NB_CLASSES
)
teX = np.array([binary_encode(i) for i in range(1, 101)])
teY = to_categorical(
    np.array([fizz_buzz_encode(i) for i in range(1, 101)]),
    NB_CLASSES
)

model = Sequential()
model.add(  # hidden layer
    Dense(
        NUM_HIDDEN,
        input_shape=(NUM_DIGITS, ),
        activation='relu',
        kernel_initializer='random_normal',
        bias_initializer='zeros'
    )
)
model.add(
    Dense(
        NB_CLASSES,
        activation='softmax',
        kernel_initializer='random_normal',
        bias_initializer='zeros'
    )
)  # output layer

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',  # stochastic gradient descent
    metrics=['accuracy']
)

model.fit(trX, trY, epochs=NB_EPOCHS, batch_size=BATCH_SIZE)


score = model.evaluate(teX, teY)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
