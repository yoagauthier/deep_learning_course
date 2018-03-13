from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# # Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

BATCH_SIZE = 50
NUM_DIGITS = 784
NUM_HIDDEN = 256
NUM_CLASSES = 10
NB_EPOCHS = 18
DISPLAY_IMGS = False

if DISPLAY_IMGS:
    for i in range(1, 10):
        print("Label: " + str(y_train[i]))  # label of i-th element of training data
        img = x_train[i].reshape((28, 28))  # saving in 'img', the reshaped i-th element of the training dataset
        plt.imshow(img, cmap='gray')  # displaying the image
        plt.show()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

model = Sequential()

model.add(Dense(NUM_HIDDEN, input_shape=(NUM_DIGITS,)))  # input
model.add(Activation('relu'))
model.add(Dense(NUM_HIDDEN, kernel_initializer='random_normal'))  # hidden
model.add(Activation('relu'))
model.add(Dense(NUM_CLASSES, kernel_initializer='random_normal'))  # output
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='sgd',  # stochastic gradient descent
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=NB_EPOCHS, batch_size=BATCH_SIZE)

# Save the learned model
model.save('my_model.h5')

# Restore learned model saved
model = load_model('my_model.h5')

# Test the trained model
score = model.evaluate(x_test, y_test)
print('Test loss', score[0])
print('Test accuracy', score[1])
