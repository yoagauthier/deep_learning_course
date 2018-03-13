from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import to_categorical

# # Loading the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

BATCH_SIZE = 50
NUM_DIGITS = 784
NUM_HIDDEN = 256
NUM_CLASSES = 10
NB_EPOCHS = 18
IMG_DIM = (28, 28)

x_train = x_train.reshape(60000, 1, IMG_DIM[0], IMG_DIM[1])
x_test = x_test.reshape(10000, 1, IMG_DIM[0], IMG_DIM[1])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)
model = Sequential()
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='random_normal',
                 use_bias=True,
                 bias_initializer=Constant(0.1),
                 padding='same',
                 input_shape=(1, IMG_DIM[0], IMG_DIM[1])))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(64,
                 kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='random_normal',
                 use_bias=True,
                 bias_initializer=Constant(0.1),
                 padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(1024,
                activation='relu',
                kernel_initializer='random_normal',
                use_bias=True,
                bias_initializer=Constant(0.1)))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES,
                activation='softmax',
                kernel_initializer='random_normal',
                use_bias=True,
                bias_initializer=Constant(0.1)))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(1e-4),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=NB_EPOCHS)

score = model.evaluate(x_test, y_test)
print('Test loss', score[0])
print('Test accuracy', score[1])
