from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt


# Loading the MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


NUM_DIGITS = 784  # 28 x 28 = 784 (taille de l'image d'input)
NUM_HIDDEN = 256
NUM_CLASSES = 10

DISPLAY_IMGS = False
if DISPLAY_IMGS:
    # >>>> Displaying some images and their labels form MNIST dataset
    for i in range(1,10):
        print("Label: " + str(mnist.train.labels[i]))  # label of i-th element of training data
        img = mnist.train.images[i].reshape((28, 28))  # saving in 'img', the reshaped i-th element of the training dataset
        plt.imshow(img, cmap='gray')  # displaying the image
        plt.show()

# >>>> Define input and ground-truth variables
X = tf.placeholder(tf.float32, [None, NUM_DIGITS])  # input data
Y = tf.placeholder(tf.float32, [None, NUM_CLASSES])  # ground-truth


# >>>> Randomly intialize the variables
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


w_h1 = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_h2 = init_weights([NUM_HIDDEN, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, NUM_CLASSES])


# >>>> Define the network model
def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.relu(tf.matmul(X, w_h1))
    h2 = tf.nn.relu(tf.matmul(h1, w_h2))
    return tf.matmul(h2, w_o)


# > Compute the predicted Y_p for an imput vector X
Y_p = model(X, w_h1, w_h2, w_o)

# > Define the cost function and the optimizer
# utiliser softmax permet d'être sur que la somme des probas de sortie (proba
# qu'un élément soit dans la classe) est égale à 1
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_p, labels=Y))
optimization_algorithm = tf.train.GradientDescentOptimizer(0.5).minimize(cost_function)

# >>>> Launch an interactive tensorflow session
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# >>>> For accuracy
# le vecteur de sortie est de la forme [0.1, 0.4, 0.9, ..., 0.0], avec chaque
# indice qui donne la proba de sortie dans l'ensemble des classes 0, 1, 2, ...
# qui sont les classes qu'on doit prédire
# tf.argmax(Y_p,1) donne l
correct_prediction = tf.equal(tf.argmax(Y_p, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# >>>> Train the network
# 1 epoch = on a parcouru le dataset en entier une fois
# besoin de faire un shuffle à chaque fois
# itération = on fait le calcul pour un batch
# ici on a 20000 * 50 / 55000 = 18 epoch
for iteration in range(20000):
    batch = mnist.train.next_batch(50)  # every batch of 50 images
    if iteration % 100 == 0:
        # batch[0] = image, batch[1] = label
        train_accuracy = accuracy.eval(feed_dict={X: batch[0], Y: batch[1]})
        print("iteration: %d, training accuracy: %g" % (iteration, train_accuracy))
        optimization_algorithm.run(feed_dict={X: batch[0], Y: batch[1]})

# >>>> Save the learned model
# > Add ops to save and restore all the variables.
saver = tf.train.Saver()
# > Variables to save
tf.add_to_collection('vars', w_h1)
tf.add_to_collection('vars', w_h2)
tf.add_to_collection('vars', w_o)
# > Save the variables to disk
save_path = saver.save(sess, "./tensorflow_model.ckpt")
print("Model saved in file: %s" % save_path)


# >>>> Restore variables saved in learned model
new_saver = tf.train.import_meta_graph('tensorflow_model.ckpt.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
i = 0
for v in all_vars:
    v_ = sess.run(v)
    if i == 0:
        w_h1 = v_  # restore w_h1
    if i == 1:
        w_h2 = v_  # restore w_h2
    if i == 2:
        w_o = v_  # restore w_o
    i = i + 1
print("Model restored correctly!")

# >>>> Test the trained model
print("\n\nTest accuracy: %g" % accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
