import tensorflow as tf
import numpy as np
import csv
import dataHandler
slim = tf.contrib.slim

#Define the batch
def get_batches(x, y, n_batches=10):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches * batch_size, batch_size):
        # If we're not on the last batch, grab data with size batch_size
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
            # On the last batch, grab the rest of the data
        else:
            X, Y = x[ii:], y[ii:]  # Grab all the remaining data
        # I love generators
        yield X, Y

# load files
with open('labels') as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader]).squeeze()
    labels = labels[:-1]
    print('loaded labels', labels.shape)


with open('codes') as f:
    codes = np.fromfile(f, dtype=np.float32)
    codes = codes.reshape((len(labels), -1))
    print('loaded codes', codes.shape)
# -------------------------------------------------------------
# split data
from sklearn.model_selection import train_test_split
labels, classes = dataHandler.one_hot_encode(labels)
X_train, X_test, y_train, y_test = train_test_split(codes, labels, test_size=0.20, random_state=42)
X_train = X_train.astype('float32')

X_train = X_train.reshape((-1,8,8,1536))
X_test = X_test.reshape((-1,8,8,1536))

print('X shape', X_train.shape)
print('y shape', y_train.shape)
# -------------------------------------------------------------

inputs_ = tf.placeholder(tf.float32, shape=[None, 8, 8, 1536])
labels_ = tf.placeholder(tf.int64, shape=[None, labels.shape[1]])

# ----
net = slim.avg_pool2d(inputs_, inputs_.get_shape()[1:3], padding='VALID')
fc3 = slim.fully_connected(net, 400, activation_fn=None)
fc2 = slim.fully_connected(fc3, 100, activation_fn=None)
fc = slim.fully_connected(fc2, 25, activation_fn=None)
logits = tf.contrib.layers.fully_connected(fc, labels.shape[1], activation_fn=None)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer().minimize(cost)

predicted = tf.nn.softmax(logits)
predicted = tf.squeeze(predicted)
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

epochs = 100
batch_size = int(len(X_train) / epochs)
print(batch_size)
iteration = 0
saver = tf.train.Saver()

# Train network and save it in GOT.ckpt
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        for x, y in get_batches(X_train, y_train):
            feed = {inputs_: x, labels_: y}
            loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            print("Epoch: {}/{}".format(e + 1, epochs), "Iteration: {}".format(iteration), "Training loss: {:.5f}".format(loss))
            iteration += 1

            if iteration % 5 == 0:
                feed = {inputs_: X_test,labels_: y_test}
                val_acc = sess.run(accuracy, feed_dict=feed)
                print("Epoch: {}/{}".format(e + 1, epochs), "Iteration: {}".format(iteration), "Validation Acc: {:.4f}".format(val_acc))

    saver.save(sess, "checkpoints/GOT.ckpt")

# Restore the network and test the accuracy
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))

    feed = {inputs_: X_test, labels_: y_test}
    test_acc = sess.run(accuracy, feed_dict=feed)
    print("Test accuracy: {:.4f}".format(test_acc))
