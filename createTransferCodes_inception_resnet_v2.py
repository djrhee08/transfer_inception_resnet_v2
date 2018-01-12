import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import numpy as np
slim = tf.contrib.slim

checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'

height = 299
width = 299
channels = 3

X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(X, num_classes=1001,is_training=False)

# Define the place to have the output
output0 = end_points["PrePool"]
output1 = end_points["PreLogitsFlatten"]
output2 = end_points["Logits"]
output3 = end_points["Predictions"]
saver = tf.train.Saver()

# a fake input image, you can use your own image
X_test = np.ones((1,height,width,channels))

# Execute graph
with tf.Session() as sess:
    saver.restore(sess, checkpoint_file)
    output0_val = output0.eval(feed_dict={X: X_test})
    output1_val = output1.eval(feed_dict={X: X_test})
    output2_val = output2.eval(feed_dict={X: X_test})
    output3_val = output3.eval(feed_dict={X: X_test})
    print(output0_val.shape)
    print(output1_val.shape)
    print(output2_val.shape)
    print(output3_val.shape)

