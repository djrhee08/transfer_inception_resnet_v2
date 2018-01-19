import tensorflow as tf
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import numpy as np
import dataHandler
import utils
import csv
slim = tf.contrib.slim

checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'
data_dir = '../TransferLearning_Inception_V4/GOT/'

height = 299
width = 299
channels = 3

X = tf.placeholder(tf.float32, shape=[None, height, width, channels])
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(X, num_classes=1001,is_training=False)

# Define the place to have the output #
output = end_points["PreLogitsFlatten"]
# ----------------------------------- #
saver = tf.train.Saver()

# a fake input image, you can use your own image
X_test = np.ones((1,height,width,channels))

batch_size = 10
codes_list = []
labels = []
batch = []

codes = None

print('_'*50)
print('Creating transfer codes:')
print('_'*50)

contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

dataHandler.transform_images(data_dir=data_dir, minimum_files_required=500)

# Execute graph
codes = None
batch = []
batch_size = 10

with tf.Session() as sess:
    saver.restore(sess, checkpoint_file)
    """
    output_val = output.eval(feed_dict={X: X_test})
    output_val = output_val.reshape((1,-1))
    print(output_val.shape)
    """

    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):

            # Add images to the current batch
            img = utils.load_image(os.path.join(class_path, file), height, width)
            batch.append(img.reshape((1, height, width, channels)))
            labels.append(each)

            # Running the batch through the network to get the codes
            if ii % batch_size == 0 or ii == len(files):

                # Image batch to pass to VGG network, from high dimension data batch, convert it into 1d
                images = np.concatenate(batch)
                output_val = output.eval(feed_dict={X: images})

                # store the codes in an array
                if codes is None:
                    codes = output_val
                else:
                    codes = np.concatenate((codes, output_val))

                batch = []
                print('{} images processed'.format(ii))
                print(codes.shape)


# -----------------------------------------------------------
# store codes locally
with open('codes', 'w') as f:
    codes.tofile(f)
    print('Transfer codes saved to file "codes" in project directory')

# store labels locally
with open('labels', 'w') as f:
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(labels)
    print('labels saved to file "labels" in project directory')
