import numpy as np
import os
import tensorflow as tf
import urllib.request
import matplotlib.pyplot as plt

# TensorFlow-Slim aka TF-Slim
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim
# https://github.com/tensorflow/models/tree/master/research/slim

# use slim from contrib
import tensorflow.contrib.slim as slim

# use slim nets from contrib rather than research/slim
from tensorflow.contrib.slim.nets import vgg

# use datasets and preprocess from research/slim as not in contrib
from datasets import imagenet

# Load the mean pixel values and the function that performs the subtraction
# Note the access to protected members of preprocessing/vgg_preprocessing.py !
from preprocessing.vgg_preprocessing import (_mean_image_subtraction, _R_MEAN, _G_MEAN, _B_MEAN)


def discrete_matshow(data, labels_names=[], title=""):
    """Function to nicely print segmentation results with colorbar showing class names"""
    fig, ax = plt.subplots(figsize=(12, 5), dpi=100)

    # get discrete colormap
    cmap = plt.get_cmap('Paired', np.max(data) - np.min(data) + 1)

    # set limits .5 outside true range
    mat = ax.matshow(data,
                     cmap=cmap,
                     vmin=np.min(data) - .5,
                     vmax=np.max(data) + .5)

    # tell the colorbar to tick at integers
    cbar = fig.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))

    # The names to be printed aside the colorbar
    if labels_names:
        cbar.ax.set_yticklabels(labels_names)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.subplots_adjust(right=0.5)
    plt.show()


with tf.Graph().as_default():
    image_string = tf.read_file('test_image.jpg')
    image = tf.image.decode_jpeg(image_string, channels=3)

    # Convert image to float32 before subtracting the
    # mean pixel value
    image_float = tf.to_float(image, name='ToFloat')

    # Subtract the mean pixel value from each pixel
    processed_image = _mean_image_subtraction(image_float, [_R_MEAN, _G_MEAN, _B_MEAN])

    input_image = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(vgg.vgg_arg_scope()):
        # spatial_squeeze option enables to use network in a fully
        # convolutional manner
        logits, _ = vgg.vgg_16(input_image,
                               num_classes=1000,
                               is_training=False,
                               spatial_squeeze=False)

    # For each pixel we get predictions for each class
    # out of 1000. We need to pick the one with the highest
    # probability. To be more precise, these are not probabilities,
    # because we didn't apply softmax. But if we pick a class
    # with the highest value it will be equivalent to picking
    # the highest value after applying softmax
    pred = tf.argmax(logits, dimension=3)

    checkpoints_dir = 'slim_pretrained'
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
        slim.get_model_variables('vgg_16'))

    with tf.Session() as sess:
        init_fn(sess)
        segmentation, np_image = sess.run([pred, image])

# Remove the first empty dimension
segmentation = np.squeeze(segmentation)

# Let's get unique predicted classes (from 0 to 1000) and
# re-label the original predictions so that classes are
# numerated starting from zero
unique_classes, relabeled_image = np.unique(segmentation, return_inverse=True)

segmentation_size = segmentation.shape

relabeled_image = relabeled_image.reshape(segmentation_size)

labels_names = []
names = imagenet.create_readable_names_for_imagenet_labels()

for index, current_class_number in enumerate(unique_classes):
    labels_names.append(str(index) + ' ' + names[current_class_number + 1])

# Show the image
plt.figure()
plt.imshow(np_image.astype(np.uint8))
plt.suptitle("Input Image", fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

# display the segmentation
discrete_matshow(data=relabeled_image, labels_names=labels_names, title="Segmentation")
