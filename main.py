import os.path
import tensorflow as tf
import helper
import warnings
import time
from distutils.version import LooseVersion
import project_tests as tests

data_dir = './data'
runs_dir = './runs'
image_shape = (160, 576)
keep_prob_value = 0.5
learning_rate_value = 1e-4

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()

    input_tensor = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_tensor, keep_prob, layer3, layer4, layer7


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    stddev = 1e-3
    l2_reg = 1e-4

    k_initializer = tf.truncated_normal_initializer(stddev=stddev)
    k_regularizier = tf.contrib.layers.l2_regularizer(l2_reg)

    one_by_one = (1, 1)
    two_by_two = (2, 2)
    four_by_four = (4, 4)

    sixteen_by_sixteen = (16, 16)
    eight_by_eight = (8, 8)

    conv_1x1_l3 = tf.layers.conv2d(inputs=vgg_layer3_out,
                                   filters=num_classes,
                                   kernel_size=one_by_one,
                                   strides=one_by_one,
                                   kernel_initializer=k_initializer,
                                   kernel_regularizer=k_regularizier,
                                   padding='same')

    conv_1x1_l4 = tf.layers.conv2d(inputs=vgg_layer4_out,
                                   filters=num_classes,
                                   kernel_size=one_by_one,
                                   strides=one_by_one,
                                   kernel_initializer=k_initializer,
                                   kernel_regularizer=k_regularizier,
                                   padding='same')

    conv_1x1_l7 = tf.layers.conv2d(inputs=vgg_layer7_out,
                                   filters=num_classes,
                                   kernel_size=one_by_one,
                                   strides=one_by_one,
                                   kernel_initializer=k_initializer,
                                   kernel_regularizer=k_regularizier,
                                   padding='same')

    deconv_layer1 = tf.layers.conv2d_transpose(inputs=conv_1x1_l7,
                                               filters=num_classes,
                                               kernel_size=four_by_four,
                                               strides=two_by_two,
                                               padding='same',
                                               kernel_regularizer=k_regularizier)

    skip_connection_1 = tf.add(deconv_layer1, conv_1x1_l4)

    deconv_layer2 = tf.layers.conv2d_transpose(inputs=skip_connection_1,
                                               filters=num_classes,
                                               kernel_size=four_by_four,
                                               strides=two_by_two,
                                               padding='same',
                                               kernel_regularizer=k_regularizier)

    skip_connection_2 = tf.add(deconv_layer2, conv_1x1_l3)
    deconv_layer3 = tf.layers.conv2d_transpose(inputs=skip_connection_2,
                                               filters=num_classes,
                                               kernel_size=sixteen_by_sixteen,
                                               strides=eight_by_eight,
                                               padding='same',
                                               kernel_regularizer=k_regularizier)

    return deconv_layer3


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(cross_entropy)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label,
             keep_prob,
             learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function

    for epoch in range(epochs):
        for batch, (images, labels) in enumerate(get_batches_fn(batch_size)):
            feed_dict = {
                input_image: images,
                correct_label: labels,
                keep_prob: keep_prob_value,
                learning_rate: learning_rate_value
            }

            start = time.time()
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)
            took = time.time() - start
            print("epoch: {} batch: {} loss: {:.6} took(s): {}".format(epoch, batch, loss, took))


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    epochs = 10
    batch_size = 2
    vgg_path = os.path.join(data_dir, 'vgg')

    tests.test_for_kitti_dataset(data_dir)

    helper.maybe_download_pretrained_vgg(data_dir)

    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)

    start = time.time()
    with tf.Session() as sess:
        image_input, keep_prob, vgg_layer3, vgg_layer4, vgg_layer7 = load_vgg(sess, vgg_path)

        fcn8_last_layer = layers(vgg_layer3, vgg_layer4, vgg_layer7, num_classes)

        logits, train_op, cross_entropy_loss = optimize(fcn8_last_layer,
                                                        correct_label,
                                                        learning_rate,
                                                        num_classes)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_nn(sess,
                 epochs,
                 batch_size,
                 get_batches_fn,
                 train_op,
                 cross_entropy_loss,
                 image_input,
                 correct_label,
                 keep_prob,
                 learning_rate)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape,
                                      logits, keep_prob, image_input)
        print("Done. Duration: {:.3} seconds".format(time.time() - start))


if __name__ == '__main__':
    run()
