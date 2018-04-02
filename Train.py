import tensorflow as tf
import numpy as np

pix_size= 100
Slice_count=40

classes= 2
batch_size= 10

# Placeholders for input data (image= 100x100x40, labels=2 (0 or 1))
x = tf.placeholder(tf.float32, shape=[None, pix_size*pix_size*Slice_count]) # [None, 100*100*40]
y_ = tf.placeholder(tf.float32, shape=[None, classes])  # [None, 2]

def CNN_Model(data, labels):
    # Model for the 3D CNN

    input_layer = tf.reshape(data["x"], [-1, pix_size, pix_size, Slice_count, 1])

    # Convolutional Layer #1 and Pooling Layer #1
    conv1 = tf.layers.conv3d(
        inputs=input_layer,
        filters=32,
        kernel_size=[3, 3, 3],
        strides=[1, 1, 1],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling3d(inputs=conv1, pool_size=[2, 2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv3d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3, 3],
        strides=[1, 1, 1],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Output Layer
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)