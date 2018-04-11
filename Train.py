import tensorflow as tf
import numpy as np

pix_size= 100
Slice_count=40

classes= 2
batch_size= 10

# Placeholders for input data (image= 100x100x40, labels=2 (0 or 1))
x = tf.placeholder(tf.float32, shape=[None, pix_size*pix_size*Slice_count]) # [None, 100*100*40]
y_ = tf.placeholder(tf.float32, shape=[None, classes])  # [None, 2]

def CNN_Model(x):
    # Model for the 3D CNN

    input_layer = tf.reshape(x, [-1, pix_size, pix_size, Slice_count, 1])

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
    
     # Convolutional Layer #3 and Pooling Layer #3
    conv3 = tf.layers.conv3d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3, 3],
        strides=[1, 1, 1],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling3d(inputs=conv3, pool_size=[2, 2, 2], strides=2)
    
    # Dense Layer
    pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 7 * 128])
    dense = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Output Layer
    output = tf.layers.dense(inputs=dropout, units=2)
    
    return output

# define the training and the input data:
data= np.load('processed_data_100_40.npy')
TrainData= data[:-100]# all the way up till the last 100
TestData= data[-100:]#last 100 onwards
x= TrainData[0]
y= TrainData[1]


# defining training:


nb_epochs= 100
def TrainCNN_model(x):
    prediction=CNN_Model(x)
    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer= tf.train.AdamOptimizer().minimize(cost)

    with tf.Session as session:
        session.run(tf.global_variables_initializer())
    
        for epoch in range(nb_epochs):
            _,run=session.run(optimizer, feed_dict={x:TrainData[0], y:TrainData[1]})
            epoch_loss+=run
        
            print('Epoch no. {} completed out of {} and loss: {}'.format(epoch,nb_epochs,epoch_loss))
    
        correct_prediction= tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy= tf.reduce_mean(tf.case(correct_prediction,'float'))
        print('Accuracy:',accuracy.eval({x:[i[0] for i in TestData],y:[i[1] for i in TestData]}))
    


TrainCNN_model(x)
    
