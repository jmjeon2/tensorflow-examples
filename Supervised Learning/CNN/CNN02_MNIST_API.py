
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


# In[2]:


training_epoch = 10
batch_size = 100
learning_rate = 0.001


# In[3]:


X=tf.placeholder(tf.float32,[None,784])
X_img = tf.reshape(X,[-1,28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])
is_training = tf.placeholder(tf.bool)

L1 = tf.layers.conv2d(inputs = X_img, filters=16, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
L1 = tf.layers.max_pooling2d(inputs=L1, pool_size=[2,2], padding='SAME', strides=2)#김성훈 교수님
#L1 = tf.layers.max_pooling2d(L1, [2,2], [2,2]) #골빈
dropout1 = tf.layers.dropout(L1, rate = 0.7, training=is_training)

L2 = tf.layers.conv2d(dropout1, filters=32, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
L2 = tf.layers.max_pooling2d(L2, [2,2], [2,2])
dropout2 = tf.layers.dropout(L2, rate=0.7, training=is_training)

L3 = tf.contrib.layers.flatten(dropout2)
L3 = tf.layers.dense(L3, 128, activation = tf.nn.relu)
L3 = tf.layers.dropout(L3, 0.5, training=is_training)

model = tf.layers.dense(L3, 10, activation = None) # None을 굳이 안해도되나?

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[4]:


total_batch = int(mnist.train.num_examples/batch_size)
sess= tf.Session()
sess.run(tf.global_variables_initializer())
print('Learning Start')
for epoch in range(training_epoch):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = sess.run([cost, optimizer], feed_dict = {X:batch_xs, Y:batch_ys, is_training:True})
        
        total_cost += c
    print('Epoch:''{:3d}'.format(epoch+1), 'Cost:','%5f'%(total_cost/total_batch))
    
print('Learning Finished!')
        


# In[8]:


is_correct = tf.equal(tf.argmax(model,axis=1), tf.argmax(Y,axis=1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

acc=sess.run(accuracy, feed_dict=
        {X:mnist.test.images[:3000], Y:mnist.test.labels[:3000], is_training:False})
print('Accuracy:','%5f'%acc)


# In[32]:


import random
a=random.randint(0,mnist.test.num_examples-1)
print("실제값:",sess.run(tf.argmax(mnist.test.labels[a:a+1],1)))
print("예측값:",sess.run(tf.argmax(model,1)
                       ,feed_dict={X:mnist.test.images[a:a+1],is_training:False}))

