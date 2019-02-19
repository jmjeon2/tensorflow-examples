
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# In[4]:


tf.set_random_seed(777)
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


# In[5]:


learning_rate = 0.01
batch_size = 100
training_epoch = 5


# In[7]:


#레이어를 하나만 사용해서 진행해보기
X=tf.placeholder(tf.float32, [None,784])

Encoder = tf.layers.dense(X, units=256, activation = tf.nn.sigmoid)
Decoder = tf.layers.dense(Encoder, units=784, activation = tf.nn.sigmoid)

cost = tf.reduce_mean(tf.pow(X-Decoder, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


# In[8]:


sess=tf.Session()
sess.run(tf.global_variables_initializer())

print('Training Start')

total_batch=int(mnist.train.num_examples/batch_size)
for epoch in range(training_epoch):
    total_cost=0
    for i in range(total_batch):
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        c,_=sess.run([cost,optimizer], feed_dict={X:batch_xs})
        
        total_cost+=c
    print("Epoch:",'{:d}'.format(epoch+1), "Cost:",'%5f'%(total_cost/total_batch))
    
print('Training Finished')
        
        


# In[9]:


sample_size = 10
samples = sess.run(Decoder, feed_dict = {X:mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize = (sample_size,2))
for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28,28)))
    ax[1][i].imshow(np.reshape(samples[i],(28,28)))

