
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)
mnist = input_data.read_data_sets("../MNIST_data/", one_hot = True)


# In[2]:


learning_rate=0.001
batch_size=100
training_epochs=15


# In[3]:


tf.reset_default_graph()#restart대신 실행시마다 그래프를 리셋시킴
class Model:
    def __init__(self,sess,name):
        self.sess=sess
        self.name=name
        self.build_net()
        
    def build_net(self):
        with tf.variable_scope(self.name):
            self.X = tf.placeholder(tf.float32, [None,784])
            X_img = tf.reshape(self.X ,[-1,28,28,1])
            self.Y = tf.placeholder(tf.float32, [None,10])
            self.is_training = tf.placeholder(tf.bool)
            
            L1=tf.layers.conv2d(X_img,16,[3,3],padding='SAME', activation=tf.nn.relu)
            L1=tf.layers.max_pooling2d(L1,[2,2],[2,2],padding='SAME')
            L1=tf.layers.dropout(L1,0.7,self.is_training)
            
            L2 = tf.layers.conv2d(L1,32,[3,3],padding='SAME',activation=tf.nn.relu)
            L2 = tf.layers.max_pooling2d(L2,[2,2],[2,2], padding='SAME')
            L2 = tf.layers.dropout(L2,0.7,self.is_training)
            
            L3 = tf.contrib.layers.flatten(L2)
            L3 = tf.layers.dense(L3, 128, activation=tf.nn.relu)
            
            self.logits = tf.layers.dense(L3, 10, activation=None)
            
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        
        correct_prediction=tf.equal(tf.argmax(self.logits,1),tf.argmax(self.Y,1))
        self.accuracy  = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    def predict(self,x_test):
        return self.sess.run(tf.argmax(self.logits,1), feed_dict={self.X:x_test,self.is_training:False})
    
    def get_accuracy(self,x_test,y_test):
        return self.sess.run(self.accuracy, feed_dict = {
            self.X:x_test, self.Y:y_test, self.is_training:False})
    
    def train(self, x_data, y_data):
        return self.sess.run([self.cost, self.optimizer], feed_dict={
            self.X:x_data, self.Y:y_data, self.is_training:True})
            
            


# In[4]:


sess=tf.Session()
m1 = Model(sess,'m1')
sess.run(tf.global_variables_initializer())

print('Learning Start')

total_batch=int(mnist.train.num_examples/batch_size)
for epoch in range(training_epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c,_=m1.train(batch_xs,batch_ys)
        avg_cost+=c/total_batch
        
    print('Epoch: %03d'%(epoch+1), 'Cost: %.9f'%(avg_cost))

print('Learning Finished')


# In[5]:


print('Acc:',m1.get_accuracy(mnist.test.images[:5000],mnist.test.labels[:5000]))


# In[6]:


import random
r = random.randint(0,mnist.test.num_examples)
print('예측:',m1.predict(mnist.test.images[r:r+1]))
print('실제:',sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))

