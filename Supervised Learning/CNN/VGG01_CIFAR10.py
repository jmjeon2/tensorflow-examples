
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets.cifar10 import load_data
(x_train, y_train), (x_test, y_test) = load_data()


# In[2]:


def next_batch(num, data, labels):
    idx = np.arange(0,len(data))
    np.random.shuffle(idx)
    idx=idx[:num]
    data_shuffle=[data[i] for i in idx]
    labels_shuffle=[labels[i] for i in idx]
    
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


# In[50]:


learning_rate = 0.001
training_epoch=20
batch_size=100
n_classes=10


# In[51]:


X = tf.placeholder(tf.float32,[None,32,32,3])
Y = tf.placeholder(tf.int64,[None,1])
Y_one_hot = tf.one_hot(Y,10) 
Y_one_hot = tf.reshape(Y_one_hot,[-1,10]) #shape맞춰줌

L1 = tf.layers.conv2d(X,64,[3,3],padding='SAME',activation=tf.nn.relu)
L1 = tf.layers.batch_normalization(L1)
L1 = tf.layers.conv2d(L1,64,[3,3],padding='SAME',activation=tf.nn.relu)
L1 = tf.layers.batch_normalization(L1)
L1 = tf.layers.max_pooling2d(L1,[2,2],[2,2],padding='SAME')

L2 = tf.layers.conv2d(L1,128,[3,3],padding='SAME',activation=tf.nn.relu)
L2 = tf.layers.batch_normalization(L2)
L2 = tf.layers.conv2d(L2,128,[3,3],padding='SAME',activation=tf.nn.relu)
L2 = tf.layers.batch_normalization(L2)
L2 = tf.layers.max_pooling2d(L2,[2,2],[2,2],padding='SAME')

L3 = tf.layers.conv2d(L2,256,[3,3],padding='SAME',activation=tf.nn.relu)
L3 = tf.layers.batch_normalization(L3)
L3 = tf.layers.conv2d(L3,256,[3,3],padding='SAME',activation=tf.nn.relu)
L3 = tf.layers.batch_normalization(L3)
L3 = tf.layers.conv2d(L3,256,[3,3],padding='SAME',activation=tf.nn.relu)
L3 = tf.layers.batch_normalization(L3)
L3 = tf.layers.max_pooling2d(L3,[2,2],[2,2],padding='SAME')

L4 = tf.layers.conv2d(L3,512,[3,3],padding='SAME',activation=tf.nn.relu)
L4 = tf.layers.batch_normalization(L4)
L4 = tf.layers.conv2d(L4,512,[3,3],padding='SAME',activation=tf.nn.relu)
L4 = tf.layers.batch_normalization(L4)
L4 = tf.layers.conv2d(L4,512,[3,3],padding='SAME',activation=tf.nn.relu)
L4 = tf.layers.batch_normalization(L4)
L4 = tf.layers.max_pooling2d(L4,[2,2],[2,2],padding='SAME')

L5 = tf.layers.conv2d(L4,512,[3,3],padding='SAME',activation=tf.nn.relu)
L5 = tf.layers.batch_normalization(L5)
L5 = tf.layers.conv2d(L5,512,[3,3],padding='SAME',activation=tf.nn.relu)
L5 = tf.layers.batch_normalization(L5)
L5 = tf.layers.conv2d(L5,512,[3,3],padding='SAME',activation=tf.nn.relu)
L5 = tf.layers.batch_normalization(L5)
L5 = tf.layers.max_pooling2d(L5,[2,2],[2,2],padding='SAME')

L6 = tf.contrib.layers.flatten(L5)#version 다름
L6 = tf.layers.dense(L6,512,activation=tf.nn.relu)
#L6 = tf.layers.dense(L6,512,activation=tf.nn.relu)

logits = tf.layers.dense(L6,10,activation=None)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

is_correct=tf.equal(tf.argmax(logits,1),tf.argmax(Y_one_hot,1))# 2번째 항tf.argmax(Y_one_hot,1)
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))


# In[52]:


sess=tf.Session()
sess.run(tf.global_variables_initializer())
total_batch = int(x_train.shape[0]/batch_size)

print('Learning Start')
for epoch in range(training_epoch):
    c_avg = 0
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(batch_size,x_train,y_train)
        c,_=sess.run([cost,optimizer],feed_dict={X:batch_xs,Y:batch_ys})
        c_avg+=c/total_batch
    
    x_s,y_s=next_batch(1000,x_test,y_test)
    print('Epoch: {:03d}'.format(epoch+1),'Cost:%.9f'%c_avg)
    #print('Acc:{:.5f}'.format(sess.run(accuracy,feed_dict={X:x_s,Y:y_s})))
    
print('Learning Finished')    


# In[53]:


#전체 테스트셋에서의 정확도 OOM때문에 분할계산 후 평균
acc_avg=0
test_size = x_test.shape[0]
total_batch = int(test_size/batch_size)
for i in range(total_batch):
    offset = i*batch_size
    x_s,y_s=x_test[offset:offset+batch_size],y_test[offset:offset+batch_size]
    c=sess.run(accuracy,feed_dict={X:x_s,Y:y_s})
    acc_avg+=c
print(acc_avg/total_batch)


# In[77]:


#랜덤한 수로 하나 뽑음
import random
a=random.randint(0,x_test.shape[0]-1)
print("실제값:",y_test[a:a+1])
print("예측값:",sess.run(tf.argmax(logits,1),feed_dict={X:x_test[a:a+1]}))

