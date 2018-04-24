import tensorflow as tf
import numpy as np

data = np.loadtxt('test.csv',delimiter=',',unpack=True,dtype='float32')

x_data=np.transpose(data[0:2])
y_data=np.transpose(data[2:])

global_step = tf.Variable(0, trainable=True, name='global_step')#초기값0 trainable-카운트

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2,10],-1.,1.),name='W1')
    b1 = tf.Variable(tf.random_normal([10],name='b1'))
    L1=tf.nn.relu(tf.add(tf.matmul(X,W1),b1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10,20],-1.,1.),name='W2')
    b2 = tf.Variable(tf.random_normal([20],name='b2'))
    L2=tf.nn.relu(tf.add(tf.matmul(L1,W2),b2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20,3],-1.,1.),name='W3')
    b3 = tf.Variable(tf.random_normal([3],name='b3'))
    model=tf.add(tf.matmul(L2,W3),b3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=Y,logits=model))

    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost,
                                                                   global_step=global_step)

    tf.summary.scalar('cost',cost) #값이 하나인 텐서를 수집할때 (추적)

sess=tf.Session()
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./BoardDemo01')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path) and 0:#and 0삭제
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all() #앞서 지정한 텐서 수집
writer = tf.summary.FileWriter('./logs', sess.graph) 

for step in range(100):
    sess.run(train_op, feed_dict={X:x_data,Y:y_data})

    print('Step: %d, ' %sess.run(global_step),
          'Cost: %.3f ' %sess.run(cost, feed_dict={X:x_data, Y:y_data}))

    summary = sess.run(merged, feed_dict= {X:x_data, Y:y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))

saver.save(sess, './BoardDemo01/TensorBoard.ckpt', global_step=global_step)
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y,1)
print('예측값:',sess.run(prediction,feed_dict={X:x_data}))
print('실제값:',sess.run(target,feed_dict={Y:y_data}))

is_correct=tf.equal(prediction,target)
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))
print('정확도:',sess.run(accuracy*100,feed_dict={X:x_data,
                                              Y:y_data}))






    
    
        
