# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:27:04 2018

@author: v_wangxiangqing
"""

#main file

import Download_ECG_signals as DS
import parameter
import process as pr
import model
import tensorflow as tf
import numpy as np

#patient_info = DS.Download(patient_name=parameter.patient_name)

if parameter.cha_flag:
    cha_size = int(sum(parameter.signals_len)-1)
else:
    cha_size = int(sum(parameter.signals_len))

training = tf.placeholder(dtype=tf.bool)
pl_learningrate = tf.placeholder(tf.float32)
Input = tf.placeholder(shape=(None, cha_size, 1), dtype=tf.float32)
Y = tf.placeholder(shape=(None, 2), dtype=tf.float32)

Output = model.model_M_bn_cnn_1_1_filter(Input, parameter.tt_ll, training)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Output, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(pl_learningrate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
    
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(Output), 1), tf.argmax(Y, 1)), tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()

Signals = [pr.seg_original_data(data = pr.load_original_data(patient_name = n, file_path=parameter.file_path), signals_len = parameter.signals_len, cha_flag=parameter.cha_flag) for n in parameter.patient_name]
pool_test_N,pool_test_notN,pool_train_N,pool_train_notN = pr.build_pool_data(data=Signals, model="N_others", test_index=parameter.patient_name[-22:], train_index=parameter.patient_name[:22])

sess.run(init)

z_loss_list = []
z_accuracy_list = []
for i in range(8000):
    lr_of_me = 0.003
    batch_list_x,batch_list_y = pr.data_batch(pool_test_N,pool_test_notN,pool_train_N,pool_train_notN,batch_size = 128, style="train")
    _, l, l_N = sess.run([train_op, loss, accuracy], feed_dict={Input: batch_list_x, 
                                                  Y: batch_list_y, 
                                                  pl_learningrate: lr_of_me, 
                                                  training:True})
    z_loss_list.append(l)
    z_accuracy_list.append(l_N)
    print('Step: %i Loss: %f: ACC: %f' % (i, l, l_N))
    """
    if (i+1)%100000 == 0:
        lr_of_me = lr_of_me*0.1
        saver.save(sess, "save/model.ckpt"+str(i+1))
    """
del i,l,l_N


z_pre_test = []
for i in range(len(pool_test_N)):
    zz_pre = []
    for j in range(128):
        zz_pre.append(pool_test_N[i][1])
    zz_pre = np.array(zz_pre).reshape(128,-1,1)
    z_pre_test.append((sess.run(accuracy, feed_dict={Input: zz_pre, Y: np.array([[1,0] for n in range(len(zz_pre))]).reshape(-1,2), training: False})))
print(sum(z_pre_test)/len(z_pre_test))

pr.error_info(z_pre_test, pool_test_N)
"""
(Counter({105: 952,
          113: 67,
          117: 826,
          121: 325,
          200: 36,
          202: 46,
          210: 41,
          212: 1,
          213: 64,
          219: 6,
          221: 4,
          222: 509,
          228: 43,
          233: 1143}),
 Counter({'N': 4063}))
"""

z_pre_test = []
for i in range(len(pool_test_notN)):
    zz_pre = []
    for j in range(128):
        zz_pre.append(pool_test_notN[i][1])
    zz_pre = np.array(zz_pre).reshape(128,-1,1)
    z_pre_test.append((sess.run(accuracy, feed_dict={Input: zz_pre, Y: np.array([[0,1] for n in range(len(zz_pre))]).reshape(-1,2), training: False})))
print(sum(z_pre_test)/len(z_pre_test))

pr.error_info(z_pre_test, pool_test_notN)
"""
(Counter({100: 33,
          103: 5,
          105: 51,
          111: 507,
          113: 4,
          121: 5,
          200: 70,
          202: 40,
          210: 48,
          212: 143,
          213: 163,
          214: 1994,
          219: 138,
          221: 10,
          222: 445,
          228: 21,
          231: 43,
          232: 498,
          233: 25,
          234: 42}),
 Counter({'A': 688,
          '~': 72,
          'V': 111,
          '|': 9,
          'Q': 2,
          'L': 2469,
          'a': 27,
          '+': 92,
          'F': 152,
          'E': 1,
          'R': 275,
          '"': 27,
          'x': 114,
          'j': 204,
          'J': 42}))
"""

z_pre_test = []
for i in range(len(pool_train_N)):
    zz_pre = []
    for j in range(128):
        zz_pre.append(pool_train_N[i][1])
    zz_pre = np.array(zz_pre).reshape(128,-1,1)
    z_pre_test.append((sess.run(accuracy, feed_dict={Input: zz_pre, Y: np.array([[1,0] for n in range(len(zz_pre))]).reshape(-1,2), training: False})))
print(sum(z_pre_test)/len(z_pre_test))
pr.error_info(z_pre_test, pool_train_N)
"""
(Counter({101: 1,
          106: 2,
          108: 434,
          112: 12,
          114: 3,
          116: 10,
          201: 9,
          203: 514,
          205: 3,
          208: 19,
          209: 9,
          215: 9,
          220: 4,
          223: 9,
          230: 2}),
 Counter({'N': 1040}))
"""

z_pre_test = []
for i in range(len(pool_train_notN)):
    zz_pre = []
    for j in range(128):
        zz_pre.append(pool_train_notN[i][1])
    zz_pre = np.array(zz_pre).reshape(128,-1,1)
    z_pre_test.append((sess.run(accuracy, feed_dict={Input: zz_pre, Y: np.array([[0,1] for n in range(len(zz_pre))]).reshape(-1,2), training: False})))
print(sum(z_pre_test)/len(z_pre_test))
pr.error_info(z_pre_test, pool_train_notN)
"""
(Counter({101: 6,
          106: 24,
          108: 27,
          109: 4,
          112: 5,
          114: 17,
          115: 2,
          116: 4,
          118: 12,
          119: 4,
          124: 17,
          201: 74,
          203: 58,
          205: 15,
          207: 98,
          208: 104,
          209: 108,
          215: 16,
          220: 44,
          223: 96}),
 Counter({'~': 69,
          'A': 255,
          'Q': 4,
          'V': 70,
          '+': 60,
          '|': 23,
          'j': 12,
          'x': 18,
          'F': 100,
          'L': 3,
          'J': 2,
          'R': 52,
          'a': 7,
          '!': 36,
          ']': 2,
          '[': 1,
          'E': 2,
          'S': 2,
          '"': 1,
          'e': 16}))
"""









"""
from collections import Counter
Counter([n[2] for n in patient_information_list])
"""





#Save(patient_name=patient_name, data=patient_info, save_path="signals/")
#result = Read(patient_name, read_path="signals/")