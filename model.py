# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:27:04 2018

@author: v_wangxiangqing
"""

#model file

import tensorflow as tf
from tflearn.layers.conv import conv_1d,max_pool_1d
from tflearn.layers.core import fully_connected, dropout
from tflearn.layers.merge_ops import merge


def M_cnn(Input, num, training, tt_ll):
    branch0 = conv_1d(Input, tt_ll, num, padding='same', activation='linear')
    branch0 = tf.layers.batch_normalization(branch0, training=training)
    branch0 = tf.nn.relu(branch0)
    """
    branch0 = conv_1d(branch0, 32, 1, padding='same', activation='linear')
    branch0 = tf.layers.batch_normalization(branch0, training=training)
    branch0 = tf.nn.relu(branch0)
    """
    return branch0
def model_M_bn_cnn_1_1_filter(Input, tt_ll, training):
    branch0 = M_cnn(Input=Input, num=53, training=training, tt_ll=tt_ll)
    branch1 = M_cnn(Input=Input, num=43, training=training, tt_ll=tt_ll)
    branch2 = M_cnn(Input=Input, num=33, training=training, tt_ll=tt_ll)
    branch3 = M_cnn(Input=Input, num=23, training=training, tt_ll=tt_ll)
    branch4 = M_cnn(Input=Input, num=13, training=training, tt_ll=tt_ll)
    branch5 = M_cnn(Input=Input, num=11, training=training, tt_ll=tt_ll)
    branch6 = M_cnn(Input=Input, num=9, training=training, tt_ll=tt_ll)
    branch7 = M_cnn(Input=Input, num=7, training=training, tt_ll=tt_ll)
    branch8 = M_cnn(Input=Input, num=5, training=training, tt_ll=tt_ll)
    branch9 = M_cnn(Input=Input, num=3, training=training, tt_ll=tt_ll)
    #TensorShape([Dimension(None), Dimension(300), Dimension(32)])
    
    network_out = merge([branch0, branch1, branch2, branch3, branch4, 
                         branch5, branch6, branch7, branch8, branch9], mode='concat', axis=2)
    #TensorShape([Dimension(None), Dimension(300), Dimension(320)])
    #network_feature_1 = fully_connected(network_out, 1280, activation='relu')
    #network_feature = fully_connected(network_feature_1, 128, activation='relu')
    network_feature = dropout(network_out, 0.9)
    Output = fully_connected(network_feature, 2, activation='relu')
    return Output