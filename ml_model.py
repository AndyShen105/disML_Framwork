#coding=utf-8
import tensorflow as tf
import numpy as np

def get_optimizer(optimizer, learning_rate):
    if optimizer == "SGD":
	return tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == "Adadelta":
	return  tf.train.AdadeltaOptimizer(learning_rate)
    elif optimizer == "Adagrad":
	return  tf.train.AdagradOptimizer(learning_rate)
    elif optimizer == "Ftrl":
        return  tf.train.FtrlOptimizer(learning_rate)
    elif optimizer == "Adam":
	return  tf.train.AdamOptimizer(learning_rate)
    elif optimizer == "Momentum":
	return  tf.train.MomentumOptimizer(learning_rate)
    elif optimizer == "RMSProp":
	return  tf.train.RMSProp(learning_rate)

def LogisticRegressionModel(weights, y, num_features, variable_partition_num=100):
    with tf.name_scope('parameter'):
	weight = tf.get_variable("weight", initializer=tf.constant(0.0, shape=[num_features, 2]),
                                 partitioner=tf.fixed_size_partitioner(variable_partition_num))
	alpha = tf.constant([0.001])
	b = tf.Variable(tf.constant(0.1, shape=[2]))
	y_ = tf.sparse_tensor_dense_matmul(weights, weight) + b
    with tf.name_scope('loss'):
	l2_norm = tf.reduce_sum(tf.square(weight))
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
	loss = tf.reduce_mean(cross_entropy)
	loss_with_l2 = tf.add(loss, tf.multiply(alpha, l2_norm))
    return loss,loss_with_l2

def SVMModel_with_linear(x_data, y, num_features):
    # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
    # L2 regularization parameter, alpha
    with tf.name_scope('parameter'):
	weight =  tf.Variable(tf.constant(0.0, shape = [num_features, 1]))
	b = tf.Variable(tf.constant(0.1, shape=[1]))
	y_ = tf.subtract(tf.sparse_tensor_dense_matmul(x_data, weight), b)
	alpha = tf.constant([0.001])
    with tf.name_scope('loss'):
	l2_norm = tf.reduce_sum(tf.square(weight))
	classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(y_, y))))
	loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
    return loss

def SVMModel_with_rfb(x_data, y, num_features, batch_size):

    weight =  tf.Variable(tf.constant(0.0, shape = [num_features, 1]))
    # Gaussian (RBF) kernel
    gamma = tf.constant(-25.0)
    b = tf.Variable(tf.constant(0.1, shape=[1,batch_size]))
    x_data_hat = tf.transpose(x_data)
    sq_dists = tf.multiply(2., tf.matmul(x_data, x_data, a_is_sparse=True, b_is_sparse=True, transpose_b=True))
    my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    # Compute SVM Model
    first_term = tf.reduce_sum(b)
    b_vec_cross = tf.matmul(tf.transpose(b), b)
    y_target_cross = tf.matmul(y, tf.transpose(y))
    second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
    loss = tf.negative(tf.subtract(first_term, second_term))

    return loss
