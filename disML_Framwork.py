#-*-coding:UTF-8-*-
from __future__ import print_function

import tensorflow as tf
import sys
import time
import os
import tensorflow as tf
from ml_model import *
from read_libsvm_data import *

#log config
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['GRPC_VERBOSITY_LEVEL']='DEBUG'
   
# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_string("ML_model", "LR", "ML model'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_float("targeted_loss", 0.05, "targted accuracy of model")
tf.app.flags.DEFINE_string("optimizer", "SGD", "optimizer we adopted")
tf.app.flags.DEFINE_integer("Batch_size", 1000, "Batch size")
tf.app.flags.DEFINE_integer("num_Features", 3231961, "number of features")
tf.app.flags.DEFINE_float("Learning_rate", 0.001, "Learning rate")
tf.app.flags.DEFINE_integer("Epoch", 1, "Epoch")
tf.app.flags.DEFINE_integer("n_intra_threads", 0, "n_intra_threads")
tf.app.flags.DEFINE_integer("n_inter_threads", 0, "n_inter_threads")
FLAGS = tf.app.flags.FLAGS


# config

num_features = FLAGS.num_Features
batch_size = FLAGS.Batch_size
learning_rate = FLAGS.Learning_rate
targeted_loss = FLAGS.targeted_loss
Optimizer = FLAGS.optimizer
Epoch = FLAGS.Epoch
n_intra_threads = FLAGS.n_intra_threads
n_inter_threads = 16 - n_intra_threads


# cluster specification
parameter_servers = sys.argv[1].split(',')
n_PS = len(parameter_servers)
workers = sys.argv[2].split(',')
n_Workers = len(workers)
cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})
server_config = tf.ConfigProto(
                intra_op_parallelism_threads=n_intra_threads,
                inter_op_parallelism_threads=n_inter_threads)

# start a server for a specific task
server = tf.train.Server(
    cluster,
    job_name=FLAGS.job_name,
    task_index=FLAGS.task_index,
    config=server_config)
	
if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":
    # Between-graph replicationee
    with tf.device(tf.train.replica_device_setter(
		worker_device="/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
	#More to come on is_chief...
        is_chief = FLAGS.task_index == 0
	# count the number of global steps
	global_step = tf.get_variable('global_step',[],initializer = tf.constant_initializer(0),trainable = False)

	#inout data
	trainset_files=[("/root/data/url_svmlight/Day%d" % i)+".svm" for i in range(121)]
    	#trainset_files=["/root/data/kdd12.tr"]
	train_filename_queue = tf.train.string_input_producer(trainset_files)
    	train_reader = tf.TextLineReader()
    	train_data_line=train_reader.read(train_filename_queue)
	with tf.name_scope('placeholder'):
	    y_shape = 2
	    if FLAGS.ML_model=="SVM":
		y_shape = 1
	    y = tf.placeholder(tf.float32, [None, y_shape]) 
            sp_indices = tf.placeholder(tf.int64)
            shape = tf.placeholder(tf.int64)
            ids_val = tf.placeholder(tf.int64)
            weights_val = tf.placeholder(tf.float32)
	 	
	with tf.name_scope('parameter'):
    	    x_data = tf.SparseTensor(sp_indices, weights_val, shape)
	    #x_data_SVM = tf.sparse_to_dense(sp_indices, shape, weights_val)
	
	with tf.name_scope('loss'):
    	    SVM_loss = SVMModel_with_linear(x_data, y, num_features)
	    LR_loss,LR_loss_l2 = LogisticRegressionModel(x_data, y, num_features)
	
	# specify optimizer
	with tf.name_scope('train'):
	    grad_op = get_optimizer( Optimizer, learning_rate)
	    LR_train_op = grad_op.minimize(LR_loss_l2, global_step=global_step)
	    SVM_train_op = grad_op.minimize(SVM_loss, global_step=global_step)

	init_op = tf.global_variables_initializer()
    	sess_config = tf.ConfigProto(
        	allow_soft_placement=True,
        	log_device_placement=False,
        	device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    sv = tf.train.Supervisor(is_chief=is_chief,
			     init_op=init_op, 
                             global_step=global_step)

    server_grpc_url = "grpc://" + workers[FLAGS.task_index]
    state = False
    with sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config) as sess:

	begin_time = time.time()
	batch_time = time.time()
	cost = 1000000.0
	step = 0
	while (not sv.should_stop()) and (step <= 20000 ):# and (cost >= targeted_loss) :#n_batches_per_epoch * Epoch
	    label_one_hot,label,indices,sparse_indices,weight_list,read_count = read_batch(sess, train_data_line, batch_size)
	    if FLAGS.ML_model=="LR":	
            	_,cost, step= sess.run([LR_train_op, LR_loss, global_step], feed_dict = { y: label_one_hot,
										sp_indices: sparse_indices,
										shape: [read_count, num_features],
										ids_val: indices,
										weights_val: weight_list})
	    else:
		_,cost, step= sess.run([SVM_train_op, SVM_loss, global_step], feed_dict = { y: label,
										sp_indices: sparse_indices,
										shape: [read_count, num_features],
										ids_val: indices,
										weights_val: weight_list})
	
	    duration = time.time()-batch_time
	    
	    re = str(step+1)+","+str(n_Workers)+","+str(n_intra_threads)+","+str(cost)+","+str(duration)
	    process = open("/root/ex_result/baseline/"+FLAGS.ML_model+"_process.csv","a+")
	    process.write(re+"\r\n")
	    process.close()
	    
	    print("Step: %d," % (step+1),
                            " Loss: %f" % cost,
                            " Bctch_Time: %fs" % float(duration))
	   
            batch_time = time.time()
	final_re = str(step+1)+","+str(n_Workers)+","+str(n_intra_threads)+","+str(cost)+","+str(float(time.time()-begin_time))
	result = open("/root/ex_result/baseline/"+FLAGS.ML_model+"_result.csv","a+")
	result.write(final_re+"\r\n")
	result.close()	
    sv.stop 
