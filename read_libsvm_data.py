#coding=utf-8
import tensorflow as tf
import numpy as np

def read_batch(sess, train_data, batch_size):
    label_list = []
    ids = []
    sp_indices = []
    weight_list = []
    label_list_one_hot = []
    for i in xrange(0, batch_size):
        try:
            line = sess.run(train_data)
        except tf.errors.OutOfRangeError as e:
            return np.reshape(label_list, (i, 1)), ids, sp_indices, weight_list, i
        label, one_hot_label,indices, values = parse_line_for_batch_for_libsvm(line)
        label_list.append(label)
	label_list_one_hot.append(one_hot_label)
        ids += indices
	#sp_indices = np.array([[i, index] for index in indices])
	for index in indices:
            sp_indices.append([i, index])
        weight_list += values
	
    return np.reshape(label_list_one_hot, (batch_size , 2)), np.reshape(label_list, (batch_size , 1)), ids, sp_indices, weight_list, batch_size
    #lablelist,id_list,

def parse_line_for_batch_for_libsvm(line):
    value = line.value.split(" ")
    label = []
    one_hot_label = []
    if value[0]=="+1":
	one_hot_label = [1, 0]
    else:
	one_hot_label = [0, 1]
    label = value[0]
    indices = []
    values = []
    for item in value[1:]:
        [index, value] = item.split(':')
	#if index start with 1, index = int(index)-1
	#else index=int(index)
        index = int(index)-1
        value = float(value)
        indices.append(index)
        values.append(value)
    return label, one_hot_label, indices, values

def parse_line_for_batch_for_libsvm2(line):
    value = line.value.split(" ")
    label = []
    label = value[0]
    indices = []
    values = []
    for item in value[1:]:
        [index, value] = item.split(':')
	#if index start with 1, index = int(index)-1
	#else index=int(index)
        index = int(index)-1
        value = float(value)
        indices.append(index)
        values.append(value)
    return label, indices, values
# label:label fo data
# indices: the list of index of featue
# indices: the value of each features

def main():
    trainset_files=["/home/andy_shen/code/test/Day0.svm"]
    print (trainset_files)
    train_filename_queue = tf.train.string_input_producer(trainset_files)
    train_reader = tf.TextLineReader()
    train_data_line=train_reader.read(train_filename_queue)
    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess , coord=coord)
    try:
	for i in range(1 ,2 ):
	    label, indices, sparse_indices, weight_list, read_count = read_batch(sess, train_data_line, 10)
            print (label)
    except tf.errors.OutOfRangeError:
	print 'Done training -- epoch limit reached'
    finally:
	coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == '__main__':
    main()

