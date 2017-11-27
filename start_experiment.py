#!/usr/bin/python
# -*- coding: UTF-8 -*-
import threading
import os
import time
import pexpect
import re
import time
import numpy as np
import logging
import sys


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )

# n_workers, n_ps, n_intra, n_iter, n_partition, optimizer, batch_size, learning_rate
Optimizer=["SGD","Adadelta","Adagrad","Ftrl","Adam","Momentum","RMSProp"]

def wait_finish(model, id):
    start_time = time.time()
    if model != "CNN":
		path = "/root/code/disML_Framwork/bin/temp0"
		maxTime = 30000.0
    else:
		path = "/root/code/disCNN_cifar/bin/temp0"
		maxTime = 18000.0
    while os.path.exists(path):
        logging.info("The job %s is not finish" % id)
		logging.info("Running time is %f s" % (time.time()-start_time))
        time.sleep(10)
		if (time.time()-start_time)>maxTime :
	    	os.system("./bin/kill_cluster_pid.sh 36 72 22222")
			break

    logging.info("The job %s is finish !" % id)

def execute(model, n_workers, n_ps, n_intra, n_partition, optimizer, batch_size, learning_rate):
    """
    :param cmd: 
    :return:
    :rtype: str 
    """
    logging.info("run config: model:%s, n_workers:%d, n_intra:%d, n_partition:%d, optimizer:%s, batch_size:%d, learning_rate:%f" % (
	model,        
	n_workers,  
	n_intra, 
	n_partition, 
	optimizer, 
	batch_size, 
	learning_rate
    ))
    if model == "SVM":
        cmd = "./bin/ps.sh %d %d %s %f 22222 %s %d %d %d 3231961 0.07" % (n_workers,
								  	n_ps,
									optimizer, 
									learning_rate,
									model,
									n_intra,
									batch_size,
									n_partition)
    elif model == "LR":
        cmd = "sh ./bin/ps.sh %d %d %s %f 22222 %s %d %d %d 54686452 0.2" % (n_workers,
								  	n_ps,
									optimizer, 
									learning_rate,
									model,
									n_intra,
									batch_size,
									n_partition)
    else:
        cmd = "~/code/disCNN_cifar/bin/ps.sh %d %d %s %f 22222 %s %d %d %d " % (n_workers,
                                                                        n_ps,
                                                                        optimizer,
                                                                        learning_rate,
                                                                        model,
                                                                        n_intra,
                                                                        batch_size,
                                                                        n_partition)
    logging.info("run command: %s" % cmd)
    os.system(cmd+" >> running_log.out")
   
    
def run(n_samples, model):
    for i in range(0, n_samples):
	n_workers = np.random.randint(1, 35)
	n_ps = 36-n_workers
	n_intra = np.random.randint(1, 15)
	n_partition = int(np.random.randint(1, 50)*n_ps/10)
	if n_partition == 0:
	    n_partition=1
	optimizer = Optimizer[np.random.randint(0, 6)]
	if model != "CNN":
	    batch_size = np.random.randint(10, 50)*100
		learning_rate = np.random.randint(1, 10)/10000.0
	else:
	    batch_size = np.random.randint(1, 10)*100
		learning_rate = np.random.randint(1, 10)/100000.0
	threads = []
	id = model+"_"+str(n_workers)+"_"+str(n_intra)+"_"+optimizer+"_"+str(learning_rate)+"_"+str(batch_size)+"_"+str(n_partition)
	t1 = threading.Thread(target=execute,args=(model, n_workers, n_ps, n_intra, n_partition, optimizer, batch_size, learning_rate))
	threads.append(t1)
	#job_id = execute(model, n_workers, n_ps, n_intra, n_partition, optimizer, batch_size, learning_rate)
        t1.setDaemon(True)
        t1.start()
	time.sleep(10)
	wait_finish(model, id)
def main():
    run(int(sys.argv[1]), sys.argv[2])
if __name__=="__main__":
    main()


