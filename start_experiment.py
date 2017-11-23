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

id = ""
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    )

# n_workers, n_ps, n_intra, n_iter, n_partition, optimizer, batch_size, learning_rate
Optimizer=["SGD","Adadelta","Adagrad","Ftrl","Adam","Momentum","RMSProp"]

def wait_finish():
    start_time = time.time()
    dir = os.path.join("temp0")
    while os.path.exists("/root/code/disML_Framwork/bin/temp0"):
        logging.info("The job %s is not finish" % id)
        time.sleep(10)
	if (time.time()-start_time)>18000:
	    os.system("./bin/kill_cluster_pid.sh 36 72 22222")

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
        cmd = "sh ./bin/ps.sh %d %d %s %f 22222 %s %d %d %d 54686452 0.05" % (n_workers,
								  	n_ps,
									optimizer, 
									learning_rate,
									model,
									n_intra,
									batch_size,
									n_partition)
    else:
        cmd = "~/code/disCNN_cifar/bin/ps.sh %d  %s %f 22222 %s %d %d %d " % (n_workers,
                                                                        n_ps,
                                                                        optimizer,
                                                                        learning_rate,
                                                                        model,
                                                                        n_intra,
                                                                        batch_size,
                                                                        n_partition)
    logging.info("run command: %s" % cmd)
    id = model+"_"+str(n_workers)+"_"+str(n_intra)+"_"+optimizer+"_"+str(learning_rate)+"_"+str(batch_size)+"_"+str(n_partition)
    #p = pexpect.spawn(cmd)
    #os.system(cmd+" >> running_log.out")
    return id
    
def run(n_samples, model):
    for i in range(0, n_samples):
	n_workers = np.random.randint(1, 35)
	n_ps = 36-n_workers
	n_intra = np.random.randint(1, 15)
	n_partition = int(np.random.randint(0.1, 2)*n_ps)
	optimizer = Optimizer[np.random.randint(0, 6)]
	if model != "CNN":
	    batch_size = np.random.randint(10, 50)*100
	else:
	    batch_size = np.random.randint(1, 10)*100
	learning_rate = np.random.randint(1, 1000)/10000.0
	threads = []
	t1 = threading.Thread(target=execute,args=(model, n_workers, n_ps, n_intra, n_partition, optimizer, batch_size, learning_rate))
	threads.append(t1)
	#job_id = execute(model, n_workers, n_ps, n_intra, n_partition, optimizer, batch_size, learning_rate)
        t1.setDaemon(True)
        t1.start()
	time.sleep(10)
	wait_finish()
def main():
    run(int(sys.argv[1]), sys.argv[2])
if __name__=="__main__":
    main()


