#!/usr/bin/python
# -*- coding: UTF-8 -*-
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

def wait_finish(job_id):
    start_time = time.time()
    dir = os.path.join("/root/pstuner/log", job_id)
    while not os.path.exists(dir):
        logging.info("The job %s is not finish" % job_id)
        if (time.time() - start_time) > best_time:
            logging.info("Run too long and break")
            pexpect.run("python client.py --action dump_log %s" % job_id)
            break
        else:
            time.sleep(10)

    logging.info("The job %s is finish !" % job_id)


def execute(model, n_workers, n_intra, n_partition, optimizer, batch_size, learning_rate):
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
        cmd = "./ps.sh %d %s %f 22222 %s %d %d %d 54686452" % (n_workers,   
								optimizer, 
								learning_rate,
								model,
								n_intra,
								batch_size,
								n_partition)
    elif model == "LR":
        cmd = "./ps.sh %d  %s %f 22222 %s %d %d 54686452" % (n_workers,   
								optimizer, 
								learning_rate,
								model,
								n_intra,
								batch_size,
								n_partition)
    else:
        cmd = "./ps.sh %d  %s %f 22222 %s %d %d 54686452" % (n_workers,   
								optimizer, 
								learning_rate,
								model,
								n_intra,
								batch_size,
								n_partition)
    logging.info("run command: %s" % cmd)
    #p = pexpect.spawn(cmd)
    print cmd
    
def run(n_samples, model):
    for i in range(0, n_samples):
	np.random.seed(i)
	n_workers = np.random.randint(1, 35)
	n_intra = np.random.randint(1, 15)
	n_partition = int(np.random.randint(0.1, 2)*(35-n_workers))
	optimizer = Optimizer[np.random.randint(0, 6)]
	batch_size = np.random.randint(10, 50)*100
	learning_rate = np.random.randint(1, 1000)/10000.0
	print learning_rate
	execute(model,n_workers, n_intra, n_partition, optimizer, batch_size, learning_rate)

def main():
    run(.argv[1], argv[2])

if __name__=="__main__":
    main()


