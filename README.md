# disML_Framwork
distributed LR, SVM on tensorflow
## cluster list
Cluster | Master Node | Slave Node
- | :-: | -: 
b1g1 | b1g36 | b1g1-b1g35 
b1g37 | b1g72 | b1g37-b1g71
## how to start
```
cd /root/code/disML_Framwork
python start_experiment.py {n_experiments_samples} {ml_job:
SVM, CNN, LR}.
 Eg: python start_experiment.py 10 SVM
```
## how to stop task
```
ssh master_node
./code/disML_Framework/bin/kill_cluster_pid.sh 1 36
```
## how to update code
```
ssh master_node
./code/disML_Framework/bin/trans_data.sh {locsl file/dir} {target dir}
```
## important log file
1. SVM_19_6_Adam_2e-05_3600_22_process.csv
 SVM: ml_job	 
 19: n_workers
 6: n_intra_op Adam:Optimizer
 2e-05: learning rate
 3600:batch size
 22:n_partations

2. SVM_result.csv records the final result of SVM all jobs.

