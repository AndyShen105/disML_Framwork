#!/bin/bash
# $1 is the number of PSs
# $2 is the number of workers
# $3 is the optimizer of model
# $4 is the targted_loss of model
# $5 is the tensorflow port
# $6 is the ml_model
# $7 batch size
# ps.sh run in ssd42

get_ps_conf(){
    ps=""
    for(( i=72; i > 72-$1; i-- ))
    do
        ps="$ps,b1g$i:"$2
    done
        ps=${ps:1}
};

get_worker_conf(){
    worker=""
    for(( i=72-$1; i > 72-$1-$2; i-- ))
    do
        worker="$worker,b1g$i:"$3
    done
    worker=${worker:1}
};

for(( i=0; i<$2; i++ ))
do
{
    echo "0"> /root/code/disML_Framwork/bin/temp$i
}
done

echo "release port occpied!"
/root/code/disML_Framwork/bin/kill_cluster_pid.sh 37 72 $5

get_ps_conf $1 $5
echo $ps
get_worker_conf $1 $2 $5
echo $worker

for(( i=36; i>72-$1-$2; i-- ))
do
{
    if [ $i == 72 ]
    then
        python /root/code/disML_Framwork/disML_Framwork.py $ps $worker --job_name=ps --task_index=0
    else
        ip="b1g"$i
        #ssh ssd$i "source activate tensorflow"
        n=`expr 72 - $1`
        if [ $i -gt $n ]
        then
            index=`expr 72 - $i`
            ssh $ip python /root/code/disML_Framwork/disML_Framwork.py $ps $worker --job_name=ps --task_index=$index
        else
            index=`expr 72 - $1 - $i`
            if [ $index != 0 ]
            then
                sleep 0.5
            fi
            ssh $ip python /root/code/disML_Framwork/disML_Framwork.py $ps $worker --job_name=worker --task_index=$index --Learning_rate=$4 --ML_model=$6 --optimizer=$3 --n_intra_threads=$7 --Batch_size=$8 --n_partitions=$9 --num_Features=${10} --targeted_loss=${11} >> /root/code/$index".temp"
	    echo "worker"$index" complated"
            echo "1"> /root/code/disML_Framwork/bin/temp$index
        fi
    fi
}&
done

while true
do
    flag=0
    for(( i=0; i<$2; i++ ))
    do
    {
        tem=`cat /root/code/disML_Framwork/bin/temp$i`
        flag=`expr $tem + $flag`
    }
    done
    if [ $flag == $2 ]
    then
        /root/code/disML_Framwork/bin/kill_cluster_pid.sh 37 72 $5
        break
    fi
done
rm -f /root/code/disML_Framwork/bin/temp*
rm -f /root/code/*.temp
echo "work done"

