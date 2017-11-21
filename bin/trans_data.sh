#!/bin/bash
#$1 id_start eg:ssd35's id is 35
#$2 id_end
#$3 local file/dir
#$4 target file/dir
echo $1
echo $2
for(( i=$1; i < $2; i++ ))
do
{
    if [ $i -lt 10 ]
    then
        scp  $3 ssd0$i:$4
    else
        scp  $3 ssd$i:$4
    fi
    echo "ssd0"$i" has done"
}
done
    

