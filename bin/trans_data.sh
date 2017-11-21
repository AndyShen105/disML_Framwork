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
    scp  $3 b1g$i:$4
    echo "b1g"$i" has done"
}
done
    

