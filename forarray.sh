#! /bin/bash 
 
# To declare static Array  
arr=(prakhar ankit 1 rishabh manish abhinav) 
 
# To print all elements of array 

echo ${arr[@]}        

# for i in $arr
# do 
#   echo "looping...  $i"
# done


# for i in hello 1 * 2 goodbye 
# do
#   echo "Looping ... i is set to $i"
# done


# for i in 1 2 3 4 5
# do
#   echo "Looping ... number $i"
# done

for i in {1..3}
do
  id=$i
  echo "Welcome $id times"
done


# if [ $# -ne 2 ]
# then
#    echo "Usage: execute_command_per_line command filename"
#    exit 1
# fi

# command=$1
# filename=$2

# while read line
# do
#    $command $line
# done < $filename


args=()
for i in "$@"; do
    args+=("$i")
done
echo "${args[@]}"


args=(1,2,4,4,5)
# for ((i=1; i<=6; i++)); do
#    args[i]=${!i}
# done

echo "${args[@]}"