#!/bin/bash

prog="/aimldl-cod/external/lanenet-lane-detection/tools/test_lanenet.py"
# echo "prog: $prog"
weights="/aimldl-cod/external/lanenet-lane-detection/model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt"
# image_path="/home/nikhil/Documents/images/"
image_path="/home/jarvis/Documents/test_cases_2/images/"
declare -a images=($(seq 1 1 420))
for img in "${images[@]}"; do
  # echo "img: $img"
  # arch_cfg=$arch-$dataset_cfg_id-$ip-$experiment_id.yml
  image="$image_path$img.jpg"
  #python tools/test_lanenet.py --weights_path ./model/tusimple_lanenet_vgg/tusimple_lanenet_vgg.ckpt  --image_path /aimldl-cod/practice/nikhil/sample-images/*.jpg
  python $prog --weights_path $weights --image_path $image
  echo "$prog --weights_path $weights --image_path $image"
done

