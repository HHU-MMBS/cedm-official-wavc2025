#!/bin/bash

# To run this: bash eval_single.sh
export CUDA_VISIBLE_DEVICES="0"
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate dgm
echo "Conda initialized and edm activated"

# bash script to evaluate the generated images using the metrics

start_time=$(date +%s.%N)

###########################################################################################################
################################### Evaluation setup ######################################################
###########################################################################################################

# Specify the variables below
dataset="cifar10"  # What dataset to evaluate, choose from cifar10, cifar100 and ffhq
clusters=100  # C in paper
ckp=200000  # M_img * 1000 in paper
duration=$((ckp / 1000))
base_folder=...  # Path to the folder where images to be evaluated are stored
out_name=...  # Name for the output folder

###########################################################################################################
################################### Dataset hyperparameters ###############################################
###########################################################################################################

output_dir=...  # Full output path
load_dir=...  # Path to save/load data embeddings from
if [ "$dataset" = cifar10 ];
then
  original="./datasets/cifar10-32x32.zip"  # Path to the dataset
elif [ "$dataset" = cifar100 ]; then
  original="./datasets/cifar100-32x32.zip"
elif [ "$dataset" = ffhq ]; then
  original="./datasets/ffhq-64x64.zip"
else
    echo "Dataset not supported"
fi

###########################################################################################################
################################### Evaluation ############################################################
###########################################################################################################

snaps=""
for subdir_id in 0000000 0050000 0100000  # Specify the subdirectories to evaluate
do
  concatenated="${base_folder}/${subdir_id}"
  snaps="$snaps $concatenated"
done

python eval_fdd.py \
  --path $original $snaps \
  --model=dinov2 --batch_size=256 \
  --save \
  --load \
  --device="cuda" \
  --load_dir=$load_dir \
  --output_dir=$output_dir \
  --exp_name="subdirs_$subdir_id" \
  --metrics fd;

end_time=$(date +%s.%N)
execution_time=$(echo "($end_time - $start_time) / 60" | bc -l)
echo "Execution time: $execution_time minutes"
