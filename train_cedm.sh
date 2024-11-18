# Launch with "bash train_cedm.sh"
######################################################################################################
export CUDA_VISIBLE_DEVICES="0,1,2,3" 
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
port=$(( 47000 + $RANDOM % 1000 ))
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate cedm

######################################################################################################
################################### Experiment setup #################################################
######################################################################################################

# Specify the following parameters
free_gpu_mem=25000  # ~25GB are needed
dataset=cifar100  # choose from cifar10, cifar100, ffhq and afhqv2
duration=100  # M_img in paper
clusters=200  # C in paper

# Optionally specify the following paths
outdir=$"./experiments/cedm/${dataset}/temi-${clusters}-clusters-${duration}M"  # Save directory for experiment
pseudo_label_path=$"./cluster_ids/${dataset}/TEMI-dino_vitb16/clusters_${clusters}/beta-0.6/cluster_ids.pt"  # Pseudo-labels
freq=$"./cluster_ids/${dataset}/TEMI-dino_vitb16/clusters_${clusters}/beta-0.6/freq.pt"  # Multinomial weights for pseudo-labels

######################################################################################################
################################### Default hyperparameters ##########################################
######################################################################################################

# Training and sampling hyperparameters that depend on the dataset
if [ $dataset = cifar10 ];
then
    lr=1e-3       # Learning rate
    batch=1024    # Batch size for training and sampling
    cres=2,2,2    # UNet channel multiplier per level
    dropout=0.13  # Dropout probability
    augment=0.12  # Augmentation probability
    num_steps=18  # Inference steps. NFE = 2*num_steps - 1
    dataset_path="./datasets/cifar10-32x32.zip"  # zip file containing the dataset after preprocessing
    fid_ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz  # FID reference file
elif [ $dataset = cifar100 ]; then
    lr=1e-3
    batch=1024
    cres=2,2,2
    dropout=0.13
    augment=0.12
    num_steps=18
    dataset_path="./datasets/cifar-100/cifar100-32x32.zip"
    fid_ref="./misc/cifar100-32x32.npz"  # Produce after preparing dataset with `python fid.py ref ...`
elif [ $dataset = ffhq ]; 
then
    lr=2e-4
    batch=512
    cres=1,2,2,2
    dropout=0.05 
    augment=0.15
    num_steps=40
    dataset_path="./datasets/vision_benchmarks/FFHQ-i/ffhq-64x64.zip"
    fid_ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz
else
    echo "Dataset not supported"
fi

######################################################################################################
################################### Training #########################################################
######################################################################################################

# See train.py for help on the arguments. Ones not listed here use their default value.
while true; do
    gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
    if [ $gpu_memory -gt $free_gpu_mem ]; then
        echo "Launching experiment..."
        torchrun --master_port=$port --nproc_per_node="$nproc_per_node" train.py \
            --outdir "$outdir" \
            --dataset $dataset \
            --cond 1 \
            --duration $duration \
            --cres $cres \
            --lr $lr \
            --batch $batch \
            --dropout $dropout \
            --augment $augment \
            --wandb 0 \
            --tick 2000 \
            --snap 10 \
            --dump 10 \
            --fp16 1 \
            --pseudo_label_path "$pseudo_label_path" \
            --dataset_path $dataset_path ;
            break
    else
        echo "Waiting for more GPU memory...last updated on $(date "+%Y-%m-%d %H:%M:%S")"
        sleep 10m
    fi
done

chmod -R 777 $outdir

######################################################################################################
################################### Sample Generation ################################################
######################################################################################################

# Evaluate the trained model
edm_model=$(ls -t "$outdir"/*.pkl | head -1)
snap=$((duration * 1000))
sample_dir=$"$outdir"$"/samples/baseline-iter-$snap"
echo "Evaluating checkpoint $edm_model"

# See generate.py for details on the available arguments. Ones not listed here use their default value.
CUDA_LAUNCH_BLOCKING=1 torchrun --master_port=$port --nproc_per_node="$nproc_per_node" generate.py \
    --seeds 0-149999 \
    --subdirs \
    --outdir "$sample_dir" \
    --network "$edm_model" \
    --batch $batch \
    --steps $num_steps \
    --sigma_min 2e-3 \
    --sigma_max 80 \
    --rho 7 \
    --freq_path "$freq" ;

######################################################################################################
################################### FID Evaluation ###################################################
######################################################################################################

# Evaluate the FID of the generated samples
for subdir_id in 0000000 0050000 0100000
do
    images_path=$"$sample_dir"/"$subdir_id"
    torchrun --master_port=$port --nproc_per_node="$nproc_per_node" fid.py calc \
        --batch $batch --ref $fid_ref --images $images_path
done
