# Rethinking cluster-conditioned diffusion models (C-EDM) <br><sub>Official PyTorch implementation</sub>

This codebase is based on NVIDIA's codebase for the [EDM](https://github.com/NVlabs/edm) Paper: `Elucidating the Design Space of Diffusion-Based Generative Models` authored by
Tero Karras, Miika Aittala, Timo Aila, Samuli Laine. A copy of the licence is provided in `LICENCE.txt`.

---
## Requirements and setup
* We recommend Linux for performance and compatibility reasons.
* 1+ high-end NVIDIA GPU for training and sampling. We have done all testing and development using V100 and A100 GPUs.
* 64-bit Python 3.8 and PyTorch 1.12.0 (or later). See https://pytorch.org for PyTorch installation instructions.
* See [environment.yml](./environment.yml) for Python library dependencies. You can use the following commands with Miniconda3 to 
create and activate your Python environment:
  - `conda env create -f environment.yml -n cedm`
  - `conda activate cedm`

---
## Dataset preprocessing as in Karras et al. ([EDM](https://github.com/NVlabs/edm))

Datasets are stored in uncompressed ZIP archives containing uncompressed PNG files and a metadata file `dataset.json` 
for labels. Custom datasets can be created from a folder containing images; see [`python dataset_tool.py --help`](./docs/dataset-tool-help.txt) 
for more information.

**CIFAR-10:** Download the [CIFAR-10 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:

```.bash
python dataset_tool.py --source=downloads/cifar10/cifar-10-python.tar.gz \
    --dest=datasets/cifar10-32x32.zip
python fid.py ref --data=datasets/cifar10-32x32.zip --dest=fid-refs/cifar10-32x32.npz
```

**CIFAR-100:** Download [CIFAR-100 python version](https://www.cs.toronto.edu/~kriz/cifar.html) and convert to ZIP archive:
```.bash
python dataset_tool.py --source=downloads/cifar100/cifar-100-python.tar.gz \
    --dest=datasets/cifar100-32x32.zip
python fid.py ref --data=datasets/cifar100-32x32.zip --dest=fid-refs/cifar100-32x32.npz
```


**FFHQ:** Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as 1024x1024 images and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/ffhq/images1024x1024 \
    --dest=datasets/ffhq-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/ffhq-64x64.zip --dest=fid-refs/ffhq-64x64.npz
```
---
## Reproduce our results
To facilitate reproduction, all cluster assignments that were used to conduct our experiments are provided locally in 
the repository. To launch training, specify the variables at the top of the `train_cedm.sh` script. 

```bash
# Specify the following parameters
free_gpu_mem=25000  # ~25GB are needed
dataset=cifar100  # choose from cifar10, cifar100, ffhq and afhqv2
duration=100  # M_img in paper
clusters=200  # C in paper
```

Then launch the training script:
```bash
bash train_cedm.sh
```

The script will run commands for training a model, generating 3 sets of 50k samples and evaluating their FID. The results 
are saved in '.csv' files for each set of images. We use the default hyperparameters from Karras et al. for training and
sampling.

| <sub>Dataset</sub>              | <sub>GPUs</sub>   | <sub>TrainingTTime</sub> | <sub>Sampling Time (50k)</sub> |
|:--------------------------------|:------------------|:-------------------------|:-------------------------------|
| <sub>cifar10&#8209;32x32</sub>  | <sub>4xA100</sub> | <sub>~1&nbsp;days</sub>  | <sub>~7&nbsp;min</sub>         |
| <sub>cifar100&#8209;32x32</sub> | <sub>4xA100</sub> | <sub>~1&nbsp;days</sub>  | <sub>~7&nbsp;min</sub>         |
| <sub>FFHQ&#8209;64x64</sub>     | <sub>4xA100</sub> | <sub>~2&nbsp;days</sub>  | <sub>~13&nbsp;min</sub>        |

If you want to train our own models, run the `train.py` script with appropriate parameters. 

---
## Generate samples
To generate samples with an already trained model, you can run the generate.py script with appropriate parameters. Below
is an example command that uses a pre-trained EDM model for CIFAR-10 to generate 50k images. See `generate.py` for more
information on the available options for image generation.

```bash
python generate.py \
    --seeds 0-49999 \
    --outdir $out_path \
    --network https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --batch 256 \
    --steps 18 \
    --sigma_min 2e-3 \
    --sigma_max 80 \
    --rho 7 ;
```

--- 
## Evaluating FID
To compute the FID for a set of 50k generated images, use the `calc` function in `fid.py`. Below is an example command
for CIFAR-10.

```bash
python fid.py calc --batch 128 --ref https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz \
    --images $images_path
```
where `$images_path` needs to be the folder containing the generated images.

---
Training, image generation and FID computation can be distributed across multiple GPUs by replacing `python` with `torchrun --master_port=$port --nproc_per_node=4`,
where `n_proc_per_node` specifies the number of GPUs to be used.
---

## Evaluating FDD (Fréchet distance with DINOv2)
The `dgm_eval` folder contains slightly modified code from the [dgm-eval](https://github.com/layer6ai-labs/dgm-eval) paper `Exposing flaws of generative model evaluation metrics and their unfair treatment of 
diffusion models` that provides code for the computation of many evaluation metrics for diffusion models, including FDD.
In order to run the evaluation, you need to specify the required paths in `eval_fdd.sh` and run the script with:

```bash
bash eval_fdd.sh
```

This will perform the evaluation and save the results in a `.csv` file. 

---