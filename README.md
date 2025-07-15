# Viewpoint-Aware Sampling (VAS) - Official PyTorch Implementation

This is the official implementation of our paper **"Viewpoint-Aware Sampling for Effective Online Domain Incremental Learning"** at 14th IFAC Symposium on Robotics 2025.

Abstract:

We investigate the problem of online domain continual learning for image classification. Within an extended series of tasks, continual learning encounters the issue of catastrophic forgetting. To mitigate this challenge, one may employ a memory-replay strategy, a technique involving the re-visitation of stored samples in a buffer when new tasks are introduced. However, the memory budget available to autonomous agents, like robots, is typically limited, making the selection of representative examples crucial. One effective strategy for ensuring representativeness is by selecting diverse examples. To this end, we propose a novel on-the-fly sampling policy called Viewpoint-Aware Sampling (VAS), maintaining diversity in the memory buffer by selecting examples from different visual perspectives. We empirically evaluate the effectiveness of VAS across the OpenLORIS-Object and the CORe50-NI benchmark and find that it consistently outperforms state-of-the-art methods in terms of average accuracy, backward transfer, and forward transfer, while requiring less computational resources.


## Requirements

* Install requirements 
  ```
  conda env create -f environment.yml
  ```
* Download dataset
  * [Download OpenLORIS-Object dataset](https://lifelong-robotic-vision.github.io/dataset/object.html)
  * [Download CORe50 dataset](https://vlomonaco.github.io/core50/)

  Extract and put the dataset in the `dataset` folder.

* Download viewpoint annotation files

  Link: [Google Drive](https://drive.google.com/file/d/1YGT7vtbVvkTwueJnwBqZkJJDnP6jFMjL/view?usp=sharing)

  Extract `viewpoint_annotations` to `dataset` directory

## Training

### Train each method

The files [run_openloris_experiments.sh](run_openloris_experiments.sh) and [run_core50_experiments.sh](run_core50_experiments.sh) contain the scripts to run the experiments for each method with different hyperparameters and random seeds.

Example of running VAS with memory size 256
```
python train.py --mode=viewpoint --stream_env online --memory_size=256 --uncertainty_measure="" \
--mem_manage=task_balanced --vas_balanced_view --rnd_seed 42 --dataset=openloris_sequential \
--exp_name=openloris_sequential --pretrain --cuda_idx=0 --result_dir=results/openloris_sequential_vas
```

### Reproduce the results in the paper

Running the whole experiments conducted in the paper may take several hours.

* Run experiments on OpenLORIS-Object dataset
  ```
  bash run_openloris_experiments.sh
  ```
* Run experiments on CORe-50-NI dataset
  ```
  bash run_core50_experiments.sh
  ```

Results are saved in pkl files under results/{openloris_sequential|core50_ni_inc}
