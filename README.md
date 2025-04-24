# Viewpoint-Aware Sampling (VAS)

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

  Coming soon...

## Run experiments
* Run experiments on OpenLORIS-Object dataset
  ```
  bash run_openloris_experiments.sh
  ```
* Run experiments on OpenLORIS-Object dataset
  ```
  bash run_core50_experiments.sh
  ```

Results are saved in pkl files under results/{openloris_sequential|core50_ni_inc}
