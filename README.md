# SiCP: Simultaneous Individual and Cooperative Perception for 3D Object Detection in Connected and Automated Vehicles
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2312.04822.pdf)

SiCP is accepted to IROS 2024.

## SiCP Architecture
![image](https://github.com/DarrenQu/SiCP/blob/main/images/sicp%20architecture.png)

## Complementary Feature Fusion
![image](https://github.com/DarrenQu/SiCP/blob/main/images/complementary%20feature%20fusion.png)

## Features
- Support SOTA detector
    - [x] [PointPillars](https://arxiv.org/abs/1812.05784)
    - [x] [SECOND](https://www.mdpi.com/1424-8220/18/10/3337)
    - [x] [Pixor](https://arxiv.org/abs/1902.06326)
    - [x] [VoxelNet](https://arxiv.org/abs/1711.06396)
          
- Support SOTA cooperative perception models
    - [x] [F-Cooper [SEC2019]](https://arxiv.org/abs/1909.06459)
    - [x] [Attentive Fusion [ICRA2022]](https://arxiv.org/abs/2109.07644)
    - [x] [V2X-ViT [ECCV2022]](https://github.com/DerrickXuNu/v2x-vit)
    - [x] [Where2comm [NeurIPS2022]](https://arxiv.org/abs/2209.12836)
    - [x] [CoBEVT [CoRL2022]](https://arxiv.org/abs/2207.02202)

- Support dataset
    - [x] [OPV2V [ICRA2022]](https://mobility-lab.seas.ucla.edu/opv2v/)
    - [ ] [V2V4Real [CVPR2023]](https://arxiv.org/abs/2303.07601) (Will relase the code soon)
 
## Dataset Preparation
- Download the [OPV2V](https://drive.google.com/drive/folders/1dkDeHlwOVbmgXcDazZvO6TFEZ6V_7WUu) and [V2V4Real](https://mobility-lab.seas.ucla.edu/v2v4real/) datasets.
- After downloading the dataset, place the data into the following structure.
```
├── opv2v_data_dumping
│   ├── train
│   │   │── 2021_08_22_22_30_58
│   ├── validate
│   ├── test
```  
## Installation
### 1. download SiCP github to your local folder
```bash
git clone https://github.com/DarrenQu/SiCP.git
cd SiCP
```
### 2. create a conda environment (python >= 3.7)
```bash
conda create -n sicp python=3.7
conda activate sicp
```
### 3. Pytorch Installation (>= 1.12.0 Required)
```bash
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
### 4. spconv 2.x Installation (if you are using CUDA 11.3)
```bash
pip install spconv-cu113
```
### 5. Install other dependencies
```bash
pip install -r requirements.txt
python setup.py develop
```
### 6. Install bbx nms calculation cuda version
```bash
python opencood/utils/setup.py build_ext --inplace
```

## Train the model
To train the model, run the following command.
```bash
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
- `hypes_yaml`: the path of configuration file, e.g. `opencood/hypes_yaml/point_pillar_sicp.yaml`.
- `model_dir`(optional): the path of checkpoint.
-  More explaination refer to [this repo](https://github.com/DerrickXuNu/OpenCOOD).
  
## Test the model
First, ensure that the `validation_dir` parameter in the `config.yaml` file, located in your checkpoint folder, is set to the path of the testing dataset, for example, `opv2v_data_dumping/test`.
```bash
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence]
```
- `model_dir`: the path of saved model.
- `fusion_method`: about the fusion strategy, 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence`: visualize in a video stream.
  
## Acknowledgement
This project is impossible without these excellent codebases [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), [CoAlign](https://github.com/yifanlu0227/CoAlign) and [V2V4Real](https://github.com/ucla-mobility/V2V4Real). 

## Citation
```bibtex
@article{qu2023sicp,
  title={SiCP: Simultaneous Individual and Cooperative Perception for 3D Object Detection in Connected and Automated Vehicles},
  author={Qu, Deyuan and Chen, Qi and Bai, Tianyu and Qin, Andy and Lu, Hongsheng and Fan, Heng and Fu, Song and Yang, Qing},
  journal={arXiv preprint arXiv:2312.04822},
  year={2023}
}
```








