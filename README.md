# 3D-DRES: Detailed 3D Referring Expression Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)

NEWS:🔥3D-DRES is accepted at AAAI 2026 !🔥

Qi Chen, Changli Wu, Jiayi Ji, Yiwei Ma, Liujuan Cao

Framework:

<img src="docs\3D-DRES.png"/>

## Introduction
Current 3D visual grounding tasks only process sentence-level detection or segmentation, which critically fails to leverage the rich compositional contextual reasonings within natural language expressions. To address this challenge, we introduce Detailed 3D Referring Expression Segmentation (3D-DRES), a new task that provides a phrase to 3D instance mapping, aiming at enhancing fine-grained 3D vision-language understanding. To support 3D-DRES, we present DetailRefer, a new dataset comprising 54,432 descriptions spanning 11,054 distinct objects. Unlike previous datasets, DetailRefer implements a pioneering phrase-instance annotation paradigm where each referenced noun phrase is explicitly mapped to its corresponding 3D elements. Additionally, we introduce DetailBase, a purposefully streamlined yet effective baseline architecture that supports dual-mode segmentation at both sentence and phrase levels. Our experimental results demonstrate that models trained on DetailRefer not only excel at phrase-level segmentation but also show surprising improvements on traditional 3D-RES benchmarks. 

## DetailRefer
Download the dataset [here](https://drive.google.com/drive/folders/1QodKFn4X6SFJS4Pr8RxBqQSUCCCVvZVg?usp=drive_link). If you come across any issues within the dataset that we have not yet identified, please feel free to leave a comment, and we will make the necessary corrections.

## DetailBase

Requirements

- Python 3.7 or higher
- Pytorch 1.12
- CUDA 11.3 or higher

The following installation suppose `python=3.8` `pytorch=1.12.1` and `cuda=11.3`.
- Create a conda virtual environment

  ```
  conda create -n detailbase python=3.8
  conda activate detailbase
  ```

- Clone this repository

  ```
  git clone https://github.com/80chen86/3D-DRES.git
  ```

- Install the dependencies

  Install [Pytorch 1.12.1](https://pytorch.org/)

  ```
  pip install spconv-cu113
  pip install https://data.pyg.org/whl/torch-1.12.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl # please check the versions in the website
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).

- Setup, Install ipdn and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd detailbase/lib/
  python setup.py develop
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` folder as follows. You need to download the ['.aggregation.json', '.txt', '_vh_clean_2.0.010000.segs.json', '_vh_clean_2.ply', '_vh_clean_2.labels.ply', '_vh_clean.aggregation.json'] files.

```
3D-DRES
├── data
│   ├── scannetv2
│   │   ├── scans
```

Split and preprocess point cloud data

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
3D-DRES
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── train
│   │   ├── val
```

### DetailRefer dataset
Download [DetailRefer](https://drive.google.com/drive/folders/1QodKFn4X6SFJS4Pr8RxBqQSUCCCVvZVg?usp=drive_link) and put them as follows.
```
3D-DRES
├── data
│   ├── DetailRefer
│   │   ├── DetailRefer_train.json
│   │   ├── DetailRefer_val.json
│   │   ├── DetailRefer_test.json
```

## Pretrained Backbone

Download [SPFormer](https://drive.google.com/drive/folders/1QodKFn4X6SFJS4Pr8RxBqQSUCCCVvZVg?usp=drive_link) pretrained model and move it to backbones.
```
mkdir backbones
mv ${Download_PATH}/sp_unet_backbone.pth backbones/
```

## Training
```
bash scripts/train.sh
```

## Inference
You can download and use our pretrain [checkpoint](https://drive.google.com/drive/folders/1QodKFn4X6SFJS4Pr8RxBqQSUCCCVvZVg?usp=drive_link).
```
bash scripts/test.sh
```

## Citation

If you find this work useful in your research, please cite:

```

```

## Ancknowledgement

Sincerely thanks for [MDIN](https://github.com/sosppxo/MDIN) and [SPFormer](https://github.com/sunjiahao1999/SPFormer) repos. This repo is build upon them.