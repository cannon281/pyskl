# PYSKL

PYSKL is a toolbox focusing on action recognition based on **SK**e**L**eton data with **PY**Torch. Various algorithms will be supported for skeleton-based action recognition. We build this project based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2).

The Safer-Activities dataset is trained using the PYSKL toolbox. The instructions to train and test PoseC3D on Safer-Activities is listed below.

## Installation
```shell
git clone https://github.com/cannon281/pyskl
cd pyskl
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```
## Data preparation to train

We preprocess our Safer-Activities dataset to generate skeleton keypoints in pkl format, download the pkl keypoints from this link [PKL-files](https://drive.google.com/file/d/1SHKJDlHRIG36eWcjQRplsNQ9jGWrxbMp).
Place the all pkl files inside "Pkl" folder in the project root directory

## Weight file preparation to test

We provide the final model weight file generated during training, download the pth files from this link [weight-files](https://drive.google.com/drive/folders/1h2_YXDFekU3Uoer3m3YXu4hLaOEo1iyL).
Place the all pth files inside "weights" folder in the project root directory.

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```
The config file paths are as listed below

```shell
# Non-Wheelchair
configs/posec3d/safer_activity_xsub/non-wheelchair.py
# Non-Wheelchair with skip
configs/posec3d/safer_activity_xsub/non-wheelchair-skip.py
# Wheelchair
configs/posec3d/safer_activity_xsub/wheelchair.py
# Wheelchair with skip
configs/posec3d/safer_activity_xsub/wheelchair-skip.py
```


## Supported Algorithms

- [x] [DG-STGCN (Arxiv)](https://arxiv.org/abs/2210.05895) [[MODELZOO](/configs/dgstgcn/README.md)]
- [x] [ST-GCN (AAAI 2018)](https://arxiv.org/abs/1801.07455) [[MODELZOO](/configs/stgcn/README.md)]
- [x] [ST-GCN++ (ACMMM 2022)](https://arxiv.org/abs/2205.09443) [[MODELZOO](/configs/stgcn++/README.md)]
- [x] [PoseConv3D (CVPR 2022 Oral)](https://arxiv.org/abs/2104.13586) [[MODELZOO](/configs/posec3d/README.md)]
- [x] [AAGCN (TIP)](https://arxiv.org/abs/1912.06971) [[MODELZOO](/configs/aagcn/README.md)]
- [x] [MS-G3D (CVPR 2020 Oral)](https://arxiv.org/abs/2003.14111) [[MODELZOO](/configs/msg3d/README.md)]
- [x] [CTR-GCN (ICCV 2021)](https://arxiv.org/abs/2107.12213) [[MODELZOO](/configs/ctrgcn/README.md)]

## Supported Skeleton Datasets

- [x] [NTURGB+D (CVPR 2016)](https://arxiv.org/abs/1604.02808) and [NTURGB+D 120 (TPAMI 2019)](https://arxiv.org/abs/1905.04757)
- [x] [Kinetics 400 (CVPR 2017)](https://arxiv.org/abs/1705.06950)
- [x] [UCF101 (ArXiv 2012)](https://arxiv.org/pdf/1212.0402.pdf)
- [x] [HMDB51 (ICCV 2021)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6126543)
- [x] [FineGYM (CVPR 2020)](https://arxiv.org/abs/2004.06704)
- [x] [Diving48 (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf)


## Demo

Check [demo.md](/demo/demo.md).


## Citation

If you use PYSKL in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry and the BibTex entry corresponding to the specific algorithm you used.

```BibTeX
@inproceedings{duan2022pyskl,
  title={Pyskl: Towards good practices for skeleton action recognition},
  author={Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={7351--7354},
  year={2022}
}
```

## Contributing

PYSKL is an OpenSource Project under the Apache2 license. Any contribution from the community to improve PYSKL is appreciated. For **significant** contributions (like supporting a novel & important task), a corresponding part will be added to our updated tech report, while the contributor will also be added to the author list.

Any user can open a PR to contribute to PYSKL. The PR will be reviewed before being merged into the master branch. If you want to open a **large** PR in PYSKL, you are recommended to first reach me (via my email dhd.efz@gmail.com) to discuss the design, which helps to save large amounts of time in the reviewing stage.

## Contact

For any questions, feel free to contact: dhd.efz@gmail.com
