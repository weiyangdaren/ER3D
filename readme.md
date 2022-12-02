# AR3D: An Accurate and Real-time 3D Object Detection Framework for Autonomous Driving
This is the official repository for ''AR3D: An Accurate and Real-time 3D Object Detection Framework for Autonomous Driving''

Due to the double-blind principle, this repository including the video 
is published as an anonymous resource.


## Overview
<div align="center">
    <img align="center" src="docs/overview.jpg" alt="drawing" width="621"/>
    <p> <b>System Overview</b> </p>
</div>

## Demo video
[[Video](https://www.youtube.com/watch?v=vVc9HqoUgc4)]

## How to run

### Environment
We provide the optimized Dynamic Library of Semi-Global Matching (SGM) of our implementation. To run it properly, installing the environments with the same version as ours would be necessary.
- GPU: TITAN RTX 
- opencv=3.4.12
- cuda=10.2
- cudnn=7.6.5

### Preparation
```shell
conda install pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=10.2
pip install spconv-cu102
cd utils/iou3d_nms
python setup.py develop
```

### Run test
```shell
python test.py
```
## Acknowledgement
- [**OpenPCDet**](https://github.com/open-mmlab/OpenPCDet)
