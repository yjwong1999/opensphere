# Face Recognition on SurvFace Dataset using OpenSphere

This repo is part of the **IEEE World AI IoT Congress 2023** </br>
Refer [YOLOv5 DeepSort Face](https://github.com/yjwong1999/Yolov5_DeepSort_Face) for the main implementation of the project. </br>
Simulation for Conference Proceedings [https://doi.org/10.1109/AIIoT58121.2023.10174362](https://doi.org/10.1109/AIIoT58121.2023.10174362) </br>
Refer [here](https://www.researchgate.net/publication/371315031_Multi-Camera_Face_Detection_and_Recognition_in_Unconstrained_Environment) for the preprint </br>
(Please note that this is an ongoing project)

## Abstract
Multi-camera face detection and recognition is an Artificial Intelligence (AI) based technology that leverages multiple cameras placed at different locations to detect and recognize human faces in real-world conditions accurately. While face detection and recognition technologies have exhibited high accuracy rates in controlled conditions, recognizing individuals in open environments remains challenging due to factors such as changes in illumination, movement, and occlusions. In this paper, we propose a multi-camera face detection and recognition (MCFDR) pipeline, which consists of three main parts - face detection, face recognition, and tracking. A series of model training is done with the open-source dataset to build a robust pipeline, and finally, the pipeline adopted trained YOLOv5n for face detection model with mAP 0.495, precision value of 0.868, and recall value of 0.781. The system also adopted the SphereFace SFNet20 model with an accuracy of 82.05% and a higher inference rate than SFNet64 for face recognition. These models are then fed into DeepSORT for multi-camera tracking. Our dataset has been applied to the pipeline and shows ideal outcomes with objectives achieved.

## Retrain OpenSphere Model

- Clone the repo
```
git clone https://github.com/yjwong1999/opensphere.git
cd opensphere
```

- Create a new conda environment to train OpenSphere
```
conda deactivate # if you are in other conda environment
conda env create -f environment.yml
conda activate opensphere

# Choose one of the following
bash scripts/dataset_setup_validation_only.sh    # if you only want to train with your custom dataset
OR
bash scripts/dataset_setup.sh                     # if you want to train with VGG Face 2 dataet
OR
bash scripts/dataset_setup_ms1m.sh                # if you want to train with ms1m dataset
```

- Get QMUL-SurvFace dataset
```
cd customize
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13ch6BPaexlKt8gXB_I8aX7p1G3yPm2Bl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13ch6BPaexlKt8gXB_I8aX7p1G3yPm2Bl" -O QMUL-SurvFace.zip && rm -rf /tmp/cookies.txt
unzip QMUL-SurvFace.zip

python3 generate_annot.py --directory QMUL-SurvFace # generate annotation for this dataset

cd ../
```

- Train OpenSphere Model using QMUL-SurvFace dataset
```
# Train SFNet20 using SphereFace loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet20_sphereface.yml
# Train SFNet20 using SphereFaceR loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet20_spherefacer.yml
# Train SFNet20 using SphereFace2 loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet20_sphereface2.yml

# Train SFNet64 using SphereFace loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet64_sphereface.yml
# Train SFNet64 using SphereFaceR loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet64_spherefacer.yml
# Train SFNet64 using SphereFace2 loss function
CUDA_VISIBLE_DEVICES=0 python train.py --config config/train/survface_sfnet64_sphereface2.yml

# NOTE THAT:
# CUDA_VISIBLE_DEVICES=0 means use 1st CUDA device to train
# CUDA_VISIBLE_DEVICES=0,1 means use 1st and 2nd CUDA devices to train
# and so on...
```

- Test OpenSphere Model using QMUL-SurvFace dataset
```
CUDA_VISIBLE_DEVICES=0 python test.py --config config/test/survface.yml --proj_dir project/<dir name>
```

- Convert OpenSphere Model to OpenVINO (for future usage)
```
CUDA_VISIBLE_DEVICES=0 python onnx_exporter.py --config config/test/survface.yml --proj_dir project/<dir name>
```


## Acknowledgement
This work was supported by the Greatech Integration (M) Sdn Bhd with project number 8084-0008.

### Reference Code
1. [OpenSphere Face Recognition](https://github.com/ydwen/opensphere) </br>
2. [YOLOv5 DeepSort Face](https://github.com/yjwong1999/Yolov5_DeepSort_Face) </br>

This repo is mainly based on ref work [1]. </br>
We intend to fork the repo from [1], however we previously have to combine the full source code with ref work [2]. </br>
Hence, we couldn't fork ref work [1] for acknowledgement purpose. </br>
Anyway, thanks a lot to [1] for the open-source + transparent + reproducible code! </br>
The codes allow us to train our own face recognition model. </br>
