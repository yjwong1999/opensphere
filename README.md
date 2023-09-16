## Retrain OpenSphere Model

- Create a new conda environment to train OpenSphere
```
conda deactivate # if you are in other conda environment
conda env create -f environment.yml
conda activate opensphere

# assuming you are in Yolov5_DeepSort_Face/Yolov5_DeepSort_Face
cd opensphere
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
