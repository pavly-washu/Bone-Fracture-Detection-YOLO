# Bone-Fracture-Detection-YOLO
Dataset augmentation/preprocessing and YOLO model training to detect bone fractures.

Create a conda enviorment
```bash
conda create -n BONE
```

Activate the conda enviorment
```bash
conda activate BONE
```

Navigate to the working directory

Install Prerequisites:
```bash
pip install -r requirements.txt
```

Dataset: https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project/data


OPTION 1(From Scratch):
Download the dataset and use augment.py to increase the contrast and brighntess of the images. Change the train directory in the data.yaml file to the new augmented directory.

To run the hyperparameter tuning script:
```bash
python test.py --tune
```


After enter the best hyperparameters into test.py

To run training script:
```bash
python test.py
```

OPTION 2 (Use already modified hyperparameters and dataset):
Navigate to working directory and run
```bash
python test.py
```
