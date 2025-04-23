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


Download the dataset and extract the valid, test, and train folders to the working directory. 

Use augment.py to increase the contrast and brighntess of the images. Change the train directory in the data.yaml file to the new augmented directory.

```bash
python augment.py --input-img train/images \
                  --input-label train/labels \
                  --output-img train_aug/images \
                  --output-label train_aug/labels \
                  --contrast 1.2 \
                  --brightness 30
```

To run the hyperparameter tuning script:
```bash
python test.py --tune
```


After enter the best hyperparameters into test.py

To run training script:
```bash
python test.py
```
