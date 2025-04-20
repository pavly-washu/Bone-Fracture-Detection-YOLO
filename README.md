# Bone-Fracture-Detection-YOLO
Dataset augmentation/preprocessing and YOLO model training to detect bone fractures.

Prerequisites:
pip install ultralytics
pip install opencv-python
pip install optuna

Dataset: https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project/data

Download the dataset and use augment.py to increase the contrast and brighntess of the images. Change the train directory in the data.yaml file to the new augmented directory.

To run the hyperparameter tuning script:
python test.py --tune

After enter the best hyperparameters into test.py

To run training script:
python test.py
