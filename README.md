# Traffic Sign Recognition (in progress)
Part of my bachelor thesis, traffic sign image recognition and classification neural network model. 

## Dataset
- 26640 images
- 43 classes
- Data is grouped in tracks of 30
- Source: [German Traffic Signs Image Dataset](https://benchmark.ini.rub.de/gtsrb_dataset.html#Downloads)

## Tech Stack
- Python
- Scikit-learn
- Matplotlib / Seaborn
- Pytorch Lightning

## Model
- Efficientnet B0
- 6 layers unfrozen
- Optimizer: Adam
- Loss function: Cross Entropy
- All experiments in model_configs.py

## Training
- 100 epochs
- Early stop on validation accuracy with patience 15
- Validation on every epoch
- Saving top 1 checkpoint

## Model Selection
- 3 iterations of cross validation

## Evaluation
- Accuracy: 98%
- Precision: 99%
- Recall: 98%
- F1-score: 98%
