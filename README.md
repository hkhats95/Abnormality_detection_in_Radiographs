# Abnormality_detection_in_Radiographs
Creating a deep neural network for detecting abnormality in radiographs obtained from MURA dataset

## Steps to be followed for training the model
1. Download the MURA dataset using [this link.](https://stanfordmlgroup.github.io/competitions/mura/)
2. Run classification_label_creation notebook to create the .csv for classification labels.
3. Run train_classifier_densenet notebook to train the classification model.
4. Run img_embeddings_generator notebook to generate the embeddings for training and validation images which contain the information about the classification.
5. Run train_study_prediction_densenet notebook to train the model to predict the probability of abnormality in radiographs utilizing the information from the classification model. Check **model.png** for the architecture of whole model.
