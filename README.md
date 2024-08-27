# Interpretability project
#### Author: Ângelo Morgado

![](/EvaluationDataset/final_results/image_interpretation.png)

This project was made in the ambit of the final project of the Computer Science degree in Universidade da Beira Interior
The objective of this project is to receive images, interpret them, and generate a description using segemntation and crowd pose estimation models.

## Setup

In order to setup this project, two key steps are fundamental, firstly, download the pre-trained weights for both Mask-RCNN and AlphaPose models, 
The duc_se.pth can be found in this [Google Drive](https://drive.google.com/open?id=1OPORTWB2cwd5YTVBX-NE8fsauZJWsrtW), and then place it in **alphapose_utils/models/sppe/**.
The yolov3-spp.weights can be found in this [Google Drive](https://drive.google.com/open?id=1D47msNOOiJKvPOXlnpyzdKA3k6E97NTC), and then place it in **alphapose_utils/models/yolo/**.

Run **train.py** to train the Mask-RCNN model.

To properly run this project, add the evaluation dataset to the EvaluationDataset, by adding the images to the **EvaluationDataset/seg_input/** folder and, to evaluate the dataset place the ground truth masks in the **EvaluationDataset/seg_true_masks/** folder and the ground true interpretations in the **EvaluationDataset/prediction_ground_truth/** folder.

## Training

To train the segmentation model simply run the **train.py** script. To fine-tune the training, adjust the parameters.

## Run the project

To run the project run the **runProject.sh** script or simply run each script individually.

