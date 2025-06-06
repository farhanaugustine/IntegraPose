[![DOI](https://zenodo.org/badge/988759361.svg)](https://doi.org/10.5281/zenodo.15565090)
# IntegraPose - Simultaneous Behavior Classification and Pose Estimation
<div align="center">
<img src="https://github.com/user-attachments/assets/2aec1a90-552d-42e6-8ae4-91ae322a6f1b" alt="IntegraPose" width="50%"/>
</div>

This repository demonstrates multi-task learning for combined object detection and pose-estimation using YOLO-Pose models. IntegraPose enables robust, simultaneous classification of complex behaviors and precise keypoint tracking from a single model.
IntegraPose is a comprehensive suite of tools designed to streamline the entire workflow of behavioral analysis, from initial video annotation to advanced, post-hoc analysis of model outputs. It consists of two primary GUI-based applications:

1. Behavior Annotation & Model Training GUI: A tool to facilitate the creation of training data, configuration of YOLO models, and launching the training process.

2. IntegraPose EDA (Exploratory Data Analysis) Tool: A powerful application to analyze, cluster, and visualize the rich datasets generated by your trained IntegraPose models.

# 🚧 Project Status 🚧
This repository is currently under active development. The tools are functional, but APIs may change, and new features will be added. Feedback and contributions are welcome!

# ✨ Key Features
1. Simultaneous Tracking & Classification: A single model predicts bounding boxes, keypoint locations, and behavioral class.

2. GUI-Driven Workflow: Easy-to-use graphical interfaces for annotation, training, and analysis, minimizing the need for complex command-line interaction.

3. Integrated Annotation Tool: Interactively label keypoints and assign behavioral classes to subjects in your images.

# 🛠️ Prerequisites
Before you begin, ensure you have the following installed:

* Python: Version >3.10 (tested with 3.11).

* Conda: Recommended for managing dependencies.

Core Libraries:

```
pip install ultralytics pandas numpy pyyaml opencv-python Pillow matplotlib scipy scikit-learn openpyxl XlsxWriter
```

🚀 Workflow & How to Use
The IntegraPose workflow is broken down into two main stages, each with its own GUI application.

## Stage 1: Annotation and Training (BehaviorAnnotation_ModelTraining)
This stage focuses on creating your custom dataset and training a model.

To get started, run `main_gui_app.py`:
```
python BehaviorAnnotation_ModelTraining/main_gui_app.py
```
1. Setup & Annotation Tab:

* Define the ordered names for your keypoints (e.g., nose,left_ear,right_ear).

* Define the list of behaviors you want to classify, each with a unique numeric ID.

* Set the directories for your source images and where the output annotation files (.txt) will be saved.

* Click "Launch Annotator" to start labeling your images. The annotator provides an OpenCV-based interface for drawing bounding boxes and placing keypoints for each behavior.

2. Model Training Tab:

* Once you have a labeled dataset, use this tab to configure your training run.

* Point to your dataset's .yaml file (which you can generate from the Setup tab).

* Select a pretrained YOLOv8-pose model variant (e.g., yolov8n-pose.pt).

* Adjust hyperparameters for training and data augmentation.

* Click "Start Training" to launch the Ultralytics training process.

3. Inference & Webcam Tabs:

* Use the "Inference" tab to run your newly trained model on videos or images.

* Use the "Webcam Inference" tab to run your model on a live camera feed.

## Stage 2: Exploratory Data Analysis (`IntegraPose-EDA`)
After you have run inference and generated output .txt files, you can use the EDA tool to perform a deep dive into the results.

To get started, run `main_EDA_IntegraPose.py`:
```
python IntegraPose-EDA/main_EDA_IntegraPose.py
```
1. Load Data & Config Tab:

* Load your inference output files (either a single file or a directory of .txt files).

* Load the data.yaml file from your project to map behavior IDs to names.

* Define the keypoint names (must match the order used during training).

* Set preprocessing options, such as coordinate normalization methods.

2. Feature Engineering Tab:

* Define a skeleton by connecting keypoint pairs.

* Generate a list of potential features based on your keypoints and skeleton.

* Select and calculate features to be used in your analysis.
3. Analysis & Clustering Tab:

* Perform deep-dive analyses on specific behaviors (e.g., view feature distributions, correlation heatmaps).

* Select features for clustering and apply PCA for dimensionality reduction.

* Run AHC or K-Means clustering to discover relationships between your labeled behaviors or to find sub-patterns within the data.

4. Video & Cluster Sync Tab:

* Load the original video and visualize the model's outputs frame-by-frame.

* See a scatter plot of your data points, with the point corresponding to the current video frame highlighted.

5. Behavioral Analytics Tab:

* Calculate behavioral bout statistics (duration, frequency, time between bouts) from raw YOLO output files.

# Showcase 🖼️
Here are some examples of IntegraPose in action, classifying distinct behaviors while simultaneously tracking keypoints.
| OpenField| Video Source | Behaviors |
|---|---|---|
|![Github_BehaviorDepot_1](https://github.com/user-attachments/assets/7946b7f0-7941-4126-b1f4-788b9a7029d8)| Video Source: [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT)| Walking, Wall-Rearing/Supported Rearing|
|![Github_BehaviorDepot_2](https://github.com/user-attachments/assets/acddccf2-7838-4131-98d1-7ef79e1d8a0a)| Video Source: [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT) | Walking, Wall-Rearing/Supported Rearing|
|![Github_BehaviorDepot_3](https://github.com/user-attachments/assets/7cce1887-1eac-439b-bb1a-9252c06df2a5)| Video Source: [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT) | Walking, Grooming|
|![Github_DeerMice_4](https://github.com/user-attachments/assets/1a86d84f-e5c9-4aae-95bf-c17c96b00f13)| Video Source: [Temporal_Behavior_Analysis](https://github.com/farhanaugustine/Temporal_Behavior_Analysis)| Exploring/Waling, Wall-Rearing/Supported Rearing|
|![Github_DeerMice_5](https://github.com/user-attachments/assets/8ef72a1a-08a9-4aba-9378-22519ba2b69d)| Video Source: [Temporal_Behavior_Analysis](https://github.com/farhanaugustine/Temporal_Behavior_Analysis) | Wall-Rearing/Supported Rearing, Jump|
|![Github_BehaviorDepot_6](https://github.com/user-attachments/assets/8e86dec0-6539-4b93-9bce-7d45aebb5353)| Video Source: [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT) | Ambulatory/Walking, Object Exploration, Object Mounting |
|![Github_BehaviorDepot_7](https://github.com/user-attachments/assets/e574b720-ce13-4190-88d5-ef2faceeefee)| Video Source: [BehaviorDEPOT](https://github.com/DeNardoLab/BehaviorDEPOT) | Ambulatory/Walking, Object Exploration |
|![Github_C57B_8](https://github.com/user-attachments/assets/bb9f0491-0d9f-4a97-80f9-6bc059b337d1)| Video Source: Self | Ambulatory/Walking, Nose-Poking, Wall-Rearing/Supported Rear |
|![Github_CHKO_9](https://github.com/user-attachments/assets/17b4d4da-fc77-4dff-a04a-000bbfef96a8)| Video Source: Self | Ambulatory/Walking, Wall-Rearing/Supported Rear |




Repo Underdevelopment 🚧🏗️👷🏼




