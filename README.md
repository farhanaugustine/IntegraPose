[![DOI](https://zenodo.org/badge/988759361.svg)](https://doi.org/10.5281/zenodo.15565090)
* Preprint availabe: [➡️📑](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5334465)

<div align="center">
  <h2 align="center">
  <em>Simultaneous 2D Pose-Estimation and Behavior Classification</em>
  <img src="https://github.com/user-attachments/assets/5bef79e4-ef99-4ca3-928a-4af75707e1a0" width="500"/>
  </h1>
</div>
</div>

This repository demonstrates multi-task learning for combined object detection and pose-estimation using YOLO-Pose models. IntegraPose enables robust, simultaneous classification of complex behaviors and precise keypoint tracking from a single model. IntegraPose is a comprehensive suite of tools designed to streamline the entire workflow of behavioral analysis, from initial video annotation to advanced, post-hoc analysis of model outputs. It consists of a primary GUI-based application that integrates various functionalities into a single, cohesive workflow.

</div>

## There two User Guides to get you started:
* 📖 Comprehensive [User Guide](https://farhanaugustine.github.io/IntegraPose/): Covers the graphical user interface (GUI), GIFs for workflow steps, advanced YOLO model customization, and example model backbone modules (for user wanting more control over the YOLO model).
* 🚀 [Quick start guide](https://farhanaugustine.github.io/IntegraPose/Quick_Start_Guide.html): A streamlined tutorial to get you up and running in minutes. Includes screenshots for references. 
</div>

# 🚧 Project Status 🚧
This repository is currently under active development. APIs may change, and new features will be added. Feedback and contributions are welcome!

</div>

# ✨ Key Features

1.  **Simultaneous Tracking & Classification:** A single model predicts bounding boxes, keypoint locations, and behavioral class.
2.  **Modular GUI-Driven Workflow:** Easy-to-use graphical interfaces for annotation, training, and analysis, minimizing the need for complex command-line interaction.
3.  **Integrated Annotation Tool:** Interactively label keypoints and assign behavioral classes to subjects in your images.
4.  **Advanced Bout Analytics:** Tools for analyzing, confirming, and scoring behavioral bouts, including manual review and correction.
5.  **Pose Clustering Analysis:** A dedicated tab for exploratory data analysis, clustering, and visualization of pose data.
6.  **Flexible ROI Management:** Define regions of interest dynamically for various analysis modes.
7.  **[Advanced Gait & Kinematic Analysis](https://github.com/farhanaugustine/IntegraPose/tree/main/Advanced%20Gait%20%26%20Kinematic%20Analysis):** A dedicated pipeline offering distinct methods for detecting strides and performing in-depth analysis of movement from keypoint data.
8.  **[HMM-VAE Video Segmentation](https://github.com/farhanaugustine/IntegraPose/tree/main/HMM-VAE-LSTM%20Video%20Segmentation):** An unsupervised workflow to automatically segment continuous pose data into discrete, meaningful behavioral motifs without prior labels.
9.  **[Sub-Behavior Discovery](https://github.com/farhanaugustine/IntegraPose/tree/main/Sub-Behavior%20Discovery%20with%20LSTM%20Autoencoders):** Employs a Seq2Seq LSTM Autoencoder to discover subtle, stereotyped variations or "sub-behaviors" within a broader action.
10. **[Tandem YOLO-LSTM Classifier](https://github.com/farhanaugustine/IntegraPose/tree/main/LSTM-Based%20Sequence%20Classifiers):** A powerful approach using YOLO for spatial pose data and an LSTM for robust, temporal behavior classification based on movement patterns. (Does require training two seperate models, but can be run in close to real-time speeds 15-18ms per frame)
   
</div>

# 🛠️ Prerequisites
Before you begin, ensure you have the following installed:

* Python: Version > 3.11 (tested with 3.11 & 3.12).
* Conda: Recommended for managing dependencies.
Core Libraries (can be installed via pip after setting up a Conda environment):
```
pip install ultralytics opencv-python pandas numpy hdbscan umap-learn matplotlib pyyaml scipy scikit-learn openpyxl XlsxWriter
```
* Alternatively, you can create a Conda environment using the provided `environment.yml` file:
```
conda env create -f environment.yml
conda activate IntegraPose
```

</div>

# 🚀 Workflow & How to Use
The IntegraPose workflow is managed through a single main GUI application.

To get started, run `main_gui_app.py` located in BehaviorAnnotation_ModelTraining Folder:

```
python main_gui_app.py
```

The application is organized into several tabs, each corresponding to a stage or type of analysis:

## 1. Setup & Annotation Tab
This tab is for defining your project's keypoints and behaviors, and for launching the integrated annotation tool.
* Define Keypoint Names: Specify the ordered names of your keypoints (e.g., nose, left_ear, right_ear).
* Define Behaviors: List the behaviors you want to classify, each with a unique numeric ID.
* Configure Annotation Directories: Set paths for your source images and where the generated annotation .txt files will be saved.
* Launch Annotator: Click "Launch Annotator" to open the OpenCV-based interactive tool for drawing bounding boxes and labeling keypoints. The annotator supports zoom, pan, and autosave.
* Generate dataset.yaml: After annotation, use this section to create the YOLO-compatible dataset.yaml file, which is crucial for model training.
## 2. Model Training Tab
Once your dataset is ready, use this tab to configure and initiate the training of your YOLO-Pose model.
* Load Dataset YAML: Point to the dataset.yaml file generated in the Setup tab.
* Select Model Variant: Choose a pre-trained YOLO-Pose model (e.g., yolov8n-pose.pt).
* Adjust Hyperparameters: Configure epochs, learning rate, batch size, and other training parameters.
* Data Augmentation Settings: Control various augmentation techniques like flip, perspective, and mosaic.
* Start Training: Launch the Ultralytics training process directly from the GUI.
## 3. Inference Tab
Use this tab to run your trained model on new video files or folders containing images.
* Load Trained Model: Select your `.pt` model file.
* Select Source: Choose a video file or a folder of images for inference.
* Configure Parameters: Adjust confidence thresholds, IOU, and other inference settings.
* Output Options: Define where output videos and detection .txt files will be saved.
* Start Inference: Process your data and generate detection outputs.
## 4. Webcam Inference Tab
Perform real-time inference using your trained model with a live webcam feed.
* Load Trained Model: Select your .pt model.
* Webcam Index: Specify the index of your webcam (e.g., 0 for the default).
* Live View & Recording: View the live inference and optionally record the annotated video.
## 5. Bout Analytics Tab
This tab provides tools for analyzing and managing behavioral bouts from your YOLO inference output.
* Load YOLO Output: Load your YOLO detection .txt files (from a folder or individual files).
* ROI Management:
   * Draw New ROI: Interactively draw regions of interest on a video frame. The tool guides you through drawing polygons.
   * Load/Save ROIs: Import and export ROI definitions to/from YAML files for reusability.
<img width="751" alt="ROI_Drawing" src="https://github.com/user-attachments/assets/5da6beba-bee1-47cd-8bf3-3ee5044099ca" />

* Behavioral Bout Analysis:
   * Calculate bout statistics (duration, frequency, time between bouts).
   * Filter and analyze bouts based on ROI entry/exit.
   * Review & Confirm Detected Bouts: Launch a dedicated tool to manually review and confirm automatically detected bouts.
   *  This tool allows for corrections and detailed scoring, saving updated bout information to a CSV file.
   <img width="976" alt="bout_confirmation_tool_v1" src="https://github.com/user-attachments/assets/98191ae4-4db2-48a4-a5ca-2638d305dbd2" />

   * Open Advanced Bout Scorer {**Work in-progress**}: Access an advanced interface for in-depth manual scoring and verification of behavioral bouts from a video.
## 6. Pose Clustering Analysis Tab
This tab allows for exploratory data analysis, including normalization, feature engineering, and clustering of pose data.
* Load Data: Import pose data (from YOLO output or other sources).
* Define Keypoints & Skeleton: Specify keypoint names and skeleton connections for feature calculations.
* Normalization & Feature Engineering: Normalize pose data and generate features based on keypoints and skeleton definitions.
* Clustering: Apply UMAP and HDBSCAN clustering to discover patterns in your behavioral data.
  ![clusters_video_1_720p_Walking_track1](https://github.com/user-attachments/assets/b566b074-c748-45b4-8c87-deeb46803c22)
  <img width="751" alt="Pose_Clustering" src="https://github.com/user-attachments/assets/c15429fb-d711-4d09-a129-fc24d8b0de14" />


* Visualization: Visualize clusters and their relationship to the original data.
</div>
# Project Structure
The repository is organized as follows:

```
BehaviorAnnotation_ModelTraining/
├── main_gui_app.py
├── keypoint_annotator_module.py
├── gui/
│   ├── advanced_bout_analyzer.py
│   ├── bout_confirmation_tool.py
│   ├── inference_tab.py
│   ├── log_tab.py
│   ├── pose_clustering_tab.py
│   ├── roi_analytics_tab.py
│   ├── setup_tab.py
│   ├── tooltips.py
│   ├── training_tab.py
│   └── webcam_tab.py
├── utils/
│   ├── bout_analyzer.py
│   ├── command_builder.py
│   ├── config_manager.py
│   ├── roi_drawing_tool.py
│   ├── roi_manager.py
│   ├── video_creator.py
│   └── webcam_manager.py
├── environment.yml
├── requirements.txt
├── README.md
└── bout_tool.log (example log file)
```
</div>

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

</div>

# Citation Suggestions:
Augustine, et al., (2025). Integrapose: A Unified Framework for Simultaneous Pose Estimation and Behavior Classification. Availabe on SSRN: https://ssrn.com/abstract=5334465

</div>

# Acknowledgments:
IntegraPose is built upon the powerful and flexible Ultralytics YOLOv8 framework. We extend our sincere gratitude to the [Ultralytics](https://www.ultralytics.com/) team for their significant contributions to the open-source community.

### Repo Underdevelopment 🚧🏗️👷🏼




