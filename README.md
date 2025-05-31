[![DOI](https://zenodo.org/badge/988759361.svg)](https://doi.org/10.5281/zenodo.15565090)
# IntegraPose - Simultaneous Behavior Classification and Pose Estimation
### This repo is to demonstrate multi-task learning for combined object detection and pose-estimations using YOLO-Pose models. IntegraPose enables robust, simultaneous classification of complex behaviors and precise keypoint tracking from a single model.

<div align="center">
<img src="https://github.com/user-attachments/assets/2aec1a90-552d-42e6-8ae4-91ae322a6f1b" alt="IntegraPose" width="50%"/>
</div>

# Pre-reqs: 
* [Ultralytics](https://docs.ultralytics.com/#what-is-ultralytics-yolo-and-how-does-it-improve-object-detection): `pip install ultralytics`
* conda environment with python >3.10 (only tested with 3.11, but other python versions should work, too!)
* numpy
* pandas
* collections
* pyyaml
* opencv-python
* json
* matplotlib
* scipy
* scikit-learn
* pillow

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




