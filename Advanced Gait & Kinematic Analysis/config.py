# config.py
# This file now contains only the DEFAULT settings for the pipeline.
# All of these values can be overridden by the user in the GUI.

# =============================================================================
# SECTION 1: DATASET CONFIGURATION
# =============================================================================
BEHAVIOR_CLASSES = {
    0: "Walking",
    1: "Wall-Rearing",
    2: "Rearing",
    3: "Grooming"
}
KEYPOINT_ORDER = [
    'Nose', 'Left Ear', 'Right Ear', 'Base of Neck', 
    'Left Front Paw', 'Right Front Paw', 'Center Spine', 
    'Left Rear Paw', 'Right Rear Paw', 'Base of Tail', 
    'Mid Tail', 'Tail Tip'
]

# =============================================================================
# SECTION 2: GAIT ANALYSIS PARAMETERS
# =============================================================================
GAIT_PAWS = ['Left Front Paw', 'Right Front Paw', 'Left Rear Paw', 'Right Rear Paw']
PAW_ORDER_HILDEBRAND = ['Left Front Paw', 'Right Front Paw', 'Left Rear Paw', 'Right Rear Paw']
PAW_SPEED_THRESHOLD_PX_PER_FRAME = 5.0
STRIDE_REFERENCE_PAW = 'Left Rear Paw'

# =============================================================================
# SECTION 3: POSE METRICS
# =============================================================================
ELONGATION_CONNECTION = ("Nose", "Base of Tail")
BODY_ANGLE_CONNECTION = ("Base of Neck", "Nose")

# =============================================================================
# SECTION 4: ANALYSIS PARAMETERS
# =============================================================================
DETECTION_CONF_THRESHOLD = 0.25
MIN_BOUT_DURATION_FRAMES = 15
