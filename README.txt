\\\TEAM POLAR BEAR \\\

Off-road semantic scene segmentation using synthetic desert data.


#OBJECTIVE

Generate synthetic desert dataset.

Train a semantic segmentation model.

Evaluate segmentation performance. 

Compare predictions with ground truth.

Create a model which can learn from the synthetic desert dataset.



#CLASSIFICATION CATEGORIES

VEGETATION : trees , lush bushes , dry grass , dry bushes. 

OBSTACLES : rocks , logs.

 ENVIRONMENT : landscape , sky.

DETAILS : flowers , ground clutter.




#METHODOLOGY

DATASET : We used the synthetic desert dataset provided by Duality AI’s Falcon platform.

PREPROCESSING : Images were resized and normalised , data augmentation was applied to improve model robustness.

PERFORMANCE EVALUATION : Compared predictions against ground truth ( the actual answer used as a reference to evaluate the model)

#TECH STACK

Python 
Javascript 
HTML and CSS
React 

#INSTALLATION 

Follow these steps to set up the environment on your local machine:

1.Clone the Repository:

Bash
git clone [https://github.com/bglord22/Off-road-Desert-Segmentation-AI] ——— GitHub link
cd [HACKATHON]——— folder link 

2.Install Dependencies:
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

Bash
pip install torch torchvision segmentation-models-pytorch albumentations opencv-python matplotlib
Hardware Requirement:
The model is optimised for NVIDIA GPUs (RTX 3050 4GB VRAM tested) using CUDA.



#USAGE 

TRAINING THE MODEL : to reproduce the results , run the training script. It will automatically use the best performing parameters and save the weights. 
Bash
	python baseline.py

#TESTING AND VISUALISATION

To run the model on test images and see colour coded segmentation masks:

Bash 
Python test_ai.py


#FEATURES 

10- CLASS SEMANTIC MAPPING: Uses a custom CLASS_MAP to translate raw sensor IDs (example- 800 for rocks , 10000 for sky) into training-ready indices.

OPTIMISED ARCHITECTURE: Utilises a U -Net with an effcientNet-B0 backbone for high accuracy with low computational overhead.

HYBRID LOSS FUNCTION: Combines weighted cross-entropy (to handle rare classes like flowers and logs) with dice loss (for boundary precision).

ADVANCED AUGMENTATION: Implements horizontal flips , brightness contrast and spatial rotations to ensure the model generalises to “ novel environments”.

PERFORMANCE MONITORING:  Real-time tracking of Mean Intersection over Union (mloU) and combo loss per batch.







