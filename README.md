# Skin Lesion Detection
Computer Vision Take Home Project

## Objective
Evaluate the given dataset to classify skin cancer images as benign or malignant.

## Table of Contents
1. [Code Repository](#code-repository)
2. [Data Management](#data-management)
3. [Preprocessing](#preprocessing)
4. [Model Creation](#model-creation)
5. [Post Processing](#post-processing)
6. [Results and Discussion](#results-and-discussion)
7. [How to Run the Repository](#how-to-run-the-repository)
8. [Dependencies](#dependencies)
9. [Contact](#contact)

## Code Repository
- **GitHub Repository:** [https://github.com/timoconnor21/Skin_lesion_detection](#)
- All code for this project is pushed to the repository.

## Data Management
- The data is provided in the zip file 'archive(5).zip'.
  - During execution of data_extraction.py, this file will be unzipped and its contents stored in 'raw_data'
  - If the raw_data file already exists, the file will not be unzipped again

## Preprocessing
- Preprocessing is handled during the dataloader
  - Images are divided by 255 to convert from 8bit integers into the range [0,1]
  - All images are normalized using ImageNet means and stds:  mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
  - Training images are further augmented with rotation, scaling, gaussian noise, and pixel dropout to diversify the training data

## Model Creation
- model_training.py will build and train a model based on the parameters outlined in model_configuration.py
  - Default model is a ResNet18 model initialized with ImageNet trained weights, and the first two layers frozen.
  - Best performing models are saved and output to model_results

## Post-Processing
- Applied post-processing techniques based on model results to improve performance.
  - Post-processing is applied in the run_me.py file for the best performing model 
  - Test time augmentation is used to generate multiple predictions on augmented versions of the test sample which are averaged to increase prediction robustness
  - Optimal thresholding of the ROC curve is also applied to find the optimal threshold for classification

## Results and Discussion
- **Training Curves**
  
![image](https://github.com/timoconnor21/Skin_lesion_detection/assets/175061865/fa216d05-326c-4ba3-8055-03e013e06f23)
- **ROC Curve**
  
![image](https://github.com/timoconnor21/Skin_lesion_detection/assets/175061865/32092618-af30-484d-a22a-0a921b62cbdb)
- **Confusion Matrix**
  
![image](https://github.com/timoconnor21/Skin_lesion_detection/assets/175061865/835a407d-63ff-4717-a1e7-fcac6797f671)

- **Model Performance:**
  - Accuracy: 90.8%
  - Sensitivity: 91.0%
  - Specificity: 90.6%
  - MCC: 0.816
  - AUC: 0.967
 
**Results after test time augmentaion:**
- **ROC Curve**
  
![image](https://github.com/timoconnor21/Skin_lesion_detection/assets/175061865/befd8310-d37a-4e3b-9a63-5ded552aec81)
- **Confusion Matrix**
  
![image](https://github.com/timoconnor21/Skin_lesion_detection/assets/175061865/0487b3d9-448c-4e13-8e25-7fc719a8f017)


- **Model Performance:**
  - Accuracy: 92.6%
  - Sensitivity: 93.4%
  - Specificity: 90.6%
  - MCC: 0.852
  - AUC: 0.975
 
**Results optimal thresholding:**
- **Confusion Matrix**
  
![image](https://github.com/timoconnor21/Skin_lesion_detection/assets/175061865/3557a39d-8b88-449d-8d3e-483a3c031c32)

- **Model Performance:**
  - Accuracy: 92.6%
  - Sensitivity: 93.4%
  - Specificity: 90.6%
  - MCC: 0.852
  - AUC: 0.975

## How to Run the Repository
1. **Clone the Repository:**
   ```bash
       git clone https://github.com/timoconnor21/Skin_lesion_detection.git
       cd Skin_lesion_detection
   ```

2. **Install Dependencies:**

   Create a new virtual enviornment
   ```bash
      conda create --name skin_lesion_detection_env python=3.8
   ```
   
   Activate the new enviornment
   ```bash
      conda activate skin_lesion_detection_env
   ```

   Navigate to the repository directory
   ```bash
       cd .../skin_lesion_detection
   ```
   Install the required dependencies
   ```bash
       pip install -r requirements.txt
   ```
   
   

3.  **Update config.py:**
     - In the config.py file, select whether to train a new model or load current best
     - If training a new model, be sure to set the desired training parameters
     - Choose whether or not to apply post-processing

3.  **Run run_me.py:**
    ```bash
        run_me.py
    ```
     - The run_me.py file will step through the appropriate steps and return the results.

## Dependencies

  - pandas==1.2.4
  - numpy==1.22.3 
  - openpyxl==3.0.10
  - scikit-learn==1.0.2
  - scipy==1.7.3
  - scikit-image==0.16.2
  - opencv==4.6.0
  - matplotlib==3.4.3
  - pip==22.2.2
  - albumentations=1.3.0=pyhd8ed1ab_0
  - opencv-python-headless==4.10.0.84
  - torch==1.12.0

## Contact
For any questions or clarifications, feel free to contact me at timroconnor21@gmail.com.
