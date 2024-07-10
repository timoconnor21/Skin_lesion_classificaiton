# Skin Lesion Classification
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
- **GitHub Repository:** [https://github.com/timoconnor21/Skin_lesion_classification](#)
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
  
![Training_curves](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/9bd7bfa9-36b0-4d6c-801f-eec6fb8caf36)


- **ROC Curve**

![ROC_curve](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/d555f89d-6dc6-4852-bbd9-fa0c0dc30ee2)

- **Confusion Matrix**

![Confusion_matrix](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/088b4474-b4b2-47f4-8463-ec02a5b7c088)

- **Model Performance:**
  - Accuracy: 90.8%
  - Sensitivity: 91.0%
  - Specificity: 90.6%
  - MCC: 0.816
  - AUC: 0.967
 
**Results after test time augmentaion:**
- **ROC Curve**

![TTA_ROC_curve](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/267ccf8a-c838-4ce3-a5d6-fabdf39a5402)

- **Confusion Matrix**
  
![TTA_ConfMat](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/8f129487-7452-44e6-8d05-231482706060)


- **Model Performance:**
  - Accuracy: 92.6%
  - Sensitivity: 93.4%
  - Specificity: 90.6%
  - MCC: 0.852
  - AUC: 0.975
 
**Results optimal thresholding:**
- **Confusion Matrix**
  
![OptimalThresh_ConfMat](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/05b18ecf-5f29-491c-b1cd-dd6219ee827e)

- **Model Performance:**
  - Accuracy: 92.6%
  - Sensitivity: 93.4%
  - Specificity: 90.6%
  - MCC: 0.852
  - AUC: 0.975

## How to Run the Repository
1. **Clone the Repository:**
   ```bash
       git clone https://github.com/timoconnor21/Skin_lesion_classification.git
       cd Skin_lesion_classification
   ```

2. **Install Dependencies:**

   Navigate to the repository directory
   ```bash
       cd .../skin_lesion_classification
   ```
   Install the required dependencies
   ```bash
       conda env create -f environment.yml
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
