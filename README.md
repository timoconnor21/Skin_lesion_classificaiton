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
  
![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/b8352c81-a26a-4552-915f-81d415d85e04)

  - Early stopped after 71 Epochs.

- **Example classified images**
  
   ![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/561fa903-8294-4860-ae3a-a47607e1f812)
   ![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/4b50d7a5-d2a4-40f8-bf99-1229c82bc3a4)
   ![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/6b2b7751-e8a8-4633-a21f-2b8dd1b76d5e)
   ![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/1c20660b-c355-48f0-992b-ef9855fb42e1)


- **ROC Curve**

![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/4d07d82e-e1f1-419f-9862-dcc2af2eeef0)

- **Confusion Matrix**

![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/4c06c4a3-d67e-488c-8f93-f45ffb66aa1c)

- **Model Performance:**
  - Accuracy: 91.5%
  - Sensitivity: 94.6%
  - Specificity: 89.3%
  - MCC: 0.832
  - AUC: 0.970
 
**Results after test time augmentaion:**
- **ROC Curve**

![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/cd2ee54f-d72e-4c7a-88f6-1710863cc21a)

- **Confusion Matrix**
  
![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/953612f4-d18a-470c-bb80-edd004a6479a)

- **Model Performance:**
  - Accuracy: 93.4%
  - Sensitivity: 94.6%
  - Specificity: 89.3%
  - MCC: 0.867
  - AUC: 0.979
 
**Results optimal thresholding:**
- **Confusion Matrix**
  
![image](https://github.com/timoconnor21/Skin_lesion_classificaiton/assets/175061865/6659e66f-8ece-480d-bd77-8e4b2a758de6)

- **Model Performance:**
  - Accuracy: 93.4%
  - Sensitivity: 94.6%
  - Specificity: 89.3%
  - MCC: 0.868
  - AUC: 0.979

## How to Run the Repository
1. **Clone the Repository:**
   ```bash
       git clone https://github.com/timoconnor21/Skin_lesion_classification.git
       cd Skin_lesion_classification
   ```

2. **Install Dependencies:**

   Update the path to environment.yml annd run the following code
   ```bash
       conda env create -f <.../skin_lesion_classification/environment.yml>
   ```
   Alternatively, all necessary packages are listed below in the dependencies section.

3.  **Update configuration.py:**
     - In the configuration.py file, select whether to train a new model or load current best
     - If training a new model, be sure to set the desired training parameters
     - Choose whether or not to apply post-processing
     - Update the main_dir to the repo folder and run the config.py to update the working directory
  
    ```bash
        model_configuration.py
    ```

3.  **Run run_me.py:**
    ```bash
        run_me.py
    ```
     - The run_me.py file will step through the appropriate steps and return the results.

## Dependencies

  channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - matplotlib
  - numpy=1.26.4
  - openpyxl
  - pandas
  - pip=24.0
  - python=3.11.9
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - scikit-image
  - scikit-learn
  - scipy
  - spyder
  - pip:
    - albumentations==1.4.11
    - annotated-types==0.7.0
    - eval-type-backport==0.2.0
    - opencv-python==4.10.0.84
    - opencv-python-headless==4.10.0.84

## Contact
For any questions or clarifications, feel free to contact me at timroconnor21@gmail.com.
