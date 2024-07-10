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
- **Model Performance:**
  - Achieved accuracy: X%
  - Confusion matrix and classification report plots
- **Discussion:**
  TBD

## How to Run the Repository
1. **Clone the Repository:**
   ```bash
      git clone https://github.com/timoconnor21/Skin_lesion_detection.git
      cd Skin_lesion_detection

2. **Install Dependencies:**
    ```bash
      python3 -m venv skin_lesion_detection_env
      source skin_lesion_detection_env/bin/activate  # On Windows use `skin_lesion_detection_env\Scripts\activate
      pip install -r requirements.txt

3.  **Update config.py:**
   - In the config.py file, select whether to train a new model or load current best
   - If training a new model, be sure to set the desired training parameters
   - Choose whether or not to apply post-processing

3.  **Run run_me.py:**
    ```bash
       run_me.py
   - The run_me.py file will step through the appropriate steps and return the results.

## Dependencies
- All dependencies are listed in requirements.txt

## Contact
For any questions or clarifications, feel free to contact me at timroconnor21@gmail.com.
