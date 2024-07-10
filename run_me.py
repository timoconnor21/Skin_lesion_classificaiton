# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:03:54 2024

@author: TimOConnor
"""

"""
RunMe file for skin lesion detection. First setup the configuration in the 
model_configuration.py file then run this code.

This code will train a new model (or load best existing model) based on the 
configuration, present the results, and run through post-processing steps.

Function of this code is to build a skin lesion detector based on images of 
skin with either malignant or benign skin lesions.

"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from data_extraction import extract_data
from model_configuration import get_config
from model_training import train_lesion_detector
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from utils import show_example_images, TTADataset, tta_predictions, AddGaussNoise
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import recall_score, matthews_corrcoef, accuracy_score, roc_auc_score
from torchvision import models
import albumentations as A
import albumentations.pytorch



def skin_lesion_detection(config):
    extract_data() # unzip the data to raw_data folder
    save_dir = os.path.join(os.getcwd(),'model_results')
    
    if config['train_model']:
        print('Trianing model')
        results = train_lesion_detector(config)
        print('Traniing complete. Now loading results from best model.')
    else:
        print('Skipping model training. Loading previous best results directly.')
    
    best_config_path = os.path.join(save_dir, 'best_config.txt')
    if os.path.exists(best_config_path):
        with open(best_config_path, 'r') as file:
            config_str = file.read()
        config_lines = config_str.split('\n')
        model_name = None
        
        # Loop through each line and extract the model name
        for line in config_lines:
            if line.startswith('model_name:'):
                _, model_name = line.split(':', 1)
                model_name = model_name.strip()
                break
            
        print('Results from best model: ', model_name)
        results_file = os.path.join(save_dir, 'best_model_results.pkl')
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
            
        print(f"Accuracy: {results['Acc']:.4f}")
        print(f"Sensitivity: {results['Sens']:.4f}")
        print(f"Specificity: {results['Spec']:.4f}")
        print(f"MCC: {results['MCC']:.4f}")
        print(f"AUC: {results['AUC']:.4f}")
        
        # Show the plots from best performing model
        # Training Curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(results['Training_curves']['Train_loss'], label='Train Loss')
        plt.plot(results['Training_curves']['Val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel(' Loss')
        plt.legend()
        plt.title('Training and Validation Loss Curves')
    
        plt.subplot(1, 2, 2)
        plt.plot(results['Training_curves']['Train_acc'], label='Train Acc')
        plt.plot(results['Training_curves']['Val_acc'], label='Val Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Classification Accuracy')
        plt.show()
        
        # ROC Curve
        auc_score = results['AUC']
        fpr, tpr, _ = roc_curve(results['Labels'], results['Probs'])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
        
        # Confusion Matrix
        confMat = confusion_matrix(results['Labels'], results['Preds'])
        disp = ConfusionMatrixDisplay(confusion_matrix=confMat, display_labels=config['classes'])
        disp.plot(cmap=plt.cm.Blues, colorbar=False)
        plt.title("Classification confusion matrix (Test data)")
        plt.show()
            
            
        # # Visualize some results
        # FP_ids = [s.split('\\')[-1] for s in results['FPs']]
        # FN_ids = [s.split('\\')[-1] for s in results['FNs']]
        # TN_ids = [sample_id for sample_id in  os.listdir(os.path.join(data_dir,'test','Benign')) if sample_id not in FP_ids]
        # TP_ids = [sample_id for sample_id in  os.listdir(os.path.join(data_dir,'test','Malignant')) if sample_id not in FN_ids]
        
        # show_example_images(TP_ids, data_dir, 'Malignant', 10, 'True Positive Examples')
        # show_example_images(TN_ids, data_dir, 'Benign', 10, 'True Negative Examples')
        # show_example_images(FP_ids, data_dir, 'Benign', 10, 'False Positive Examples')
        # show_example_images(FN_ids, data_dir, 'Malignant', 10, 'False Negative Examples')
    
    
    ## Apply post-processing techniques
    if config['post_processing']:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print('Applying post processing')
        model =  torch.load(os.path.join(save_dir,model_name+'.pth'))

        
        data_transforms = {
            'train': A.Compose([
                A.RandomRotate90(p=1),   
                # A.ShiftScaleRotate(shift_limit=0.05, scale_limit = .05, rotate_limit=45, p=.2),
                AddGaussNoise(mean_val=0, var_val=.0001, p=.1),
                # A.PixelDropout (dropout_prob=0.1, per_channel=True, drop_value=0, mask_drop_value=None, always_apply=False, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2()
            ]),
            'test': A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                A.pytorch.ToTensorV2()
            ])
        } 

        # Load test data paths and labels
        test_root_dir = 'raw_data/test'
        test_image_paths = []
        test_labels = []
        for label, subdir in enumerate(['Benign', 'Malignant']):
            subdir_path = os.path.join(os.getcwd(),test_root_dir, subdir)
            for img_name in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, img_name)
                test_image_paths.append(img_path)
                test_labels.append(label)
                 
        tta_dataset = TTADataset(test_image_paths, test_labels,transform=data_transforms['train']) 
        tta_loader = DataLoader(tta_dataset, batch_size=64, shuffle=False)


        # Perform TTA predictions
        print('Apply test time augmentation for more robust predictions on the test dataset')
        labels, tta_preds, tta_probs, tta_fp_paths, tta_fn_paths = tta_predictions(model, tta_loader, device)
        
        # Calculate metrics
        tta_accuracy = accuracy_score(labels, tta_preds)
        tta_sensitivity = recall_score(labels, tta_preds)
        tta_confMat = confusion_matrix(labels, tta_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=tta_confMat, display_labels=config['classes'])
        disp.plot(cmap=plt.cm.Blues, colorbar=False)
        plt.title("Classification confusion matrix (Test data after TTA)")
        plt.savefig(os.path.join(save_dir,'TTA_ConfMat.png'))
        plt.show()
        
        tta_tn, tta_fp, tta_fn, tta_tp = confMat.ravel()
        tta_specificity = tta_tn / (tta_tn + tta_fp)
        tta_mcc = matthews_corrcoef(labels, tta_preds)
        tta_auc = roc_auc_score(labels, tta_probs)
        
        print('Results after test time augmentation')
        print(f"Accuracy: {tta_accuracy:.4f}")
        print(f"Sensitivity (Recall): {tta_sensitivity:.4f}")
        print(f"Specificity: {tta_specificity:.4f}")
        print(f"MCC: {tta_mcc:.4f}")
        print(f"AUC: {tta_auc:.4f}")
        
        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(labels, tta_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {tta_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve after TTA')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(save_dir,'TTA_ROC_curve.png'))
        plt.show()
        
        print('Calculating optimal classification threshold and recomputing metrics')
        # Chose optimal classification threshold
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        opt_preds = np.where(tta_probs >= optimal_threshold, 1, 0)
        opt_accuracy = accuracy_score(labels, opt_preds)
        opt_sensitivity = recall_score(labels, opt_preds)
        opt_confMat = confusion_matrix(labels, opt_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=opt_confMat, display_labels=config['classes'])
        disp.plot(cmap=plt.cm.Blues, colorbar=False)
        plt.title("Classification confusion matrix (Test data after choosing optimal threshold)")
        plt.savefig(os.path.join(save_dir,'OptimalThresh_ConfMat.png'))
        plt.show()
        
        opt_tn, opt_fp, opt_fn, opt_tp = confMat.ravel()
        opt_specificity = opt_tn / (opt_tn + opt_fp)
        opt_mcc = matthews_corrcoef(labels, opt_preds)
        
        print('Results after optimal thresholding')
        print(f"Accuracy: {opt_accuracy:.4f}")
        print(f"Sensitivity (Recall): {opt_sensitivity:.4f}")
        print(f"Specificity: {opt_specificity:.4f}")
        print(f"MCC: {opt_mcc:.4f}")
        
        
        # Store results and paths
        post_processing_results = {
            'Acc': tta_accuracy, 'Sens': tta_sensitivity, 'Spec': tta_specificity, 'MCC': tta_mcc, 'AUC': tta_auc,
            'TP': tta_tp, 'TN': tta_tn, 'FP': tta_fp, 'FN': tta_fn,
            'FPs': list(tta_fp_paths), 'FNs': list(tta_fn_paths),
            'Labels': labels, 'Preds': tta_preds, 'Probs': tta_probs, 'Opt_Acc': opt_accuracy,'Opt_Sens': opt_sensitivity,'Opt_Spec': opt_specificity,'Opt_MCC': opt_mcc, 'Opt_CM': opt_confMat}
    
    else:
        print('No previously existing models. Please set train_model to True in the model_configuration')
    
        
    return results, post_processing_results
    
if __name__ == '__main__':
    config = get_config()
    results = skin_lesion_detection(config)