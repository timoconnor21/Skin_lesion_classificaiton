# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:34:23 2024

@author: TimOConnor
"""

"""
This code extracts the zipped image data to an ouput folder 'raw_data' for 
further processing and training. 

Code expects an original zipped file 'archive (5).zip' to be in the working
directory and creates the raw_data folder within the same directory.

If the raw_data folder already exists, it is assumed the data is already 
unzipped and will not unzip the original file again.
"""

import zipfile
import os


def extract_zip_to_raw_data(zip_path, extract_to='raw_data'):
    # Check if the raw_data directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print('Done extracting zip file to "raw_data"')
    else:
        print('Zip file already extracted to "raw_data"')
            
        
def extract_data():
    # Extract the zip_files to input data if it is not already there        
    zip_file_path = os.path.join(os.getcwd(),'archive (5).zip')
    extract_zip_to_raw_data(zip_file_path)
    
if __name__ == '__main__':
    extract_data()