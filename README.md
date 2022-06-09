# Segmentation

This repository contains scripts for 3D Unet architechture which is based on integrated MONAI into an existing PyTorch medical DL program. 

* Install the requirements.txt file by using 'pip install -r requirements.txt'
* The instalation steps for pytorch and monai is written in "installation.txt" file.

Data extraction and CSV file:
1) The Pelvis dataset can be downloaded https://zenodo.org/record/4588403#.Ym-WotpBy3B
* Download Images: CTPelvic1K_dataset6_data.tar.gz
* Download Masks: CTPelvic1K_dataset6_Anonymized_mask.tar.gz
2) Create a "Data" folder on your D drive where all images and masks will be extracted. 
3) To extract images in Command Prompt write the command below and in your "Data" folder you will have a "CTPelvic1K_dataset6_data" folder with 103 images.
* tar -xvzf C:\Users\Yourname\Downloads\CTPelvic1K_dataset6_data.tar.gz -C D:\Data
5) To extract masks in Command Prompt write the command below and in your "Data" folder you will have a "ipcai2021_dataset6_Anonymized" folder with 103 masks.
* tar -xvzf C:\Users\Yourname\Downloads\CTPelvic1K_dataset6_Anonymized_mask.tar.gz -C D:\Data
6) Use "train_data.csv" and "test_data.csv" files, which will help you to automatically select images/masks for training and testing. Please place these csv files in your "Data" folder.


Dataset description:
* Dataset: dataset6 (CLINIC) 
* Target: Pelvis
* Modality: CT
* Format: NIFTI
* Size: 55 cases (41 Training + 14 Testing)

Labels: 
* 0: background, 
* 1: sacrum, 
* 2: right_hip, 
* 3: left_hip, 
* 4: lumbar_vertebra    




Description of python scripts:

1) train.py --> 3D Unet training which works with Early Stopping. 

2) evaluate.py --> Evaluates the trained model on testing dataset and verify the dice score of each case. 
 
3) predict.py --> Uploads one image and output gives one segmented mask: 
* python predict.py -i image.nii.gz -o seg.nii.gz

4)  predict_2.py --> Uploads dicom series and output gives a segmented image in nifti format and another in Dicom format (a folder containing a series of ".dcm" files): 
* python predict_2.py -i dicom_path -o image_name
     
5) slicer.py -->  Transforms segmented mask to make it suitable for 3D slicer visualization: 
* python slicer.py -i seg.nii.gz -o name.nii.gz

6) pytorchtools.py --> for early stopping and must be placed in the same folder where the train.py file is located. 
