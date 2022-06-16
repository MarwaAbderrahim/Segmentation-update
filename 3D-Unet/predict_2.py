import logging
import os
import sys
import glob
from monai.transforms import LoadImage
import matplotlib.pyplot as plt
import numpy
import time
import monai
import numpy as np
import torch
from monai.handlers.utils import from_engine
from monai.visualize import blend_images
import argparse
import sys
import os
from DicomRTTool import DicomReaderWriter
import nibabel
import SimpleITK as sitk
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.inferers import sliding_window_inference
from pathlib import Path
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    ScaleIntensityRanged,
    Spacingd,
    CropForegroundd,
    Orientationd,
    SaveImaged,
    EnsureTyped,
)



#Using argparse module which makes it easy to write user-friendly command-line interfaces.
parser = argparse.ArgumentParser(description='Predict masks from input images')
#parser.add_argument("-i", "--input", type=str, required=True, help="path to input image")    #input CT image we can call by "-i" command

parser.add_argument("-i", "--input", type=Path, default=Path(__file__).absolute().parent / "data", help="Path to the data directory",)
parser.add_argument("-o", "--output", type=str, help="Name of predicted Mask")                 #output segmented mask we can call by "-o" command
cwd = os.getcwd() 


# p = parser.parse_args()
# print(p.data_dir, type(p.data_dir))



def main():
    #print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    args = parser.parse_args()
    
    #The path to the folder where the trained model is saved
    model_dir  = cwd

    #-------------------------------------- Convert dicom to nifti----------------------------------------------

    in_path_dicom_nifti=args.input
    output=args.output
    image="raw_data_Dicom2Nii.nii.gz"
    #Initialize the reader
    reader = DicomReaderWriter()

    # Provide a path through which the reader should search for DICOM
    reader.walk_through_folders(in_path_dicom_nifti)

    #  Load the images
    reader.get_images()

    # Write .nii images
    sitk.WriteImage(reader.dicom_handle,  image)

    # path of the .nii image
    img=cwd + "/" + image
    #The path of the image that will be used for the segmentation, user of the code can chose an image using command-line.
    test_images = sorted(glob.glob(os.path.join(img)))
    test_dicts = [{"image": image_name} for image_name in test_images]
    files = test_dicts[:]
    
    
    #-------------------------------------- Convert nitfti to dicom ----------------------------------------------
    def writeSlices(series_tag_values, new_img, i, out_dir):
            image_slice = new_img[:,:,i]
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()

            # Tags shared by the series.
            list(map(lambda tag_value: image_slice.SetMetaData(tag_value[0], tag_value[1]), series_tag_values))

            # Slice specific tags.
            image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
            image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time

            # Setting the type to CT preserves the slice location.
            image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over

            # (0020, 0032) image position patient determines the 3D spacing between slices.
            image_slice.SetMetaData("0020|0032", '\\'.join(map(str,new_img.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
            image_slice.SetMetaData("0020,0013", str(i)) # Instance Number

            # Write to the output directory and add the extension dcm, to force writing in DICOM format.
            writer.SetFileName(os.path.join(out_dir,'slice' + str(i).zfill(4) + '.dcm'))
            writer.Execute(image_slice)


    def convert_nifti_to_dicom(nifti_dir, out_dir):
        pixel_data = sitk.sitkUInt16
        new_img = sitk.ReadImage(nifti_dir, pixel_data) 
        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")
        # out_path = os.path.join(out_dir)
        direction = new_img.GetDirection()
        series_tag_values = [("0008|0031",modification_time), # Series Time
                        ("0008|0021",modification_date), # Series Date
                        ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                        ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                        ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                            direction[1],direction[4],direction[7])))),
                        ("0008|103e", "Created-SimpleITK")] # Series Description

        # Write slices to output directory
        list(map(lambda i: writeSlices(series_tag_values, new_img, i, out_dir), range(new_img.GetDepth())))
        







    # Formula to compute minimum/maximum scale internsity based on known Wide window and Window level. For bones L=500 and W=2000

    L=500
    W=2000
    i_min = L - (W/2)   #output  -500
    i_max =  L + (W/2)  #output  1500


    # define pre transforms
    pre_transforms = Compose([
        LoadImaged(keys="image"),  #Dictionary-based wrapper which can load both images and labels
        EnsureChannelFirstd(keys="image"), #Ensures the channel first input for both images and labels
        Orientationd(keys=["image"], axcodes="RAS"), #Reorienting the input array
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"), #Resampling the input array, interpolation mode to calculate output values.
        ScaleIntensityRanged(keys=["image"], a_min=i_min, a_max=i_max,b_min=0.0, b_max=4.0, clip=True,), #Scale min/max intensity ranges for image and mask, clip after scaling
        CropForegroundd(keys=["image"], source_key="image"),  #Crops only the foreground object of the expected images. 
        EnsureTyped(keys="image"), #Ensure the input data to be a PyTorch Tensor or numpy array
        ])
    

    #Data Loading and augmentation
    dataset = CacheDataset(data=files, transform=pre_transforms, cache_rate=1.0, num_workers=0)
    test_loader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    
    # define post transforms
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=pre_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            ),
        
        AsDiscreted(keys="pred", argmax=True, to_onehot=5),
        #SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./out_seg", output_postfix="seg", resample=False),

    ])
    


    #create 3D UNet architecture
    device = torch.device("cuda:0")  #define the device which should be either GPU or CPU
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=5,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    

    #loading the saved model of previously trained 3D U-Net model
    model.load_state_dict(torch.load(os.path.join(model_dir, "best_metric_model.pth"))) 

    
    #evaluation of the model on inference mode, upload one CT image and as an outcome we get its segmented mask 
    model.eval()
    with torch.no_grad():
        for test_data in test_loader:
            start_time = time.time()
            test_inputs = test_data["image"].to(device)
            roi_size = (160, 160, 160)
            sw_batch_size = 4  # In order to excute the code with "CUDA out of memory" error, update the value of batch_size to 3, 2 or 1. 
            test_data["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, model)
            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            test_outputs = from_engine(["pred"])(test_data)
            

            loader = LoadImage()
            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])[0]
            test_output_argmax = torch.argmax(test_outputs[0], dim=0, keepdim=True)
            
            rety = blend_images(image=original_image[None], label=test_output_argmax, cmap="jet", alpha=0.5, rescale_arrays=True)
            
            print("Total time seconds: {:.2f}".format((time.time()- start_time)))
    
            
            # fig, (ax1, ax2, ax3) = plt.subplots(figsize=(18, 6), ncols=3)
            # ax1.title.set_text('Image')
            # ax2.title.set_text('Predicted Mask')
            # ax3.title.set_text('Segmented Image')
            # bar1 = ax1.imshow(original_image[None][ 0, :, :, 90], cmap="gray")
            # fig.colorbar(bar1, ax=ax1)
            # bar2 = ax2.imshow(test_output_argmax [ 0, :, :, 90],  cmap="jet")
            # fig.colorbar(bar2, ax=ax2)
            # ax3.imshow(torch.moveaxis(rety[:, :, :, 90], 0, -1))
            # fig.colorbar(bar1, ax=ax3)
            # plt.show()
            # fig.savefig('segmentation.png', bbox_inches='tight')  #the visualization will be saved in the same folder where is your predict.py file.



            data = np.asarray(test_outputs[0]) # an array of dimension (5, 512,512,350), which contains the masks
            
            
            data_multiplied = []
            for i in range(test_outputs[0].shape[0]):
                data_Multibly = data[i] * i
                data_multiplied.append(data_Multibly)

            data_multiplied = np.asarray(data_multiplied)
            data_labels = data_multiplied.sum(axis=0)       # an array of dimension ((512,512,350), which contains the segmented image.


            nifti_file = nibabel.Nifti1Image(data_labels, None)
            nibabel.save(nifti_file, os.path.join(args.output))  # save the segmented image
            
            os.mkdir(output)
            convert_nifti_to_dicom(os.path.join(args.output), output)  
            print(f'The segmented dicom is saved in the {cwd}\{output}')
            


if __name__ == '__main__':
    main()








