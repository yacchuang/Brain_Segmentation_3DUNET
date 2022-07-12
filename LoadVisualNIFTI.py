import glob  # For retrieving files/pathnames matching a specified pattern
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, IntSlider, ToggleButtons


BrainT1Subjs = glob.glob("/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/BrainMRI_train/T1_LPI_*.nii");
PFMaskSubjs = glob.glob("/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/PFMask_train/PFseg_LPI_*.nii");

def read_img_sitk(img_path):
  image_data = sitk.ReadImage(img_path)
  return image_data

def read_img_nii(img_path):
  image_data = np.array(nib.load(img_path).get_fdata())
  return image_data

for subj in BrainT1Subjs:
    np_BrainImg = read_img_nii(subj)
    sitk_BrainImg = read_img_sitk(subj)
    ## Conversion between numpy and SimpleITK
    sitk_BrainImg2 = sitk.GetImageFromArray(np_BrainImg)
    np_BrainImg2 = sitk.GetArrayFromImage(sitk_BrainImg2)
    # print(sitk_BrainImg2.GetSize())
    # print(np_BrainImg2.shape)


for mask in PFMaskSubjs:
    np_PFMaskImg = read_img_nii(mask)
    sitk_PFMaskImg = read_img_sitk(mask)
    ## Conversion between numpy and SimpleITK
    sitk_PFMaskImg2 = sitk.GetImageFromArray(np_PFMaskImg)
    np_PFMaskImg2 = sitk.GetArrayFromImage(sitk_PFMaskImg2)
    ## Check shape of images
    # np_shape = np_PFMaskImg.shape
    # sitk_shape = sitk_PFMaskImg.GetSize()
    # print("Shape of np_PFMaskImg : ", np_shape)
    # print("Shape of sitk_PFMaskImg : ", sitk_shape)


'''
## Visualize samples
BrainT1Subj02 = "/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/BrainMRI_train/T1_LPI_Subj02.nii";
PFMaskSubj02 = "/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/PFMask_train/PFseg_LPI_Subj02.nii";

@interact
def explore_3dimage(layer = (0,255) , modality=['BrainT1Subj02', 'PFMaskSubj02'] , view = ['axial' , 'sagittal' , 'coronal']):
    if modality == 'BrainT1Subj02':
      modal = BrainT1Subj02
    elif modality == 'PFMaskSubj02':
      modal = PFMaskSubj02
    else :
      print("Error")

    image = read_img_sitk(BrainT1Subj02)
    array_view = sitk.GetArrayViewFromImage(image)

    if view == 'axial':
        array_view = array_view[layer, :, :]
    elif view == 'coronal':
        array_view = array_view[:, layer, :]
    elif view == 'sagittal':
        array_view = array_view[:, :, layer]
    else:
        print("Error")


    plt.figure(figsize=(10, 5))
    plt.imshow(array_view, cmap='gray')
    plt.title('Explore Layers of Brain', fontsize=10)
    plt.axis('off')
'''
