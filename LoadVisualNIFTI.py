import glob  # For retrieving files/pathnames matching a specified pattern
import re # specifies a set of strings that matches it
import SimpleITK as sitk
import items as items
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# from ipywidgets import interact, interactive, IntSlider, ToggleButtons


BrainT1Subjs = glob.glob("/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/BrainMRI_train/T1_LPI_*.nii");
PFMaskSubjs = glob.glob("/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/BrainSeg/PFMask_train/PFseg_LPI_*.nii");

'''
BrainT1_path = glob.glob("/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/AutomaticSegmentationData/Combined/Chiari/*/T1.nii");
PFMask_path = glob.glob("/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/AutomaticSegmentationData/Combined/Chiari/*/CerebralTonsilMask.nii");
pattern = re.compile("/Volumes/Kurtlab/Chiari_Morpho_Segmentation/Segmentation/AutomaticSegmentationData/Combined/Chiari/.*_(\w*)\.nii");


data_paths = [{
    pattern.findall(item)[0]:item for item in items
}
for items in list(BrainT1_path)]

print(list(BrainT1_path))
print('number of training examples' ,len(data_paths))
'''
def read_img_sitk(img_path):
  image_data = sitk.ReadImage(img_path)
  return image_data

def read_img_nii(img_path):
  image_data = np.array(nib.load(img_path).get_fdata())
  return image_data

for subj in BrainT1Subjs:
    np_BrainImg = read_img_nii(subj)
    sitk_BrainImg = read_img_sitk(subj)
    sitk_BrainImg2 = sitk.GetImageFromArray(np_BrainImg)
    np_BrainImg2 = sitk.GetArrayFromImage(sitk_BrainImg2)
    print(sitk_BrainImg2.GetSize())
    print(np_BrainImg2.shape)


for mask in PFMaskSubjs:
    np_PFMaskImg = read_img_nii(mask)
    sitk_PFMaskImg = read_img_sitk(mask)
    sitk_PFMaskImg2 = sitk.GetImageFromArray(np_PFMaskImg)
    np_PFMaskImg2 = sitk.GetArrayFromImage(sitk_PFMaskImg2)
    # np_shape = np_PFMaskImg.shape
    # sitk_shape = sitk_PFMaskImg.GetSize()
    # print("Shape of np_PFMaskImg : ", np_shape)
    # print("Shape of sitk_PFMaskImg : ", sitk_shape)

'''
## Check shape of images

np_shape = np_PFMaskImg.shape
sitk_shape = sitk_PFMaskImg.GetSize()
print("Shape of np_PFMaskImg : ", np_shape)
print("Shape of sitk_PFMaskImg : ", sitk_shape)
'''

## Conversion between numpy and SimpleITK

sitk_BrainImg2 = sitk.GetImageFromArray(np_BrainImg)
sitk_PFMaskImg2 = sitk.GetImageFromArray(np_PFMaskImg)

print(sitk_BrainImg2.GetSize())

np_BrainImg2 = sitk.GetArrayFromImage(sitk_BrainImg2)
np_PFMaskImg2 = sitk.GetArrayFromImage(sitk_PFMaskImg2)
print(np_BrainImg2.shape)


## Visualize samples

@interact
def explore_3dimage(layer = (0,255) , modality=['BrainT1_path', 'PFMask_path'] , view = ['axial' , 'sagittal' , 'coronal']):
    if modality == 'BrainT1_path':
      modal = 'BrainT1_path'
    elif modality == 'PFMask_path':
      modal = 'PFMask_path'
    else :
      print("Error")

    image = read_img_sitk(BrainT1_path)
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
