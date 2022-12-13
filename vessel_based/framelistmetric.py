'''
  Adapted from Sophia's code https://github.com/sophiabano/FetReg2021/blob/master/fetreg-reg/FrameListMetricScript.py
  To obtain output csv files for the boxplot of vessel based SSIM boxplots in the future
'''
import cv2
import numpy as np
import pandas as pd
import os
import argparse
import ssim_processing as sp
from tqdm import tqdm

# from utilsReg import *
from PIL import Image, ImageFilter

def get_mask_im(fullImgPaths, mask_path, crop_top, crop_bottom):
    """
    :param fullImgPaths: Image path of images to be processed, need only the size of one image there
    :param mask_path: path to the mask image to be used
    :param crop_top: the amount of pixels to be removed at the top due to dead pixels
    :param crop_bottom:  the amount of pixels to be removed at the bottom due to dead pixels
    :return: returns mask image
    """
    img_1 = cv2.imread(fullImgPaths[0])
    img_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB)


    mask_im = Image.open(mask_path)
    mask_im = mask_im.resize((img_1.shape[0], img_1.shape[1]), Image.ANTIALIAS)  
    mask_im = np.array(mask_im)
    mask_im = mask_im * np.uint8(255)

    # crop mask
    mask_im[:crop_top] = 0
    mask_im[(mask_im.shape[0] - crop_bottom):] = 0

    mask_im = cv2.resize(mask_im, img_1.shape[1::-1])

    return mask_im

def readHfromTXT(miccai_Hpath):
    """
    :param miccai_Hpath: Text files containing H (homography) matrix for each consecutive image pairs in the video sequence
    :return: returns H_array - a numpy array of size (N,3,3) where N is the number of images/homography matrices
    """
    fullTxtPaths =  [ miccai_Hpath + '/' + f  for f  in sorted(os.listdir(miccai_Hpath))]
    
    Fileopen = open(fullTxtPaths[0], 'r').read().strip()
    Hstr3 = Fileopen.split("\n")
    
    H_indv = np.array([])
    for H1 in Hstr3:
        Hstr1 = H1.split(" ")
        Hfloat = [float(s) for s in Hstr1]
        H_indv = np.append(H_indv, Hfloat)
        
    H_array = np.array([H_indv.reshape(3,3)])
        
    for i in range(len(fullTxtPaths)-1):
        Fileopen = open(fullTxtPaths[i+1], 'r').read().strip()
        Hstr3 = Fileopen.split("\n")
        
        H_indv = np.array([])
        for H1 in Hstr3:
            Hstr1 = H1.split(" ")
            Hfloat = [float(s) for s in Hstr1]
            H_indv = np.append(H_indv, Hfloat)
        #print(H_indv)
            
        H_indv =np.array([H_indv.reshape(3,3)])
    
        H_array  = np.append(H_array,H_indv, axis = 0)
    return H_array

def readFrameListfromTXT(path):
    """
    :param miccai_Hpath: Text files containing H (homography) matrix for each consecutive image pairs in the video sequence
    :return: returns H_array - a numpy array of size (N,3,3) where N is the number of images/homography matrices
    """
    Fileopen = open(path, 'r').read().strip()
    FramePairs = Fileopen.split("\n")
    frame_list = np.loadtxt(path, dtype="int")
    return frame_list

images_path = "Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video06/masks_transform" 
mask_path = "Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/mask.png"
framelist_path = "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/framelist_"
prediction_path = "Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/masks_transform_txt" 

transformation = "Homography"
padding_size = 2000
showImages = False

window_size = 50
frame_distance = 5

fullImgPaths =  [ images_path + '/' + f  for f  in sorted(os.listdir(images_path))]

mask_im   = get_mask_im(fullImgPaths, mask_path, 0, 0)
mask_area = mask_im.sum()/255;
crop_size = (0.6)*np.sqrt(mask_area)
crop_size = crop_size.astype(int)


# Read H from registration results
H_array = readHfromTXT(prediction_path)

length= len(H_array)-1
df_ssim = pd.DataFrame()
df_overlap = pd.DataFrame()
for i in tqdm(range(1,6)):
    frame_list = readFrameListfromTXT(framelist_path+ str(i)+".txt")
    print("name of txt is: ", framelist_path+ str(i)+".txt")
    SSIM, overlap  = sp.getSSIMForRegOverlapFrameList(frame_list, H_array, mask_im, transformation, crop_size, fullImgPaths, showImages)
    
    #added while conditions since columns with unequal numbers of rows cannot be added to a dataframe
    while len(SSIM) != length:
        SSIM= np.append(SSIM, np.nan)
    while len(overlap) != length:
        overlap= np.append(overlap, np.nan)
        
    #create a new column
    df_ssim[str(i)]= SSIM
    df_overlap[str(i)]= overlap
df_ssim.to_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/anon001_ssim_dsm.csv", index= False)
df_overlap.to_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/anon001_overlap_dsm.csv", index= False)

print("Done!")

