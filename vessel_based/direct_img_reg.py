'''
  Referenced Sophia's repository https://github.com/sophiabano/EndoVis-FetReg2021/blob/033f40826ee7e50b61a76ad929efb66fb67bb8f6/FetReg-registration-docker-example/code/main.py
  Obtaining the images and homography txt files for relative transformations between adjacent frames
'''
from glob import glob  # For listing files
import cv2  # For handling image loading & processing
import os  # for path creation
import numpy as np
import argparse

#for displaying image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

parser = argparse.ArgumentParser(description='obtain relative transformation')
parser.add_argument("--inputfolder", type=str, default = "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/masks_predicted", help="vessel based images")
parser.add_argument("--outputpath", type=str, default = "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video06/masks_transform", help="output of code with images of relative transformation between adjacent frames")
parser.add_argument("--outputtxt", type=str, default = "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video06/masks_transform_txt")


args = parser.parse_args()

INPUT_PATH= args.inputfolder
OUTPUT_PATH= args.outputpath
OUTPUT_PATH_TXT= args.outputtxt

if __name__ == "__main__":
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(OUTPUT_PATH +' created')
    else:
        print(OUTPUT_PATH +' exists')

    if not os.path.exists(OUTPUT_PATH_TXT):
        os.makedirs(OUTPUT_PATH_TXT)
        print(OUTPUT_PATH_TXT +' created')
    else:
        print(OUTPUT_PATH_TXT +' exists')
        
    input_file_list = sorted(glob(INPUT_PATH + "/*.png"))
    
    for i in range (len(input_file_list)-1):
        img1 =cv2.imread(input_file_list[i])
        img2 =cv2.imread(input_file_list[i+1])

        #convert to grey image
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

        warp_matrix = np.eye(2, 3, dtype=np.float32) 
        warp_mode = cv2.MOTION_AFFINE #set an affine motion model
        imgshape= img1.shape

        # taken from the internet
        termination_eps = 1e-10
        number_of_iterations = 5000
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

        #I swapped img2 and img1. because first param is template image, second is input img --> want img 1 to look like 2
        retval, warpmatrix= cv2.findTransformECC(img2, img1, warp_matrix, warp_mode, criteria) #the default is an affine model
        
        im2_aligned = cv2.warpAffine(img1, warpmatrix, (imgshape[1],imgshape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        #append the 001 to the bottom of affine
        warpmatrix= np.append(warpmatrix, [0,0,1])
        warpmatrix= np.resize(warpmatrix,(3,3))
        
        #file naming convention 
        basename= os.path.basename(input_file_list[i])
        front= os.path.splitext(basename)[0]
        basename2= os.path.basename(input_file_list[i+1])
        backk= os.path.splitext(basename2)[0]
        back= backk.split("_")[1]
        filename= front+'_'+back
        cv2.imwrite(OUTPUT_PATH+'/'+filename+'.png', im2_aligned)
        result = np.savetxt(OUTPUT_PATH_TXT + "/" + filename+'.txt' , warpmatrix)
