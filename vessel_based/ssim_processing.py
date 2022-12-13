import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
cv2.ocl.setUseOpenCL(False)
import sys
from PIL import Image
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from glob import glob  # For listing files
import os
from os import listdir
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from datetime import datetime

# import flow_file_processing as fp
# import visualize as vis



def getWarpedSrcImg(srcImg, H, showImages=True):
    """
    Get warped image, for comparison
    :param srcImg: image to be warped back to the destination
    :param H: the H matrix
    :param showImages: display or not
    :return: warped image
    """

    sh = srcImg.shape
    sh = (448, 448)
    ht = sh[0]
    wd = sh[1]
    ww = wd
    hh = ht

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    result = cv2.warpPerspective(srcImg, H, (ww, hh))

    if showImages:
        plt.figure(figsize=(10, 4))
        plt.imshow(result)

        plt.show()

    return result


def getIntersection(warp_srcImg, destImg, showImages):
    """
    Obsolete function currently, was to be used to find the square intersection pixels between two circular images.
    Planned to work more on this and improve it to not just square intersection but any shape intersection
    :param warp_srcImg:
    :param destImg:
    :param showImages:
    :return:
    """
    warp_srcImg_gray = cv2.cvtColor(warp_srcImg, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero(warp_srcImg_gray)  # Find all non-zero points (text)

    x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
    cropped_src_img = warp_srcImg[y:y + h, x:x + w]  # Crop the image - note we do this on the original image

    if showImages:
        print("cropped_src_img")
        plt.figure(figsize=(10, 4))
        plt.imshow(cropped_src_img)
        plt.show()

    color = (0, 0, 0)
    masked_dest_img = np.full(destImg.shape, color, dtype=np.uint8)
    I, J = np.transpose(coords)
    masked_dest_img[J, I] = destImg[J, I]
    cropped_dest_img = masked_dest_img[y:y + h, x:x + w]

    if showImages:
        print("cropped_dest_img")
        plt.figure(figsize=(10, 4))
        plt.imshow(cropped_dest_img)
        plt.show()

    return [cropped_src_img, cropped_dest_img]


def getSSIM(src_img, dest_img):
    """
    Get SSIM between two images, note that with a little change to the function ssim , chaning a default parameter
    can get you SSIM map as well. YOu can also use multichannel versions, but this configurations worked best for me.
    :param src_img: src_img
    :param dest_img: dest_img
    :return: SSIM Value.
    """
    src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
    dest_img = cv2.cvtColor(dest_img, cv2.COLOR_RGB2GRAY)

    src_img = cv2.GaussianBlur(src_img, (9, 9), 2)
    dest_img = cv2.GaussianBlur(dest_img, (9, 9), 2)

    ssim_value = ssim(src_img, dest_img)
    # ssim_value = ssim(src_img, dest_img, multichannel = True)

    return ssim_value

def getSSIMForRegOverlap(src_img, dest_img,H,mask,squareLength,showImages):
    # print("warp mask ongoing.")
    warp_mask   = getWarpedSrcImg(mask, H, showImages)
    # print("done with warp mask")
    overlap_mask = cv2.bitwise_and(mask, warp_mask)
    warp_srcImg = getWarpedSrcImg(src_img, H, showImages)

    overlap_ratio = overlap_mask.sum()/mask.sum()

    cropped_warp_srcImg = get_square_in_image(warp_srcImg, squareLength, showImages)
    cropped_destImg = get_square_in_image(dest_img, squareLength, showImages)
    ssim_value = getSSIM(cropped_warp_srcImg, cropped_destImg)
    return ssim_value, overlap_ratio

def getHRefToBase(H_frame_distance, transformation):
    """
    Gets transformation to the first frame for a particular distance
    :param H_frame_distance: a distance(e.g 5) x transformation array that contains all pairwise transformations for that length
    :param transformation: Affine or Homography
    :return: returns transformation to base frame.
    """
    frame_distance = len(H_frame_distance)
    H_ref_to_base = np.zeros((frame_distance, 3, 3))

    for i in range(1, frame_distance + 1):
        new_H = np.eye(3);

        if transformation == "Homography":
            for j in range(i):
                new_H = np.matmul(new_H, H_frame_distance[j])
        elif transformation == "Affine":
            for j in range(i):
                new_H = np.matmul(new_H, np.vstack([H_frame_distance[j], [0, 0, 1]]))
        H_ref_to_base[i - 1] = new_H

    return H_ref_to_base

def getHRefToBase(H_frame_distance, transformation):
    """
    Gets transformation to the first frame for a particular distance
    :param H_frame_distance: a distance(e.g 5) x transformation array that contains all pairwise transformations for that length
    :param transformation: Affine or Homography
    :return: returns transformation to base frame.
    """
    frame_distance = len(H_frame_distance)
    H_ref_to_base = np.zeros((frame_distance, 3, 3))

    for i in range(1, frame_distance + 1):
        new_H = np.eye(3);

        if transformation == "Homography":
            for j in range(i):
                new_H = np.matmul(new_H, H_frame_distance[j])
        elif transformation == "Affine":
            for j in range(i):
                new_H = np.matmul(new_H, np.vstack([H_frame_distance[j], [0, 0, 1]]))
        H_ref_to_base[i - 1] = new_H

    return H_ref_to_base

# got from Sophia's visualisation code
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
    #print(H_indv)
        
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

def getSSIMForRegOverlapFrameList(frame_list, H_array, mask, transformation, squareLength, imgPaths, showImages):
    """
    Get SSIM for a given list of frame pairsframe distance away.
    :param frame_list: numpy Nx2 matrix with frame indices.
    :param H_array: pair wise transformation
    :param transformation: Affine or Homography
    :param squareLength: length of square which we would crop and calculate SSIM for
    :param imgPaths: paths to the image
    :param showImages: display flag
    :return: matrix of SSIM values.
    """

    ssimVector    = np.zeros(frame_list.shape[0])
    overlapVector = np.zeros(frame_list.shape[0])

    for i in np.arange(0,len(ssimVector)):
        begin = frame_list[i,0]
        end   = frame_list[i,1]
        destImgPath = imgPaths[begin]  # the previous image
        srcImgPath = imgPaths[end];
        total_H = np.eye(3)
        for j in np.arange(begin,end): # r
            if transformation == "Homography":
                total_H = np.matmul(total_H, H_array[j])
            elif transformation == "Affine":
                total_H = np.matmul(total_H, np.vstack([H_array[j], [0, 0, 1]]))
        
        srcImg, destImg = inputAndVisualizeStitchPair(srcImgPath, destImgPath, showImages)
        # srcImg, destImg = fp.inputAndVisualizeStitchPair(srcImgPath, destImgPath, showImages)
        ssim_value, overlap_ratio = getSSIMForRegOverlap(srcImg, destImg, total_H, mask, squareLength, showImages)
        ssimVector[i] = ssim_value
        overlapVector[i] = overlap_ratio

    return ssimVector, overlapVector

#got from https://github.com/sophiabano/FetReg2021/blob/master/fetreg-reg/flow_file_processing.py
def inputAndVisualizeStitchPair(srcImgPath, destImgPath, showImages=True):
    """
    :param srcImgPath: path to image to be projected back to the destination image
    :param destImgPath: path to the image which is the destination
    :param showImages: bool to show images or not
    :return: returns src image and dest image pair
    """
    srcImg = cv2.imread(srcImgPath)
    srcImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2RGB)
    srcImg_gray = cv2.cvtColor(srcImg, cv2.COLOR_RGB2GRAY)

    destImg = cv2.imread(destImgPath)
    destImg = cv2.cvtColor(destImg, cv2.COLOR_BGR2RGB)
    destImg_gray = cv2.cvtColor(destImg, cv2.COLOR_RGB2GRAY)

    if showImages:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(10, 4))
        ax1.imshow(destImg, cmap="gray")
        ax1.set_xlabel("dest image", fontsize=14)

        ax2.imshow(srcImg, cmap="gray")
        ax2.set_xlabel("Src image (Image to be transformed)", fontsize=14)

        plt.show()
    return srcImg, destImg
    # return [srcImg, destImg] #don't think that it should be a list?

def getSSIMForRegOverlapFrameDistance(window_size, frame_distance, H_array, mask, transformation, squareLength, imgPaths, showImages):
    """
    Get SSIM for a given frame distance away.
    :param window_size: size of stride, normally should be 1.
    :param frame_distance: normally should be 6
    :param H_array: pair wise transformation
    :param transformation: Affine or Homography
    :param squareLength: length of square which we would crop and calculate SSIM for
    :param imgPaths: paths to the image
    :param showImages: display flag
    :return: matrix of SSIM values.
    """

    r = (len(imgPaths) - frame_distance) // window_size
    ssimMatrix    = np.zeros((r, frame_distance))
    overlapMatrix = np.zeros((r, frame_distance))

    if transformation == "Homography":
        H_frame_distance = np.zeros((frame_distance, 3, 3))
    elif transformation == "Affine":
        H_frame_distance = np.zeros((frame_distance, 2, 3))

    # for i in range(r):  # r
    for i in tqdm(range(r)):
        begin = i
        end = i + frame_distance
        img_indexes = np.arange(begin, end + 1)
        # print("img_indexes",img_indexes)
        img_indexes_paths = [imgPaths[i] for i in img_indexes]
        # print("img_indexes_paths", img_indexes_paths)

        # fill up H_ref_to_base for this list
        H_frame_distance = H_array[begin:end]

        # Get affine matrix (done in visualisation code too)
        H_affine = np.zeros( (len(H_array), 2,3))
        for m in range(len(H_array)):
            H = H_array[m,:,:]
            H_affine[m] =  H[:2, :]

        # H_ref_to_base = getHRefToBase(H_frame_distance, transformation)
        H_ref_to_base = getHRefToBase(H_affine, transformation)

        for j in range(frame_distance):
            destImgPath = img_indexes_paths[0]  # the previous image
            srcImgPath = img_indexes_paths[j + 1]

            #removed fp in front of inputAndVisualizeStitchPair
            # print(srcImgPath, destImgPath)
            # srcImg, destImg = inputAndVisualizeStitchPair(srcImgPath, destImgPath, showImages)
            # print(srcImg)
            srcImg= cv2.imread(srcImgPath)
            # print(srcImg.shape)
            destImg= cv2.imread(destImgPath)
            # exit()
            ssim_value, overlap_ratio = getSSIMForRegOverlap(srcImg, destImg, H_ref_to_base[j], mask, squareLength, showImages)
            ssimMatrix[i, j] = ssim_value
            overlapMatrix[i] = overlap_ratio
            #overlapVector[i] = overlap_ratio
    return ssimMatrix, overlapMatrix

def findDenseTransformation(srcImg, destImg):
  """
  Get Lucas Kanade pyrammidal dense transformation, between a srcImage and a destimage, using a contrib branch of opencv.
  I do not know why I added a status there, I think it was to help with errors
  but I would check and remove it if it is not needed
  Improvement on this would be to perform the pyramidal transformation myself which would allow me to be able to uses
  our circular mask. But currently it works.
  :param srcImg: srcImg
  :param destImg: destImg
  :return: Dense transformation
  """

  mapper = cv2.reg_MapperGradAffine()
  mapperPyramid = cv2.reg_MapperPyramid(mapper)
  # mapperPyramid.numIterPerScale_ = 3
  # mapperPyramid.numLev_ = 3

  result_pointer = mapperPyramid.calculate(srcImg.astype(float), destImg.astype(float))
  result_array = cv2.reg.MapTypeCaster_toAffine(result_pointer)

  H = np.concatenate([result_array.getLinTr(), result_array.getShift()], axis = 1)
  status = False
  return [H, status]


def get_square_in_image(image, squareLength, showImages):
    """
    Get square images in the middle of warped src image and destination image for performing SSIM
    currently in use, if time allows would change to any shape of intersection via getIntersection()
    :param image: image
    :param squareLength: middle square length to be considered
    :param showImages: display flag
    :return: return square image.
    """
    h_start = int((image.shape[0] - squareLength) / 2)
    w_start = int((image.shape[1] - squareLength) / 2)

    crop_img = image[h_start:h_start + squareLength, w_start:w_start + squareLength]
    if showImages:
        print("cropped Image...............")
        vis.visualizeImg(crop_img)

    return crop_img

#added this in
def getmask(mask_path, img_1):
    mask_im = Image.open(mask_path)
    mask_im = mask_im.resize((img_1.shape[0], img_1.shape[1]), Image.ANTIALIAS)  
    mask_im = np.array(mask_im)
    mask_im = mask_im * np.uint8(255)

    mask_im = cv2.resize(mask_im, img_1.shape[1::-1])
    return mask_im

