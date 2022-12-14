"""
  Obtain square images from circular photos captured from the fetoscope
"""
import cv2
import numpy as np
from glob import glob
import math

#from https://stackoverflow.com/questions/10469235/opencv-apply-mask-to-a-color-image

def get_mask_dim(image):
  oneimg= cv2.imread(image)
  img = cv2.cvtColor(oneimg, cv2.COLOR_BGR2RGB)
  # detect circles
  gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=50, minRadius=0, maxRadius=0)
  circles = np.uint16(np.around(circles))[0][0]
  print(circles)
  return circles

def get_circle(image, circledim):
  oneimg= cv2.imread(image)
  img = cv2.cvtColor(oneimg, cv2.COLOR_BGR2RGB)

  # draw mask
  center= round(img.shape[0]/2)
  mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # mask is only 
  cv2.circle(mask, (center, center), circledim[2], (255, 0, 0), -1)

  # get first masked value (foreground)
  fg = cv2.bitwise_or(img, img, mask=mask)

  # get second masked value (background) mask must be inverted
  mask = cv2.bitwise_not(mask)
  background = np.full(img.shape, 255, dtype=np.uint8)
  bk = cv2.bitwise_or(background, background, mask=mask)

  # combine foreground+background
  final = cv2.bitwise_or(fg, bk)
  return final
  # cv2_imshow(final)

def crop_square(circles, final):
  square= round(math.sqrt(2) * (circles[2]))
  centerx= circles[0]
  centery= circles[1]
  edgex= round(centerx-(square/2))
  edgey= round(centery-(square/2))

  endx= round(centerx+(square/2))
  endy= round(centery+(square/2))
  crop_img = final[edgex:endx, edgey:endy]
  return crop_img

drive_root= "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01"
images = glob(drive_root+"/images/*png")
save_dir= "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/cropped_images"
circles= get_mask_dim(images[0])
img= cv2.imread(images[0])
mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8)  # mask is only 
cv2.circle(mask, (circles[0], circles[1]), circles[2], (255, 0, 0), -1)
cv2.imwrite(save_dir+ "mask.png", mask)

for i in images:
  final= get_circle(i, circles)
  cropped= crop_square(circles, final)
  path= save_dir +"cropped_images/"+ i.rsplit('/', 1)[1]
  cv2.imwrite(path, cropped)
print("done!")

