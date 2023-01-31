'''
    Generate framelist txt file
'''
import pandas as pd
import os
from tqdm import tqdm

if __name__ == "__main__":
    vals= 5
    images_path = "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video06/masks_transform"
    # savetxt= "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/framelist_2.txt"
    length= len(os.listdir(images_path))-1
    
    for i in tqdm(range(1,6)):
        savetxt= "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video06/framelist_"+ str(i)+".txt"
        with open(savetxt, 'w') as fp:
            for j in range(0, length):
                firstnum= j
                secondnum= j+i

                if secondnum <= length:
                    fp.write("%s %s\n" % (firstnum,secondnum))
        # print('Done')