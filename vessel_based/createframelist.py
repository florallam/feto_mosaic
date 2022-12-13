"""
    Generate framelist txt file for the framelistmetric.py input
"""
import pandas as pd
import os
from tqdm import tqdm

    images_path = "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video06/masks_transform"
    length= len(os.listdir(images_path))-1
    
 if __name__ == "__main__":   
    for i in tqdm(range(1,6)):
        savetxt= "/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video06/framelist_"+ str(i)+".txt"
        with open(savetxt, 'w') as fp:
            for j in range(0, length):
                firstnum= j
                secondnum= j+i
                if secondnum <= length:
                    fp.write("%s %s\n" % (firstnum,secondnum))
                    
                    
