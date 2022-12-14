"""
  Obtain the boxplots for the framelistmetric SSIM values
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime

df1 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video02/anon002_ssim.csv")#.assign(Trial=1)
ax = sns.boxplot(data=df1, orient='v')
ax.set_ylim([0, 1.01]) 
plt.xlabel('Frame Index')
plt.ylabel('Structural Similarity Index (SSIM)')
plt.title("Video 2 (300 Frames)")
now= datetime.today()
ddmmyy= now.strftime('%d%m%y')
time= now.strftime('%H%M')
plt.savefig('anon002_ssimboxplot_'+"_"+ddmmyy+"_"+time+'.png')
