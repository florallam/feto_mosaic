import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime

#for boxplots
df1 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/anon001_ssim_dsm.csv")
df2 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video02/anon002_ssim_dsm.csv")
df3 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video03/anon003_ssim_dsm.csv")
df4 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video04/anon005_ssim_dsm.csv")
df5 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video05/anon010_ssim_dsm.csv")
df6 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video06/anon012_ssim_dsm.csv")
df11 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video01/anon001_ssim.csv")
df12 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video02/anon002_ssim.csv")
df13 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video03/anon003_ssim.csv")
df14 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video04/anon005_ssim.csv")
df15 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video05/anon010_ssim.csv")
df16 = pd.read_csv("/home/flora/data/Fetoscopy Placenta Dataset/Vessel_registration_unannotated_clips/video06/anon012_ssim.csv")
df21 = pd.read_csv("/home/flora/data/anon001_ssim.csv")
df22 = pd.read_csv("/home/flora/data/anon002_ssim.csv")
df23 = pd.read_csv("/home/flora/data/anon003_ssim.csv")
df24 = pd.read_csv("/home/flora/data/anon005_ssim.csv")
df25 = pd.read_csv("/home/flora/data/anon010_ssim.csv")
df26 = pd.read_csv("/home/flora/data/anon012_ssim.csv")

#for individual boxplots
def plotbox():
    fig, ax = plt.subplots(2,3, figsize=(25,10))
    fig.tight_layout(pad=5.0)
    sns.boxplot(data=df21, orient='v', ax= ax[0,0])
    ax[0,0].set_title('Video 1 (400 Frames)', fontsize=20)
    sns.boxplot(data=df22, orient='v', ax= ax[0,1])
    ax[0,1].set_title('Video 2 (200 Frames)', fontsize=20)
    sns.boxplot(data=df23, orient='v', ax= ax[0,2])
    ax[0,2].set_title('Video 3 (49 Frames)', fontsize=20)
    sns.boxplot(data=df24, orient='v', ax=ax[1,0])
    ax[1,0].set_title('Video 4 (100 Frames)', fontsize=20)
    sns.boxplot(data=df25, orient='v', ax=ax[1,1])
    ax[1,1].set_title('Video 5 (100 Frames)', fontsize=20)
    sns.boxplot(data=df26, orient='v', ax=ax[1,2])
    ax[1,2].set_title('Video 6 (100 Frames)', fontsize=20)
    for i in ax.flat:
        i.set(xlabel='Frame Index', ylabel='Structural Similarity Index (SSIM)')
        i.xaxis.label.set_size(15)
        i.yaxis.label.set_size(15)
    for j in ax.flat:
        j.set_ylim([0, 1.01]) 
    now= datetime.today()
    ddmmyy= now.strftime('%d%m%y')
    time= now.strftime('%H%M')
    # fig.tight_layout(pad=5.0)
    fig.savefig('optical_ssimboxplot'+"_"+ddmmyy+"_"+time+'.png')

#for individual time plots
def plottime():
    fig1, ax1 = plt.subplots(2,3, figsize=(25,10))
    fig1.tight_layout(pad=5.0)
    sns.lineplot(data=df21['1'], ax= ax1[0,0])
    ax1[0,0].set_title('Video 1 (400 Frames)', fontsize=20)
    sns.lineplot(data=df22['1'], ax= ax1[0,1])
    ax1[0,1].set_title('Video 2 (200 Frames)',fontsize=20)
    sns.lineplot(data=df23['1'], ax= ax1[0,2])
    ax1[0,2].set_title('Video 3 (49 Frames)',fontsize=20)
    sns.lineplot(data=df24['1'], ax=ax1[1,0])
    ax1[1,0].set_title('Video 4 (100 Frames)',fontsize=20)
    sns.lineplot(data=df25['1'], ax=ax1[1,1])
    ax1[1,1].set_title('Video 5 (100 Frames)',fontsize=20)
    sns.lineplot(data=df26['1'], ax=ax1[1,2])
    ax1[1,2].set_title('Video 6 (100 Frames)',fontsize=20)
    for i in ax1.flat:
        i.set(xlabel='Frame Index', ylabel='Structural Similarity Index (SSIM)')
        i.xaxis.label.set_size(15)
        i.yaxis.label.set_size(15)
    for j in ax1.flat:
        j.set_ylim([0, 1.01]) 
    now= datetime.today()
    ddmmyy= now.strftime('%d%m%y')
    time= now.strftime('%H%M')
    fig1.savefig('vessel_ssimtimeplot'+"_"+ddmmyy+"_"+time+'.png')

#get overall boxplot
def getallboxplot():
    fig2, ax2 = plt.subplots(2,3, figsize=(25,10))
    fig2.tight_layout(pad=7.0)
    df1= pd.melt(df1)
    df11= pd.melt(df11)
    df21= pd.melt(df21)
    df1['ttype']= 'dsm'
    df11['ttype'] = 'vessel'
    df21['ttype'] = 'optical'
    df31= pd.concat([df1, df11, df21], ignore_index=True)
    sns.boxplot(hue='ttype', x='variable', y='value',data=df31, ax= ax2[0,0])
    ax2[0,0].get_legend().remove()


    df2= pd.melt(df2)
    df12= pd.melt(df12)
    df22= pd.melt(df22)
    df2['ttype']= 'dsm'
    df12['ttype'] = 'vessel'
    df22['ttype'] = 'optical'
    df32= pd.concat([df2, df12, df22], ignore_index=True)
    sns.boxplot(hue='ttype', x='variable', y='value',data=df32, ax= ax2[0,1])
    ax2[0,1].get_legend().remove()

    df3= pd.melt(df3)
    df13= pd.melt(df13)
    df23= pd.melt(df23)
    df3['ttype']= 'dsm'
    df13['ttype'] = 'vessel'
    df23['ttype'] = 'optical'
    df33= pd.concat([df3, df13, df23], ignore_index=True)
    sns.boxplot(hue='ttype', x='variable', y='value',data=df33, ax= ax2[0,2])
    ax2[0,2].get_legend().remove()

    df4= pd.melt(df4)
    df14= pd.melt(df14)
    df24= pd.melt(df24)
    df4['ttype']= 'dsm'
    df14['ttype'] = 'vessel'
    df24['ttype'] = 'optical'
    df34= pd.concat([df4, df14, df24], ignore_index=True)
    sns.boxplot(hue='ttype', x='variable', y='value',data=df34, ax= ax2[1,0])
    ax2[1,0].get_legend().remove()

    df5= pd.melt(df5)
    df15= pd.melt(df15)
    df25= pd.melt(df25)
    df5['ttype']= 'dsm'
    df15['ttype'] = 'vessel'
    df25['ttype'] = 'optical'
    df35= pd.concat([df5, df15, df25], ignore_index=True)
    sns.boxplot(hue='ttype', x='variable', y='value',data=df35, ax= ax2[1,1])
    ax2[1,1].get_legend().remove()

    df6= pd.melt(df6)
    df16= pd.melt(df16)
    df26= pd.melt(df26)
    df6['ttype']= 'dsm'
    df16['ttype'] = 'vessel'
    df26['ttype'] = 'optical'
    df36= pd.concat([df6, df16, df26], ignore_index=True)
    sns.boxplot(hue='ttype', x='variable', y='value',data=df36, ax= ax2[1,2])
    # ax2[0,0].get_legend().remove()
    handles, labels = ax2[1,2].get_legend_handles_labels()
    ax2[1,2].get_legend().remove()

    ax2[0,0].set_title('Video 1 (400 Frames)', fontsize = 20)
    ax2[0,1].set_title('Video 2 (200 Frames)', fontsize = 20)
    ax2[0,2].set_title('Video 3 (49 Frames)', fontsize = 20)
    ax2[1,0].set_title('Video 4 (100 Frames)', fontsize = 20)
    ax2[1,1].set_title('Video 5 (100 Frames)', fontsize = 20)
    ax2[1,2].set_title('Video 6 (100 Frames)', fontsize = 20)
    for i in ax2.flat:
        i.set(xlabel='Frame Index', ylabel='Structural Similarity Index (SSIM)')
        i.xaxis.label.set_size(20)
        i.yaxis.label.set_size(20)

        # i.rcParams.update({'font.size': 10})
    for j in ax2.flat:
        j.set_ylim([0, 1.01]) 
        # j.rcParams.update({'font.size': 10})
    # fig2.rc({'font.size': 10})

    fig2.legend(handles, labels, loc='upper center', ncol=3, fontsize = 20)#, bbox_to_anchor=(.75, 0.98))
    fig2.savefig('allboxplot.png')

# getallboxplot()

#for lineplot (all)
def getalllineplot():
    fig3, ax3 = plt.subplots(2,3, figsize=(25,10))
    fig3.tight_layout(pad=5.0)
    df1['ttype']= 'dsm'
    df11['ttype'] = 'vessel'
    df21['ttype'] = 'optical'
    df1['frame index']= df1.index
    df11['frame index']= df11.index
    df21['frame index']= df21.index
    df31= pd.concat([df1, df11, df21], ignore_index=True)
    sns.lineplot(hue='ttype', x='frame index', y='1',data=df31, ax= ax3[0,0])
    ax3[0,0].get_legend().remove()

    df2['ttype']= 'dsm'
    df12['ttype'] = 'vessel'
    df22['ttype'] = 'optical'
    df2['frame index']= df2.index
    df12['frame index']= df12.index
    df22['frame index']= df22.index
    df32= pd.concat([df2, df12, df22], ignore_index=True)
    sns.lineplot(hue='ttype', x='frame index', y='1',data=df32, ax= ax3[0,1])
    ax3[0,1].get_legend().remove()

    df3['ttype']= 'dsm'
    df13['ttype'] = 'vessel'
    df23['ttype'] = 'optical'
    df3['frame index']= df3.index
    df13['frame index']= df13.index
    df23['frame index']= df23.index
    df33= pd.concat([df3, df13, df23], ignore_index=True)
    sns.lineplot(hue='ttype', x='frame index', y='1',data=df33, ax= ax3[0,2])
    ax3[0,2].get_legend().remove()

    df4['ttype']= 'dsm'
    df14['ttype'] = 'vessel'
    df24['ttype'] = 'optical'
    df4['frame index']= df4.index
    df14['frame index']= df14.index
    df24['frame index']= df24.index
    df34= pd.concat([df4, df14, df24], ignore_index=True)
    sns.lineplot(hue='ttype', x='frame index', y='1',data=df34, ax= ax3[1,0])
    ax3[1,0].get_legend().remove()

    df5['ttype']= 'dsm'
    df15['ttype'] = 'vessel'
    df25['ttype'] = 'optical'
    df5['frame index']= df5.index
    df15['frame index']= df15.index
    df25['frame index']= df25.index
    df35= pd.concat([df5, df15, df25], ignore_index=True)
    sns.lineplot(hue='ttype', x='frame index', y='1',data=df35, ax= ax3[1,1])
    ax3[1,1].get_legend().remove()

    df6['ttype']= 'dsm'
    df16['ttype'] = 'vessel'
    df26['ttype'] = 'optical'
    df6['frame index']= df6.index
    df16['frame index']= df16.index
    df26['frame index']= df26.index
    df36= pd.concat([df6, df16, df26], ignore_index=True)
    sns.lineplot(hue='ttype', x='frame index', y='1',data=df36, ax= ax3[1,2])

    ax3[0,0].set_title('Video 1 (400 Frames)', fontsize = 15)
    ax3[0,1].set_title('Video 2 (200 Frames)', fontsize = 15)
    ax3[0,2].set_title('Video 3 (49 Frames)', fontsize = 15)
    ax3[1,0].set_title('Video 4 (100 Frames)', fontsize = 15)
    ax3[1,1].set_title('Video 5 (100 Frames)', fontsize = 15)
    ax3[1,2].set_title('Video 6 (100 Frames)', fontsize = 15)

    for i in ax3.flat:
        i.set(xlabel='Frame Index', ylabel='Structural Similarity Index (SSIM)')
        i.xaxis.label.set_size(15)
        i.yaxis.label.set_size(15)
        i.set_ylim([0, 1.01]) 
    handles, labels = ax3[1,2].get_legend_handles_labels()
    ax3[1,2].get_legend().remove()
    fig3.legend(handles, labels, loc='upper center', ncol=3, fontsize= 15)#, bbox_to_anchor=(.75, 0.98))
    fig3.savefig('alllineplot.png')