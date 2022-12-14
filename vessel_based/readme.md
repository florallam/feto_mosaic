# Vessel Based Approach
This section references the [Robust fetoscopic mosaicking from deep learned flow fields](https://link.springer.com/article/10.1007/s11548-022-02623-1) paper <br>
as well as Sophia Bano's [repository](https://github.com/sophiabano/EndoVis-FetReg2021/blob/033f40826ee7e50b61a76ad929efb66fb67bb8f6/FetReg-registration-docker-example/code/main.py) for the purpose of obtaining the homography files as well as [visualisation](https://github.com/sophiabano/EndoVis-FetReg2021/blob/master/Visualisation/Registration-Mosaic/fetreg2021_registration_vis.py) purposes. <br>

## createframelist.py
- Obtain framelist for the framelistmetric input

## direct_img_reg.py
- Obtain vessel based image transformations as well as homography txt values

## framelistmetric.py
- Evaluate the performance of vessel based approach

## ssim_processing.py
- Has required functions referenced by framelistmetric.py

## visualiseboxplots.py
- Obtain boxplots from the results obtained from framelistmetric.py

## visualise_transform.py
- Visualise the transformations for vessel based approach in the form of a video
