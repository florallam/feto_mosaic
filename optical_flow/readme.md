# Optical Flow Approach
The codes in this folder are reproduced from the [Deep Flow Fields For Fetoscopic Mosaicking](https://github.com/labdeeman7/deepFlowFieldsForFetoscopeMosaicing) repository with slight modifications to remove the dark circle outlines around the mosaics in the ```get_flowNet_2_mosaic.ipynb``` code. <br>

## Files:
1. get_optical_flow_with_flownet_2.ipynb
- To obtain flo files <br>
2. get_flowNet_2_mosaic.ipynb
- To obtain mosaic video
- modified to crop the mosaic to remove the dark circles around the mosaic (previously it was done using enblend, but time was rather tight!)
