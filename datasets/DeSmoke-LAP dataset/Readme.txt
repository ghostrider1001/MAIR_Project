Dataset Description
--------------------
Laparoscopic surgery dataset contains two folders. Information of data organisation is explained \cite{Pan2022DeSmokeLAP} and the dataset is associated with the proposed model, DeSmoke-LAP, shared on Github (https://github.com/yiroup20/DeSmoke-LAP). Details of each folder is as follows:


1. Dataset
-----------------------------------
The folder contains frames applied for training and evaluation process in our research. Data was collected from 10 robot-assisted laparoscopic hysterectomy procedure recordings, which were decomposed into frames at 1 fps. From each procedure, 300 hazy images and 300 clear images were manually selected. A short video clip of 50 frames from each procedure was also selected that was utilised for testing. Thus, for each video folder, there are three sub-folders, containing clear, hazy and test clip. For further details, please refer to \cite{Pan2022DeSmokeLAP}.

The following number of frames are included in each sub-folder:
TLH_2  - 650 frames
TLH_6  - 650 frames
TLH_7  - 650 frames
TLH_8  - 650 frames
TLH_9  - 650 frames
TLH_10 - 650 frames
TLH_11 - 650 frames
TLH_12 - 650 frames
TLH_16 - 650 frames
TLH_17 - 650 frames


2. Video_Clips_Evaluation
-----------------------------------------
The folder contains video clips showing the output of different desmoking methods. There are 6 sub-folders and each represents a unique method including the original unprocessed data. In each sub-folder named by methods, outputs on the 10 test video clips are shown. A presentation video (https://youtu.be/JCdQNbg0WqY) is also added along with sub-folders, which explains the details between outputs by different methods.

The following number of videos are included in each sub-folder:
Input        -  6 videos
CycleGAN     -  6 videos
FastCUT      -  6 videos
Colores      -  6 videos
Cycle-Dehaze -  6 videos
DeSmoke-LAP  -  6 videos


License
--------
The Laparoscopic Surgery Dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
https://creativecommons.org/licenses/by-nc-sa/4.0/


Reference
----------
We request to cite the following publication whenever research making use of this dataset is reported in any academic publication or research report:

@article{Pan2022DeSmokeLAP,
  title={DeSmoke-LAP: Improved Unpaired Image-to-Image Translation for Desmoking in Laparoscopic Surgery},
  author={Bano, Sophia and Vasconcelos, Francisco and Park, Hyun and Jeong, Taikyeong Ted and Stoyanov, Danail},
  journal={International journal of computer assisted radiology and surgery},
  year={2022},
  publisher={Springer}
}


Contact
-------
For comments, suggestions or feedback, or if you experience any problems with this website or the
dataset, please contact Yirou Pan (yirou.pan.20@ucl.ac.uk) and Sophia Bano (sophia.bano@ucl.ac.uk).


