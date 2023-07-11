# Dental Assistant with AI

This project aims to extract dental information on radiograph images using AI. 
Utilized MaskRCNN model to detect and segment teeth and post-process algortihm to numbering teeth according to the FDI numbering rule.
Utilized mmdetection framework for instance segmentation model.

![fdi](https://github.com/7ZFG1/Dental-Assistant-with-AI/assets/64545114/272621ae-d212-4ab4-b528-9167b08a5728)

```AI_Engine.py``` runs MaskRCNN model to detect and segment teeth. The model trained 8 class. For example, there are 4 molars in the mouth and this molar is labeled and trained as label 8 regardless of the region of the molar.

```dental_postprocess.py``` finds teeth region and missing teeth using AI model outputs. The code calculates the distance from the center point of each tooth to the corner points of the image, calculates in which region the teeth should be, and assigns regions to the teeth. Thus, the teeth are named according to the FDI numbering system. 


You can run all the project by following command ```python3 dental_main.py```

Images result will be saved in ```result_images``` folder and information result will be saved ```dental_report.txt```.

![1](https://github.com/7ZFG1/Dental-Assistant-with-AI/assets/64545114/77b0e8ab-87bc-4d70-a57a-ad9f09d2fc45)
![2](https://github.com/7ZFG1/Dental-Assistant-with-AI/assets/64545114/a79ec7b6-d60c-4998-8d1b-35976201b254)
![3](https://github.com/7ZFG1/Dental-Assistant-with-AI/assets/64545114/5915891f-5da8-4d2d-9215-ab2d452c0109)
![4](https://github.com/7ZFG1/Dental-Assistant-with-AI/assets/64545114/edff5051-32b7-4b32-aa9f-c9950bc3fc70)


>Dental Report
-------------------------------------
----IMAGE: 1.png----
Missing Teeth: 11 12 18 21 22 31 38 
Total Number of Teeth: 25

----IMAGE: 2.png----
Missing Teeth: 11 12 22 28 41 
Total Number of Teeth: 27

----IMAGE: 3.png----
Missing Teeth: 11 12 13 14 15 16 17 18 21 22 23 24 25 26 27 28 31 32 33 34 35 36 41 42 43 44 45 46 47 48 
Total Number of Teeth: 2

----IMAGE: 25.png----
Missing Teeth: 0
Total Number of Teeth: 32

TODO
---------------------------
Segmented areas is not appeared in result images even if the model is instance segmentance model. This is because the model output cannot be parsed correctly. This bug will be fixed.
