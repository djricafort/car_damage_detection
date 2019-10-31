# car_damage_detection
This is an automated car scratch detection using [Mask R-CNN](https://github.com/matterport/Mask_RCNN). This code follows the guide in this [link.](https://towardsdatascience.com/cnn-application-detecting-car-exterior-damage-full-implementable-code-1b205e3cb48c)

## How to use:
1. Run setup.py
``` bash
pip install .
``` 
2. Run demo.py to test it in your terminal or run app.py to run it in your browser

3. To use as api:
```python
from damage_detector.detector import Detector
if __name__ == '__main__':
    image_path = '<input image path>'
    Detector.detect_scratches(image_path)
```

**if an error is encountered due to your tensorflow version, please install tensorflow version 1.14.0

## When running using demo.py
1. Run demo.py:
``` bash
python demo.py
```
2. Enter image directory.

3. Check the ouput inside `/static/predictions/` folder.

## When running using app.py
1. Run app.py :
``` bash
python app.py
``` 
2. Open `http://127.0.0.1:5000/` in your browser. The landing page should look like the image below:

![landing page](https://drive.google.com/uc?export=view&id=1ELTir5W1QRL-N2sOjB1S2rPVnFlLqVWI)

3. Click the "Select Image" button and select a car image where you would like to detect damages. Select `.jpg` image file formats only.
![landing page](https://drive.google.com/uc?export=view&id=16J3X37fGULoRmghV0xgoOEK_d_kzEoDu)

4. Click the "Detect!" button to start processing your image.
![landing page](https://drive.google.com/uc?export=view&id=181Qwcr0Qk1LvGHtSAusQQfv1cT7FbMQN)

5. A preview will be shown showing the detected damage from the car image:
![landing page](https://drive.google.com/uc?export=view&id=1oi0Q7V1-Hk_hETGeBqguk1xa6NWdUJTB)

6. The image with detections will also be saved locally and can be found inside `/static/predictions/` folder.
![landing page](https://drive.google.com/uc?export=view&id=1ecXra_VlaQVxEHxq1Ujj7j4rsRsbTIWu)
