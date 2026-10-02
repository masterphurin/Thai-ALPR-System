# Automatic-Number-Plate-Recognition-YOLOv8
## Demo


https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/assets/79400407/1af57131-3ada-470a-b798-95fff00254e6



## Data

The video used in the tutorial can be downloaded [here](https://drive.google.com/file/d/1JbwLyqpFCXmftaJY1oap8Sa6KfjoWJta/view?usp=sharing).

## Model

A Yolov8 pre-trained model (YOLOv8n) was used to detect vehicles.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8 using [this dataset](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4). 
- The model is available [here](https://drive.google.com/file/d/1Zmf5ynaTFhmln2z7Qvv-tgjkWQYQ9Zdw/view?usp=sharing).

## Dependencies

The sort module needs to be downloaded from [this repository](https://github.com/abewley/sort).

```bash
git clone https://github.com/abewley/sort
```

## Project Setup

* Make an environment with python=3.10 using the following command 
``` bash
conda create --prefix ./env python==3.10 -y
```
* Activate the environment
``` bash
source activate ./env
``` 

* Install the project dependencies using the following command 
```bash
pip install -r requirements.txt
```

The preview needs the GUI build of OpenCV. If another package replaces it with a headless build, run:

```bash
pip install --force-reinstall "opencv-python<4.10" "numpy<2"
```
* Run main.py with the sample video file to generate the test.csv file 
``` python
python main.py
```
* Run the add_missing_data.py file for interpolation of values to match up for the missing frames and smooth output.
```python
python add_missing_data.py
```

* Finally run the visualize.py passing in the interpolated csv files and hence obtaining a smooth output for license plate detection.
```python
python visualize.py
```

## Video ALPR with automatic plate capture

Use `webcam_detect.py` with a video file instead of a webcam. The program uses `license_plate_detector.pt`, accepts plate detections with at least 50% model confidence, runs YOLO at 1280px every two frames by default, and keeps the latest detection box visible between inference frames. It keeps several sharp crops from each tracked plate and waits for a close/near-exit view before saving an image. It does not read or save plate text.

```bash
python webcam_detect.py path/to/video.mp4
```

To prioritize detection on every frame, or to trade some image detail for additional speed on a slower CPU:

```bash
python webcam_detect.py path/to/video.mp4 --detect-every 1
python webcam_detect.py path/to/video.mp4 --imgsz 960 --detect-every 2
```

Captured plate crops are saved in the `plates` folder. Press `q` to stop playback.

## Read a captured plate image

Run `scan.py` to open two windows: one for selecting and viewing a plate image, and another for the recognized registration number, province, and OCR text. Small captured images are enlarged up to 2.5x in the preview. OCR tries the original enlarged image and enhanced variants; common letter/number lookalikes (such as `O`/`0` and `I`/`1`) are guessed in the numeric part and marked as guesses. Guesses can still be wrong, especially for blurry or very small images. The `plates` folder is opened by default; use the image buttons to choose or browse images in that folder.

```bash
python scan.py
```

You can also start the scanner with a different image folder:

```bash
python scan.py path/to/images
```
