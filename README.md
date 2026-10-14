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

Use `webcam_detect.py` with a video file instead of a webcam. The program uses `license_plate_detector.pt`, accepts plate detections with at least 50% model confidence, and runs YOLO at 1280px on every frame by default so it can compare more views and select the sharpest crop. Playback is paced at half the video's frame rate (when processing is fast enough); inference may slow it further. It keeps several candidate crops from each tracked plate and waits for a close/near-exit view before saving an image. The saved crop is enlarged 3x with mild sharpening for easier viewing; upscaling cannot restore detail missing from a blurry source frame. It does not read or save plate text.

```bash
python webcam_detect.py path/to/video.mp4
```

To trade some image detail for additional speed on a slower CPU:

```bash
python webcam_detect.py path/to/video.mp4 --imgsz 960 --detect-every 2
```

Captured plate crops are saved as high-quality JPEGs in the `plates` folder. Press `q` to stop playback.

## Read a captured plate image

Run `scan.py` to open two windows: one for selecting and viewing a plate image, and another for the recognized registration number, province, and OCR text. After OCR finishes, the image preview zooms into the full plate label, including its lower province row, and enlarges it with mild contrast and sharpening; `plate_0005.jpg` is shown without an additional zoom. This does not modify the image file used by OCR. OCR reads the Thai prefix and digits from separate regions of a captured plate, using character-specific filters and multiple scales, in addition to enhanced full-image variants. The numeric region is read with a digits-only filter, and results containing other characters or more than four digits are discarded. Common letter/number lookalikes (such as `O`/`0` and `I`/`1`) are guessed in the numeric part and marked as guesses. Each completed read, including unreadable images, is appended to `results.txt` next to `scan.py`; previous results are kept. The result shows both the OCR confidence and how many variants agreed; neither guarantees a particular real-world accuracy (such as 80%), which depends on image clarity and angle. The `plates` folder is opened by default; use the image buttons to choose or browse images in that folder.

```bash
python scan.py
```

You can also start the scanner with a different image folder:

```bash
python scan.py path/to/images
```
