## requirements
pip install insightface onnxruntime opencv-python tqdm
    

## TO RUN
python scrfd_to_cvat.py /path/to/images -o annotations.xml

## README
It creates a prelabels folder containing:

annotations.xml: CVAT 1.1 format. In CVAT, upload it with the CVAT 1.1 format.
yolo_labels.zip: YOLO format. In CVAT, upload it with Ultralytics YOLO Detection 1.0.
yolo_labels/: the same YOLO files unzipped. There's one .txt per image, with one line per face: class x_center y_center width height, all scaled to 0–1.

To get only one format, use --format cvat or --format yolo. To write somewhere other than prelabels, use -o some_folder.