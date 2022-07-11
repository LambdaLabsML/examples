import sys

from download_dataset import download_dataset
from convert_to_yolov5_format import convert_to_yolov5_format
from create_yolov5_dataset_yaml import create_yolov5_dataset_yaml
from yolov5.utils.downloads import attempt_download


if __name__ == "__main__":
    wider_face = download_dataset()

    convert_to_yolov5_format(wider_face, yolo_train_dir="./yolov5/data/train")
    convert_to_yolov5_format(wider_face, yolo_test_dir="./yolov5/data/test")
    create_yolov5_dataset_yaml(yolo_train_dir, yolo_test_dir)

    sys.path.append('yolov5')
    attempt_download('yolov5/weights/yolov5s.pt')
    attempt_download('yolov5/weights/yolov5m.pt')
    attempt_download('yolov5/weights/yolov5l.pt')
