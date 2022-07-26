import sys
from pathlib import Path

from download_dataset import download_dataset
from convert_to_yolov5_format import convert_to_yolov5_format
from create_yolov5_dataset_yaml import create_yolov5_dataset_yaml
from yolov5.utils.downloads import attempt_download


if __name__ == "__main__":
    (wider_face_train, wider_face_val, wider_face_test) = download_dataset()

    convert_to_yolov5_format(wider_face_train, dst_dir=Path("./yolov5/data/train"))
    convert_to_yolov5_format(wider_face_val, dst_dir=Path("./yolov5/data/val"))
    convert_to_yolov5_format(wider_face_test, dst_dir=Path("./yolov5/data/test"))

    create_yolov5_dataset_yaml("data/train", "data/val", "data/val")

    sys.path.append('yolov5')
    attempt_download('yolov5/weights/yolov5s.pt')
    attempt_download('yolov5/weights/yolov5m.pt')
    attempt_download('yolov5/weights/yolov5l.pt')
