import sys

from download_dataset import download_dataset
from convert_to_yolov5_format import convert_to_yolov5_format
from create_yolov5_dataset_yaml import create_yolov5_dataset_yaml
from yolov5.utils.downloads import attempt_download


if __name__ == "__main__":
    (wider_face_train, wider_face_test, wider_face_val) = download_dataset()

    convert_to_yolov5_format(wider_face_train, dst_dir="./yolov5/data/train")
    convert_to_yolov5_format(wider_face_test, dst_dir="./yolov5/data/test")
    convert_to_yolov5_format(wider_face_val, dst_dir="./yolov5/data/val")

    create_yolov5_dataset_yaml("./yolov5/data/train", "./yolov5/data/test")

    sys.path.append('yolov5')
    attempt_download('yolov5/weights/yolov5s.pt')
    attempt_download('yolov5/weights/yolov5m.pt')
    attempt_download('yolov5/weights/yolov5l.pt')
