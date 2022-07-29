import os
import sys
from joblib import Parallel, delayed
from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datasets import load_dataset
from datasets import IterableDataset
from tqdm import tqdm
from yolov5.utils.downloads import attempt_download


def create_yolov5_dataset_yaml(
    yolo_train_dir: str,
    yolo_val_dir: str,
    yolo_test_dir: str
) -> None:
    """
    yolo_X_dir args should be relative to the yolov5 ultralytics
    repo root, not relative to prepare_dataset.py
    """
    yaml_file = "./yolov5/data/wider_face.yaml"
    train_images_dir = os.path.join(yolo_train_dir, "images")
    val_images_dir = os.path.join(yolo_val_dir, "images")
    test_images_dir = os.path.join(yolo_test_dir, "images")

    classes = ['Face']
    names_str = ""
    for item in classes:
        names_str = names_str + ", \'%s\'" % item
    names_str = "names: [" + names_str[1:] + "]"

    with open(yaml_file, "w") as wobj:
        wobj.write("train: %s\n" % train_images_dir)
        wobj.write("val: %s\n" % val_images_dir)
        wobj.write("test: %s\n" % test_images_dir)
        wobj.write("nc: %d\n" % len(classes))
        wobj.write(names_str + "\n")


def download_dataset(show_example: bool = False):
    wider_face_train = load_dataset('wider_face', split='train')
    wider_face_val = load_dataset('wider_face', split='validation')
    wider_face_test = load_dataset('wider_face', split='test')

    print("Num images in wider_face training set: %i" % (len(wider_face_train)))
    print("Num images in wider_face val set: %i" % (len(wider_face_val)))
    print("Num images in wider_face test set: %i" % (len(wider_face_test)))

    img = np.array(wider_face_train[110]['image'], dtype=np.uint8)
    faces = wider_face_train[110]['faces']
    bboxes = faces['bbox']

    fig, ax = plt.subplots()
    ax.imshow(img)

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    if show_example:
        plt.show()

    return wider_face_train, wider_face_val, wider_face_test


def _write_files(data_point, dst_dir: Path, i: int):
    pil_img = data_point['image']
    label = data_point['faces']
    img_filename = str(i) + ".png"

    dst_image_file = dst_dir / "images" / f"{img_filename}"
    dst_label_file = dst_dir / "labels" / f"{img_filename}"
    dst_label_file = dst_label_file.with_suffix(".txt")
    if dst_label_file.exists():
        return

    # we're only detecting faces, so class_id is constant
    class_id = 0
    img_width, img_height = pil_img.size
    with dst_label_file.open("w") as wobj:
        for bbox in label['bbox']:
            cx = (bbox[0] + (bbox[2]/2.0)) / img_width
            cy = (bbox[1] + (bbox[3]/2.0)) / img_height

            # output annotation is:
            #   class_id, center_x, center_y, box_width, box_height
            # image width and height normalized to (0, 1)
            box_width = bbox[2]/img_width
            box_height = bbox[3]/img_height
            output_line = f"{class_id} {cx} {cy} {box_width} {box_height}\n"
            wobj.write(output_line)
    pil_img.save(str(dst_image_file))


def convert_to_yolov5_format(
    dataset: IterableDataset,
    dst_dir: Path,
) -> None:
    (dst_dir / Path("images")).mkdir(parents=True, exist_ok=True)
    (dst_dir / Path("labels")).mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=32)(delayed(_write_files)(dataset[i], dst_dir, i)
                        for i in tqdm(range(len(dataset))))


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
