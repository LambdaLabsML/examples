import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Tuple

from datasets import IterableDataset
from tqdm import tqdm


def convert_to_yolov5_format(
    dataset: IterableDataset,
    dst_dir: str,
) -> None:
    Path(os.path.join(dst_dir, "images")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(dst_dir, "labels")).mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(dataset))):
        data_point = dataset[i]
        pil_img = data_point['image']
        label = data_point['faces']
        img_filename = str(i) + ".png"

        dst_image_file = os.path.join(dst_dir, "images/%s" % (img_filename))
        dst_label_file = os.path.join(dst_dir, "labels/%s" % (img_filename.replace(".png", ".txt")))
        if os.path.exists(dst_label_file):
            continue

        class_name = "face"  # we're only detecting faces, so these are constants
        class_id = 0
        img_width, img_height = pil_img.size
        with open(dst_label_file, "w") as wobj:
            for bbox in label['bbox']:
                cx = (bbox[0] + (bbox[2]/2.0)) / img_width
                cy = (bbox[1] + (bbox[3]/2.0)) / img_height

                # output annotation is: class_id, center_x, center_y, box_width, box_height,
                # image width and height normalized to (0, 1)
                output_line = "%d %f %f %f %f\n" % (class_id, cx, cy, bbox[2]/img_width, bbox[3]/img_height)
                wobj.write(output_line)
        pil_img.save(dst_image_file)

