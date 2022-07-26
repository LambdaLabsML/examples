from joblib import Parallel, delayed
from pathlib import Path

from datasets import IterableDataset
from tqdm import tqdm


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
    (dst_dir / "images").mkdir(parents=True, exist_ok=True)
    (dst_dir / "labels").mkdir(parents=True, exist_ok=True)

    Parallel(n_jobs=32)(delayed(_write_files)(dataset[i], dst_dir, i)
                        for i in tqdm(range(len(dataset))))
