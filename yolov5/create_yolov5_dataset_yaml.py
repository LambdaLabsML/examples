import os


def create_yolov5_dataset_yaml(yolo_train_dir: str, yolo_test_dir: str):
    yaml_file = "./yolov5/data/wider_face.yaml"
    train_images_dir = os.path.join(yolo_train_dir, "images")
    val_images_dir = os.path.join(yolo_test_dir, "images")

    classes = ['Face']
    names_str = ""
    for item in classes:
        names_str = names_str + ", \'%s\'" % item
    names_str = "names: [" + names_str[1:] + "]"

    with open(yaml_file, "w") as wobj:
        wobj.write("train: %s\n" % train_images_dir)
        wobj.write("val: %s\n" % val_images_dir)
        wobj.write("nc: %d\n" % len(classes))
        wobj.write(names_str + "\n")
