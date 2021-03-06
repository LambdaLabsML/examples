import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from datasets import load_dataset


def download_dataset():
    wider_face = load_dataset('wider_face', split='train')
    print("Num images in wider_face training set: %i" % (len(wider_face)))

    img = np.array(wider_face[110]['image'], dtype=np.uint8)
    faces = wider_face[110]['faces']
    bboxes = faces['bbox']

    fig, ax = plt.subplots()
    ax.imshow(img)

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]),
                                 bbox[2], bbox[3],
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Uncomment to see an example of WiderFace, with bboxes drawn
    # plt.show()

    return wider_face
