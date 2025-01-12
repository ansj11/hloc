
from glob import glob
import os
import cv2
import shutil
from tqdm import tqdm
import numpy as np
from pdb import set_trace


path = "./datasets/masks"

def rename_files(path):
    outdir = path.replace("masks", "carV3/masks")
    os.makedirs(outdir, exist_ok=True)
    img_paths = sorted(glob(os.path.join(path, "*.jpg")))

    for idx, img_path in enumerate(img_paths):
        if not (img_name.endswith('0.jpg') or img_name.endswith('5.png')):
            continue
        img_name = os.path.basename(img_path).split('.')[0]
        index = int(img_name / 5) + 61
        new_name = '%02d.jpg' % (index)
        new_path = os.path.join(outdir, new_name)
        shutil.copy(img_path, new_path)

def resize_and_mask2bbox(path):
    img_paths = sorted(glob(os.path.join(path, "images/*.jpg")))
    
    bbox = np.loadtxt(os.path.join(path, "bbox.txt"))
    
    bboxes = []
    height, width = 1448, 2560
    count = 0
    for img_path in tqdm(img_paths):
        image = cv2.imread(img_path, -1)
        if image.shape[2] == 4:
            image = cv2.resize(image, (width, height), cv2.INTER_LINEAR)
            x, y, w, h = cv2.boundingRect(image[..., -1])
            bboxes.append([x, y, x+w, y+h])
            # cv2.imwrite(img_path, image)
        else:
            rgba = np.zeros((height, width, 4), dtype=np.uint8)
            rgba[..., :3] = image
            box = [int(x) for x in bbox[count]]
            rgba[box[1]:box[3], box[0]:box[2], 3] = 255
            cv2.imwrite(img_path[:-3] + 'png', rgba)
            count += 1
    
    bboxes = np.array(bboxes)
    bboxes = np.concatenate([bboxes, bbox], axis=0)
    
    # np.savetxt(os.path.join(path, "bboxes.txt"), bboxes, fmt='%d')

if __name__ == '__main__':
    path = "./datasets/masks/"
    # resize_and_mask2bbox(path)
    rename_files(path)