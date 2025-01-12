
import sys
import tqdm
import os
import cv2
import json
from pathlib import Path
import numpy as np

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
    pairs_from_sequence,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from pdb import set_trace
import pycolmap

from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from pdb import set_trace

path = sys.argv[1]

root = os.path.dirname(path)

images = Path(path)
outputs = Path(path)
sfm_pairs = outputs / "pairs-sfm.txt"
loc_pairs = outputs / "pairs-loc.txt"
sfm_dir = outputs / "sfm"
features = outputs / "features.h5"
matches =  outputs / "matches.h5"

feature_conf = extract_features.confs["disk"]
matcher_conf = match_features.confs["disk+lightglue"]

model_path = sfm_dir
model = pycolmap.Reconstruction(model_path)

references = []
for image in model.images.values():
    references.append(image.name)
references = sorted(references)
print(references)

image_dir = Path(os.path.join(root, "images/02/"))
# query = "DCSF2324300339R000001.png"
# queries = [query]
queries = []
for root, _, files in os.walk(image_dir):
    for fname in files:
        if fname.endswith(".jpg") or fname.endswith(".png"):
            queries.append(fname)

print(queries)
bboxes = {}
for image_file in queries:
    mask_file = os.path.join(image_dir, image_file).replace('images', 'masks')
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    height, width = mask.shape
    basename = os.path.basename(image_file)
    if '01/' in mask_file:
        key = os.path.join('images/01', basename)
        bbox = [0, 0, width, height]
        height1, width1 = height, width
    elif '02/' in mask_file:
        key = basename
        ys, xs = np.where(mask > 0)
        xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()
        bh, bw = ymax - ymin, xmax - xmin
        x0 = max(0, xmin - bw // 10)
        x1 = min(width, xmax + bw // 10)
        y0 = max(0, ymin - bh // 10)
        y1 = min(height, ymax + bh // 10)
        bbox = [x0, y0, x1, y1]
        height2, width2 = height, width
    bboxes[key] = bbox
print(bboxes)

extract_features.main(
    feature_conf, image_dir, image_list=queries, feature_path=features, overwrite=True, bboxes=bboxes
)
pairs_from_exhaustive.main(loc_pairs, image_list=queries, ref_list=references)
match_features.main(
    matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
)

conf = {
    "estimation": {"ransac": {"max_error": 12}},
    "refinement": {"refine_focal_length": False, "refine_extra_params": False},
}
localizer = QueryLocalizer(model, conf)
ref_ids = [model.find_image_with_name(r).image_id for r in references]

with open('/gemini/data-1/2d-gaussian-splatting/video/phone/calibration_new.json', 'r') as f:
    data_dict = json.load(f)

cameras = []
poses = {}
for query in queries:
    key = query[:-10]
    K = np.array(data_dict[key]['K'])
    fx, fy, cx, cy = K[[0, 4, 2, 5]]
    camera = pycolmap.infer_camera_from_image(image_dir / query)
    camera.model = 'PINHOLE'
    camera.params = K[[0, 4, 2, 5]]
    cameras.append(camera)

    ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

    pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
    poses[query] = {
        'cam_from_world': {"rotation": 
            {"quat": pose.cam_from_world.rotation.quat[[3,0,1,2]].tolist()},
            "translation": pose.cam_from_world.translation.tolist()},
        "camera": {"height": height, "width": width, 'params': [fx, fy, cx, cy]},
        }
set_trace()
with open(outputs / 'refine_poses.json', 'w') as f:
    json.dump(poses, f)


