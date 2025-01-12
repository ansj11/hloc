#!/usr/bin/env python
# coding: utf-8

# In this notebook, we will build a 3D map of a scene from a small set of images and then localize an image downloaded from the Internet. This demo was contributed by [Philipp Lindenberger](https://github.com/Phil26AT/).


import sys
import tqdm
import os
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

# path = "datasets/carV8/"
path = sys.argv[1]
outpath = os.path.join(path, "hloc")
bbox_file = os.path.join(path, "bbox_dict.json")

images = Path(path)
outputs = Path(outpath)
# os.system('rm -rf $outputs')
sfm_pairs = outputs / "pairs-sfm.txt"
loc_pairs = outputs / "pairs-loc.txt"
sfm_dir = outputs / "sfm"
features = outputs / "features.h5"
matches =  outputs / "matches.h5"

feature_conf = extract_features.confs["disk"]
matcher_conf = match_features.confs["disk+lightglue"]


# # 3D mapping
# First we list the images used for mapping. These are all day-time shots of Sacre Coeur.
references = sorted([p.relative_to(images).as_posix() for p in (images / "images/").iterdir()])
print(references)
print(len(references), "mapping images")
# plot_images([read_image(images / r) for r in references], dpi=25)
bboxes = None
if os.path.exists(bbox_file):
    with open(bbox_file, 'r') as f:
        dic = json.load(f)
    bboxes = {}
    for k, v in dic.items():
        key = os.path.join('images', k)
        bboxes[key] = v
print(bboxes)

# Then we extract features and match them across image pairs. Since we deal with few images, 
# we simply match all pairs exhaustively. For larger scenes, we would use image retrieval, 
# as demonstrated in the other notebooks.
extract_features.main(
    feature_conf, images, image_list=references, feature_path=features, bboxes=bboxes
)
pairs_from_exhaustive.main(sfm_pairs, image_list=references)
# pairs_from_sequence.main(sfm_pairs, image_list=references, features=features, window_size=3)
match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches);


# The we run incremental Structure-From-Motion and display the reconstructed 3D model.
model = reconstruction.main(
    sfm_dir, images, sfm_pairs, features, matches, image_list=references,
    camera_mode='SINGLE', min_match_score = 0.5
)
# fig = viz_3d.init_figure()
# viz_3d.plot_reconstruction(
#     fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True
# )
# fig.show()


# We also visualize which keypoints were triangulated into the 3D model.
visualization.visualize_sfm_2d(model, images, color_by="visibility", n=2)


# # Localization
url = "https://upload.wikimedia.org/wikipedia/commons/5/53/Paris_-_Basilique_du_Sacr%C3%A9_Coeur%2C_Montmartre_-_panoramio.jpg"
query = "query/night.jpg"
# get_ipython().system('mkdir -p $images/query && wget $url -O $images/$query -q')
# plot_images([read_image(images / query)], dpi=75)


# Again, we extract features for the query and match them exhaustively.
# extract_features.main(
#     feature_conf, images, image_list=[query], feature_path=features, overwrite=True
# )
# pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
# match_features.main(
#     matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
# );


# # We read the EXIF data of the query to infer a rough initial estimate of camera parameters like the focal length. Then we estimate the absolute camera pose using PnP+RANSAC and refine the camera parameters.
# import pycolmap
# from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

# camera = pycolmap.infer_camera_from_image(images / query)
# ref_ids = [model.find_image_with_name(r).image_id for r in references]
# conf = {
#     "estimation": {"ransac": {"max_error": 12}},
#     "refinement": {"refine_focal_length": True, "refine_extra_params": True},
# }
# localizer = QueryLocalizer(model, conf)
# ret, log = pose_from_cluster(localizer, query, camera, ref_ids, features, matches)

# print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
# visualization.visualize_loc_from_log(images, query, log, model)


# We visualize the correspondences between the query images a few mapping images. We can also visualize the estimated camera pose in the 3D map.
# pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
# viz_3d.plot_camera_colmap(
#     fig, pose, camera, color="rgba(0,255,0,0.5)", name=query, fill=True
# )
# # visualize 2D-3D correspodences
# inl_3d = np.array(
#     [model.points3D[pid].xyz for pid in np.array(log["points3D_ids"])[ret["inliers"]]]
# )
# viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)
# fig.show()

