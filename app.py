import os
import os.path as osp
import argparse

import cv2
import sys

import numpy.linalg

sys.path.insert(1, "TensorFlow_yolo")
sys.path.insert(0, osp.join('i2l_meshnet', 'main'))
sys.path.insert(0, osp.join('i2l_meshnet', 'data'))
sys.path.insert(0, osp.join('i2l_meshnet', 'common'))

import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

import time
import numpy as np
from flask import Flask, jsonify

from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image
from utils.vis import vis_keypoints

sys.path.insert(0, cfg.smpl_path)
# noinspection PyUnresolvedReferences
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

app = Flask(__name__)
distance_between_hands = 0
root = 0

width = 1280
height = 1024
vid = cv2.VideoCapture(0)
vid.set(3, width)
vid.set(4, height)

# by default VideoCapture returns float instead of int
width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

bbox = [0, 0, width, height]
bbox = process_bbox(bbox, width, height)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0', type=str, dest='gpu_ids')
    parser.add_argument('--stage', default='param', type=str, dest='stage')
    parser.add_argument('--test_epoch', default='8', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    if not args.stage:
        assert 0, "Please set training stage among [lixel, param]"

    assert args.test_epoch, 'Test epoch is required.'
    return args


# argument parsing
args = parse_args()
cfg.set_args(args.gpu_ids, args.stage)
cudnn.benchmark = True

# SMPL joint set
joint_num = 29  # original: 24. manually add nose, L/R eye, L/R ear
joints_name = (
    'Pelvis', 'L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe', 'R_Toe',
    'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
    'L_Hand', 'R_Hand', 'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear')
flip_pairs = ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
skeleton = (
    (0, 1), (1, 4), (4, 7), (7, 10), (0, 2), (2, 5), (5, 8), (8, 11), (0, 3), (3, 6), (6, 9), (9, 14), (14, 17),
    (17, 19),
    (19, 21), (21, 23), (9, 13), (13, 16), (16, 18), (18, 20), (20, 22), (9, 12), (12, 24), (24, 15), (24, 25),
    (24, 26),
    (25, 27), (26, 28))

# SMPl mesh
vertex_num = 6890
smpl_layer = SMPL_Layer(gender='neutral', model_root=cfg.smpl_path + '/smplpytorch/native/models')
face = smpl_layer.th_faces.numpy()
joint_regressor = smpl_layer.th_J_regressor.numpy()
root_joint_idx = 0

# snapshot load
model_path = './snapshot_%d.pth.tar' % int(args.test_epoch)
assert osp.exists(model_path), 'Cannot find model at ' + model_path
model = get_model(vertex_num, joint_num, 'test')

model = DataParallel(model)
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# prepare input image
transform = transforms.ToTensor()


def main():
    times = []
    # by default VideoCapture returns float instead of int

    # obtain this from RootNet (https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/tree/master/demo)
    root_depth = 11250.5732421875
    root_depth /= 1000  # output of RootNet is milimeter. change it to meter
    with torch.no_grad():
        while True:
            t1 = time.time()
            _, frame = vid.read()

            out, bb2img_trans = run_nn(frame)
            points = out['joint_coord_img'].cpu().numpy()[0]
            points[:, 0] = points[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
            points[:, 1] = points[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
            points[:, :2] = np.dot(bb2img_trans, np.concatenate((points[:, :2], np.ones_like(points[:, :1])), 1).transpose(1, 0)).transpose(1, 0)
            points[:, 2] = (points[:, 2] / cfg.output_hm_shape[0] * 2. - 1) * (cfg.bbox_3d_size / 2)

            l_hand = points[22]
            r_hand = points[23]

            n = abs(((l_hand[0] - r_hand[0]) ** 2 + (l_hand[1] - r_hand[1]) ** 2 + (l_hand[2] - r_hand[2]) ** 2) ** 0.5)

            frame = vis_keypoints(frame, points)
            cv2.imshow('output', frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
            print(points[:1])
            print(n)

            t2 = time.time()
            times.append(t2 - t1)
            times = times[-20:]

            ms = sum(times) / len(times) * 1000
            fps = 1000 / ms
            print("Time: {:.2f}ms, {:.1f} FPS".format(ms, fps))


def run_nn(frame):
    original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_frame, bbox, 1.0, 0.0, False,
                                                           cfg.input_img_shape)
    img = transform(img.astype(np.float32)) / 255
    img = img.cuda()[None, :, :, :]

    # forward
    inputs = {'img': img}
    targets = {}
    meta_info = {'bb2img_trans': bb2img_trans}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    return out, bb2img_trans


@app.route('/calibrate')
def calibration(run_time=5):
    global distance_between_hands, root

    _, frame = vid.read()
    out, bb2img_trans = run_nn(frame)
    norm = 0
    while run_time > 0:
        t1 = time.time()
        _, frame = vid.read()
        points = out["joint_coord_img"].cpu().numpy()[0]

        l_hand = points[22]
        r_hand = points[23]

        n = abs(((l_hand[0] - r_hand[0]) ** 2 + (l_hand[1] - r_hand[1]) ** 2 + (l_hand[2] - r_hand[2]) ** 2) ** 0.5)
        if n > norm:
            norm = n
            root = points[0]
            print(norm)

        frame= vis_keypoints(frame, points[18:24])
        cv2.imshow('output', frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

        out, bb2img_trans = run_nn(frame)

        t2 = time.time()
        run_time -= t2 - t1

    distance_between_hands = norm
    return jsonify({'success': 'OK'})


@app.route("/points")
def calculate_points():
    _, frame = vid.read()
    out, bb2img_trans = run_nn(frame)
    points = out["joint_coord_img"].cpu().numpy()[0]

    normalized_points = [(point - root) / distance_between_hands for point in points[18: 24]]
    l_elbow, r_elbow, *_, l_hand, r_hand = [tuple(map(float, point)) for point in normalized_points]
    print(f'Elbow: {points[18]}, Normalized Elbow: {l_elbow}')

    data = {"right_hand": {'joint_target': r_elbow, 'effector': r_hand},
            "left_hand": {'joint_target': l_elbow, 'effector': l_hand}}

    pprint(f'Points: {normalized_points}\nNormalized Points {points[18:24]}')
    return jsonify(data)


from pprint import pprint


if __name__ == "__main__":
    # main()
    app.run(host='127.0.0.1', port='8080')
