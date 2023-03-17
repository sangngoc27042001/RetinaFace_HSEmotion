from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import sys
sys.path.append('./Face_Detector_1MB_with_landmark/')
from config import config

from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import pickle

# parser = argparse.ArgumentParser(description='Test')
# parser.add_argument('-m', '--trained_model', default='./weights_RFB/RBF_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
# parser.add_argument('--origin_size', default=False, type=str, help='Whether use origin image size to evaluate')
# parser.add_argument('--long_side', default=320, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
# parser.add_argument('--save_folder', default='./img', type=str, help='Dir to save txt results')
# parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--dataset_folder', default='./img', type=str, help='dataset path')
# parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
# parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
# parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
# args = parser.parse_args()
args = pickle.load(open('./Face_Detector_1MB_with_landmark/args.pkl','rb'))
args.trained_model = './Face_Detector_1MB_with_landmark/weights/mobilenet0.25_Final.pth'
args.confidence_threshold = config['RetinaFaceInterference']['conf_thres']
args.cpu = True if config['device']=='cpu' else False

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

cfg = cfg_mnet
net = RetinaFace(cfg = cfg, phase = 'test')
net = load_model(net, args.trained_model, args.cpu)
net.eval()
# print('Finished loading model!')
# print(net)
cudnn.benchmark = True
device = torch.device("cpu" if args.cpu else "cuda")
net = net.to(device)

def detect_face_retinaface(img_raw):
    img = np.float32(img_raw)
    # testing scale
    target_size = args.long_side
    max_size = args.long_side
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    loc, conf, landms = net(img)  # forward pass
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args.confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    order = scores.argsort()[::-1]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, args.nms_threshold)

    boxes = boxes[keep, :]
    if len(boxes) == 0:
        return None
    return boxes.astype(np.int)



if __name__ == '__main__':
    torch.set_grad_enabled(False)
    img_raw = cv2.imread('./20180720_174416.jpg')
    bbs = detect_face_retinaface(img_raw)
    
    for bb in bbs:
        l, t, r, b = bb
        cv2.rectangle(img_raw, (l, t), (r, b), (36,255,12), 5)   
    
    cv2.imwrite('./out3.jpg', img_raw)