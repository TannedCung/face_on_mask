from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import time
from data import cfg_mnet, cfg_slim, cfg_rfb
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from models.net_slim import Slim
from models.net_rfb import RFB
from utils.box_utils import decode, decode_landm
from utils.timer import Timer
import onnxruntime as rt
import torchvision.transforms as transforms
import cv2
from skimage import transform
from imutils.video import VideoStream

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='./weights/RBF_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='RFB', help='Backbone network mobile0.25 or slim or RFB')
parser.add_argument('--emb_net', default='./weights/arcfaceR50_asia.onnx', help='onnx file')
# parser.add_argument('--emb_net', default='./weights/arcfaceR50_masked_2.onnx', help='onnx file')
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--long_side', default=640, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
args = parser.parse_args()

RGB_MEAN = [0.5, 0.5, 0.5]
RGB_STD = [0.5, 0.5, 0.5]

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def emb_preprocess(img):
    train_transform = transforms.Compose([ # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN,
                                std = RGB_STD),
        ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2,0,1))/255.0
    img = ((img.transpose() - RGB_MEAN)/RGB_STD).transpose()
    img = np.expand_dims(img, 0)
    return img
     
def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def embbeding(img, net):
    img = emb_preprocess(img).astype(np.float32)
    input_name = net.get_inputs()[0].name
    label_name = net.get_outputs()[0].name
    output_onnx = net.run([label_name], {input_name: img})[0]
    emb = _l2_norm(output_onnx)
    return emb

def align_face(cv_img, dst, size=(112,112)):
    """align face theo widerface
    
    Arguments:
        cv_img {arr} -- Ảnh gốc
        dst {arr}} -- landmark 5 điểm theo mtcnn
    
    Returns:
        arr -- Ảnh face đã align
    """
    face_img = np.zeros(size, dtype=np.uint8)
    # Matrix standard lanmark same wider dataset
    src = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041] ], dtype=np.float32) * (size[0] / 112)
    
    tform = transform.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2,:]
    face_img = cv2.warpAffine(cv_img, M, size, borderValue=0.0)
    return face_img

def _l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output

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


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg = cfg, phase = 'test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg = cfg, phase = 'test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg = cfg, phase = 'test')
    else:
        print("Don't support network!")
        exit(0)

    net = load_model(net, args.trained_model, args.cpu)
    net.eval()

    emb_net = rt.InferenceSession(args.emb_net)

    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)
    # vid = VideoStream(src=0).start()
    vid = VideoStream("rtsp://admin:meditech123@192.168.100.143:554").start()
    # vid = cv2.VideoCapture(0)
    # testing begin
    ret = True
    img_raw = vid.read()
    milstone = None
    milstone_img = None
    
    while img_raw is not None:
        # image_path = "./img/sample.jpg"

        # img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
        big_img = vid.read()
        ratio = 4
        img_raw = cv2.resize(big_img, (480, 270))
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
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        im_height, im_width, _ = img.shape
        # img = cv2.resize(img, (270, 480))


        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        scores_mask = conf.squeeze(0).data.cpu().numpy()[:, 2]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.unique(np.concatenate((np.where(scores > args.confidence_threshold)[0], np.where(scores_mask > args.confidence_threshold)[0])))
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        scores_mask = scores_mask[inds]
        _mask = np.greater(scores_mask, scores)
        scores = scores_mask*_mask + scores*(~_mask)
        _cls = _mask + 1
        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        _cls = _cls[order]
        # print(_cls)

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis], _cls[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        dis = 0
        dets = dets[0] * ratio
        # show image
        # if args.save_image:
        new_lamk = dets[6:16].reshape(-1 ,2)
        this_img = align_face(big_img, new_lamk)
        this_emb = embbeding(this_img, emb_net)
        if milstone is None:
            milstone_img = this_img
            milstone = this_emb
            continue
        else:
            dis = np.linalg.norm(milstone-this_emb)
        
        big_img[0:112, 0:112] = milstone_img
        big_img[112:224, 0:112] = this_img

        if dets[4] < args.vis_thres:
            continue
        text = "{:.4f} - dis: {:.4f}".format(dets[4]/ratio, dis)
        color = (0, 0, 255) if dets[5]/ratio==1 else (0, 255, 0)
        dets = list(map(int, dets))
        cv2.rectangle(big_img, (dets[0], dets[1]), (dets[2], dets[3]), color, 2)
        cx = dets[0]
        cy = dets[1] + 12
        cv2.putText(big_img, text, (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # for b in dets:
        #     if b[4] < args.vis_thres:
        #         continue
        #     text = "{:.4f}".format(b[4])
        #     color = (0, 0, 255) if b[5]==1 else (0, 255, 0)
        #     b = list(map(int, b))
        #     cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), color, 2)
        #     cx = b[0]
        #     cy = b[1] + 12
        #     cv2.putText(img_raw, text, (cx, cy),
        #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        #     # landms
        #     cv2.circle(img_raw, (b[6], b[7]), 1, (0, 0, 255), 4)
        #     cv2.circle(img_raw, (b[8], b[9]), 1, (0, 255, 255), 4)
        #     cv2.circle(img_raw, (b[10], b[11]), 1, (255, 0, 255), 4)
        #     cv2.circle(img_raw, (b[12], b[13]), 1, (0, 255, 0), 4)
        #     cv2.circle(img_raw, (b[14], b[15]), 1, (255, 0, 0), 4)
            # save image

        name = "test.jpg"
        cv2.imshow(name, big_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


