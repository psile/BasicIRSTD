import argparse
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from net import Net
# from dataset import *
import matplotlib.pyplot as plt
# from metrics import *
import os
import time
import pdb
from torchvision.transforms import ToTensor
from PIL import Image
import torch.nn.functional as F
from models.common import DetectMultiBackend
from utils_asf.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer, xyxy2xywh)
from utils_asf.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from pathlib import Path
from utils_asf.plots import Annotator, colors, save_one_box
from utils_asf.segment.general import masks2segments, process_mask, process_mask_native
from torchvision import transforms
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['ACM'], type=list, 
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")#['ACM', 'ALCNet','DNANet', 'ISNet', 'RDIAN', 'ISTDU-Net']
parser.add_argument("--pth_dirs", default=['PRCV2024改fusion/ACM_150.pth.tar'], type=list, help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='/home/public/PRCV2024/PRCV_yolo/images/train', type=str, help="train_dataset_dir") #'NUAA-SIRST/ACM_200.pth.tar'
parser.add_argument("--dataset_names", default=['PRCV2024'], type=list, 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")#, 'NUDT-SIRST', 'IRSTD-1K'
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.25)

def downsample_if_needed(img, size_limit=512):
    """如果图像尺寸超过限制，进行下采样"""
    _,_,h, w = img.shape
    if max(h, w) > size_limit:
        scale_factor = size_limit / max(h, w)
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        img=F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        #img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        return img, h,w
    else:
        return img, h,w

global opt
opt = parser.parse_args()

def test(): 
    imgsz=[640, 640]
    source=opt.dataset_dir
    weights='best.pt'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data='PRCV.yaml'
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    model.warmup(imgsz=(1 , 3, *imgsz))
    dataset = LoadImages(source, img_size=imgsz, stride=32, auto=True, vid_stride=1)
    #test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    #test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    # net = Net(model_name=opt.model_name, mode='test').cuda()
    # net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    # net.eval()

    
    # eval_mIoU = mIoU() 
    # eval_PD_FA = PD_FA()
    #for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).to(model.device)
        im =  im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # img = Variable(img).cuda()
        pred, proto = model(im.repeat(1,3,1,1), augment=False, visualize=False)[:2]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000, nm=32)
        for i, det in enumerate(pred):
            im0=im0s.copy()
            p=path
            p = Path(p)
            save_dir='./results/' + 'WideIRSTD' + '/' +'AGPCNet'
            if not os.path.exists(save_dir):
                os.makedirs( save_dir)
            save_dir=Path(save_dir)
            save_path = str(save_dir / p.name)    
            im0=torch.zeros_like(torch.tensor(im0)).numpy()
            imc = im0
            im_new=torch.zeros_like(im)
            annotator = Annotator(im0, line_width=1, example=str({0: 'airplane'}))
            retina_masks=False
            if len(det):
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    #masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    #masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
        #pred = model.forward(img)
            plot_img = torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() / 255. \
                        if retina_masks else im_new[i]
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                annotator.box_label(xyxy, '', color='')#colors('', True)
        im0 = annotator.result()
        im0=torch.tensor(im0)
        im0=transforms.ToPILImage()(((im0[:,:,0]>0.2).float()).cpu())
        im0.save(save_path)



        


    
    # results1 = eval_mIoU.get()
    # results2 = eval_PD_FA.get()
    # print("pixAcc, mIoU:\t" + str(results1))
    # print("PD, FA:\t" + str(results2))
    # opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    # opt.f.write("PD, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_Sun_Jun__2_16_56_49_2024.txt','w')#'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
    #pdb.set_trace()
    if opt.pth_dirs == None:
        for i in range(len(opt.model_names)):
            opt.model_name = opt.model_names[i]
            print(opt.model_name)
            opt.f.write(opt.model_name + '_400.pth.tar' + '\n')
            for dataset_name in opt.dataset_names:
                opt.dataset_name = dataset_name
                opt.train_dataset_name = opt.dataset_name
                opt.test_dataset_name = opt.dataset_name
                print(dataset_name)
                opt.f.write(opt.dataset_name + '\n')
                opt.pth_dir = opt.save_log + opt.dataset_name + '/' + opt.model_name + '_400.pth.tar'
                test()
            print('\n')
            opt.f.write('\n')
        opt.f.close()
    else:
        for model_name in opt.model_names:
            for dataset_name in opt.dataset_names:
                for pth_dir in opt.pth_dirs:
                    if dataset_name in pth_dir and model_name in pth_dir:
                        #pdb.set_trace()
                        opt.test_dataset_name = dataset_name
                        opt.model_name = model_name
                        opt.train_dataset_name = dataset_name #pth_dir.split('/')[0]
                        print(pth_dir)
                        opt.f.write(pth_dir)
                        print(opt.test_dataset_name)
                        opt.f.write(opt.test_dataset_name + '\n')
                        opt.pth_dir = opt.save_log + pth_dir
                        test()
                        print('\n')
                        opt.f.write('\n')
        opt.f.close()
        
