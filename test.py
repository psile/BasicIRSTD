import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import os
import time
from tqdm import tqdm
# from pydensecrf import densecrf
# from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD test")
parser.add_argument("--model_names", default=['ACM', 'ALCNet','DNANet', 'ISNet', 'RDIAN', 'ISTDU-Net'], nargs='+',  
                    help="model_name: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")
parser.add_argument("--pth_dirs", default=None, nargs='+', help="checkpoint dir, default=None or ['NUDT-SIRST/ACM_400.pth.tar','NUAA-SIRST/ACM_400.pth.tar']")
parser.add_argument("--dataset_dir", default='./datasets', type=str, help="train_dataset_dir")
parser.add_argument("--dataset_names", default=['NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K'], nargs='+', 
                    help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea'")
parser.add_argument("--img_norm_cfg", default=None, type=dict,
                    help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                    help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                    help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

parser.add_argument("--save_img", default=True, type=bool, help="save image of or not")
parser.add_argument("--save_img_dir", type=str, default='./results/', help="path of saved image")
parser.add_argument("--save_log", type=str, default='./log/', help="path of saved .pth")
parser.add_argument("--threshold", type=float, default=0.5)

global opt
opt = parser.parse_args()
## Set img_norm_cfg
if opt.img_norm_cfg_mean != None and opt.img_norm_cfg_std != None:
  opt.img_norm_cfg = dict()
  opt.img_norm_cfg['mean'] = opt.img_norm_cfg_mean
  opt.img_norm_cfg['std'] = opt.img_norm_cfg_std

# def crf_refine(img, pred_mask, iter_num=5):
#     """
#     使用CRF后处理来优化预测的分割掩码。
#     :param img: 原始图像，形状为(H, W, C)，其中C为通道数。
#     :param pred_mask: 预测的分割掩码，形状为(H, W)，值域为{0, 1}。
#     :param iter_num: CRF迭代次数。
#     :return: 经过CRF优化后的分割掩码。
#     """
#     h, w = pred_mask.shape
#     d = densecrf.DenseCRF2D(w, h, 2)  # 2表示两类：背景和前景
#     #pdb.set_trace()
#     U = unary_from_labels(pred_mask, 2, gt_prob=0.7, zero_unsure=False)
#     d.setUnaryEnergy(U)
    
#     # 添加空间特征
#     feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
#     d.addPairwiseEnergy(feats, compat=3, kernel=densecrf.DIAG_KERNEL,
#                         normalization=densecrf.NORMALIZE_SYMMETRIC)
    
#     # 添加颜色特征
#     #pdb.set_trace()
#     feats = create_pairwise_bilateral(sdims=(80, 80), schan=(20, 20, 20),img=img, chdim=1)
#     d.addPairwiseEnergy(feats, compat=10, kernel=densecrf.DIAG_KERNEL,normalization=densecrf.NORMALIZE_SYMMETRIC)
    
#     Q = d.inference(iter_num)
#     refined_mask = np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8)
#     return refined_mask
def downsample_if_needed(img, size_limit=512):
    """如果图像尺寸超过限制，进行下采样"""
    _,_,h, w = img.shape
    if max(h, w) > size_limit:
        scale_factor = size_limit / max(h, w)
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        # 调整尺寸为8的倍数
        new_h = ((new_h + 15) // 16) * 16  # 向上取最接近的8的倍数
        new_w = ((new_w + 15) // 16) * 16
        img=F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)
        #img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        return img, h,w #,True
    else:
        return img, h,w  #,False  
import pdb
def test(): 
    test_set = TestSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    # test_set = InferenceSetLoader(opt.dataset_dir, opt.train_dataset_name, opt.test_dataset_name, opt.img_norm_cfg)
    # test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    
    # net = Net(model_name=opt.model_name, mode='test').cuda()
    # try:
    #     net.load_state_dict(torch.load(opt.pth_dir)['state_dict'])
    # except:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     net.load_state_dict(torch.load(opt.pth_dir, map_location=device)['state_dict'])
    # net.eval()
    
    eval_mIoU = mIoU() 
    eval_PD_FA = PD_FA()
   
    max_block_size = (512, 512)
    with torch.no_grad():
        for idx_iter, (img, gt_mask, size, img_dir) in enumerate(test_loader):
            for pth_dir in opt.pth_dirs:
                model_name=pth_dir.split("_")[0].split("/")[-1]
                #pdb.set_trace()
                if model_name=="AGPCNet":
                    net = Net(model_name="AGPCNet", mode='test').cuda()
                    try:
                        net.load_state_dict(torch.load("log/"+pth_dir)['state_dict'])
                    except:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        net.load_state_dict(torch.load("log/"+pth_dir, map_location=device)['state_dict'])
                    net.eval()
                elif model_name=="DNANet":
                    net = Net(model_name="DNANet", mode='test').cuda()
                    try:
                        net.load_state_dict(torch.load("log/"+pth_dir)['state_dict'])
                    except:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        net.load_state_dict(torch.load("log/"+pth_dir, map_location=device)['state_dict'])
                    net.eval()
                elif model_name=="SSTNet":
                    net = Net(model_name="SSTNet", mode='test').cuda()
                    try:
                        net.load_state_dict(torch.load("log/"+pth_dir)['state_dict'])
                    except:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        net.load_state_dict(torch.load("log/"+pth_dir, map_location=device)['state_dict'])
                    net.eval()

            
                img = Variable(img).cuda()
                _, _, height, width = img.size()

                # 计算需要填充的尺寸
                pad_height = (max_block_size[0] - height % max_block_size[0]) % max_block_size[0] # 512 - 832 % 512 = 192
                pad_width = (max_block_size[1] - width % max_block_size[1]) % max_block_size[1] # 512 - 1088 % 512 = 448
                
                # 对图像进行填充
                # img = F.pad(img, (0, 0, pad_width, pad_height), mode='constant', constant_values=0)#padding_mode
                img=F.pad(img, (0, pad_width,0, pad_height),mode='constant',value=0)
                _, _, padded_height, padded_width = img.size()

                num_blocks_height = (padded_height + max_block_size[0] - 1) // max_block_size[0]
                num_blocks_width = (padded_width + max_block_size[1] - 1) // max_block_size[1]

                # 动态分块推理
                output = torch.zeros_like(img)
                '''集成'''
                pred=torch.zeros_like(img)
                '''集成'''
                for i in range(num_blocks_height):
                    for j in range(num_blocks_width):
                        block_y = i * max_block_size[0]
                        block_x = j * max_block_size[1]
                        block_height = min(max_block_size[0], padded_height - block_y)
                        block_width = min(max_block_size[1], padded_width - block_x)

                        # 确保块的尺寸大于0
                        if block_height <= 0 or block_width <= 0:
                            print(f'Skipping block at (i={i}, j={j}) due to zero or negative size: height={block_height}, width={block_width}')
                            continue

                        block = img[:, :, block_y:block_y + block_height, block_x:block_x + block_width]
                        '''repeat'''
                        block=block.repeat(1,3,1,1)
                        '''end'''
                        try:
                            pred_block = net.forward(block)
                        except RuntimeError as e:
                            print(f'Error processing block at (i={i}, j={j}): {str(e)}')
                            continue

                        output[:, :, block_y:block_y + block_height, block_x:block_x + block_width] = pred_block

           
            # '''crf'''
            # output= crf_refine(img[0].permute(1, 2, 0).cpu().numpy(), (output[0][0]>opt.threshold).cpu().numpy().astype(np.uint8))
            # '''crf'''
            # 去除填充部分 
                pred+=output
            

            # output = output[:,:,:size[0],:size[1]]
            # pred = output
            pred=pred[:,:,:size[0],:size[1]]
            pred=pred/len(opt.pth_dirs)
            gt_mask = gt_mask[:,:,:size[0],:size[1]]
           
            
            eval_mIoU.update((pred>opt.threshold).cpu(), gt_mask)
            eval_PD_FA.update((pred[0,0,:,:]>opt.threshold).cpu(), gt_mask[0,0,:,:], size)   
            
            ### save img
            if opt.save_img == True:
                img_save = transforms.ToPILImage()((pred[0,0,:,:]).cpu())
                if not os.path.exists(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name):
                    os.makedirs(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name)
                img_save.save(opt.save_img_dir + opt.test_dataset_name + '/' + opt.model_name + '/' + img_dir[0] + '.png')  
    
    results1 = eval_mIoU.get()
    results2 = eval_PD_FA.get()
    print("pixAcc, mIoU:\t" + str(results1))
    print("PD, FA:\t" + str(results2))
    opt.f.write("pixAcc, mIoU:\t" + str(results1) + '\n')
    opt.f.write("PD, FA:\t" + str(results2) + '\n')

if __name__ == '__main__':
    opt.f = open(opt.save_log + 'test_' + (time.ctime()).replace(' ', '_').replace(':', '_') + '.txt', 'w')
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
                #for pth_dir in opt.pth_dirs:
                    #if dataset_name in pth_dir and model_name in pth_dir:
                opt.test_dataset_name = dataset_name
                opt.model_name = model_name
                opt.train_dataset_name =dataset_name# pth_dir.split('/')[0]
                # print(pth_dir)
                for pth_dir in opt.pth_dirs:
                    opt.f.write(pth_dir)
                    print(pth_dir)
                print(opt.test_dataset_name)
                opt.f.write(opt.test_dataset_name + '\n')
                # opt.pth_dir = opt.save_log + pth_dir
                test()
                print('\n')
                opt.f.write('\n')
        opt.f.close()
        
