from pytorch_grad_cam import GradCAM,EigenCAM,AblationCAM,GradCAMElementWise,GradCAMPlusPlus,ScoreCAM
import torch
import numpy as np
import cv2



def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

def gradCAM_vis(model, target_layers, input_tensor, gt,
                rgb_img=None, categories=[1, 2]):
    class SemanticSegmentationTarget:
        def __init__(self, category, mask):
            self.category = category
            self.mask = mask

        def __call__(self, model_output):
            #import pdb;pdb.set_trace()
            if model_output.shape[1]>1:
                return (model_output[0, self.category, :, :] * self.mask).sum()
            else:
                return (model_output[0, 0, :, :] * self.mask).sum()

    gt_float = gt != 0


    img = np.zeros_like(np.float32(gt_float.cpu().detach().numpy()))[0]
    input_tensor = input_tensor.requires_grad_()
    for category in categories:
        targets = [SemanticSegmentationTarget(category, gt_float)]
        with GradCAM(model=model,
                     target_layers=target_layers,
                     use_cuda=torch.cuda.is_available(),
                     ) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # cam_image = show_cam_on_image(normalize(rgb_img), grayscale_cam[0], use_rgb=True)
            # img_c1 = cv2.applyColorMap(np.uint8(grayscale_cam[1] * 255), cv2.COLORMAP_JET)
            img += normalize(grayscale_cam[0])

    img = np.clip(img, a_min=0., a_max=1.)
    img = cv2.applyColorMap(np.uint8(img * 255), cv2.COLORMAP_JET)

    return img

def save_imgtensor_func(img_tensor,out_dir=None,mask=None):
    img = img_tensor.detach().cpu().numpy()
    # if (mask==1).sum()>(mask==2).sum():
    #     img = -img
    img = np.uint8(normalize(img)*255)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imwrite(out_dir,img)