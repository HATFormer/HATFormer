import numpy as np
import numpy.ma as ma
#from utils.chamfer_3D.dist_chamfer_3D import chamfer_3DDist_nograd as CD
import torch
#from ssim import ssim
from sklearn import metrics
from utils.mIoU import IoUMetric

class MetricEvaluator:

    def __init__(self, num_classes=3):

        self.results = []
        self.num_classes = num_classes
        self.IoUMetric = IoUMetric(num_classes=3)
        self.batch_idx = 0
        self.bacth_nochange = 0

    def get_metrics(self):
        N = self.batch_idx #+ 1
        mIoU, mean_f1 = self.IoUMetric.compute_metrics()
        mean_mae, rmse1, rmse2, rel, zncc, chamferDist, *arg = self.results

        rt = {
            'mean_mae': mean_mae / N,
            'RMSE1': np.sqrt(rmse1 / N),
            'RMSE2': np.sqrt(rmse2 / (N - self.bacth_nochange)),
            'cRel': rel / (N - self.bacth_nochange),
            'cZNCC': zncc / (N - self.bacth_nochange),
            'cd': chamferDist / N,
            'mIoU': mIoU,
            'mean_f1': mean_f1,
        }
        if arg != []:
            rt['mean_mae_bg'] = arg[0]/N
            rt['RMSE_bg'] = np.sqrt(arg[1]/N)
        return rt

    def compute_metrics(self, out3d, mask3d, out2d, mask2d, out3d_bg=None, dsm_mask=None):
        eval_out2d = out2d.cpu().numpy()
        eval_out3d = out3d.detach().cpu().numpy().ravel()
        eval_mask3d = mask3d.cpu().numpy().ravel()
        eval_mask2d = mask2d.cpu().numpy().ravel()
        eval_mask2d[eval_mask2d == 3] = 2
        eval_out2d[eval_out2d == 3] = 2
        self.IoUMetric.process(eval_mask2d.ravel(), eval_out2d.ravel())
        eval_mask2d[eval_mask2d > 0] = 1
        eval_out2d[eval_out2d > 0] = 1

        mean_ae = metrics.mean_absolute_error(eval_mask3d, eval_out3d)

        s_rmse1 = metric_mse(eval_out3d, eval_mask3d, eval_mask2d, exclude_zeros=False)
        s_rmse2 = metric_mse(eval_out3d, eval_mask3d, eval_mask2d, exclude_zeros=True)

        rel = metric_rel(eval_out3d, eval_mask3d, eval_mask2d)
        zncc = metric_ncc(eval_out3d, eval_mask3d, eval_mask2d)

        chamferDist = 0  # self.chamferDist.func(out3d, mask3d)
        if zncc == 0:
            self.bacth_nochange += 1

        eval_results = [mean_ae, s_rmse1, s_rmse2, rel, zncc, chamferDist]

        if out3d_bg is not None:
            eval_out3d_bg = out3d_bg.detach().cpu().numpy().ravel()
            eval_dsm_mask = dsm_mask.cpu().numpy().ravel()
            mean_ae_bg = metrics.mean_absolute_error(eval_dsm_mask, eval_out3d_bg)
            s_rmse1_bg = metric_mse(eval_out3d_bg, eval_dsm_mask, eval_mask2d, exclude_zeros=False)
            eval_results += [mean_ae_bg,s_rmse1_bg]
        if sum(self.results)==0:
            self.results = [0 for _ in range(len(eval_results))]
        self.batch_idx += 1
        self.results = [x + y for x, y in zip(eval_results, self.results)]


eps = 1e-10
#cd = CD()
def get_mask_array(inputs, targets, mask, thresh=1e-3):
    # The thresh is used for excluding really small height values that less than 0.001 m
    # since the small non-zero values like 1e-5 would cause uncorrect Rel output
    mask_ = mask.copy()
    indices_one = mask_ == 1
    indices_zero = mask_ == 0
    mask_[indices_one] = 0  # replacing 1s with 0s
    mask_[np.abs(targets)<thresh] = 1
    mask_[indices_zero] = 1  # replacing 0s with 1s
    inputs = ma.masked_array(inputs, mask=mask_)
    targets = ma.masked_array(targets, mask=mask_)

    return inputs, targets

# to calculate rmse
def metric_mse(inputs, targets, mask, exclude_zeros = False):
    if exclude_zeros:
        if mask.sum()!=0:
            inputs, targets = get_mask_array(inputs, targets, mask)
            loss = (inputs - targets) ** 2
            n_pixels = np.count_nonzero(targets)
            #import pdb;pdb.set_trace()
            #n_pixels = 1 if n_pixels==0 else n_pixels
            return np.sum(loss)/n_pixels
        else:
            return 0.0
    else:
        loss = (inputs - targets) ** 2

        return np.mean(loss)

def metric_rel(inputs, targets, mask):
    
    if targets.sum()==0:
        return 0
    
    inputs, targets = get_mask_array(inputs, targets, mask)
    result = (inputs-targets)/(targets+eps)
    #if np.mean(np.abs(result))>5:
    #    import pdb;pdb.set_trace()
    return np.mean(np.abs(result))

def metric_rellog10(inputs, targets, mask):
    if targets.sum()==0:
        return 0
    inputs, targets = get_mask_array(inputs, targets, mask)
    result = np.log10(inputs+eps) - np.log10(targets+eps, where=(mask!=0))

    return np.mean(np.abs(result))

def metric_ncc(inputs, targets, mask):

    if targets.sum()==0:
        return 0
    #inputs, targets = get_mask_array(inputs, targets, mask)
    mean_He, mean_Hr = inputs.mean(), targets.mean()
    std_He, std_Hr = np.std(inputs)+eps, np.std(targets)+eps
    ncc = (inputs-mean_He)*(targets-mean_Hr)/(std_He*std_Hr)

    return np.mean(ncc)

def metric_ssim(inputs, targets):
    return ssim(inputs.cuda(), targets.cuda())




class metric_chamferDist(torch.nn.Module):
    def __init__(self,H=512,W=512,grid_size=128,res=0.25):
        super(metric_chamferDist, self).__init__()
        self.grid_x, self.grid_y = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
        self.grid_x, self.grid_y = self.grid_x.ravel()*res, self.grid_y.ravel()*res
        self.grid_size = grid_size
        
        #grid_x, grid_y = grid_x.to(inputs.device), grid_y.to(inputs.device)
    def func(self, inputs, targets):
        self.grid_x, self.grid_y = self.grid_x.to(inputs.device), self.grid_y.to(inputs.device)
        #import pdb;pdb.set_trace()
        input_grids = inputs[0][0].unfold(0, self.grid_size, self.grid_size).unfold(1, self.grid_size, self.grid_size)
        input_grids = inputs.reshape(-1,self.grid_size, self.grid_size)
        target_grids = targets[0].unfold(0, self.grid_size, self.grid_size).unfold(1, self.grid_size, self.grid_size)
        target_grids = targets.reshape(-1,self.grid_size, self.grid_size)
        dist = []
        for grid_pred, grid_gt in zip(input_grids, target_grids):
            pred = torch.stack((self.grid_x, self.grid_y,grid_pred.ravel()),0).transpose(1,0).unsqueeze(0)
            gt = torch.stack((self.grid_x, self.grid_y,grid_gt.ravel()),0).transpose(1,0).unsqueeze(0)
        
            dist1, dist2, idx1, idx2 = cd(gt,pred)
            dist.append((dist1.sum()+dist2.sum()).detach().cpu().numpy())
        #import pdb;pdb.set_trace()

        return np.mean(dist)