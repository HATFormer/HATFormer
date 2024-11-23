import os
import cv2

import torch
import torch.nn.functional as F
import lightning.pytorch as pl

import numpy as np
import rasterio as ro, logging

from utils.optim import set_scheduler
from utils.metrics import MetricEvaluator
from utils.vis_feature import save_imgtensor_func
from utils.evaluation import getHist

from pl_models.common import pl_trainer_base


log = logging.getLogger()
log.setLevel(logging.ERROR)

np.random.seed(seed=0)

sample_ratio = 0.001*0.1 # 0.5 for smars
sample_size = int(512*512*sample_ratio)
sample_idx = np.random.choice(512*512, size=sample_size, replace=False)

# define the LightningModule
class pl_trainer(pl_trainer_base):
    def __init__(self, model=None, exp_config=None, criterion2d=None,
                 criterion3d=None, save_img=False, aux=False):
        super().__init__()

        self.model = model

        self.exp_config = exp_config
        self.optim_params = exp_config['optim']
        self.min_scale = exp_config['data']['min_value']
        self.max_scale = exp_config['data']['max_value']
        try:
            self.min_scale_dsm = exp_config['data']['min_value_dsm']
            self.max_scale_dsm = exp_config['data']['max_value_dsm']
        except:
            pass
        self.lweight2d, self.lweight3d = exp_config['model']['loss_weights']
        self.criterion2d = criterion2d
        self.criterion3d = criterion3d
        self.kl_loss = torch.nn.KLDivLoss(size_average=False, reduce=False)
        self.prob_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.log_sm = torch.nn.LogSoftmax(dim=1)
        self.save_img = save_img
        self.aux = aux

        self.custom_lut = self.genColormap()
        self.sigma = 1

    def contra_loss(self, seg_label, dist, m1=0.3, m2=2.2):

        contra_label = seg_label.clone().detach()

        mdist_pos = torch.clamp(dist - m1, min=0.0)
        mdist_neg = torch.clamp(m2 - dist, min=0.0)
        labeled_points = (contra_label != 0) == (contra_label != 255)

        loss_neg = labeled_points * mdist_neg
        loss_pos = (contra_label == 0) * (mdist_pos)

        return loss_pos.mean() + loss_neg.mean()

    def background_uncertainty_suppression(self, inputs):
        n = len(inputs)
        avg = 1/n*sum(inputs)
        unc_maps = []
        for i in range(n):
            #unc_maps.append(self.kl_loss(inputs[i],avg))
            unc_maps.append((inputs[i]-avg)**2)

        return unc_maps

    def cal_loss(self, out2d, mask2d, out3d, mask3d, out3d_bg=None,
                 dsm_mask=None, out_aux=None, prob=None, dist=None):

        if type(out2d) == list:
            loss2d = 0.0
            for idx,sub_out2d in enumerate(out2d):
                if 'weight2d' in self.exp_config['model']:
                    weight2d = self.exp_config['model']['weight2d']
                    loss2d += weight2d[idx]*self.criterion2d(F.interpolate(sub_out2d, size=out2d[0].shape[2:]), mask2d.long())
                else:
                    loss2d += self.criterion2d(F.interpolate(sub_out2d, size=out2d[0].shape[2:]), mask2d.long())
        else:
            loss2d = self.criterion2d(out2d, mask2d.long())
        if self.aux:
            # 0.4 is the weight of auxiliary loss from P2VNet:https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9975266
            loss2d += 0.4 * self.criterion2d(out_aux, mask2d.long())
        if type(out3d) == list:
            loss3d = 0.0
            for idx, sub_out3d in enumerate(out3d):
                if 'weight3d' in self.exp_config['model']:
                    weight3d = self.exp_config['model']['weight3d']
                    loss3d += weight3d[idx]*self.criterion3d(F.interpolate(sub_out3d, size=out3d[0].shape[2:]).squeeze(dim=1), mask3d)
                else:
                    loss3d += self.criterion3d(F.interpolate(sub_out3d, size=out3d[0].shape[2:]).squeeze(dim=1), mask3d)
        else:
            # import pdb;pdb.set_trace()
            loss3d = self.criterion3d(out3d.squeeze(dim=1), mask3d)
        if out3d_bg is not None and type(out3d_bg) == list:
            if 'weightunc' in self.exp_config['model'] and self.exp_config['model']['weightunc']!=0.0:

                unc_maps = self.background_uncertainty_suppression(out3d_bg)
                loss_unc = 0.0

                for unc in unc_maps:
                    loss_unc += torch.mean(unc)
            #loss_unc *= 1/len(unc_maps)

            loss3d_bg = 0.0
            for idx, sub_out3d in enumerate(out3d_bg):
                if 'loss_unc' in locals():
                    unc_mask = torch.exp(-unc_maps[idx])
                    loss3d_bg += self.criterion3d(
                        unc_mask*F.interpolate(sub_out3d,
                                      size=out3d_bg[0].shape[2:]).squeeze(dim=1),
                        dsm_mask*unc_mask)
                else:
                    loss3d_bg += self.criterion3d(
                        F.interpolate(sub_out3d, size=out3d_bg[0].shape[2:]).squeeze(dim=1),
                        dsm_mask)
        elif out3d_bg is not None:
            # import pdb;pdb.set_trace()
            loss3d_bg = self.criterion3d(out3d_bg.squeeze(dim=1), dsm_mask)
        # probability model loss
        if prob is not None:
            # prob_mask = mask3d.abs().unsqueeze(1)
            if type(prob) == list:
                loss_prob = 0.0
                #loss_unc = 0.0
                for sub_prob in prob:
                    mu, logvar, prob_x, uncertainty = sub_prob
                    # loss_prob += torch.mean(-0.5*torch.sum(1+logvar-mu**2-logvar.exp(),dim=1))
                    loss_prob += self.criterion3d(
                        F.interpolate(prob_x,
                                      size=out3d[0].shape[2:]).squeeze(dim=1),
                        mask3d)
                    #loss_unc += uncertainty.mean()
                    # loss_prob += self.prob_criterion(sub_prob, prob_mask)

            else:
                loss_prob = self.prob_criterion(prob, prob_mask) \
                            + self.kl_loss(prob.sigmoid().log(), prob_mask).mean()

        if dist is not None:
            if type(dist) == list:
                loss_contra = 0.0
                for sub_dist in dist:
                    loss_contra += self.contra_loss(mask2d, sub_dist)

        if 'dynamicweight' in self.exp_config['model']:
            lweight = 1 - (self.current_epoch + 1) / self.exp_config['optim']['num_epochs']
            loss = lweight * loss2d + self.sigma * (1 - lweight) * loss3d
        else:
            loss = self.lweight2d * loss2d + self.lweight3d * loss3d

        if 'loss_unc' in locals():
            weightunc = 1.0
            if 'weightunc' in self.exp_config['model']:
                weightunc = self.exp_config['model']['weightunc']
            loss += weightunc * loss_unc
        if 'loss3d_bg' in locals():
            weightbg = 1.0
            if 'weightbg' in self.exp_config['model']:
                weightbg = self.exp_config['model']['weightbg']
            loss += weightbg * loss3d_bg
            if 'loss_prob' in locals():
                lweight = 0.1 * (self.current_epoch + 1) / self.exp_config['optim']['num_epochs']
                loss += 0.1 * loss_prob
                loss += 0.1 * loss_unc
                # print(loss_prob)
                return loss, loss2d.item(), loss3d.item(), loss3d_bg.item(), loss_prob.item(), loss_unc.item()
            if 'loss_contra' in locals():
                loss += 0.05 * loss_contra
                return loss, loss2d.item(), loss3d.item(), loss3d_bg.item(), loss_contra.item()

            return loss, loss2d.item(), loss3d.item(), loss3d_bg.item()

        else:
            return loss, loss2d.item(), loss3d.item()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        t1, t2, mask2d, mask3d, dsm_mask = batch
        dsm_mask = self.norm_train_dsm(dsm_mask)
        # dsm_mask *= (mask2d==0)
        mask3d = self.norm_train(mask3d)

        if self.aux:
            out2d, out3d, out_aux = self.model(t1, t2)
        else:
            results = self.model(t1, t2)

        if len(results) == 4:
            out2d, out3d, out3d_bg, prob = results
        elif len(results) == 3:
            out2d, out3d, out3d_bg = results
        elif len(results) == 2:
            out2d, out3d = results
        else:
            out3d_bg = results

        if 'out3d_bg' in locals():
            if 'prob' in locals():
                loss, loss2d, loss3d, loss3d_bg, loss_prob, loss_unc = self.cal_loss(out2d, mask2d, out3d,
                                                                                     mask3d, out3d_bg, dsm_mask,
                                                                                     prob=prob,
                                                                                     out_aux=out_aux if self.aux else None)
            elif 'out2d' in locals():
                loss, loss2d, loss3d, loss3d_bg = self.cal_loss(out2d, mask2d, out3d,
                                                                mask3d, out3d_bg, dsm_mask,
                                                                out_aux=out_aux if self.aux else None)

            else:
                loss = self.criterion3d(out3d_bg.squeeze(dim=1), dsm_mask)
                loss2d = 0
                loss3d = 0
        else:
            loss, loss2d, loss3d = self.cal_loss(out2d, mask2d, out3d, mask3d,
                                                 out_aux=out_aux if self.aux else None)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("2d_loss", loss2d, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("3d_loss", loss3d, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if 'out3d_bg' in locals():
            self.log("bg3d_loss", loss3d_bg, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if 'prob' in locals():
            self.log("prob_loss", loss_prob, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if 'loss_dist' in locals():
            self.log("loss_dist", loss_dist, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        if 'loss_unc' in locals():
            self.log("loss_unc", loss_unc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_test_epoch_end(self):

        rt = self.MetricEvaluator.get_metrics()
        mIoU, mean_f1, cd, RMSE1 = rt['mIoU'], rt['mean_f1'], rt['cd'], rt['RMSE1']
        mean_mae, RMSE2, cRel, cZNCC = rt['mean_mae'], rt['RMSE2'], rt['cRel'], rt['cZNCC']
        print('|metrics|mIoU|F1-score|ChamferDist|RMSE|MAE|cRMSE|cRel|cZNCC|')
        print('|--|--|--|--|--|--|--|--|--|--|')
        print(
            f'|{mIoU * 100:.3f}|{mean_f1 * 100:.3f}|{cd:.3f}|{RMSE1:.3f}|{mean_mae:.3f}|{RMSE2:.3f}|{cRel:.3f}|{cZNCC:.3f}|')

        if 'mean_mae_bg' in rt and 'RMSE_bg' in rt:
            print('MAE_bg:{:.3f},RMSE_bg:{:.3f}'.format(rt['mean_mae_bg'], rt['RMSE_bg']))


    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        t1, t2, mask2d, mask3d, dsm_mask, self.img_path = batch
        #dsm_mask *= (mask2d == 0)

        if self.aux:
            out2d, out3d, out_aux = self.model(t1, t2)
        else:
            results = self.model(t1, t2)

        VIS_FEATURE = False
        if VIS_FEATURE:
            try:
                out2d, out3d, vis_feature = results
            except:
                out2d, out3d, _, vis_feature = results
        else:
            if 'results' not in locals():
                pass
            elif len(results) == 5:
                out2d, out3d, out3d_bg, _1, _2 = results
            elif len(results) == 4:
                out2d, out3d, out3d_bg, _ = results
            elif len(results) == 3:
                out2d, out3d, out3d_bg = results
            elif len(results) == 2:
                out2d, out3d = results
            else:
                out3d = results
        # print(out3d_bg[0].max(), out3d_bg[0].min())
        DMI_or_ICIF = False
        if DMI_or_ICIF:
            #out2d = out2d[1] + out2d[2]
            out3d = (out3d[0] + out3d[1]) / 2

        if type(out3d) == list:
            out3d = [self.norm_infer(_) for _ in out3d]
        else:
            out3d = self.norm_infer(out3d)

        if type(out2d) == list:
            out2d = out2d[0]
        if type(out3d) == list:
            out3d = out3d[0]
        out2d = out2d.detach().argmax(dim=1)


        # metric evalutation
        if 'out3d_bg' in locals():

            out3d_bg = [self.norm_infer_dsm(out3d_bg[i]) for i in range(len(out3d_bg))]
            unc_maps = self.background_uncertainty_suppression(out3d_bg)
            #out3d_bg = out3d_bg[0]
            # out3d_bg = out3d_bg[0]
            self.MetricEvaluator.compute_metrics(out3d, mask3d, out2d, mask2d, out3d_bg[0], dsm_mask)
        else:
            self.MetricEvaluator.compute_metrics(out3d, mask3d, out2d, mask2d)

        if self.save_img and (mask2d >= 1).sum() > 0:

            mask = mask2d.detach().cpu().numpy()[0] != 0
            gt_hist, self.bins = getHist(mask3d.detach().cpu().numpy()[0][mask], self.num_bins)
            pred_hist, self.bins = getHist(out3d.detach().cpu().numpy()[0][0][mask], self.num_bins)
            self.total_hist += pred_hist
            self.total_gt_hist += gt_hist

            f3d_out_path = os.path.join(self.logger.log_dir.rstrip(self.logger.log_dir.split('/')[-1]), 'f3d')
            os.makedirs(f3d_out_path, exist_ok=True)
            save_imgtensor_func(out3d[0][0], out_dir=os.path.join(f3d_out_path, self.img_path[0].split('/')[-1]),
                                mask=mask2d)

            self.mask3d = mask3d.detach().cpu().numpy()
            self.save_img_func(self.applyColor(out2d.cpu().numpy()[0]).transpose(2, 0, 1), self.img_path, '/out2d',
                               dim=3)
            self.save_img_func(self.applyColor(mask2d.cpu().numpy()[0]).transpose(2, 0, 1), self.img_path, '/gt2d',
                               dim=3)
            self.save_img_func_3d_backup(out3d.detach().cpu().numpy()[0][0], self.img_path, '/out3d')
            self.save_img_func_3d_backup(mask3d.detach().cpu().numpy()[0], self.img_path, '/gt3d')
            self.dsm_mask = dsm_mask.detach().cpu().numpy()
            self.save_img_func_3d_backup(dsm_mask.detach().cpu().numpy()[0], self.img_path, '/dsm_mask', dsm_mask=True)
            if 'out3d_bg' in locals():
                img_name = self.img_path[0].split('/')[-1].strip('.tif')
                self.save_img_func_3d_backup(out3d_bg[0].detach().cpu().numpy()[0][0],
                                             [self.img_path[0].replace(img_name,img_name+'_s1')],
                                             '/out3d_bg', dsm_mask=True)
                self.save_img_func_3d_backup(out3d_bg[1].detach().cpu().numpy()[0][0],
                                             [self.img_path[0].replace(img_name,img_name+'_s2')],
                                             '/out3d_bg', dsm_mask=True)
                self.save_img_func_3d_backup(out3d_bg[2].detach().cpu().numpy()[0][0],
                                             [self.img_path[0].replace(img_name,img_name+'_s3')],
                                             '/out3d_bg', dsm_mask=True)


    def on_test_epoch_start(self):

        self.num_bins = 200
        self.total_hist = np.zeros(200)
        self.total_gt_hist = np.zeros(200)
        self.MetricEvaluator = MetricEvaluator()

