import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import torch
import numpy as np
import os
import yaml
from argparse import ArgumentParser

from utils.losses import choose_criterion3d, choose_criterion2d
import torch.backends.cudnn as cudnn
import random


def get_args():
    parser = ArgumentParser(description="Hyperparameters", add_help=True)
    parser.add_argument('-c', '--config-name', type=str, help='YAML Config name',
                        dest='CONFIG', default='CD_hmcdnet_crossattv1_3d_plabel')
    parser.add_argument('--ckpt_version', type=str, help='checkpoint version',
                        dest='ckpt_version', default='-1')
    parser.add_argument('-nw', '--num-workers', type=int, help='Number of workers',
                        dest='num_workers', default=8)
    parser.add_argument('-v', '--verbose', type=bool, help='Verbose validation metrics',
                        dest='verbose', default=False)
    parser.add_argument('--eval', help='train or evalution',
                        dest='eval', action='store_true')
    parser.add_argument('--save_img', help='save img or not',
                        dest='save_img', action='store_true')
    parser.add_argument('-d', '--device', type=int, help='device ID',
                        dest='device', default=0)
    parser.add_argument('--ckpt_idx', type=int, help='the index of checkpoint',
                        dest='ckpt_idx', default=1)
    parser.add_argument('--show_size', help='train or evalution',
                        dest='show_size', action='store_true')
    return parser.parse_args()


def parse_args():
    args = get_args()
    config_name = args.CONFIG
    config_path = './configs/' + config_name
    default_dst_dir = "./results/"
    out_file = default_dst_dir + config_name + '/'
    os.makedirs(out_file, exist_ok=True)

    # Load the configuration params of the experiment
    full_config_path = config_path + ".yaml"
    print(f"Loading experiment {full_config_path}")
    with open(full_config_path, "r") as f:
        exp_config = yaml.load(f, Loader=yaml.SafeLoader)
    return args, exp_config, out_file


def define_dataset(exp_config, args):
    mean = exp_config['data']['mean']
    std = exp_config['data']['std']
    if exp_config['data']['augmentations']:
        train_transform = get_training_augmentations(m=mean, s=std)
    else:
        train_transform = get_validation_augmentations(m=mean, s=std)
    valid_transform = get_validation_augmentations(m=mean, s=std)

    x_train_dir = exp_config['data']['train']['path']
    x_valid_dir = exp_config['data']['val']['path']
    x_test_dir = exp_config['data']['test']['path']

    valid_dataset = Dataset(x_valid_dir, exp_config['data']['sets'],
                            augmentation=valid_transform, return_crs=True)
    test_dataset = Dataset(x_test_dir, exp_config['data']['sets'],
                           augmentation=valid_transform, return_crs=True)
    train_dataset = Dataset(x_train_dir, exp_config['data']['sets'],
                            augmentation=train_transform)

    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=args.num_workers,
                              batch_size=exp_config['data']['batch_size'],pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=args.num_workers)

    return train_loader, valid_loader, test_loader


def define_loss(exp_config):
    name_3dloss = exp_config['model']['3d_loss']
    weights3d = 0.75
    if '3d_loss_weights' in list(exp_config['model'].keys()):
        weights3d = exp_config['model']['3d_loss_weights']
    criterion3d = choose_criterion3d(name=name_3dloss, class_weights=weights3d)

    weights2d = exp_config['model']['2d_loss_weights']
    class_weights2d = torch.FloatTensor(weights2d)  # .cuda()
    name_2dloss = exp_config['model']['2d_loss']
    criterion2d = choose_criterion2d(name_2dloss, class_weights2d)

    return criterion2d, criterion3d


def print_model_size(model):
    summary = pl.utilities.model_summary.ModelSummary(model, max_depth=-1)
    print(summary)

    from thop import profile, clever_format
    input_tensor1 = torch.randn((1, 3, 512, 512))
    input_tensor2 = torch.randn((1, 3, 512, 512))
    inputs = (input_tensor1, input_tensor2)

    # Calculate the GFLOPs of the model
    flops, params = profile(model.model, inputs=inputs)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    # print(f"Model GFLOPs: {flops/1e9:.2f}")
    import pdb
    pdb.set_trace()


def load_finetune_checkpoint(model, path):
    # import pdb;pdb.set_trace()
    try:
        m = torch.load(path, map_location=torch.device('cpu'))['state_dict']
    except:
        m = torch.load(path, map_location=torch.device('cpu'))['optimizer_G_state_dict']
    model_dict = model.state_dict()
    for k in m.keys():

        if k in model_dict and model_dict[k].shape == m[k].shape:
            pname = k
            pval = m[k]
            # import pdb;pdb.set_trace()
            model_dict[pname] = pval.clone().to(model_dict[pname].device)

    model.load_state_dict(model_dict)

def load_finetune_checkpoint_res18(model, path):
    # import pdb;pdb.set_trace()
    m = torch.load(path, map_location=torch.device('cpu'))

    model_dict = model.state_dict()
    pre = 'model.Tenc_x2.resnet.resnet.'
    match_keys = []
    for k in m.keys():

        if pre+k in model_dict and model_dict[pre+k].shape == m[k].shape:
            match_keys.append(k)
            pname = pre+k
            pval = m[k]
            # import pdb;pdb.set_trace()
            model_dict[pname] = pval.clone().to(model_dict[pname].device)


    model.load_state_dict(model_dict)


def main_train(args, exp_config, out_file):
    from pytorch_lightning import seed_everything
    seed_everything(18,workers=True)

    criterion2d, criterion3d = define_loss(exp_config)
    train_loader, valid_loader, test_loader = define_dataset(exp_config, args)
    model_name = exp_config['model']['model']
    # import pdb;pdb.set_trace()
    exec(f'from models.{model_name} import {model_name}')
    model = eval(f'{model_name}()')
    save_ckpt_func= ModelCheckpoint(save_top_k=-1, every_n_epochs=10, save_on_train_epoch_end=True)
    try:
        pretrain_path = exp_config['model']['pretrain_path']
    except:
        pretrain_path = None
    try:
        resume_path = exp_config['model']['resume_path']
    except:
        resume_path = None
    
    if not pretrain_path or args.eval:
        pl_model = pl_trainer(model=model, exp_config=exp_config,
                              criterion2d=criterion2d, criterion3d=criterion3d,
                              aux='return_aux' in exp_config['model'].keys(),
                              save_img=args.save_img)
    elif args.eval != True:
        print(f'Load pretrain ckpt from {pretrain_path}')

        try:
            pl_model = pl_trainer.load_from_checkpoint(
                checkpoint_path=pretrain_path,
                strict=False, model=model, exp_config=exp_config,
                criterion2d=criterion2d, criterion3d=criterion3d,
                aux='return_aux' in exp_config['model'].keys(),
                save_img=args.save_img)
            #load_finetune_checkpoint_res18(pl_model, './results/resnet_weights/resnet18-5c106cde.pth')
        except:

            pl_model = pl_trainer(model=model, exp_config=exp_config,
                                  criterion2d=criterion2d, criterion3d=criterion3d,
                                  aux='return_aux' in exp_config['model'].keys(),
                                  save_img=args.save_img)
            load_finetune_checkpoint(pl_model, pretrain_path)

    if args.show_size:
        print_model_size(pl_model)
    if args.eval != True:
        #import pdb;pdb.set_trace()
        exp_name = input('Please specify the experiment name:')
        logger = pl.loggers.CSVLogger(save_dir=os.path.join('results', args.CONFIG), version=f'version_{exp_name}')
        trainer = pl.Trainer(default_root_dir=out_file,
                             max_epochs=exp_config['optim']['num_epochs'], devices=[args.device],
                             accumulate_grad_batches=2,#deterministic=True,
                             callbacks=[save_ckpt_func],
                             logger=logger)
        #import pdb;pdb.set_trace()
        if not resume_path:
            trainer.fit(model=pl_model, train_dataloaders=train_loader)
        else:
            trainer.fit(model=pl_model, train_dataloaders=train_loader, ckpt_path=resume_path)

        trainer.test(model=pl_model, dataloaders=test_loader)
    else:
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(save_dir=os.path.join('results', args.CONFIG, 'lightning_logs'),
                                    name='test_'+args.ckpt_version,
                                   version='version_test_'+args.ckpt_version)
        trainer = pl.Trainer(default_root_dir=out_file,
                             max_epochs=exp_config['optim']['num_epochs'],
                             devices=[args.device],
                             num_nodes=1,
                             logger=logger,
                             inference_mode=False)

        if args.ckpt_version != '-1':
            ckpt_path = os.path.join('results', args.CONFIG, 'lightning_logs',
            'version_' + args.ckpt_version, 'checkpoints') + '/'
        else:
            ckpt_path = os.path.join('results', args.CONFIG) + '/'

        ckpt_names = [x for x in os.listdir(ckpt_path) 
                      if x.endswith('.ckpt')]
        ckpt_names.sort(key=lambda x:whichepoch(x))
        ckpt_path += ckpt_names[-args.ckpt_idx]

        trainer.test(model=pl_model, dataloaders=test_loader,
                     ckpt_path=ckpt_path)

def whichepoch(x):
    return int(x.split('=')[1].split('-')[0])

if __name__ == '__main__':

    manual_seed = 3407

    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    cudnn.deterministic = True
    cudnn.benchmark = False



    args, exp_config, out_file = parse_args()

    d_flag = exp_config['data']['train']['path']
    if type(d_flag) == list:
        d_flag = d_flag[0]
    if 'hdf5' in d_flag:
        from utils.dataloader_hd5 import Dataset
    else:
        from utils.dataloader import Dataset
    from utils.augmentations import get_training_augmentations, get_validation_augmentations

    if '2d' in exp_config['model']['model'].lower() and 'levir' not in args.CONFIG:
        from pl_models.pl_model_2d import pl_trainer
    elif '3d' in exp_config['model']['model'].lower() and 'levir' not in args.CONFIG:
        from pl_models.pl_model_3d import pl_trainer

    elif 'levir' in args.CONFIG:
        from pl_models.pl_model_levir import pl_trainer
        from utils.augmentations_levir import get_training_augmentations, get_validation_augmentations
        from utils.dataloader_levir import Dataset
    elif 'plabel' in args.CONFIG:
        from pl_models.pl_model_plabel import pl_trainer
        from utils.dataloader_plabel import Dataset

    elif len(exp_config['data']['sets']) == 4:
        from pl_models.pl_model import pl_trainer


    main_train(args, exp_config, out_file)

