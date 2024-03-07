import logging
import torch
from os import path as osp
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def image_sr(args):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(args.root_path, args.SR, is_train=False)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        dataset_opt['dataroot_gt'] = osp.join(args.output_dir, f'temp_HR')
        if args.SR == 'x4': 
            dataset_opt['dataroot_lq'] = osp.join(args.output_dir, f'temp_LR/X4')
        if args.SR == 'x2': 
            dataset_opt['dataroot_lq'] = osp.join(args.output_dir, f'temp_LR/X2')
        
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        test_loaders.append(test_loader)

    # create model
    model = build_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # print(root_path)
    # image_sr(root_path)
