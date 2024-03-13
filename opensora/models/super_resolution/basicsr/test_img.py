import logging
import torch
from os import path as osp
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def image_sr(args):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(args.root_path, is_train=False)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        dataset_opt['dataroot_lq'] = osp.join(args.output_dir, f'temp_LR')
        if args.SR == 'x4': 
            opt['upscale'] = opt['network_g']['upscale'] = 4
            opt['val']['suffix'] = 'x4'
            opt['path']['pretrain_network_g'] = osp.join(args.root_path, f'experiments/pretrained_models/RGT_x4.pth')
        if args.SR == 'x2': 
            opt['upscale'] = opt['network_g']['upscale'] = 2
            opt['val']['suffix'] = 'x2'
            
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        test_loaders.append(test_loader)

    opt['path']['pretrain_network_g'] = args.ckpt_path
    opt['val']['use_chop'] = args.use_chop
    opt['path']['visualization'] = osp.join(args.output_dir, f'temp_results')
    opt['path']['results_root'] = osp.join(args.output_dir, f'temp_results')

    # create model
    model = build_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # print(root_path)
    # image_sr(root_path)
