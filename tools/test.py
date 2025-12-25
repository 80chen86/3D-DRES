import argparse
import gorilla, os
import torch
from tqdm import tqdm
import numpy as np
from detailbase.dataset import build_dataloader, build_dataset
from detailbase.model import DetailBase
from detailbase.utils.mask_encoder import rle_decode, rle_encode
from detailbase.utils import get_root_logger, save_pred_instances
import json

def get_args():
    parser = argparse.ArgumentParser('3D-DRES')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', default=None, type=str, help='directory for output results')
    parser.add_argument('--num_gpus', type=int, default=1, help='number of gpus')
    parser.add_argument('--gpu_id', type=int, default=[0], nargs='+', help='ids of gpus to use')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    gorilla.set_cuda_visible_devices(gpu_ids=args.gpu_id, num_gpu=args.num_gpus)

    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger(log_file=args.checkpoint.replace('.pth', '.log'))

    model = DetailBase(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)

    keys, gt_pmasks, pred_pmasks, attn_map, pious, sious, long_ious, long_sious, complex_ious, complex_sious = [], [], [], [], [], [], [], [], [], []
    iou_dict = {}
    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            res = model(batch, mode='predict')
            keys.extend(res['key'])
            pious.extend(res['piou'])
            sious.extend(res['siou'])
            long_ious.extend(res['long_iou'])
            long_sious.extend(res['long_siou'])
            complex_ious.extend(res['complex_iou'])
            complex_sious.extend(res['complex_siou'])
            gt_pmasks.extend(res['gt_pmask'])
            attn_map.extend([map.numpy() for map in res['pred_pmask']])
            
            pred_pmasks.extend(
                [
                    rle_encode((pred_pmask>0.5).int().numpy())
                    for pred_pmask in res['pred_pmask']
                ]
            )
            progress_bar.update()
            #break
            
        progress_bar.close()

    for idx, key in enumerate(keys):
        piou = pious[idx]
        iou_dict[key] = piou.item()
    iou_path = os.path.join(os.path.dirname(args.checkpoint), 'ious.json')
    # write to json
    with open(iou_path, 'w') as f:
        json.dump(iou_dict, f)
    
    if len(long_ious) == 0: long_ious.append(torch.tensor(0))
    if len(long_sious) == 0: long_sious.append(torch.tensor(0))
    if len(complex_ious) == 0: complex_ious.append(torch.tensor(0))
    if len(complex_sious) == 0: complex_sious.append(torch.tensor(0))
    
    logger.info('Evaluate referring segmentation')
    # point-level metrics
    pious = torch.stack(pious, dim=0).cpu().numpy()
    precision_half = (pious > 0.5).sum().astype(float) / pious.size
    precision_quarter = (pious > 0.25).sum().astype(float) / pious.size
    miou = pious.mean()
    sious = torch.stack(sious, dim=0).cpu().numpy()
    smiou = sious.mean()
        
    long_ious = torch.stack(long_ious, dim=0).cpu().numpy()
    long_precision_half = (long_ious > 0.5).sum().astype(float) / long_ious.size
    long_precision_quarter = (long_ious > 0.25).sum().astype(float) / long_ious.size
    long_miou = long_ious.mean()
    long_sious = torch.stack(long_sious, dim=0).cpu().numpy()
    long_smiou = long_sious.mean()
        
    complex_ious = torch.stack(complex_ious, dim=0).cpu().numpy()
    complex_precision_half = (complex_ious > 0.5).sum().astype(float) / complex_ious.size
    complex_precision_quarter = (complex_ious > 0.25).sum().astype(float) / complex_ious.size
    complex_miou = complex_ious.mean()
    complex_sious = torch.stack(complex_sious, dim=0).cpu().numpy()
    complex_smiou = complex_sious.mean()
    
    logger.info('mIOU: {:.3f}. Acc_50: {:.3f}. Acc_25: {:.3f}. smIOU: {:.3f}'.format(miou, precision_half, precision_quarter, smiou))
    logger.info('l_mIOU: {:.3f}. l_Acc_50: {:.3f}. l_Acc_25: {:.3f}. l_smIOU: {:.3f}'.format(long_miou, long_precision_half, long_precision_quarter, long_smiou))
    logger.info('c_mIOU: {:.3f}. c_Acc_50: {:.3f}. c_Acc_25: {:.3f}. c_smIOU: {:.3f}'.format(complex_miou, complex_precision_half, complex_precision_quarter, complex_smiou))
    
    #logger.info('multi_mIOU: {:.3f}. multi_Acc_50: {:.3f}. multi_Acc_25: {:.3f}'.format(multi_miou, multi_precision_half, multi_precision_quarter))
    #for u in [0.25, 0.5]:
    #    logger.info(f'Acc@{u}: \tunique: '+str(round((unique>u).mean(), 4))+' \tmulti: '+str(round((multi>u).mean(), 4))+' \tall: '+str(round((pious>u).mean(), 4)))
    #logger.info('mIoU:\t \tunique: '+str(round(unique.mean(), 4))+' \tmulti: '+str(round(multi.mean(), 4))+' \tall: '+str(round(pious.mean(), 4)))
    
    # save output
    if args.out is None:
        output = input('If you want to save the results? (y/n)')
        if output == 'y':
            args.out = os.path.join(os.path.dirname(args.checkpoint), 'results')
        else:
            logger.info('Not saving results.')
            exit()
        
    if args.out:
        logger.info('Saving results...')
        save_pred_instances(args.out, 'pred_instance', keys, pred_pmasks)
        logger.info('Done.')

if __name__ == '__main__':
    main()
