import os
import datetime
import time
import argparse


import torch
from torch.utils.data import DataLoader

from model import create_crf_model, create_model
from lr_scheduler import poly_lr_scheduler, warmup_lr_scheduler
from trainer import train_one_epoch, evaluate
from augmentations import get_transform

from colorpalette import show_config
from dataset import VOCSegmentation, VOCSegmentationVal

import yaml

def parse_args():

    parser = argparse.ArgumentParser(description="pytorch fcn training")

    with open('config.yml', 'r') as f:
        data = yaml.safe_load(f)
        args = data['train_config']
        crf_args = data['crf_config']

    parser.add_argument("--data-path", default=args['data_path'], help="VOCdevkit root")
    parser.add_argument("--aux", default=False, type=bool, help="auxilier loss")

    parser.add_argument("--num-classes", default=args['num_classes'], type=int)
    parser.add_argument("--crf", default=args['crf'], type=bool)
    parser.add_argument("--fullscaleFeat", default=args['fullscaleFeat'])
    parser.add_argument("--held-out-images", default=args['held_out_images'])
    parser.add_argument("--train-with-held-out", default=args['train_with_held_out'], type=bool)
    parser.add_argument("--lr_scheduler", default=args['lr_scheduler'], type=str)
    parser.add_argument("--loss", default=args['loss'], type=str)

    parser.add_argument("--device", default=args['device'], help="training device")

    parser.add_argument("-b", "--batch-size", default=args['batch_size'], type=int)

    parser.add_argument("--epochs", default=args['epochs'], type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=args['lr'], type=float, help='initial learning rate')

    parser.add_argument('--wd', '--weight-decay', default=args['weight_decay'], type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=args['print_freq'], type=int, help='print frequency')

    parser.add_argument('--resume', default=args['resume'], help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=args['start_epoch'], type=int, metavar='N',
                        help='start epoch')
    parser.add_argument("--amp", default=args['amp'], type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args(args=[])

    return args

args = parse_args()

def main(args):

    with open('config.yml', 'r') as f:
        data = yaml.safe_load(f)
        crf_args = data['crf_config']
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    train_transforms = get_transform(train=True)
    train_dataset = VOCSegmentation(args.data_path,
                                    year='2012',
                                    transforms=train_transforms,
                                    held_out=args.held_out_images,
                                    train_with_held_out=args.train_with_held_out,
                                    txt_name="train.txt")
    num_train_images = train_dataset.__len__()

    val_transforms = get_transform(train=False)
    val_dataset = VOCSegmentationVal(args.data_path,
                                  year='2012',
                                  transforms=val_transforms,
                                  txt_name="val.txt")
    
    num_val_images = val_dataset.__len__()
    
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=train_dataset.collate_fn)

    val_loader = DataLoader(val_dataset,
                            batch_size=4,
                            num_workers=num_workers,
                            pin_memory=True,
                            collate_fn=val_dataset.collate_fn)
    num_steps_per_epoch = len(train_loader)

    if args.crf:
        model = create_crf_model(pretrian_path=None, config=crf_args.config, num_classes=num_classes, freeze_backbone=True, fullscaleFeat=args.fullscaleFeat)
    else:
        model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.parameters() if p.requires_grad]},
    ]
    
    named_parameter = [
        name for name, param in model.named_parameters() if param.requires_grad
    ]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    assert args.lr_scheduler in ['warmup', 'poly']

    if args.lr_scheduler == 'warmup':
        lr_scheduler = warmup_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)
    elif args.lr_scheduler == 'poly':
        lr_scheduler = poly_lr_scheduler(optimizer, len(train_loader), args.epochs)

    args.start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])
            
    if args.lr_scheduler == 'warmup':
        named = 'GradualWarmupScheduler'
    elif args.lr_scheduler == 'poly':
        named = 'PolynomialScheduler'
        
    if args.loss == 'CE':
        loss = 'CrossEntropy'
            
    titles = {
        'device': device, 'batch size': batch_size, 'num workers': num_workers, 'learning rate': args.lr, 'loss function': loss,
        'params to optimize': named_parameter, 'optimizer': optimizer, 'scaler': scaler, 'epochs': args.epochs, 
        'weight decay': args.weight_decay, 'lr_scheduler': named, 'start epochs': args.start_epoch, 'aux': args.aux, 
        'crf': args.crf, 'fullscaleFeat': args.fullscaleFeat, 'held out images': args.held_out_images, 
        'train with held out': args.train_with_held_out, 'num train images': num_train_images,
        'num val images': num_val_images, 'num steps per epoch': num_steps_per_epoch,
        'train transfrom': train_transforms.trans, 'val transforms': val_transforms.trans,
    }
    if not args.crf:
        titles.pop('params to optimize', None)
            
    config = show_config(titles)
    config.log_every()
    
    print(config)

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        
        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)

        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n"
            f.write(train_info + val_info + "\n\n")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, f"model{epoch}.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

if __name__ == '__main__':
    main(args)
