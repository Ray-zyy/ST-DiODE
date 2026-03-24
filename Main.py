from __future__ import annotations

import os

import torch

from config import parse_args
from data_preparation import load_data
from engine import Engine
from utils import count_parameters, get_log_dir, get_logger, init_seed

try:
    from models.ST_DiODE import ST_DiODE
except ImportError:
    from ST_DiODE import ST_DiODE


def build_device(args) -> torch.device:
    use_cuda = args.use_gpu and torch.cuda.is_available() and args.device.startswith('cuda')
    if use_cuda:
        return torch.device(f'cuda:{args.gpu}')
    return torch.device('cpu')


def main():
    args = parse_args()
    device = build_device(args)

    init_seed(args.seed)

    args.log_dir = get_log_dir(args.work_dir, args.model, args.dataname)
    os.makedirs(args.log_dir, exist_ok=True)
    if not args.checkpoint:
        args.checkpoint = os.path.join(args.log_dir, f'{args.dataname}_{args.model}_best.pth')

    logger = get_logger(args.log_dir, name=args.model, debug=args.debug)
    logger.info('Using device: %s', device)
    logger.info('Log directory: %s', args.log_dir)
    logger.info('Checkpoint path: %s', args.checkpoint)

    train_loader, valid_loader, test_loader, data_mean, data_std = load_data(**vars(args))
    valid_loader = test_loader if valid_loader is None else valid_loader

    target_T_out = 20 if args.dataname == 'kth' else args.in_shape[0]
    model = ST_DiODE(
        shape_in=tuple(args.in_shape),
        T_out=target_T_out,
        hid_S=args.hid_S,
        hid_T=args.hid_T,
        N_S=args.N_S,
        N_T=args.N_T,
    )

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        logger.info('Detected %d GPUs. Enabling DataParallel.', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    logger.info('Trainable parameters: %d', count_parameters(model))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
    )
    criterion = torch.nn.MSELoss()

    engine = Engine(
        args=args,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        scaler=(data_mean, data_std),
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        logger=logger,
        device=device,
    )

    if args.debug:
        logger.info('Debug mode enabled. Running test only.')
        engine.test()
        return

    logger.info('>>>>>>>>>>>>>>>>>>>> Training starts >>>>>>>>>>>>>>>>>>>>')
    engine.train()
    logger.info('>>>>>>>>>>>>>>>>>>>> Testing starts  >>>>>>>>>>>>>>>>>>>>')
    engine.test()


if __name__ == '__main__':
    main()
