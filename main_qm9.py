import argparse
import torch
import wandb
import copy
from tqdm import tqdm
from qm9.utils import calc_mean_mad
from utils import get_model, get_loaders, set_seed


def main(args):
    # # Generate model
    model = get_model(args).to(args.device)
    # Setup wandb
    wandb.init(project=f"QM9-{args.target_name}")
    wandb.config.update(vars(args))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')
    # # Get loaders
    train_loader, val_loader, test_loader = get_loaders(args)
    mean, mad = calc_mean_mad(train_loader)
    mean, mad = mean.to(args.device), mad.to(args.device)

    # Get optimization objects
    criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_train_mae, best_val_mae, best_model = float('inf'), float('inf'), None

    for _ in tqdm(range(args.epochs)):
        epoch_mae_train, epoch_mae_val = 0, 0

        model.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(args.device)

            pred = model(batch)
            loss = criterion(pred, (batch.y - mean) / mad)
            mae = criterion(pred * mad + mean, batch.y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optimizer.step()
            epoch_mae_train += mae.item()

        model.eval()
        for _, batch in enumerate(val_loader):
            batch = batch.to(args.device)
            pred = model(batch)
            mae = criterion(pred * mad + mean, batch.y)

            epoch_mae_val += mae.item()

        epoch_mae_train /= len(train_loader.dataset)
        epoch_mae_val /= len(val_loader.dataset)

        if epoch_mae_val < best_val_mae:
            best_val_mae = epoch_mae_val
            best_model = copy.deepcopy(model)

        scheduler.step()

        wandb.log({
            'Train MAE': epoch_mae_train,
            'Validation MAE': epoch_mae_val
        })

    test_mae = 0
    best_model.eval()
    for _, batch in enumerate(test_loader):
        batch = batch.to(args.device)
        pred = best_model(batch)
        mae = criterion(pred * mad + mean, batch.y)
        test_mae += mae.item()

    test_mae /= len(test_loader.dataset)
    print(f'Test MAE: {test_mae}')

    wandb.log({
        'Test MAE': test_mae,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='empsn',
                        help='model')
    parser.add_argument('--max_com', type=str, default='1_2',  # e.g. 1_2
                        help='model type')
    parser.add_argument('--num_hidden', type=int, default=77,
                        help='hidden features')
    parser.add_argument('--num_layers', type=int, default=7,
                        help='number of layers')
    parser.add_argument('--act_fn', type=str, default='silu',
                        help='activation function')
    parser.add_argument('--lift_type', type=str, default='rips',
                        help='lift type')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16,
                        help='learning rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='gradient clipping')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='dataset')
    parser.add_argument('--target_name', type=str, default='H',
                        help='regression task')
    parser.add_argument('--dim', type=int, default=2,
                        help='ASC dimension')
    parser.add_argument('--dis', type=float, default=4.0,
                        help='radius Rips complex')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')

    parsed_args = parser.parse_args()
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(parsed_args.seed)
    main(parsed_args)
