import argparse
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from models.GTM import GTM
from models.FCN import FCN
from utils.data_multitrends import ZeroShotDataset
from sklearn.metrics import mean_absolute_error
from pathlib import Path


def compute_forecast_metrics_np(y_true, y_pred, erp_epsilon=0.1):
    abs_err = np.abs(y_true - y_pred)

    mae = abs_err.mean()
    wape = 100.0 * abs_err.sum() / max(y_true.sum(), 1e-12)

    mae_per_series = abs_err.mean(axis=1)
    mae_per_series = np.maximum(mae_per_series, 1e-12)

    signed_error_per_series = (y_true - y_pred).sum(axis=1)
    ts_per_series = signed_error_per_series / mae_per_series
    ts = ts_per_series.mean()

    erp_per_series = (abs_err >= erp_epsilon).sum(axis=1)
    erp = erp_per_series.mean()

    return round(wape, 3), round(mae, 3), round(ts, 3), round(erp, 3)


def print_error_metrics(y_true, y_pred, rescaled_y_true, rescaled_y_pred):
    wape, mae, ts, erp = compute_forecast_metrics_np(y_true, y_pred)
    rwape, rmae, rts, rerp = compute_forecast_metrics_np(rescaled_y_true, rescaled_y_pred)

    print("Normalized:", {"WAPE": wape, "MAE": mae, "TS": ts, "ERP": erp})
    print("Rescaled:", {"WAPE": rwape, "MAE": rmae, "TS": rts, "ERP": rerp})


def run(args):
    print(args)

    # Set up CUDA
    device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')

    # Seeds for reproducibility
    pl.seed_everything(args.seed)

    # Load sales data
    test_df = pd.read_csv(Path(args.data_folder + 'test.csv'), parse_dates=['release_date'])
    item_codes = test_df['external_code'].values

    # Load category and color encodings
    cat_dict = torch.load(Path(args.data_folder + 'category_labels.pt'), weights_only=False)
    col_dict = torch.load(Path(args.data_folder + 'color_labels.pt'), weights_only=False)
    fab_dict = torch.load(Path(args.data_folder + 'fabric_labels.pt'), weights_only=False)

    # Load Google trends
    gtrends = pd.read_csv(Path(args.data_folder + 'gtrends.csv'), index_col=[0], parse_dates=True)

    test_loader = ZeroShotDataset(test_df, Path(args.data_folder + '/images'), gtrends, cat_dict, col_dict, \
                                  fab_dict, args.trend_len).get_loader(batch_size=1, train=False)

    model_savename = f'{args.wandb_run}_model{args.model_output_dim}_eval{args.eval_horizon}'

    # Create model
    model = None
    if args.model_type == 'FCN':
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.model_output_dim,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_trends=args.use_trends,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            gpu_num=args.gpu_num
        )
    else:
        model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.model_output_dim,
            num_heads=args.num_attn_heads,
            num_layers=args.num_hidden_layers,
            cat_dict=cat_dict,
            col_dict=col_dict,
            fab_dict=fab_dict,
            use_text=args.use_text,
            use_img=args.use_img,
            trend_len=args.trend_len,
            num_trends=args.num_trends,
            use_encoder_mask=args.use_encoder_mask,
            autoregressive=args.autoregressive,
            gpu_num=args.gpu_num
        )

    # Fail if the checkpoint and model architecture don't match
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=True)

    # Forecast the testing set
    model.to(device)
    model.eval()
    gt, forecasts, attns = [], [], []
    for test_data in tqdm(test_loader, total=len(test_loader), ascii=True):
        with torch.no_grad():
            test_data = [tensor.to(device) for tensor in test_data]
            item_sales, category, color, textures, temporal_features, gtrends, images = test_data
            y_pred, att = model(category, color, textures, temporal_features, gtrends, images)

            # Make sure you cut the evaluation horizon instead of the model architecture
            y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
            y_true_np = item_sales.detach().cpu().numpy().reshape(-1)
            forecasts.append(y_pred_np[:args.eval_horizon])
            gt.append(y_true_np[:args.eval_horizon])

            attns.append(att.detach().cpu().numpy())

    attns = np.stack(attns)
    forecasts = np.array(forecasts)
    gt = np.array(gt)

    # rescale_vals = np.load(args.data_folder + 'normalization_scale.npy')[:args.eval_horizon]

    # Rescale the values in such a way that it won't end up with a 0-dimentional vector
    scale = float(np.load(Path(args.data_folder) / 'normalization_scale.npy'))
    rescale_vals = np.full(args.eval_horizon, scale, dtype=np.float32)

    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals
    print_error_metrics(gt, forecasts, rescaled_gt, rescaled_forecasts)

    Path('results').mkdir(parents=True, exist_ok=True)
    torch.save({'results': rescaled_forecasts, 'gts': rescaled_gt, 'codes': item_codes.tolist()},
               Path('results/' + model_savename + '.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')

    # General arguments
    parser.add_argument('--data_folder', type=str, default='dataset/')
    parser.add_argument('--ckpt_path', type=str, default='log/path-to-model.ckpt')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--seed', type=int, default=21)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM', help='Choose between GTM or FCN')
    parser.add_argument('--use_trends', type=int, default=1)
    parser.add_argument('--use_img', type=int, default=1)
    parser.add_argument('--use_text', type=int, default=1)
    parser.add_argument('--trend_len', type=int, default=52)
    parser.add_argument('--num_trends', type=int, default=3)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=64)

    # Instead of using the output_dim for the model and for the evaluation we split this such that you do not change
    # the model when having two different values
    parser.add_argument('--model_output_dim', type=int, default=12)
    parser.add_argument('--eval_horizon', type=int, default=12)

    parser.add_argument('--use_encoder_mask', type=int, default=1)
    parser.add_argument('--autoregressive', type=int, default=0)
    parser.add_argument('--num_attn_heads', type=int, default=4)
    parser.add_argument('--num_hidden_layers', type=int, default=1)

    # wandb arguments
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()

    # Add to check if the evaluation size is smaller than the model output size
    if args.eval_horizon > args.model_output_dim:
        raise ValueError(
            f"eval_horizon ({args.eval_horizon}) cannot be bigger than "
            f"model_output_dim ({args.model_output_dim})."
        )

    run(args)