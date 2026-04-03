import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from models.FCN import FCN
from models.GTM import GTM
from utils.data_multitrends import ZeroShotDataset


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


def build_test_loader(test_df, args, gtrends, cat_dict, col_dict, fab_dict):
    use_competition = bool(args.use_competition_extension) and args.model_type == "GTM"

    dataset = ZeroShotDataset(
        data_df=test_df,
        img_root=Path(args.data_folder) / "images",
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        trend_len=args.trend_len,
        target_cols=args.target_cols,
        temporal_cols=args.temporal_cols,
        text_cols=args.text_cols,
        trend_cols=args.trend_cols,
        image_col=args.image_col,
        use_competition_extension=use_competition,
        competition_reference_df=test_df if use_competition else None,
        competition_topk_indices_path=args.competition_topk_indices_path if use_competition else None,
        competition_topk_values_path=args.competition_topk_values_path if use_competition else None,
        competition_row_mapping_path=args.competition_row_mapping_path if use_competition else None,
        competition_meta_path=args.competition_meta_path if use_competition else None,
        competition_top_k=args.competition_top_k,
    )
    return dataset.get_loader(batch_size=1, train=False)


def build_model(args, cat_dict, col_dict, fab_dict):
    if args.model_type == "FCN":
        return FCN(
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
            gpu_num=args.gpu_num,
        )

    if args.model_type == "GTM":
        return GTM(
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
            gpu_num=args.gpu_num,
            use_competition_extension=args.use_competition_extension,
            competition_top_k=args.competition_top_k,
        )

    raise ValueError("model_type must be either 'FCN' or 'GTM'")


def load_checkpoint_state_dict(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError("Checkpoint format not recognized.")

    model.load_state_dict(state_dict, strict=True)
    return model


def unpack_and_forward(model, batch, model_type):
    if len(batch) == 7:
        item_sales, category, color, fabric, temporal_features, gtrends, images = batch

        if model_type == "FCN":
            y_pred = model(
                category,
                color,
                fabric,
                temporal_features,
                gtrends,
                images,
            )
            attn = None
        else:
            y_pred, attn, _ = model(
                category,
                color,
                fabric,
                temporal_features,
                gtrends,
                images,
            )

        return item_sales, y_pred, attn

    if model_type == "FCN":
        raise ValueError(
            "FCN does not support the competition-extension batch format."
        )

    (
        item_sales,
        category,
        color,
        fabric,
        temporal_features,
        gtrends,
        images,
        neighbor_categories,
        neighbor_colors,
        neighbor_fabrics,
        neighbor_images,
        neighbor_scores,
        neighbor_mask,
    ) = batch

    y_pred, attn, _ = model(
        category,
        color,
        fabric,
        temporal_features,
        gtrends,
        images,
        neighbor_categories=neighbor_categories,
        neighbor_colors=neighbor_colors,
        neighbor_fabrics=neighbor_fabrics,
        neighbor_images=neighbor_images,
        neighbor_scores=neighbor_scores,
        neighbor_mask=neighbor_mask,
    )
    return item_sales, y_pred, attn


def load_rescale_values(data_folder, eval_horizon):
    scale_data = np.load(Path(data_folder) / "normalization_scale.npy")

    if np.ndim(scale_data) == 0:
        scale = float(scale_data)
        return np.full(eval_horizon, scale, dtype=np.float32)

    scale_data = np.array(scale_data).reshape(-1)
    if len(scale_data) < eval_horizon:
        raise ValueError(
            f"normalization_scale.npy has length {len(scale_data)}, "
            f"but eval_horizon is {eval_horizon}."
        )

    return scale_data[:eval_horizon].astype(np.float32)


def run(args):
    print(args)
    device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu")
    pl.seed_everything(args.seed)

    data_folder = Path(args.data_folder)

    test_df = pd.read_csv(
        data_folder / "test.csv",
        parse_dates=["release_date"],
    )
    item_codes = test_df["external_code"].astype(str).values

    cat_dict = torch.load(
        data_folder / "category_labels.pt",
        weights_only=False,
    )
    col_dict = torch.load(
        data_folder / "color_labels.pt",
        weights_only=False,
    )
    fab_dict = torch.load(
        data_folder / "fabric_labels.pt",
        weights_only=False,
    )
    gtrends = pd.read_csv(
        data_folder / "gtrends.csv",
        index_col=[0],
        parse_dates=True,
    )

    if args.use_competition_extension and args.model_type != "GTM":
        raise ValueError("The competition extension is implemented only for model_type='GTM'.")

    test_loader = build_test_loader(test_df, args, gtrends, cat_dict, col_dict, fab_dict)
    model = build_model(args, cat_dict, col_dict, fab_dict)
    model = load_checkpoint_state_dict(model, args.ckpt_path, device)
    model.to(device)
    model.eval()

    model_savename = (
        f"{args.model_type}_{args.wandb_run}_"
        f"model{args.model_output_dim}_eval{args.eval_horizon}"
    )

    gt = []
    forecasts = []
    attns = []

    for batch in tqdm(test_loader, total=len(test_loader), ascii=True):
        with torch.no_grad():
            batch = [tensor.to(device) if torch.is_tensor(tensor) else tensor for tensor in batch]
            item_sales, y_pred, attn = unpack_and_forward(model, batch, args.model_type)

        y_pred_np = y_pred.detach().cpu().numpy().reshape(-1)
        y_true_np = item_sales.detach().cpu().numpy().reshape(-1)

        forecasts.append(y_pred_np[: args.eval_horizon])
        gt.append(y_true_np[: args.eval_horizon])

        if attn is not None:
            attns.append(attn.detach().cpu().numpy())

    forecasts = np.array(forecasts)
    gt = np.array(gt)

    rescale_vals = load_rescale_values(args.data_folder, args.eval_horizon)
    rescaled_forecasts = forecasts * rescale_vals
    rescaled_gt = gt * rescale_vals

    print_error_metrics(gt, forecasts, rescaled_gt, rescaled_forecasts)

    Path("results").mkdir(parents=True, exist_ok=True)

    save_payload = {
        "results": rescaled_forecasts,
        "gts": rescaled_gt,
        "codes": item_codes.tolist(),
    }

    if len(attns) > 0:
        save_payload["attns"] = attns

    torch.save(
        save_payload,
        Path("results") / f"{model_savename}.pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot sales forecasting")

    parser.add_argument("--data_folder", type=str, default=".")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--seed", type=int, default=21)

    parser.add_argument("--model_type", type=str, default="GTM", help="Choose between GTM or FCN")
    parser.add_argument("--use_trends", type=int, default=1)
    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--model_output_dim", type=int, default=12)
    parser.add_argument("--eval_horizon", type=int, default=12)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)

    # dataset columns
    parser.add_argument("--target_cols", nargs="+", default=[str(i) for i in range(12)])
    parser.add_argument("--temporal_cols", nargs="+", default=["day", "week", "month", "year"])
    parser.add_argument("--text_cols", nargs="+", default=["category", "color", "fabric"])
    parser.add_argument("--trend_cols", nargs="+", default=["category", "color", "fabric"])
    parser.add_argument("--image_col", type=str, default="image_path")

    parser.add_argument("--use_competition_extension", type=int, default=0)
    parser.add_argument("--competition_top_k", type=int, default=4)
    parser.add_argument(
        "--competition_topk_indices_path",
        type=str,
        default="test_topk_indices.npy",
    )
    parser.add_argument(
        "--competition_topk_values_path",
        type=str,
        default="test_topk_values.npy",
    )
    parser.add_argument(
        "--competition_row_mapping_path",
        type=str,
        default="test_availability_row_mapping.csv",
    )
    parser.add_argument(
        "--competition_meta_path",
        type=str,
        default="test_extension_meta.json",
    )

    parser.add_argument("--wandb_run", type=str, default="Run1")

    args = parser.parse_args()

    if args.eval_horizon > args.model_output_dim:
        raise ValueError(
            f"eval_horizon ({args.eval_horizon}) cannot be bigger than "
            f"model_output_dim ({args.model_output_dim})."
        )

    run(args)