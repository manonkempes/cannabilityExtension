import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from models.FCN import FCN
from models.GTM import GTM
from utils.data_multitrends import ZeroShotDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_dataset(
    data_df,
    full_reference_df,
    args,
    gtrends,
    cat_dict,
    col_dict,
    fab_dict,
    train_mode,
):
    use_competition = bool(args.use_competition_extension) and args.model_type == "GTM"

    dataset = ZeroShotDataset(
        data_df=data_df,
        img_root=Path(args.data_folder) / "images",
        img_tensor_root=Path(args.data_folder) / "images_pt",
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
        competition_reference_df=full_reference_df if use_competition else None,
        competition_topk_indices_path=args.competition_topk_indices_path if use_competition else None,
        competition_topk_values_path=args.competition_topk_values_path if use_competition else None,
        competition_row_mapping_path=args.competition_row_mapping_path if use_competition else None,
        competition_meta_path=args.competition_meta_path if use_competition else None,
        competition_top_k=args.competition_top_k,
    )

    cache_name = "train_cache.pt" if train_mode else "val_cache.pt"

    return dataset.get_loader(
        batch_size=args.batch_size if train_mode else 1,
        train=train_mode,
        cache_path=Path(args.data_folder) / "cache" / cache_name,
    )

def run(args):
    print(args)
    pl.seed_everything(args.seed)

    data_folder = Path(args.data_folder)
    log_dir = Path(args.log_dir)
    (log_dir / args.model_type).mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(
        data_folder / "train.csv",
        parse_dates=["release_date"],
    )

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

    train_df = train_df.sort_values("release_date").reset_index(drop=True)
    val_size = max(1, int(0.15 * len(train_df)))
    subtrain_df = train_df.iloc[:-val_size].copy()
    val_df = train_df.iloc[-val_size:].copy()

    if args.use_competition_extension and args.model_type != "GTM":
        raise ValueError("The competition extension is implemented only for model_type='GTM'.")

    train_loader = build_dataset(
        data_df=subtrain_df,
        full_reference_df=train_df,
        args=args,
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        train_mode=True,
    )

    val_loader = build_dataset(
        data_df=val_df,
        full_reference_df=train_df,
        args=args,
        gtrends=gtrends,
        cat_dict=cat_dict,
        col_dict=col_dict,
        fab_dict=fab_dict,
        train_mode=False,
    )

    if args.model_type == "FCN":
        model = FCN(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
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
    elif args.model_type == "GTM":
        model = GTM(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
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
    else:
        raise ValueError("model_type must be either 'FCN' or 'GTM'")

    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    model_savename = f"{args.model_type}_{args.wandb_run}"

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=str(log_dir / args.model_type),
        filename=model_savename + "---{epoch}---" + dt_string,
        monitor="val_mae",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_mae",
        min_delta=0.0,
        patience=5,
        verbose=True,
        mode="min",
    )

    tb_logger = pl_loggers.TensorBoardLogger(str(log_dir), name=model_savename)

    use_gpu = torch.cuda.is_available()

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        precision="16-mixed" if use_gpu else 32,
        max_epochs=args.epochs,
        check_val_every_n_epoch=1,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("Best model path:", checkpoint_callback.best_model_path)
    print("Last model path:", checkpoint_callback.last_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot sales forecasting")

    # data / output
    parser.add_argument("--data_folder", type=str, default=".")
    parser.add_argument("--log_dir", type=str, default="log")

    # training
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)

    # model
    parser.add_argument("--model_type", type=str, default="GTM", help="Choose between GTM or FCN")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=12)
    parser.add_argument("--num_attn_heads", type=int, default=4)
    parser.add_argument("--num_hidden_layers", type=int, default=1)

    # input usage
    parser.add_argument("--use_trends", type=int, default=1)
    parser.add_argument("--use_img", type=int, default=1)
    parser.add_argument("--use_text", type=int, default=1)
    parser.add_argument("--trend_len", type=int, default=52)
    parser.add_argument("--num_trends", type=int, default=3)
    parser.add_argument("--use_encoder_mask", type=int, default=1)
    parser.add_argument("--autoregressive", type=int, default=0)

    # dataset columns
    parser.add_argument("--target_cols", nargs="+", default=[str(i) for i in range(12)])
    parser.add_argument("--temporal_cols", nargs="+", default=["day", "week", "month", "year"])
    parser.add_argument("--text_cols", nargs="+", default=["category", "color", "fabric"])
    parser.add_argument("--trend_cols", nargs="+", default=["category", "color", "fabric"])
    parser.add_argument("--image_col", type=str, default="image_path")

    # competition extension
    parser.add_argument("--use_competition_extension", type=int, default=0)
    parser.add_argument("--competition_top_k", type=int, default=4)
    parser.add_argument(
        "--competition_topk_indices_path",
        type=str,
        default="train_topk_indices.npy",
    )
    parser.add_argument(
        "--competition_topk_values_path",
        type=str,
        default="train_topk_values.npy",
    )
    parser.add_argument(
        "--competition_row_mapping_path",
        type=str,
        default="train_availability_row_mapping.csv",
    )
    parser.add_argument(
        "--competition_meta_path",
        type=str,
        default="train_extension_meta.json",
    )

    # naming
    parser.add_argument("--wandb_entity", type=str, default="username-here")
    parser.add_argument("--wandb_proj", type=str, default="GTM")
    parser.add_argument("--wandb_run", type=str, default="Run1")

    args = parser.parse_args()
    run(args)