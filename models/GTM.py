import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from transformers import pipeline


def compute_forecast_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    erp_epsilon: float = 0.1,
):
    """Compute the validation metrics used by the repository."""
    y_true = y_true.float()
    y_pred = y_pred.float()

    abs_err = torch.abs(y_true - y_pred)
    mae = abs_err.mean()
    wape = 100.0 * abs_err.sum() / y_true.sum().clamp(min=1e-12)

    mae_per_series = abs_err.mean(dim=1).clamp(min=1e-12)
    signed_error_per_series = (y_true - y_pred).sum(dim=1)
    ts_per_series = signed_error_per_series / mae_per_series
    ts = ts_per_series.mean()

    erp_per_series = (abs_err >= erp_epsilon).float().sum(dim=1)
    erp = erp_per_series.mean()

    return wape, mae, ts, erp


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TimeDistributed(nn.Module):
    """Apply a module on the last dimension for every time step."""

    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        if self.batch_first:
            return y.contiguous().view(x.size(0), -1, y.size(-1))
        return y.view(-1, x.size(1), y.size(-1))


class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_img, use_text, dropout=0.2):
        super().__init__()
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.use_img = use_img
        self.use_text = use_text

        input_dim = embedding_dim + (embedding_dim * use_img) + (embedding_dim * use_text)
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
        )

    def forward(self, img_encoding, text_encoding, dummy_encoding):
        pooled_img = self.img_pool(img_encoding)
        condensed_img = self.img_linear(pooled_img.flatten(1))

        decoder_inputs = []
        if self.use_img == 1:
            decoder_inputs.append(condensed_img)
        if self.use_text == 1:
            decoder_inputs.append(text_encoding)
        decoder_inputs.append(dummy_encoding)

        concat_features = torch.cat(decoder_inputs, dim=1)
        return self.feature_fusion(concat_features)


class MultimodalProductEncoder(nn.Module):
    """
    Build the product representation h_i used inside the competition module.

    This representation uses image and text only, in line with the proposal's
    "multimodal embedding" option for h_i and h_j.
    """

    def __init__(self, embedding_dim, hidden_dim, use_img, use_text, dropout=0.2):
        super().__init__()
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.use_img = use_img
        self.use_text = use_text

        input_dim = (embedding_dim * use_img) + (embedding_dim * use_text)
        if input_dim == 0:
            raise ValueError("At least one of use_img or use_text must be enabled.")

        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
        )

    def forward(self, img_encoding, text_encoding):
        features = []
        if self.use_img == 1:
            pooled_img = self.img_pool(img_encoding)
            condensed_img = self.img_linear(pooled_img.flatten(1))
            features.append(condensed_img)
        if self.use_text == 1:
            features.append(text_encoding)

        concat_features = torch.cat(features, dim=1)
        return self.feature_fusion(concat_features)


class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dropout=0.2,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask

    def _generate_encoder_mask(self, size, forecast_horizon, device):
        mask = torch.zeros((size, size), device=device)
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i : i + split, i : i + split] = 1
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        return mask

    def forward(self, gtrends):
        gtrend_emb = self.input_linear(gtrends.permute(0, 2, 1))
        gtrend_emb = self.pos_embedding(gtrend_emb.permute(1, 0, 2))

        if self.use_mask == 1:
            input_mask = self._generate_encoder_mask(
                gtrend_emb.shape[0],
                self.forecast_horizon,
                device=gtrend_emb.device,
            )
            gtrend_emb = self.encoder(gtrend_emb, input_mask)
        else:
            gtrend_emb = self.encoder(gtrend_emb)

        return gtrend_emb


class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim, cat_dict, col_dict, fab_dict):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cat_dict = {v: k for k, v in cat_dict.items()}
        self.col_dict = {v: k for k, v in col_dict.items()}
        self.fab_dict = {v: k for k, v in fab_dict.items()}
        self.word_embedder = pipeline("feature-extraction", model="bert-base-uncased")
        self.fc = nn.Linear(768, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, category, color, fabric):
        textual_description = [
            self.col_dict[color.detach().cpu().numpy().tolist()[i]]
            + " "
            + self.fab_dict[fabric.detach().cpu().numpy().tolist()[i]]
            + " "
            + self.cat_dict[category.detach().cpu().numpy().tolist()[i]]
            for i in range(len(category))
        ]

        word_embeddings = self.word_embedder(textual_description)
        word_embeddings = [torch.FloatTensor(x[1:-1]).mean(axis=0) for x in word_embeddings]
        word_embeddings = torch.stack(word_embeddings).to(category.device)
        word_embeddings = self.dropout(self.fc(word_embeddings))
        return word_embeddings


class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            resnet = models.resnet50(pretrained=True)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):
        img_embeddings = self.resnet(images)
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2], -1)
        return out.view(*size).contiguous()


class DummyEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.day_embedding = nn.Linear(1, embedding_dim)
        self.week_embedding = nn.Linear(1, embedding_dim)
        self.month_embedding = nn.Linear(1, embedding_dim)
        self.year_embedding = nn.Linear(1, embedding_dim)
        self.dummy_fusion = nn.Linear(embedding_dim * 4, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, temporal_features):
        d = temporal_features[:, 0].unsqueeze(1)
        w = temporal_features[:, 1].unsqueeze(1)
        m = temporal_features[:, 2].unsqueeze(1)
        y = temporal_features[:, 3].unsqueeze(1)

        d_emb = self.day_embedding(d)
        w_emb = self.week_embedding(w)
        m_emb = self.month_embedding(m)
        y_emb = self.year_embedding(y)

        temporal_embeddings = self.dummy_fusion(torch.cat([d_emb, w_emb, m_emb, y_emb], dim=1))
        temporal_embeddings = self.dropout(temporal_embeddings)
        return temporal_embeddings


class TransformerDecoderLayer(nn.Module):
    """Lightweight cross-attention decoder layer that also returns attention weights."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2, attn_weights = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn_weights


class SimpleTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        attn_weights = None
        output = tgt

        for layer in self.layers:
            output, attn_weights = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )

        return output, attn_weights


class GTM(pl.LightningModule):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_heads,
        num_layers,
        use_text,
        use_img,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len,
        num_trends,
        gpu_num,
        use_encoder_mask=1,
        autoregressive=False,
        use_competition_extension=False,
        competition_top_k=10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_dim
        self.use_encoder_mask = use_encoder_mask
        self.autoregressive = autoregressive
        self.gpu_num = gpu_num
        self.use_competition_extension = bool(use_competition_extension)
        self.competition_top_k = competition_top_k

        self.save_hyperparameters()
        self.validation_outputs = []

        self.dummy_encoder = DummyEmbedder(embedding_dim)
        self.image_encoder = ImageEmbedder()
        self.text_encoder = TextEmbedder(embedding_dim, cat_dict, col_dict, fab_dict)
        self.gtrend_encoder = GTrendEmbedder(
            output_dim,
            hidden_dim,
            use_encoder_mask,
            trend_len,
            num_trends,
        )
        self.static_feature_encoder = FusionNetwork(embedding_dim, hidden_dim, use_img, use_text)
        self.multimodal_product_encoder = MultimodalProductEncoder(
            embedding_dim,
            hidden_dim,
            use_img,
            use_text,
        )

        decoder_layer = TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
        )
        if self.autoregressive:
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len=output_dim)
        self.decoder = SimpleTransformerDecoder(decoder_layer, num_layers)

        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len if not self.autoregressive else 1),
            nn.Dropout(0.2),
        )
        self.decoder_step_fc = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Dropout(0.2),
        )

        self.rel_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        self.comp_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def _generate_square_subsequent_mask(self, size, device):
        mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        return mask

    def encode_multimodal_embedding(self, category, color, fabric, images):
        """Compute h_i using only image and textual metadata."""
        img_encoding = self.image_encoder(images)
        text_encoding = self.text_encoder(category, color, fabric)
        x_i = self.multimodal_product_encoder(img_encoding, text_encoding)
        return x_i

    def _build_launch_week_competition_context(
        self,
        target_product_repr,
        static_feature_fusion,
        neighbor_categories,
        neighbor_colors,
        neighbor_fabrics,
        neighbor_images,
        neighbor_scores,
        neighbor_mask,
    ):
        """
        Build the launch-week competition context.

        The neighborhood is selected with the precomputed launch-week Top-K tensors.
        The resulting context is repeated across the forecast horizon so it can be
        consumed by the existing decoder.
        """
        batch_size, top_k = neighbor_scores.shape

        flat_neighbor_categories = neighbor_categories.reshape(-1)
        flat_neighbor_colors = neighbor_colors.reshape(-1)
        flat_neighbor_fabrics = neighbor_fabrics.reshape(-1)
        flat_neighbor_images = neighbor_images.reshape(
            batch_size * top_k,
            neighbor_images.size(-3),
            neighbor_images.size(-2),
            neighbor_images.size(-1),
        )

        neighbor_img_encoding = self.image_encoder(flat_neighbor_images)
        neighbor_text_encoding = self.text_encoder(
            flat_neighbor_categories,
            flat_neighbor_colors,
            flat_neighbor_fabrics,
        )
        neighbor_repr = self.multimodal_product_encoder(
            neighbor_img_encoding,
            neighbor_text_encoding,
        ).reshape(batch_size, top_k, self.hidden_dim)

        target_expanded = target_product_repr.unsqueeze(1).expand(-1, top_k, -1)
        pair_features = torch.cat(
            [target_expanded, neighbor_repr, neighbor_scores.unsqueeze(-1)],
            dim=-1,
        )

        relation_logits = self.rel_mlp(pair_features).squeeze(-1)
        signed_weights = neighbor_scores * torch.tanh(relation_logits)
        signed_weights = signed_weights * neighbor_mask

        c_negative = (
            torch.clamp(-signed_weights, min=0.0).unsqueeze(-1) * neighbor_repr
        ).sum(dim=1)
        c_positive = (
            torch.clamp(signed_weights, min=0.0).unsqueeze(-1) * neighbor_repr
        ).sum(dim=1)

        z_bar = self.comp_mlp(
            torch.cat([static_feature_fusion, c_negative, c_positive], dim=-1)
        )
        z_bar_sequence = z_bar.unsqueeze(1).repeat(1, self.output_len, 1)

        diagnostics = {
            "signed_weights": signed_weights.detach(),
            "c_negative": c_negative.detach(),
            "c_positive": c_positive.detach(),
        }
        return z_bar_sequence, diagnostics

    def forward(
        self,
        category,
        color,
        fabric,
        temporal_features,
        gtrends,
        images,
        neighbor_categories=None,
        neighbor_colors=None,
        neighbor_fabrics=None,
        neighbor_images=None,
        neighbor_scores=None,
        neighbor_mask=None,
    ):
        img_encoding = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        gtrend_encoding = self.gtrend_encoder(gtrends)
        static_feature_fusion = self.static_feature_encoder(
            img_encoding,
            text_encoding,
            dummy_encoding,
        )
        target_product_repr = self.multimodal_product_encoder(img_encoding, text_encoding)

        competition_enabled = (
            self.use_competition_extension
            and neighbor_categories is not None
            and neighbor_scores is not None
            and neighbor_mask is not None
        )

        if competition_enabled:
            if self.autoregressive:
                raise ValueError(
                    "The competition extension in this implementation expects autoregressive=0."
                )

            z_bar_sequence, diagnostics = self._build_launch_week_competition_context(
                target_product_repr=target_product_repr,
                static_feature_fusion=static_feature_fusion,
                neighbor_categories=neighbor_categories,
                neighbor_colors=neighbor_colors,
                neighbor_fabrics=neighbor_fabrics,
                neighbor_images=neighbor_images,
                neighbor_scores=neighbor_scores,
                neighbor_mask=neighbor_mask,
            )

            tgt = z_bar_sequence.permute(1, 0, 2)
            memory = gtrend_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory)
            forecast = self.decoder_step_fc(decoder_out).squeeze(-1).transpose(0, 1)
            return forecast, attn_weights, diagnostics

        if self.autoregressive == 1:
            tgt = torch.zeros(
                self.output_len,
                gtrend_encoding.shape[1],
                gtrend_encoding.shape[-1],
                device=gtrend_encoding.device,
            )
            tgt[0] = static_feature_fusion
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(
                self.output_len,
                device=gtrend_encoding.device,
            )
            memory = gtrend_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory, tgt_mask=tgt_mask)
            forecast = self.decoder_fc(decoder_out).squeeze(-1).transpose(0, 1)
        else:
            tgt = static_feature_fusion.unsqueeze(0)
            memory = gtrend_encoding
            decoder_out, attn_weights = self.decoder(tgt, memory)
            forecast = self.decoder_fc(decoder_out).squeeze(0)

        return forecast.view(-1, self.output_len), attn_weights, None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return [optimizer]

    def _forward_from_batch(self, batch):
        if len(batch) == 7:
            item_sales, category, color, fabric, temporal_features, gtrends, images = batch
            forecasted_sales, _, _ = self.forward(
                category,
                color,
                fabric,
                temporal_features,
                gtrends,
                images,
            )
            return item_sales, forecasted_sales

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

        forecasted_sales, _, _ = self.forward(
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
        return item_sales, forecasted_sales

    def training_step(self, train_batch, batch_idx):
        item_sales, forecasted_sales = self._forward_from_batch(train_batch)
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        item_sales, forecasted_sales = self._forward_from_batch(val_batch)
        self.validation_outputs.append(
            {
                "item_sales": item_sales.squeeze(),
                "forecasted_sales": forecasted_sales.squeeze(),
            }
        )

    def on_validation_epoch_start(self):
        self.validation_outputs = []

    def on_validation_epoch_end(self):
        if len(self.validation_outputs) == 0:
            return

        item_sales = torch.stack([x["item_sales"] for x in self.validation_outputs])
        forecasted_sales = torch.stack(
            [x["forecasted_sales"] for x in self.validation_outputs]
        )

        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())

        rescaled_item_sales = item_sales * 1065
        rescaled_forecasted_sales = forecasted_sales * 1065

        val_wape_norm, val_mae_norm, val_ts_norm, val_erp_norm = compute_forecast_metrics(
            item_sales,
            forecasted_sales,
            erp_epsilon=0.1,
        )
        val_wape, val_mae, val_ts, val_erp = compute_forecast_metrics(
            rescaled_item_sales,
            rescaled_forecasted_sales,
            erp_epsilon=0.1,
        )

        self.log("val_loss", loss)
        self.log("val_wape_norm", val_wape_norm, prog_bar=False)
        self.log("val_mae_norm", val_mae_norm, prog_bar=False)
        self.log("val_ts_norm", val_ts_norm, prog_bar=False)
        self.log("val_erp_norm", val_erp_norm, prog_bar=False)

        self.log("val_wape", val_wape, prog_bar=False)
        self.log("val_mae", val_mae, prog_bar=True)
        self.log("val_ts", val_ts, prog_bar=False)
        self.log("val_erp", val_erp, prog_bar=False)

        print(
            f"Validation normalized | "
            f"WAPE: {val_wape_norm.item():.3f} | "
            f"MAE: {val_mae_norm.item():.3f} | "
            f"TS: {val_ts_norm.item():.3f} | "
            f"ERP: {val_erp_norm.item():.3f}"
        )
        print(
            f"Validation rescaled | "
            f"WAPE: {val_wape.item():.3f} | "
            f"MAE: {val_mae.item():.3f} | "
            f"TS: {val_ts.item():.3f} | "
            f"ERP: {val_erp.item():.3f}"
        )

        self.validation_outputs.clear()