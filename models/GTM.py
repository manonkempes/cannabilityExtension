import copy
import math
from collections import OrderedDict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import AutoModel, AutoTokenizer


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
    def __init__(self, d_model, dropout=0.1, max_len=52, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, : x.size(1)]
        else:
            x = x + self.pe[:, : x.size(0)].transpose(0, 1)
        return self.dropout(x)


class TimeDistributed(nn.Module):
    """Apply a module on the last dimension for every time step."""

    def __init__(self, module, batch_first=True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if x.dim() <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)

        if self.batch_first:
            return y.contiguous().view(x.size(0), -1, y.size(-1))
        return y.view(-1, x.size(1), y.size(-1))


class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_img, use_text, dropout=0.2):
        super().__init__()
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

    def forward(self, img_features, text_encoding, dummy_encoding):
        decoder_inputs = []
        if self.use_img == 1:
            decoder_inputs.append(self.img_linear(img_features))
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

    def forward(self, img_features, text_encoding):
        features = []
        if self.use_img == 1:
            features.append(self.img_linear(img_features))
        if self.use_text == 1:
            features.append(text_encoding)

        concat_features = torch.cat(features, dim=1)
        return self.feature_fusion(concat_features)


class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim), batch_first=True)
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dropout=0.2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask
        self._mask_cache = {}

    def _generate_encoder_mask(self, size, forecast_horizon, device):
        cache_key = (size, forecast_horizon, str(device))
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        mask = torch.zeros((size, size), device=device)
        split = math.gcd(size, forecast_horizon)
        for i in range(0, size, split):
            mask[i : i + split, i : i + split] = 1
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, gtrends):
        # [B, num_trends, trend_len] -> [B, trend_len, num_trends]
        gtrend_emb = self.input_linear(gtrends.permute(0, 2, 1))
        gtrend_emb = self.pos_embedding(gtrend_emb)

        if self.use_mask == 1:
            input_mask = self._generate_encoder_mask(
                gtrend_emb.shape[1],
                self.forecast_horizon,
                device=gtrend_emb.device,
            )
            return self.encoder(gtrend_emb, mask=input_mask)

        return self.encoder(gtrend_emb)


class TextEmbedder(nn.Module):
    def __init__(
        self,
        embedding_dim,
        cat_dict,
        col_dict,
        fab_dict,
        bert_model_name="bert-base-uncased",
        max_cache_size=4096,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.cat_dict = {v: k for k, v in cat_dict.items()}
        self.col_dict = {v: k for k, v in col_dict.items()}
        self.fab_dict = {v: k for k, v in fab_dict.items()}

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(self.bert.config.hidden_size, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        self.max_cache_size = max_cache_size
        self._embedding_cache = OrderedDict()

    def _evict_if_needed(self):
        while len(self._embedding_cache) > self.max_cache_size:
            self._embedding_cache.popitem(last=False)

    def _build_descriptions(self, category, color, fabric):
        category_ids = category.detach().cpu().tolist()
        color_ids = color.detach().cpu().tolist()
        fabric_ids = fabric.detach().cpu().tolist()

        return [
            f"{self.col_dict[col]} {self.fab_dict[fab]} {self.cat_dict[cat]}"
            for cat, col, fab in zip(category_ids, color_ids, fabric_ids)
        ]

    def _mean_pool(self, hidden_state, attention_mask, special_tokens_mask=None):
        mask = attention_mask.clone()
        if special_tokens_mask is not None:
            mask = mask * (1 - special_tokens_mask)
        mask = mask.unsqueeze(-1).type_as(hidden_state)
        masked_hidden = hidden_state * mask
        summed = masked_hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def _encode_missing_descriptions(self, missing_descriptions, device):
        if not missing_descriptions:
            return

        encoded = self.tokenizer(
            missing_descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            special_tokens_mask = encoded.pop("special_tokens_mask")
            hidden = self.bert(**encoded).last_hidden_state
            pooled = self._mean_pool(hidden, encoded["attention_mask"], special_tokens_mask)

        pooled_cpu = pooled.detach().cpu()
        for description, embedding in zip(missing_descriptions, pooled_cpu):
            self._embedding_cache[description] = embedding
        self._evict_if_needed()

    def forward(self, category, color, fabric):
        descriptions = self._build_descriptions(category, color, fabric)
        missing = [d for d in descriptions if d not in self._embedding_cache]
        self._encode_missing_descriptions(missing, category.device)

        cached_embeddings = [self._embedding_cache[d] for d in descriptions]
        word_embeddings = torch.stack(cached_embeddings, dim=0).to(category.device)
        word_embeddings = self.dropout(self.fc(word_embeddings))
        return word_embeddings


class ImageEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            resnet = models.resnet50(pretrained=True)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):
        with torch.no_grad():
            img_embeddings = self.resnet(images)
        return img_embeddings.flatten(1)


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
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        del tgt_mask  # not used in this lightweight decoder implementation
        tgt2, attn_weights = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            need_weights=True,
        )
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))

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
        self._decoder_mask_cache = {}

        self.save_hyperparameters()

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
            self.pos_encoder = PositionalEncoding(hidden_dim, max_len=output_dim, batch_first=True)
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
        cache_key = (size, str(device))
        if cache_key in self._decoder_mask_cache:
            return self._decoder_mask_cache[cache_key]

        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        self._decoder_mask_cache[cache_key] = mask
        return mask

    def encode_multimodal_embedding(self, category, color, fabric, images):
        """Compute h_i using only image and textual metadata."""
        img_features = self.image_encoder(images)
        text_encoding = self.text_encoder(category, color, fabric)
        x_i = self.multimodal_product_encoder(img_features, text_encoding)
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

        neighbor_img_features = self.image_encoder(flat_neighbor_images)
        neighbor_text_encoding = self.text_encoder(
            flat_neighbor_categories,
            flat_neighbor_colors,
            flat_neighbor_fabrics,
        )
        neighbor_repr = self.multimodal_product_encoder(
            neighbor_img_features,
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
        z_bar_sequence = z_bar.unsqueeze(1).expand(-1, self.output_len, -1)

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
        img_features = self.image_encoder(images)
        dummy_encoding = self.dummy_encoder(temporal_features)
        text_encoding = self.text_encoder(category, color, fabric)
        gtrend_encoding = self.gtrend_encoder(gtrends)
        static_feature_fusion = self.static_feature_encoder(
            img_features,
            text_encoding,
            dummy_encoding,
        )
        target_product_repr = self.multimodal_product_encoder(img_features, text_encoding)

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

            decoder_out, attn_weights = self.decoder(z_bar_sequence, gtrend_encoding)
            forecast = self.decoder_step_fc(decoder_out).squeeze(-1)
            return forecast, attn_weights, diagnostics

        if self.autoregressive == 1:
            tgt = torch.zeros(
                gtrend_encoding.shape[0],
                self.output_len,
                gtrend_encoding.shape[-1],
                device=gtrend_encoding.device,
            )

            tgt[:, 0, :] = static_feature_fusion
            tgt = self.pos_encoder(tgt)
            tgt_mask = self._generate_square_subsequent_mask(
                self.output_len,
                device=gtrend_encoding.device,
            )
            decoder_out, attn_weights = self.decoder(tgt, gtrend_encoding, tgt_mask=tgt_mask)
            forecast = self.decoder_fc(decoder_out).squeeze(-1)
        else:
            tgt = static_feature_fusion.unsqueeze(1)
            decoder_out, attn_weights = self.decoder(tgt, gtrend_encoding)
            forecast = self.decoder_fc(decoder_out).squeeze(1)

        return forecast.view(-1, self.output_len), attn_weights, None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

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
        del batch_idx
        item_sales, forecasted_sales = self._forward_from_batch(train_batch)
        loss = F.mse_loss(item_sales, forecasted_sales)
        self.log("train_loss", loss, prog_bar=True, batch_size=item_sales.size(0))
        return loss

    def validation_step(self, val_batch, batch_idx):
        del batch_idx

        item_sales, forecasted_sales = self._forward_from_batch(val_batch)

        item_sales = item_sales.detach().view(item_sales.size(0), -1)
        forecasted_sales = forecasted_sales.detach().view(forecasted_sales.size(0), -1)

        abs_err = torch.abs(item_sales - forecasted_sales)
        sq_err = (item_sales - forecasted_sales) ** 2

        mae_per_series = abs_err.mean(dim=1).clamp(min=1e-12)
        ts_per_series = (item_sales - forecasted_sales).sum(dim=1) / mae_per_series
        erp_per_series = (abs_err >= 0.1).float().sum(dim=1)

        scale = 1065.0
        rescaled_item_sales = item_sales * scale
        rescaled_abs_err = abs_err * scale
        erp_per_series_rescaled = (rescaled_abs_err >= 0.1).float().sum(dim=1)

        self.val_sq_err_sum += sq_err.sum()
        self.val_abs_err_sum += abs_err.sum()
        self.val_true_sum += item_sales.sum()
        self.val_ts_sum += ts_per_series.sum()
        self.val_erp_sum += erp_per_series.sum()

        self.val_abs_err_sum_rescaled += rescaled_abs_err.sum()
        self.val_true_sum_rescaled += rescaled_item_sales.sum()
        self.val_erp_sum_rescaled += erp_per_series_rescaled.sum()

        self.val_series_count += item_sales.size(0)
        self.val_point_count += item_sales.numel()

    def on_validation_epoch_start(self):
        device = self.device

        self.val_sq_err_sum = torch.tensor(0.0, device=device)
        self.val_abs_err_sum = torch.tensor(0.0, device=device)
        self.val_true_sum = torch.tensor(0.0, device=device)
        self.val_ts_sum = torch.tensor(0.0, device=device)
        self.val_erp_sum = torch.tensor(0.0, device=device)

        self.val_abs_err_sum_rescaled = torch.tensor(0.0, device=device)
        self.val_true_sum_rescaled = torch.tensor(0.0, device=device)
        self.val_erp_sum_rescaled = torch.tensor(0.0, device=device)

        self.val_series_count = 0
        self.val_point_count = 0

    def on_validation_epoch_end(self):
        if self.val_series_count == 0 or self.val_point_count == 0:
            return

        loss = self.val_sq_err_sum / self.val_point_count

        val_mae_norm = self.val_abs_err_sum / self.val_point_count
        val_wape_norm = 100.0 * self.val_abs_err_sum / self.val_true_sum.clamp(min=1e-12)
        val_ts_norm = self.val_ts_sum / self.val_series_count
        val_erp_norm = self.val_erp_sum / self.val_series_count

        val_mae = self.val_abs_err_sum_rescaled / self.val_point_count
        val_wape = 100.0 * self.val_abs_err_sum_rescaled / self.val_true_sum_rescaled.clamp(min=1e-12)
        val_ts = val_ts_norm
        val_erp = self.val_erp_sum_rescaled / self.val_series_count

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