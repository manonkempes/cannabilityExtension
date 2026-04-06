import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def floor_to_monday(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Return the Monday of the calendar week for a timestamp."""
    timestamp = pd.Timestamp(timestamp)
    return (timestamp - pd.Timedelta(days=timestamp.weekday())).normalize()


def safe_minmax_scale(values: np.ndarray) -> np.ndarray:
    """Scale a 1D array to [0, 1] while handling constant arrays safely."""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values

    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        return np.zeros_like(values, dtype=np.float32)

    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def precompute_image_tensors(
    img_root,
    img_tensor_root,
    image_size=(256, 256),
    use_float16=True,
):
    img_root = Path(img_root)
    img_tensor_root = Path(img_tensor_root)
    img_tensor_root.mkdir(parents=True, exist_ok=True)

    try:
        resize = Resize(image_size, interpolation=Image.Resampling.BILINEAR)
    except AttributeError:
        resize = Resize(image_size)

    transform = Compose(
        [
            resize,
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    exts = {".jpg", ".jpeg", ".png", ".webp"}

    image_paths = [p for p in img_root.rglob("*") if p.suffix.lower() in exts]

    for img_path in tqdm(image_paths, total=len(image_paths), ascii=True):
        rel_path = img_path.relative_to(img_root).with_suffix(".pt")
        out_path = img_tensor_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            continue

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                tensor = transform(img)

            if use_float16:
                tensor = tensor.to(torch.float16)

            torch.save(tensor, out_path)
        except Exception:
            continue

class LazyDataset(Dataset):
    """
    Load target and neighbor images lazily to keep RAM usage manageable.

    When the competition extension is disabled, the dataset returns the original
    7-tuple used by the baseline code.

    When the competition extension is enabled, the dataset returns:
    (
        item_sales,
        category,
        color,
        fabric,
        temporal_features,
        gtrends,
        target_image,
        neighbor_categories,
        neighbor_colors,
        neighbor_fabrics,
        neighbor_images,
        neighbor_scores,
        neighbor_mask,
    )
    """

    def __init__(
            self,
            item_sales,
            categories,
            colors,
            fabrics,
            temporal_features,
            gtrends,
            img_paths,
            img_root,
            img_tensor_root=None,
            neighbor_categories=None,
            neighbor_colors=None,
            neighbor_fabrics=None,
            neighbor_img_paths=None,
            neighbor_scores=None,
            neighbor_mask=None,
            image_cache_size=4096,
    ):
        self.item_sales = item_sales
        self.categories = categories
        self.colors = colors
        self.fabrics = fabrics
        self.temporal_features = temporal_features
        self.gtrends = gtrends
        self.img_paths = img_paths
        self.img_root = img_root
        self.img_tensor_root = str(img_tensor_root) if img_tensor_root is not None else None

        self.neighbor_categories = neighbor_categories
        self.neighbor_colors = neighbor_colors
        self.neighbor_fabrics = neighbor_fabrics
        self.neighbor_img_paths = neighbor_img_paths
        self.neighbor_scores = neighbor_scores
        self.neighbor_mask = neighbor_mask

        self.use_competition_extension = neighbor_categories is not None

        try:
            resize = Resize((256, 256), interpolation=Image.Resampling.BILINEAR)
        except AttributeError:
            resize = Resize((256, 256))

        self.transforms = Compose(
            [
                resize,
                ToTensor(),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.zero_image = torch.zeros(3, 256, 256, dtype=torch.float32)
        self.image_cache_size = max(0, int(image_cache_size))
        self._image_cache = OrderedDict()

    def __len__(self):
        return len(self.item_sales)

    def _evict_if_needed(self):
        while len(self._image_cache) > self.image_cache_size:
            self._image_cache.popitem(last=False)

    def _read_image(self, relative_path: str) -> torch.Tensor:
        if relative_path is None or relative_path == "":
            return self.zero_image

        if self.img_tensor_root is not None:
            tensor_path = Path(self.img_tensor_root) / Path(relative_path).with_suffix(".pt")
            if tensor_path.exists():
                try:
                    tensor = torch.load(tensor_path, map_location="cpu", weights_only=False)
                    return tensor.float()
                except Exception:
                    pass

        image_path = os.path.join(self.img_root, relative_path)
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                return self.transforms(img)
        except Exception:
            return self.zero_image

    def _load_image(self, relative_path: str) -> torch.Tensor:
        if relative_path is None or relative_path == "":
            return self.zero_image

        if self.image_cache_size > 0 and relative_path in self._image_cache:
            cached = self._image_cache.pop(relative_path)
            self._image_cache[relative_path] = cached
            return cached

        image_tensor = self._read_image(relative_path)

        if self.image_cache_size > 0:
            self._image_cache[relative_path] = image_tensor
            self._evict_if_needed()

        return image_tensor

    def __getitem__(self, idx):
        target_image = self._load_image(self.img_paths[idx])

        base_tuple = (
            self.item_sales[idx],
            self.categories[idx],
            self.colors[idx],
            self.fabrics[idx],
            self.temporal_features[idx],
            self.gtrends[idx],
            target_image,
        )

        if not self.use_competition_extension:
            return base_tuple

        row_neighbor_paths = self.neighbor_img_paths[idx]
        row_neighbor_mask = self.neighbor_mask[idx]

        neighbor_imgs = [
            self._load_image(rel_path) if float(valid) > 0 else self.zero_image
            for rel_path, valid in zip(row_neighbor_paths, row_neighbor_mask.tolist())
        ]
        neighbor_imgs = torch.stack(neighbor_imgs, dim=0)

        return base_tuple + (
            self.neighbor_categories[idx],
            self.neighbor_colors[idx],
            self.neighbor_fabrics[idx],
            neighbor_imgs,
            self.neighbor_scores[idx],
            self.neighbor_mask[idx],
        )


class ZeroShotDataset:
    def __init__(
            self,
       data_df,
        img_root,
        img_tensor_root,
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len,
        target_cols=None,
        temporal_cols=None,
        text_cols=None,
        trend_cols=None,
        image_col="image_path",
        use_competition_extension=False,
        competition_reference_df=None,
        competition_topk_indices_path=None,
        competition_topk_values_path=None,
        competition_row_mapping_path=None,
        competition_meta_path=None,
        competition_top_k=10,
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = int(trend_len)
        self.img_root = str(img_root)
        self.img_tensor_root = str(img_tensor_root) if img_tensor_root is not None else None

        self.target_cols = target_cols or [str(i) for i in range(12)]
        self.temporal_cols = temporal_cols or ["day", "week", "month", "year"]
        self.text_cols = text_cols or ["category", "color", "fabric"]
        self.trend_cols = trend_cols or ["category", "color", "fabric"]
        self.image_col = image_col

        self.use_competition_extension = bool(use_competition_extension)
        self.competition_top_k = int(competition_top_k)

        self.competition_reference_df = None
        self.topk_indices = None
        self.topk_values = None
        self.week_to_index = None
        self.external_code_to_product_index = None

        self.reference_category_ids = None
        self.reference_color_ids = None
        self.reference_fabric_ids = None
        self.reference_image_paths = None

        self._gtrend_cache = {}
        self._competition_snapshot_cache = {}

        if len(self.text_cols) != 3:
            raise ValueError("text_cols must contain exactly 3 columns: [category, color, fabric]")

        if len(self.trend_cols) != 3:
            raise ValueError("trend_cols must contain exactly 3 columns: [category, color, fabric]")

        if self.use_competition_extension:
            if competition_reference_df is None:
                raise ValueError(
                    "competition_reference_df must be provided when the competition extension is enabled."
                )

            required_paths = [
                competition_topk_indices_path,
                competition_topk_values_path,
                competition_row_mapping_path,
                competition_meta_path,
            ]
            if any(path is None for path in required_paths):
                raise ValueError(
                    "All competition paths must be provided when the competition extension is enabled."
                )

            self.competition_reference_df = competition_reference_df.reset_index(drop=True)
            self.competition_reference_df["external_code"] = (
                self.competition_reference_df["external_code"].astype(str)
            )
            self._load_competition_artifacts(
                competition_topk_indices_path=competition_topk_indices_path,
                competition_topk_values_path=competition_topk_values_path,
                competition_row_mapping_path=competition_row_mapping_path,
                competition_meta_path=competition_meta_path,
            )

    def _resolve_single_column(self, data: pd.DataFrame, col_name):
        """
        Resolve column names robustly for cases where 0..11 may appear as strings
        or integers, depending on how the DataFrame was loaded.
        """
        if col_name in data.columns:
            return col_name

        if isinstance(col_name, str) and col_name.isdigit():
            as_int = int(col_name)
            if as_int in data.columns:
                return as_int

        if isinstance(col_name, int):
            as_str = str(col_name)
            if as_str in data.columns:
                return as_str

        raise ValueError(f"Missing column: {col_name}")

    def _resolve_columns(self, data: pd.DataFrame, columns):
        return [self._resolve_single_column(data, c) for c in columns]

    def _load_competition_artifacts(
        self,
        competition_topk_indices_path,
        competition_topk_values_path,
        competition_row_mapping_path,
        competition_meta_path,
    ):
        """Load the precomputed Top-K competition neighborhood files."""
        self.topk_indices = np.load(competition_topk_indices_path, mmap_mode="r")
        self.topk_values = np.load(competition_topk_values_path, mmap_mode="r")

        row_mapping = pd.read_csv(competition_row_mapping_path, usecols=["external_code"])
        row_mapping["external_code"] = row_mapping["external_code"].astype(str)

        meta = json.loads(Path(competition_meta_path).read_text())
        week_columns = meta["week_columns"]
        self.week_to_index = {week_label: idx for idx, week_label in enumerate(week_columns)}
        self.external_code_to_product_index = {
            code: idx for idx, code in enumerate(row_mapping["external_code"].tolist())
        }

        reference_df = self.competition_reference_df.set_index("external_code")

        cat_text_col, color_text_col, fabric_text_col = self.text_cols

        num_products = len(row_mapping)
        self.reference_category_ids = np.zeros(num_products, dtype=np.int64)
        self.reference_color_ids = np.zeros(num_products, dtype=np.int64)
        self.reference_fabric_ids = np.zeros(num_products, dtype=np.int64)
        self.reference_image_paths = [""] * num_products

        for idx, code in enumerate(row_mapping["external_code"].tolist()):
            if code not in reference_df.index:
                raise ValueError(
                    f"Product '{code}' was found in the competition files but not in competition_reference_df."
                )

            row = reference_df.loc[code]
            self.reference_category_ids[idx] = self.cat_dict[str(row[cat_text_col])]
            self.reference_color_ids[idx] = self.col_dict[str(row[color_text_col])]
            self.reference_fabric_ids[idx] = self.fab_dict[str(row[fabric_text_col])]
            self.reference_image_paths[idx] = str(row[self.image_col])

        expected_shape = (len(week_columns), num_products)
        if self.topk_indices.shape[:2] != expected_shape:
            raise ValueError(
                "The Top-K tensors do not match the metadata dimensions. "
                f"Expected {expected_shape}, got {self.topk_indices.shape[:2]}."
            )

    def _extract_temporal_features(self, data: pd.DataFrame, temporal_cols=None) -> torch.FloatTensor:
        cols = temporal_cols or self._resolve_columns(data, self.temporal_cols)
        arr = data[cols].to_numpy(dtype=np.float32, copy=True)
        return torch.from_numpy(arr)

    def _get_sales_tensor(self, data: pd.DataFrame, target_cols=None) -> torch.FloatTensor:
        cols = target_cols or self._resolve_columns(data, self.target_cols)
        arr = data[cols].to_numpy(dtype=np.float32, copy=True)
        return torch.from_numpy(arr)

    def _get_scaled_gtrend(self, label: str, start_date: pd.Timestamp) -> np.ndarray:
        """
        Return the normalized Google Trends history for a single attribute.

        The series is padded with zeros if it is shorter than trend_len.
        """
        label = str(label)
        start_date = pd.Timestamp(start_date).normalize()
        cache_key = (label, start_date)

        if cache_key in self._gtrend_cache:
            return self._gtrend_cache[cache_key]

        gtrend_start = start_date - pd.DateOffset(weeks=self.trend_len)
        try:
            series = self.gtrends.loc[gtrend_start:start_date, label].values[-self.trend_len :]
        except KeyError:
            series = np.zeros(self.trend_len, dtype=np.float32)

        series = np.asarray(series, dtype=np.float32)
        if len(series) >= self.trend_len:
            series = series[-self.trend_len :]
        else:
            pad = np.zeros(self.trend_len - len(series), dtype=np.float32)
            series = np.concatenate([pad, series], axis=0)

        scaled = safe_minmax_scale(series)
        self._gtrend_cache[cache_key] = scaled
        return scaled

    def _build_competition_snapshot(self, row):
        """
        Build the launch-week Top-K neighborhood for one target product.
        """
        target_code = str(row.external_code)
        launch_week_label = floor_to_monday(pd.Timestamp(row.release_date)).strftime("%Y-%m-%d")
        cache_key = (target_code, launch_week_label)

        if cache_key in self._competition_snapshot_cache:
            return self._competition_snapshot_cache[cache_key]

        if target_code not in self.external_code_to_product_index:
            raise ValueError(f"Product '{target_code}' was not found in the competition mapping.")

        if launch_week_label not in self.week_to_index:
            raise ValueError(
                f"Launch week '{launch_week_label}' was not found in the competition metadata."
            )

        product_index = self.external_code_to_product_index[target_code]
        week_index = self.week_to_index[launch_week_label]

        neighbor_indices = np.asarray(
            self.topk_indices[week_index, product_index, : self.competition_top_k]
        ).astype(np.int64, copy=False)
        neighbor_scores = np.asarray(
            self.topk_values[week_index, product_index, : self.competition_top_k]
        ).astype(np.float32, copy=False)

        valid = (neighbor_indices != -1) & (neighbor_scores > 0)
        safe_indices = neighbor_indices.copy()
        safe_indices[~valid] = 0

        neighbor_categories = self.reference_category_ids[safe_indices].astype(np.int64, copy=True)
        neighbor_colors = self.reference_color_ids[safe_indices].astype(np.int64, copy=True)
        neighbor_fabrics = self.reference_fabric_ids[safe_indices].astype(np.int64, copy=True)
        neighbor_mask = valid.astype(np.float32, copy=False)

        neighbor_img_paths = [self.reference_image_paths[idx] for idx in safe_indices.tolist()]
        for i, is_valid in enumerate(valid.tolist()):
            if not is_valid:
                neighbor_img_paths[i] = ""

        result = (
            neighbor_categories,
            neighbor_colors,
            neighbor_fabrics,
            neighbor_img_paths,
            neighbor_scores.astype(np.float32, copy=True),
            neighbor_mask.astype(np.float32, copy=True),
        )
        self._competition_snapshot_cache[cache_key] = result
        return result

    def preprocess_payload(self):
        data = self.data_df
        num_rows = len(data)

        target_cols = self._resolve_columns(data, self.target_cols)
        temporal_cols = self._resolve_columns(data, self.temporal_cols)
        text_cols = self._resolve_columns(data, self.text_cols)
        trend_cols = self._resolve_columns(data, self.trend_cols)
        image_col = self._resolve_single_column(data, self.image_col)
        release_date_col = self._resolve_single_column(data, "release_date")

        cat_text_col, color_text_col, fabric_text_col = text_cols
        cat_trend_col, color_trend_col, fabric_trend_col = trend_cols

        multitrends = np.empty((num_rows, 3, self.trend_len), dtype=np.float32)
        image_paths = [""] * num_rows

        if self.use_competition_extension:
            external_code_col = self._resolve_single_column(data, "external_code")
            _ = external_code_col  # validatie

            neighbor_categories_all = np.empty((num_rows, self.competition_top_k), dtype=np.int64)
            neighbor_colors_all = np.empty((num_rows, self.competition_top_k), dtype=np.int64)
            neighbor_fabrics_all = np.empty((num_rows, self.competition_top_k), dtype=np.int64)
            neighbor_img_paths_all = [None] * num_rows
            neighbor_scores_all = np.empty((num_rows, self.competition_top_k), dtype=np.float32)
            neighbor_mask_all = np.empty((num_rows, self.competition_top_k), dtype=np.float32)

        iterator = data.itertuples(index=False, name="Row")
        for i, row in enumerate(tqdm(iterator, total=num_rows, ascii=True)):
            row_dict = row._asdict()

            cat_gtrend = self._get_scaled_gtrend(
                row_dict[cat_trend_col],
                row_dict[release_date_col],
            )
            col_gtrend = self._get_scaled_gtrend(
                row_dict[color_trend_col],
                row_dict[release_date_col],
            )
            fab_gtrend = self._get_scaled_gtrend(
                row_dict[fabric_trend_col],
                row_dict[release_date_col],
            )

            multitrends[i] = np.stack([cat_gtrend, col_gtrend, fab_gtrend], axis=0)
            image_paths[i] = str(row_dict[image_col])

            if self.use_competition_extension:
                (
                    n_cat,
                    n_col,
                    n_fab,
                    n_img_paths,
                    n_scores,
                    n_mask,
                ) = self._build_competition_snapshot(row)

                neighbor_categories_all[i] = n_cat
                neighbor_colors_all[i] = n_col
                neighbor_fabrics_all[i] = n_fab
                neighbor_img_paths_all[i] = n_img_paths
                neighbor_scores_all[i] = n_scores
                neighbor_mask_all[i] = n_mask

        multitrends = torch.from_numpy(multitrends)

        category_values = data[cat_text_col].astype(str).to_numpy()
        color_values = data[color_text_col].astype(str).to_numpy()
        fabric_values = data[fabric_text_col].astype(str).to_numpy()

        categories = torch.from_numpy(
            np.fromiter(
                (self.cat_dict[val] for val in category_values),
                dtype=np.int64,
                count=num_rows,
            )
        )
        colors = torch.from_numpy(
            np.fromiter(
                (self.col_dict[val] for val in color_values),
                dtype=np.int64,
                count=num_rows,
            )
        )
        fabrics = torch.from_numpy(
            np.fromiter(
                (self.fab_dict[val] for val in fabric_values),
                dtype=np.int64,
                count=num_rows,
            )
        )

        payload = {
            "item_sales": self._get_sales_tensor(data, target_cols=target_cols),
            "categories": categories,
            "colors": colors,
            "fabrics": fabrics,
            "temporal_features": self._extract_temporal_features(data, temporal_cols=temporal_cols),
            "gtrends": multitrends,
            "img_paths": image_paths,
        }

        if self.use_competition_extension:
            payload.update(
                {
                    "neighbor_categories": torch.from_numpy(neighbor_categories_all),
                    "neighbor_colors": torch.from_numpy(neighbor_colors_all),
                    "neighbor_fabrics": torch.from_numpy(neighbor_fabrics_all),
                    "neighbor_img_paths": neighbor_img_paths_all,
                    "neighbor_scores": torch.from_numpy(neighbor_scores_all),
                    "neighbor_mask": torch.from_numpy(neighbor_mask_all),
                }
            )

        return payload

    def _build_dataset_from_payload(self, payload):
        return LazyDataset(
            item_sales=payload["item_sales"],
            categories=payload["categories"],
            colors=payload["colors"],
            fabrics=payload["fabrics"],
            temporal_features=payload["temporal_features"],
            gtrends=payload["gtrends"],
            img_paths=payload["img_paths"],
            img_root=self.img_root,
            img_tensor_root=self.img_tensor_root,
            neighbor_categories=payload.get("neighbor_categories"),
            neighbor_colors=payload.get("neighbor_colors"),
            neighbor_fabrics=payload.get("neighbor_fabrics"),
            neighbor_img_paths=payload.get("neighbor_img_paths"),
            neighbor_scores=payload.get("neighbor_scores"),
            neighbor_mask=payload.get("neighbor_mask"),
        )


    def preprocess_data(self):
        payload = self.preprocess_payload()
        return self._build_dataset_from_payload(payload)

    def get_loader(self, batch_size, train=True, cache_path=None, rebuild_cache=False):
        if cache_path is not None:
            cache_path = Path(cache_path)

        if cache_path is not None and cache_path.exists() and not rebuild_cache:
            print(f"Loading cached preprocessing from: {cache_path}")
            payload = torch.load(cache_path, weights_only=False)
        else:
            print("Starting dataset creation process...")
            payload = self.preprocess_payload()

            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, cache_path)
                print(f"Saved cached preprocessing to: {cache_path}")

        data_with_gtrends = self._build_dataset_from_payload(payload)

        cpu_count = os.cpu_count() or 0
        num_workers = min(4, cpu_count)
        pin_memory = torch.cuda.is_available()

        loader_kwargs = {
            "dataset": data_with_gtrends,
            "batch_size": batch_size,
            "shuffle": train,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2

        return DataLoader(**loader_kwargs)