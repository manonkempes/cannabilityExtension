import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.preprocessing import MinMaxScaler
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
    if len(values) == 0:
        return values
    if np.allclose(values.max(), values.min()):
        return np.zeros_like(values, dtype=np.float32)
    return MinMaxScaler().fit_transform(values.reshape(-1, 1)).flatten().astype(np.float32)


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
        neighbor_categories=None,
        neighbor_colors=None,
        neighbor_fabrics=None,
        neighbor_img_paths=None,
        neighbor_scores=None,
        neighbor_mask=None,
    ):
        self.item_sales = item_sales
        self.categories = categories
        self.colors = colors
        self.fabrics = fabrics
        self.temporal_features = temporal_features
        self.gtrends = gtrends
        self.img_paths = img_paths
        self.img_root = img_root

        self.neighbor_categories = neighbor_categories
        self.neighbor_colors = neighbor_colors
        self.neighbor_fabrics = neighbor_fabrics
        self.neighbor_img_paths = neighbor_img_paths
        self.neighbor_scores = neighbor_scores
        self.neighbor_mask = neighbor_mask
        self.use_competition_extension = neighbor_categories is not None

        self.transforms = Compose(
            [
                Resize((256, 256)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.zero_image = torch.zeros(3, 256, 256, dtype=torch.float32)

    def __len__(self):
        return len(self.item_sales)

    def _load_image(self, relative_path: str) -> torch.Tensor:
        """
        Load an RGB image and apply the same transforms as the baseline code.
        Invalid or missing files are replaced by a zero image so that the
        dataloader does not crash mid-training.
        """
        if relative_path is None or relative_path == "":
            return self.zero_image.clone()

        image_path = os.path.join(self.img_root, relative_path)
        try:
            img = Image.open(image_path).convert("RGB")
            return self.transforms(img)
        except Exception:
            return self.zero_image.clone()

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

        neighbor_imgs = []
        for rel_path, valid in zip(
            self.neighbor_img_paths[idx],
            self.neighbor_mask[idx].tolist(),
        ):
            if valid > 0:
                neighbor_imgs.append(self._load_image(rel_path))
            else:
                neighbor_imgs.append(self.zero_image.clone())
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
        gtrends,
        cat_dict,
        col_dict,
        fab_dict,
        trend_len,
        use_competition_extension=False,
        competition_reference_df=None,
        competition_topk_indices_path=None,
        competition_topk_values_path=None,
        competition_row_mapping_path=None,
        competition_meta_path=None,
        competition_top_k=10,
    ):
        self.data_df = data_df.copy().reset_index(drop=True)
        self.gtrends = gtrends
        self.cat_dict = cat_dict
        self.col_dict = col_dict
        self.fab_dict = fab_dict
        self.trend_len = trend_len
        self.img_root = str(img_root)

        self.use_competition_extension = bool(use_competition_extension)
        self.competition_top_k = competition_top_k
        self.competition_reference_df = None
        self.topk_indices = None
        self.topk_values = None
        self.week_to_index = None
        self.external_code_to_product_index = None
        self.product_index_to_reference_row = None

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

            self.competition_reference_df = competition_reference_df.copy().reset_index(
                drop=True
            )
            self._load_competition_artifacts(
                competition_topk_indices_path=competition_topk_indices_path,
                competition_topk_values_path=competition_topk_values_path,
                competition_row_mapping_path=competition_row_mapping_path,
                competition_meta_path=competition_meta_path,
            )

    def _load_competition_artifacts(
        self,
        competition_topk_indices_path,
        competition_topk_values_path,
        competition_row_mapping_path,
        competition_meta_path,
    ):
        """Load the precomputed Top-K competition neighborhood files."""
        self.topk_indices = np.load(competition_topk_indices_path)
        self.topk_values = np.load(competition_topk_values_path)

        row_mapping = pd.read_csv(competition_row_mapping_path)
        row_mapping["external_code"] = row_mapping["external_code"].astype(str)

        meta = json.loads(Path(competition_meta_path).read_text())
        week_columns = meta["week_columns"]

        self.week_to_index = {week_label: idx for idx, week_label in enumerate(week_columns)}
        self.external_code_to_product_index = {
            code: idx for idx, code in enumerate(row_mapping["external_code"].tolist())
        }

        reference_df = self.competition_reference_df.copy()
        reference_df["external_code"] = reference_df["external_code"].astype(str)
        reference_by_code = {
            code: row for code, row in reference_df.set_index("external_code").iterrows()
        }

        self.product_index_to_reference_row = []
        for code in row_mapping["external_code"].tolist():
            if code not in reference_by_code:
                raise ValueError(
                    f"Product '{code}' was found in the competition files but not in competition_reference_df."
                )
            self.product_index_to_reference_row.append(reference_by_code[code])

        if self.topk_indices.shape[:2] != (
            len(week_columns),
            len(self.product_index_to_reference_row),
        ):
            raise ValueError(
                "The Top-K tensors do not match the metadata dimensions. "
                f"Expected {(len(week_columns), len(self.product_index_to_reference_row))}, "
                f"got {self.topk_indices.shape[:2]}."
            )

    def _extract_temporal_features(self, data: pd.DataFrame) -> torch.FloatTensor:
        """
        Extract the four temporal features used by the baseline model.
        If the named columns are present, use them explicitly.
        Otherwise, fall back to the original positional slicing.
        """
        expected_cols = ["day", "week", "month", "year"]
        if all(col in data.columns for col in expected_cols):
            return torch.FloatTensor(data[expected_cols].values)
        return torch.FloatTensor(data.iloc[:, 13:17].values)

    def _get_sales_tensor(self, data: pd.DataFrame) -> torch.FloatTensor:
        """
        Return the 12-week target horizon.
        The original repository takes the first 12 columns as the target horizon,
        so the same assumption is preserved here.
        """
        return torch.FloatTensor(data.iloc[:, :12].values)

    def _get_scaled_gtrend(self, label: str, start_date: pd.Timestamp) -> np.ndarray:
        """
        Return the normalized Google Trends history for a single attribute.
        The series is padded with zeros if it is shorter than trend_len.
        """
        gtrend_start = start_date - pd.DateOffset(weeks=52)
        try:
            series = self.gtrends.loc[gtrend_start:start_date][label].values[-52:]
        except KeyError:
            series = np.zeros(self.trend_len, dtype=np.float32)

        series = np.asarray(series, dtype=np.float32)
        if len(series) >= self.trend_len:
            series = series[-self.trend_len :]
        else:
            pad = np.zeros(self.trend_len - len(series), dtype=np.float32)
            series = np.concatenate([pad, series], axis=0)

        return safe_minmax_scale(series)

    def _build_competition_snapshot(self, row: pd.Series):
        """
        Build the launch-week Top-K neighborhood for one target product.
        This first integration uses the launch week as the assortment snapshot
        to stay compatible with the current GTM decoder.
        """
        target_code = str(row["external_code"])
        product_index = self.external_code_to_product_index[target_code]

        launch_week_label = floor_to_monday(pd.Timestamp(row["release_date"])).strftime(
            "%Y-%m-%d"
        )
        if launch_week_label not in self.week_to_index:
            raise ValueError(
                f"Launch week '{launch_week_label}' was not found in the competition metadata."
            )
        week_index = self.week_to_index[launch_week_label]

        neighbor_indices = self.topk_indices[
            week_index, product_index, : self.competition_top_k
        ]
        neighbor_scores = self.topk_values[
            week_index, product_index, : self.competition_top_k
        ]

        neighbor_categories = []
        neighbor_colors = []
        neighbor_fabrics = []
        neighbor_img_paths = []
        neighbor_mask = []

        for neighbor_index, score in zip(neighbor_indices.tolist(), neighbor_scores.tolist()):
            if neighbor_index == -1 or score <= 0:
                neighbor_categories.append(0)
                neighbor_colors.append(0)
                neighbor_fabrics.append(0)
                neighbor_img_paths.append("")
                neighbor_mask.append(0.0)
                continue

            neighbor_row = self.product_index_to_reference_row[neighbor_index]
            neighbor_categories.append(self.cat_dict[neighbor_row["category"]])
            neighbor_colors.append(self.col_dict[neighbor_row["color"]])
            neighbor_fabrics.append(self.fab_dict[neighbor_row["fabric"]])
            neighbor_img_paths.append(neighbor_row["image_path"])
            neighbor_mask.append(1.0)

        return (
            neighbor_categories,
            neighbor_colors,
            neighbor_fabrics,
            neighbor_img_paths,
            neighbor_scores.astype(np.float32),
            np.asarray(neighbor_mask, dtype=np.float32),
        )

    def preprocess_data(self):
        data = self.data_df.copy()

        multitrends = []
        image_paths = []

        neighbor_categories_all = []
        neighbor_colors_all = []
        neighbor_fabrics_all = []
        neighbor_img_paths_all = []
        neighbor_scores_all = []
        neighbor_mask_all = []

        for _, row in tqdm(data.iterrows(), total=len(data), ascii=True):
            cat = row["category"]
            col = row["color"]
            fab = row["fabric"]
            start_date = pd.Timestamp(row["release_date"])
            img_path = row["image_path"]

            cat_gtrend = self._get_scaled_gtrend(cat, start_date)
            col_gtrend = self._get_scaled_gtrend(col, start_date)
            fab_gtrend = self._get_scaled_gtrend(fab, start_date)

            multitrends.append(np.vstack([cat_gtrend, col_gtrend, fab_gtrend]))
            image_paths.append(img_path)

            if self.use_competition_extension:
                (
                    n_cat,
                    n_col,
                    n_fab,
                    n_img_paths,
                    n_scores,
                    n_mask,
                ) = self._build_competition_snapshot(row)
                neighbor_categories_all.append(n_cat)
                neighbor_colors_all.append(n_col)
                neighbor_fabrics_all.append(n_fab)
                neighbor_img_paths_all.append(n_img_paths)
                neighbor_scores_all.append(n_scores)
                neighbor_mask_all.append(n_mask)

        multitrends = torch.FloatTensor(np.asarray(multitrends, dtype=np.float32))

        data = data.drop(
            ["external_code", "season", "release_date", "image_path"],
            axis=1,
            errors="ignore",
        )

        item_sales = self._get_sales_tensor(data)
        temporal_features = self._extract_temporal_features(data)
        categories = torch.LongTensor([self.cat_dict[val] for val in data["category"].values])
        colors = torch.LongTensor([self.col_dict[val] for val in data["color"].values])
        fabrics = torch.LongTensor([self.fab_dict[val] for val in data["fabric"].values])

        if not self.use_competition_extension:
            return LazyDataset(
                item_sales=item_sales,
                categories=categories,
                colors=colors,
                fabrics=fabrics,
                temporal_features=temporal_features,
                gtrends=multitrends,
                img_paths=image_paths,
                img_root=self.img_root,
            )

        return LazyDataset(
            item_sales=item_sales,
            categories=categories,
            colors=colors,
            fabrics=fabrics,
            temporal_features=temporal_features,
            gtrends=multitrends,
            img_paths=image_paths,
            img_root=self.img_root,
            neighbor_categories=torch.LongTensor(np.asarray(neighbor_categories_all)),
            neighbor_colors=torch.LongTensor(np.asarray(neighbor_colors_all)),
            neighbor_fabrics=torch.LongTensor(np.asarray(neighbor_fabrics_all)),
            neighbor_img_paths=neighbor_img_paths_all,
            neighbor_scores=torch.FloatTensor(np.asarray(neighbor_scores_all)),
            neighbor_mask=torch.FloatTensor(np.asarray(neighbor_mask_all)),
        )

    def get_loader(self, batch_size, train=True):
        print("Starting dataset creation process...")
        data_with_gtrends = self.preprocess_data()

        if train:
            return DataLoader(
                data_with_gtrends,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
            )
        return DataLoader(
            data_with_gtrends,
            batch_size=1,
            shuffle=False,
            num_workers=2,
        )