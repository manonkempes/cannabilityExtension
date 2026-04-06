from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HORIZON_WEEKS = 12
DEFAULT_TOP_K = 10


def floor_to_monday(series: pd.Series) -> pd.Series:
    """Return the Monday of the calendar week for each timestamp."""
    dt = pd.to_datetime(series)
    return (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()


def validate_unique_product_ids(data: pd.DataFrame, id_col: str) -> pd.Series:
    """
    Validate that each product has a unique identifier.
    The competition tensors are product-by-product, so each row must map
    to exactly one product ID.
    """
    product_ids = data[id_col].astype(str)
    duplicated = product_ids[product_ids.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            f"Column '{id_col}' must contain unique product IDs. "
            f"Found duplicates such as: {duplicated[:10]}"
        )
    return product_ids


def build_availability_matrix(
    data: pd.DataFrame,
    horizon_weeks: int = DEFAULT_HORIZON_WEEKS,
    id_col: str = "external_code",
    date_col: str = "release_date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the weekly activity matrix A.

    Rows = products
    Columns = calendar weeks
    Values = a_i(t), where:
        a_i(t) = 1 if product i is active in week t
        a_i(t) = 0 otherwise

    A product is assumed to be active from its release week for
    `horizon_weeks` consecutive weeks.
    """
    required_cols = {id_col, date_col}
    missing_cols = required_cols - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    working = data.copy()
    working[date_col] = pd.to_datetime(working[date_col], errors="coerce")
    if working[date_col].isna().any():
        bad_rows = working.index[working[date_col].isna()].tolist()
        raise ValueError(
            f"Column '{date_col}' contains invalid dates at rows: {bad_rows[:10]}"
        )

    working["product_id"] = validate_unique_product_ids(working, id_col)
    working["release_week_start"] = floor_to_monday(working[date_col])

    min_week = working["release_week_start"].min()
    max_week = working["release_week_start"].max() + pd.Timedelta(
        weeks=horizon_weeks - 1
    )
    calendar_weeks = pd.date_range(start=min_week, end=max_week, freq="7D")
    week_labels = [week.strftime("%Y-%m-%d") for week in calendar_weeks]

    # Vectorized availability computation
    release_weeks = working["release_week_start"].to_numpy(dtype="datetime64[D]")
    cal_weeks = calendar_weeks.to_numpy(dtype="datetime64[D]")

    # Shape: (num_products, num_weeks)
    diff_days = (cal_weeks[None, :] - release_weeks[:, None]).astype("timedelta64[D]").astype(int)
    diff_weeks = diff_days // 7

    availability = ((diff_weeks >= 0) & (diff_weeks < horizon_weeks)).astype(np.int8)

    availability_matrix = pd.DataFrame(
        availability,
        index=working["product_id"].astype(str),
        columns=week_labels,
        dtype=np.int8,
    )

    row_mapping = working[[id_col, date_col, "release_week_start", "product_id"]].copy()
    row_mapping[date_col] = row_mapping[date_col].dt.strftime("%Y-%m-%d")
    row_mapping["release_week_start"] = row_mapping["release_week_start"].dt.strftime(
        "%Y-%m-%d"
    )

    return availability_matrix, row_mapping


def load_similarity_matrix(similarity_path: Path, product_ids: list[str]) -> np.ndarray:
    """
    Load a precomputed cosine similarity matrix.

    Supported formats:
    - CSV: product IDs must be present as both index and columns
    - NPY: must already be saved in the same product order as `product_ids`
    """
    suffix = similarity_path.suffix.lower()

    if suffix == ".csv":
        sim_df = pd.read_csv(similarity_path, index_col=0)
        sim_df.index = sim_df.index.astype(str)
        sim_df.columns = sim_df.columns.astype(str)

        missing_in_index = set(product_ids) - set(sim_df.index)
        missing_in_columns = set(product_ids) - set(sim_df.columns)
        if missing_in_index:
            raise ValueError(
                "The similarity CSV is missing product IDs in its index: "
                f"{sorted(list(missing_in_index))[:10]}"
            )
        if missing_in_columns:
            raise ValueError(
                "The similarity CSV is missing product IDs in its columns: "
                f"{sorted(list(missing_in_columns))[:10]}"
            )

        sim_df = sim_df.loc[product_ids, product_ids]
        similarity = sim_df.to_numpy(dtype=np.float32)

    elif suffix == ".npy":
        similarity = np.load(similarity_path).astype(np.float32)
        expected_shape = (len(product_ids), len(product_ids))
        if similarity.shape != expected_shape:
            raise ValueError(
                f"Similarity NPY has shape {similarity.shape}, expected {expected_shape}. "
                "For .npy input, the product order must exactly match the availability matrix."
            )
    else:
        raise ValueError("Unsupported similarity file format. Use .csv or .npy.")

    return similarity


def build_positive_similarity(similarity: np.ndarray) -> np.ndarray:
    """
    Apply the positive-part operator:
        s_ij_plus = max(s_ij, 0)
    """
    return np.maximum(similarity, 0.0).astype(np.float32)


def build_activity_mask_tensor(availability_matrix: pd.DataFrame) -> np.ndarray:
    """
    Build the time-dependent activity mask:
        m_ij_comp(t) = a_i(t) * a_j(t)
    """
    availability = availability_matrix.to_numpy(dtype=np.int8)  # (products, weeks)
    activity_mask_tensor = np.einsum("pw,qw->wpq", availability, availability).astype(
        np.int8
    )
    return activity_mask_tensor


def build_masked_candidate_similarity_tensor(
    activity_mask_tensor: np.ndarray,
    positive_similarity: np.ndarray,
) -> np.ndarray:
    """
    Build the masked candidate similarity tensor:
        s_tilde_ij_comp(t) = s_ij_plus * m_ij_comp(t)
    """
    masked_candidate_similarity = (
        activity_mask_tensor.astype(np.float32) * positive_similarity[None, :, :]
    ).astype(np.float32)
    return masked_candidate_similarity

def build_topk_neighbors(
    masked_candidate_similarity: np.ndarray,
    top_k: int,
    exclude_self: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each week t and target product i, select the Top-K active similar products.

        N_k_comp(i, t) = TopK_j(s_tilde_ij_comp(t), k_comp)

    Returns:
    - topk_indices: shape (num_weeks, num_products, top_k)
    - topk_values: shape (num_weeks, num_products, top_k)
    """
    num_weeks, num_products, _ = masked_candidate_similarity.shape

    topk_indices = np.full((num_weeks, num_products, top_k), -1, dtype=np.int32)
    topk_values = np.zeros((num_weeks, num_products, top_k), dtype=np.float32)

    effective_top_k = min(top_k, num_products - (1 if exclude_self else 0))
    if effective_top_k <= 0:
        return topk_indices, topk_values

    scores = masked_candidate_similarity.copy()

    if exclude_self:
        idx = np.arange(num_products)
        scores[:, idx, idx] = -1.0

    # Neem per (week, product_i) de indices van de grootste effective_top_k scores
    kth = num_products - effective_top_k
    part = np.argpartition(scores, kth=kth, axis=2)[:, :, -effective_top_k:]
    part_scores = np.take_along_axis(scores, part, axis=2)

    # Sorteer die Top-K kandidaten alsnog aflopend
    order = np.argsort(part_scores, axis=2)[:, :, ::-1]
    ranked_indices = np.take_along_axis(part, order, axis=2).astype(np.int32)
    ranked_values = np.take_along_axis(part_scores, order, axis=2).astype(np.float32)

    # Alleen positieve scores zijn geldig
    invalid = ranked_values <= 0
    ranked_indices[invalid] = -1
    ranked_values[invalid] = 0.0

    topk_indices[:, :, :effective_top_k] = ranked_indices
    topk_values[:, :, :effective_top_k] = ranked_values

    return topk_indices, topk_values


def build_topk_long_dataframe(
    topk_indices: np.ndarray,
    topk_values: np.ndarray,
    week_labels: list[str],
    product_ids: list[str],
) -> pd.DataFrame:
    """Convert the Top-K tensors into a readable long-format table."""
    rows: list[dict] = []
    num_weeks, num_products, top_k = topk_indices.shape

    for t in range(num_weeks):
        for i in range(num_products):
            for rank in range(top_k):
                neighbor_idx = topk_indices[t, i, rank]
                score = topk_values[t, i, rank]
                if neighbor_idx == -1:
                    continue

                rows.append(
                    {
                        "week_start": week_labels[t],
                        "product_i": product_ids[i],
                        "rank": rank + 1,
                        "product_j": product_ids[neighbor_idx],
                        "masked_similarity": float(score),
                    }
                )

    return pd.DataFrame(rows)


def save_outputs(
    availability_matrix: pd.DataFrame,
    row_mapping: pd.DataFrame,
    positive_similarity: np.ndarray,
    activity_mask_tensor: np.ndarray,
    masked_candidate_similarity: np.ndarray,
    topk_indices: np.ndarray,
    topk_values: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    """Save all preprocessing outputs to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    availability_csv_path = output_dir / f"{stem}_availability_matrix.csv"
    availability_npy_path = output_dir / f"{stem}_availability_matrix.npy"
    availability_long_path = output_dir / f"{stem}_availability_long.csv"
    row_mapping_path = output_dir / f"{stem}_availability_row_mapping.csv"

    positive_similarity_npy_path = output_dir / f"{stem}_positive_similarity.npy"
    activity_mask_npy_path = output_dir / f"{stem}_activity_mask_tensor.npy"
    masked_similarity_npy_path = (
        output_dir / f"{stem}_masked_candidate_similarity_tensor.npy"
    )

    topk_indices_npy_path = output_dir / f"{stem}_topk_indices.npy"
    topk_values_npy_path = output_dir / f"{stem}_topk_values.npy"
    topk_long_csv_path = output_dir / f"{stem}_topk_neighbors_long.csv"
    meta_json_path = output_dir / f"{stem}_extension_meta.json"

    availability_matrix.to_csv(
        availability_csv_path,
        index=True,
        index_label="product_id",
    )
    np.save(availability_npy_path, availability_matrix.to_numpy(dtype=np.int8))

    availability_long = (
        availability_matrix.reset_index()
        .melt(id_vars="product_id", var_name="week_start", value_name="is_active")
        .sort_values(["product_id", "week_start"])
        .reset_index(drop=True)
    )
    availability_long.to_csv(availability_long_path, index=False)

    row_mapping.to_csv(row_mapping_path, index=False)

    np.save(positive_similarity_npy_path, positive_similarity)
    np.save(activity_mask_npy_path, activity_mask_tensor)
    np.save(masked_similarity_npy_path, masked_candidate_similarity)
    np.save(topk_indices_npy_path, topk_indices)
    np.save(topk_values_npy_path, topk_values)

    topk_long_df = build_topk_long_dataframe(
        topk_indices=topk_indices,
        topk_values=topk_values,
        week_labels=availability_matrix.columns.tolist(),
        product_ids=availability_matrix.index.tolist(),
    )
    topk_long_df.to_csv(topk_long_csv_path, index=False)

    meta = {
        "formula_activity_mask": "m_ij_comp(t) = a_i(t) * a_j(t)",
        "formula_positive_similarity": "s_ij_plus = max(s_ij, 0)",
        "formula_masked_candidate_similarity": "s_tilde_ij_comp(t) = s_ij_plus * m_ij_comp(t)",
        "formula_topk": "N_k_comp(i, t) = TopK_j(s_tilde_ij_comp(t), k_comp)",
        "num_products": int(availability_matrix.shape[0]),
        "num_calendar_weeks": int(availability_matrix.shape[1]),
        "availability_matrix_shape": [
            int(availability_matrix.shape[0]),
            int(availability_matrix.shape[1]),
        ],
        "positive_similarity_shape": [
            int(positive_similarity.shape[0]),
            int(positive_similarity.shape[1]),
        ],
        "activity_mask_tensor_shape": [
            int(activity_mask_tensor.shape[0]),
            int(activity_mask_tensor.shape[1]),
            int(activity_mask_tensor.shape[2]),
        ],
        "masked_candidate_similarity_shape": [
            int(masked_candidate_similarity.shape[0]),
            int(masked_candidate_similarity.shape[1]),
            int(masked_candidate_similarity.shape[2]),
        ],
        "topk_indices_shape": [
            int(topk_indices.shape[0]),
            int(topk_indices.shape[1]),
            int(topk_indices.shape[2]),
        ],
        "topk_values_shape": [
            int(topk_values.shape[0]),
            int(topk_values.shape[1]),
            int(topk_values.shape[2]),
        ],
        "tensor_axis_order": ["week_index", "product_i_index", "product_j_index"],
        "week_columns": availability_matrix.columns.tolist(),
        "product_ids_in_row_order": availability_matrix.index.tolist(),
        "availability_dtype": "int8",
        "similarity_dtype": "float32",
        "topk_indices_dtype": "int32",
        "topk_values_dtype": "float32",
        "availability_csv": availability_csv_path.name,
        "availability_npy": availability_npy_path.name,
        "availability_long_csv": availability_long_path.name,
        "row_mapping_csv": row_mapping_path.name,
        "positive_similarity_npy": positive_similarity_npy_path.name,
        "activity_mask_tensor_npy": activity_mask_npy_path.name,
        "masked_candidate_similarity_tensor_npy": masked_similarity_npy_path.name,
        "topk_indices_npy": topk_indices_npy_path.name,
        "topk_values_npy": topk_values_npy_path.name,
        "topk_neighbors_long_csv": topk_long_csv_path.name,
    }
    meta_json_path.write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the competition-aware preprocessing tensors from release dates and cosine similarity."
    )
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--similarity_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="dataset")
    parser.add_argument("--horizon_weeks", type=int, default=DEFAULT_HORIZON_WEEKS)
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--id_col", type=str, default="external_code")
    parser.add_argument("--date_col", type=str, default="release_date")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    similarity_path = Path(args.similarity_path)
    output_dir = Path(args.output_dir)
    stem = input_path.stem

    data = pd.read_csv(input_path)

    availability_matrix, row_mapping = build_availability_matrix(
        data=data,
        horizon_weeks=args.horizon_weeks,
        id_col=args.id_col,
        date_col=args.date_col,
    )

    product_ids = availability_matrix.index.astype(str).tolist()
    similarity = load_similarity_matrix(
        similarity_path=similarity_path,
        product_ids=product_ids,
    )

    positive_similarity = build_positive_similarity(similarity)
    activity_mask_tensor = build_activity_mask_tensor(availability_matrix)
    masked_candidate_similarity = build_masked_candidate_similarity_tensor(
        activity_mask_tensor=activity_mask_tensor,
        positive_similarity=positive_similarity,
    )

    topk_indices, topk_values = build_topk_neighbors(
        masked_candidate_similarity=masked_candidate_similarity,
        top_k=args.top_k,
        exclude_self=True,
    )

    save_outputs(
        availability_matrix=availability_matrix,
        row_mapping=row_mapping,
        positive_similarity=positive_similarity,
        activity_mask_tensor=activity_mask_tensor,
        masked_candidate_similarity=masked_candidate_similarity,
        topk_indices=topk_indices,
        topk_values=topk_values,
        output_dir=output_dir,
        stem=stem,
    )

    print(f"Saved competition preprocessing files for '{stem}' to:")
    print(output_dir.resolve())
    print(f"Availability matrix shape: {availability_matrix.shape}")
    print(f"Positive similarity shape: {positive_similarity.shape}")
    print(f"Activity mask tensor shape: {activity_mask_tensor.shape}")
    print(f"Masked candidate similarity shape: {masked_candidate_similarity.shape}")
    print(f"Top-k indices shape: {topk_indices.shape}")
    print(f"Top-k values shape: {topk_values.shape}")


if __name__ == "__main__":
    main()