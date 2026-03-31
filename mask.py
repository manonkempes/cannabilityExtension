import numpy as np
import pandas as pd


def build_activity_indicator(matrix, t, horizon_weeks=12):
    """
    Voegt per product de indicator a_i(t) toe:
    mask = 1 als product actief is op tijdstip t, anders 0
    """
    matrix = matrix.copy()
    matrix["release_date"] = pd.to_datetime(matrix["release_date"])
    t = pd.Timestamp(t)

    matrix["mask"] = (
        (matrix["release_date"] <= t) &
        (t < matrix["release_date"] + pd.Timedelta(weeks=horizon_weeks))
    ).astype(np.float32)

    return matrix


def build_pairwise_activity_mask(matrix, t, horizon_weeks=12):
    """
    Bouwt de pairwise activity mask M_ij(t) = a_i(t) * a_j(t)
    """
    matrix = build_activity_indicator(matrix, t=t, horizon_weeks=horizon_weeks)

    a_t = matrix["mask"].to_numpy(dtype=np.float32)   # vector met a_i(t)
    M_t = np.outer(a_t, a_t).astype(np.float32)       # matrix met M_ij(t)

    return matrix, M_t


import numpy as np
import pandas as pd


def build_activity_mask_over_time(matrix, week_dates, horizon_weeks=12):
    """
    Constructs activity indicators and pairwise activity masks over multiple weeks.

    A[i, t]   = a_i(t)
    M[t,i,j]  = a_i(t) * a_j(t)

    Returns:
    - A: shape [n_products, n_weeks]
    - M: shape [n_weeks, n_products, n_products]
    """
    matrix = matrix.copy()

    release_dates = pd.to_datetime(matrix["release_date"]).values.astype("datetime64[ns]")
    week_dates = pd.to_datetime(week_dates).values.astype("datetime64[ns]")
    horizon = np.timedelta64(horizon_weeks * 7, "D")

    A = (
        (release_dates[:, None] <= week_dates[None, :]) &
        (week_dates[None, :] < release_dates[:, None] + horizon)
    ).astype(np.float32)

    M = np.einsum("it,jt->tij", A, A).astype(np.float32)

    return A, M