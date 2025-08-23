import os
import glob
from typing import Optional, Tuple

import pandas as pd


def resolve_artifacts_dir(explicit_dir: Optional[str] = None, base_dir: str = "artifacts/qvm_v221_flat_vectorized") -> str:
    """
    Resolve the artifacts directory to use.

    Preference order:
    1) explicit_dir argument if provided and exists
    2) environment variable QVM_ARTIFACTS_DIR if exists
    3) latest subdirectory under base_dir
    """
    if explicit_dir and os.path.isdir(explicit_dir):
        return explicit_dir
    env_dir = os.environ.get("QVM_ARTIFACTS_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    # Fallback to latest run dir
    dirs = [p for p in glob.glob(os.path.join(base_dir, "*/")) if os.path.isdir(p)]
    if not dirs:
        raise FileNotFoundError(f"No artifact directories found under {base_dir}")
    dirs_sorted = sorted(dirs, key=os.path.getmtime)
    return dirs_sorted[-1].rstrip("/")


def _load_series_with_flexible_column(path: str) -> pd.Series:
    """
    Load a CSV with columns [date, value] but allow flexible value column naming.
    Returns a pandas Series indexed by datetime.
    """
    df = pd.read_csv(path)
    if 'date' not in df.columns:
        raise KeyError(f"CSV at {path} does not contain a 'date' column. Columns: {list(df.columns)}")
    value_col = None
    # prefer standard names first
    for c in ("return", "returns", "ret", "close_price", "value"):
        if c in df.columns:
            value_col = c
            break
    if value_col is None:
        # fall back to second column
        non_date_cols = [c for c in df.columns if c != 'date']
        if not non_date_cols:
            raise KeyError(f"CSV at {path} does not contain a value column besides 'date'")
        value_col = non_date_cols[0]
    s = (
        df.assign(date=pd.to_datetime(df['date']))
          .set_index('date')[value_col]
          .astype(float)
          .sort_index()
    )
    return s


def load_returns_series(artifacts_dir: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load portfolio no-risk daily returns and benchmark daily returns from the artifacts directory.
    Returns (portfolio_returns, benchmark_returns) as daily series aligned on the intersection of dates.
    """
    nr_path = os.path.join(artifacts_dir, 'no_risk_returns.csv')
    bm_path = os.path.join(artifacts_dir, 'benchmark_returns.csv')
    if not os.path.exists(nr_path):
        raise FileNotFoundError(f"Missing no_risk_returns.csv at {nr_path}")
    if not os.path.exists(bm_path):
        raise FileNotFoundError(f"Missing benchmark_returns.csv at {bm_path}")
    nr = _load_series_with_flexible_column(nr_path)
    bm = _load_series_with_flexible_column(bm_path)
    idx = nr.index.intersection(bm.index)
    nr = nr.loc[idx]
    bm = bm.loc[idx]
    return nr, bm


def to_monthly_compounded(daily_returns: pd.Series) -> pd.Series:
    """
    Convert a daily returns series to monthly compounded returns at month end.
    """
    if daily_returns.empty:
        return daily_returns
    daily_returns = daily_returns.sort_index()
    monthly = (1.0 + daily_returns).groupby([daily_returns.index.to_period('M')]).prod() - 1.0
    # convert PeriodIndex to Timestamp at month end
    monthly.index = monthly.index.to_timestamp('M')
    return monthly


def compute_rolling_beta(portfolio: pd.Series, benchmark: pd.Series, window: int = 6) -> pd.Series:
    """
    Compute rolling beta using covariance/variance over a specified window length.
    Assumes both series are aligned and at the same frequency.
    """
    portfolio = portfolio.sort_index()
    benchmark = benchmark.sort_index()
    idx = portfolio.index.intersection(benchmark.index)
    portfolio = portfolio.loc[idx]
    benchmark = benchmark.loc[idx]
    betas = []
    for i in range(len(idx)):
        j = max(0, i - window + 1)
        r = portfolio.iloc[j:i+1]
        b = benchmark.iloc[j:i+1]
        if len(r) >= 3 and float(b.var()) > 0.0:
            cov = float(pd.concat([r, b], axis=1).cov().iloc[0, 1])
            var_b = float(b.var())
            betas.append(cov / var_b)
        else:
            betas.append(float('nan'))
    return pd.Series(betas, index=idx)


