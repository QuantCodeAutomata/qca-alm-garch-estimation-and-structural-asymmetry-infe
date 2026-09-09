import numpy as np
import pandas as pd
import yfinance as yf

# Asset configuration: ticker, download start/end, return type
# yfinance end date is exclusive, so end is one calendar day past the last desired date.
ASSET_CONFIG = {
    'SP500':     {'ticker': '^GSPC',   'start': '2000-01-03', 'end': '2018-06-28', 'return_type': 'open_close'},
    'FTSE100':   {'ticker': '^FTSE',   'start': '2000-01-04', 'end': '2018-06-28', 'return_type': 'open_close'},
    'DAX':       {'ticker': '^GDAXI',  'start': '2000-01-03', 'end': '2018-06-28', 'return_type': 'open_close'},
    'Nikkei225': {'ticker': '^N225',   'start': '2000-02-02', 'end': '2018-06-28', 'return_type': 'open_close'},
    'KOSPI':     {'ticker': '^KS11',   'start': '2000-01-04', 'end': '2018-06-28', 'return_type': 'open_close'},
    'Bitcoin':   {'ticker': 'BTC-USD', 'start': '2012-01-02', 'end': '2025-01-07', 'return_type': 'close_close'},
}


def _download_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance and return a flat-column DataFrame."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance 0.2+ returns (Price, Ticker) MultiIndex for all downloads
        df = df.xs(ticker, axis=1, level=1)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    return df


def load_asset_returns(asset: str) -> pd.Series:
    """
    Download daily OHLCV data and return log returns as a clean pd.Series.

    Equity indices (SP500, FTSE100, DAX, Nikkei225, KOSPI): open-to-close log returns,
      r_t = log(Close_t) - log(Open_t).
    Bitcoin: close-to-close log returns, r_t = log(Close_t) - log(Close_{t-1}).

    Parameters
    ----------
    asset : str
        One of 'SP500', 'FTSE100', 'DAX', 'Nikkei225', 'KOSPI', 'Bitcoin'.

    Returns
    -------
    pd.Series
        Daily log returns with a DatetimeIndex, NaNs dropped.
    """
    cfg = ASSET_CONFIG[asset]
    df = _download_ohlcv(cfg['ticker'], cfg['start'], cfg['end'])

    if cfg['return_type'] == 'open_close':
        log_close = np.log(df['Close'].astype(float))
        log_open = np.log(df['Open'].astype(float))
        returns = log_close - log_open
    else:  # close_close (Bitcoin)
        log_close = np.log(df['Close'].astype(float))
        returns = log_close.diff()

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    returns.name = asset
    return returns


def load_all_assets() -> dict:
    """
    Download and return log return series for all six assets.

    Returns
    -------
    dict[str, pd.Series]
        Keys are asset names; values are daily log return Series.
    """
    return {asset: load_asset_returns(asset) for asset in ASSET_CONFIG}


def load_realized_variance(asset: str) -> pd.Series:
    """
    Return a realized variance (RV) proxy for the given asset.

    For all assets the proxy is the squared daily log return produced by
    load_asset_returns (open-to-close squared for equity indices,
    close-to-close squared for Bitcoin).  Five-minute intraday bars are
    not available for the full historical period via yfinance, so this
    daily-squared-return proxy is used throughout.

    Parameters
    ----------
    asset : str
        One of 'SP500', 'FTSE100', 'DAX', 'Nikkei225', 'KOSPI', 'Bitcoin'.

    Returns
    -------
    pd.Series
        Daily RV proxy (squared log returns) with a DatetimeIndex.
    """
    returns = load_asset_returns(asset)
    rv = returns ** 2
    rv.name = f'{asset}_RV'
    return rv


def load_all_rv() -> dict:
    """
    Return RV proxy series for all six assets.

    Returns
    -------
    dict[str, pd.Series]
        Keys are asset names; values are daily RV proxy Series.
    """
    return {asset: load_realized_variance(asset) for asset in ASSET_CONFIG}
