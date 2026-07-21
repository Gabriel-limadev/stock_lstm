import pandas as pd

from .indicators import (
    calculate_returns,
    calculate_sma,
    calculate_rsi,
    calculate_volatility
)

def create_features(df: pd.DataFrame)-> pd.DataFrame:
    '''
    Cria características a partir dos dados do DataFrame.
    '''
    df["Date"] = pd.to_datetime(df["Date"])

    df = calculate_returns(df)
    df = calculate_sma(df)
    df = calculate_rsi(df)
    df = calculate_volatility(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df