import pandas as pd

from .indicators import (
    calculate_returns,
    calculate_sma,
    calculate_rsi,
    calculate_volatility
)

def create_features(processed_path: str, features_path: str)-> pd.DataFrame:
    '''
    Cria características a partir dos dados do DataFrame.
    '''
    df = pd.read_csv(processed_path)
    df["Date"] = pd.to_datetime(df["Date"])

    df = calculate_returns(df)
    df = calculate_sma(df)
    df = calculate_rsi(df)
    df = calculate_volatility(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(features_path, index=False)

    return df