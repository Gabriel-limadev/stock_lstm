from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

def download_market_data(stock: str, start_date: str = "2020-01-01", end_date: str | None = None) -> pd.DataFrame:
    '''
    Download stock data from Yahoo Finance.

    Parameters:
    stock (str): The stock symbol.
    start_date (str): The start date for the data.
    end_date (str): The end date for the data.

    Returns:
    pd.DataFrame: The downloaded stock data.
    '''
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(
        stock,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise ValueError(
            f"A ação '{stock}' não foi identificada no Yahoo Finance."
        )
        
    # Converte o índice (Date) em coluna
    df.reset_index(inplace=True)

    # Corrige MultiIndex criado pelo yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[1] == "" else col[0]
            for col in df.columns
        ]

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df