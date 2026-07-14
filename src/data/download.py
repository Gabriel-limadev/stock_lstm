import pandas as pd
import yfinance as yf

def download_market_data(stock: str, raw_path: str, start_date: str = "2020-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    '''
    Download stock data from Yahoo Finance.

    Parameters:
    stock (str): The stock symbol.
    raw_path (str): The path to save the raw data.
    start_date (str): The start date for the data.
    end_date (str): The end date for the data.

    Returns:
    pd.DataFrame: The downloaded stock data.
    '''
    df = yf.download(
        stock,
        start=start_date,
        end=end_date
    )

    if df.empty:
        raise ValueError(
            f"Nenhum dado encontrado para '{stock}'."
        )
    df.sort_values("Date", inplace=True)
    df.reset_index(inplace=True)

    df.to_csv(
        raw_path,
        index=False
    )