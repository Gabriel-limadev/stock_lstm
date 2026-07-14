import pandas as pd


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Calcula a mudança percentual diária do preço de fechamento.
    '''

    # Calculando a mudança percentual diária do preço de fechamento
    df["Return"] = df["Close"].pct_change()
    return df

def calculate_sma(df: pd.DataFrame, windows: list = [10, 50]) -> pd.DataFrame:
    '''
    Calcula a Média Móvel Simples (SMA) para os dados do DataFrame.
    '''
    
    for window in windows:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
    return df

def calculate_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    '''
    Calcula o Índice de Força Relativa (RSI) para os dados do DataFrame.
    '''
    # Calculando as mudanças de preço
    delta = df["Close"].diff()

    # Separando as mudanças positivas e negativas
    gain = (delta.where(delta > 0)).fillna(0)
    loss = (-delta.where(delta < 0)).fillna(0)

    # Calculando a média móvel de ganhos e perdas
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    # Calculando o RSI
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def calculate_volatility(df: pd.DataFrame, windows=[10,50]) -> pd.DataFrame:
    ''' 
    Calcula a volatilidade para os dados do DataFrame.
    '''
    for window in windows:
        df[f"Volatility_{window}"] = df["Return"].rolling(window=window).std()

    return df