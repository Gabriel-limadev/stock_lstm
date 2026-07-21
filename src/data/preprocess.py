import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Limpeza inicial dos dados.
    '''

    # Converter data
    df["Date"] = pd.to_datetime(df["Date"])

    # Remover duplicados
    df = df.drop_duplicates()

    # Remover linhas vazias
    df = df.dropna()

    return df

def validate_data(df: pd.DataFrame):
    '''
    Valida se os dados estão corretos.
    '''

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Colunas ausentes: {missing}")

    if df.empty:
        raise ValueError("Dataset vazio")

    return df

def preprocess_data(df: pd.DataFrame)-> pd.DataFrame:
    '''
    Preprocessa os dados brutos e salva o resultado.
    '''
    df_validated = validate_data(df)

    df_cleaned = clean_data(df_validated)

    return df_cleaned