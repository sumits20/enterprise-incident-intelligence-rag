import pandas as pd

def load_incidents(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df