import pandas as pd


TEXT_FIELDS = [
    "Issue",
    "Issue_Description",
    "Root_Cause",
    "Resolution",
    "Responsible_Team",
    "Priority",
    "Typical_Resolution_Hours",
    "Date",
]

def build_documents(df: pd.DataFrame) -> pd.Series:

    def row_to_text(r: pd.Series) -> str:
        parts = []
        for col in TEXT_FIELDS:
            if col in r and pd.notna(r[col]):
                parts.append(f"{col.replace('_',' ')}: {r[col]}")
        return "\n".join(parts)

    return df.apply(row_to_text, axis=1)
