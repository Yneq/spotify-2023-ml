from pathlib import Path
import pandas as pd

def load_spotify_data(path: str | Path) -> pd.DataFrame:
    """Load Spotify 2023 dataset from CSV with safe defaults."""
    return pd.read_csv(
        path,
        encoding="latin1",
        on_bad_lines="skip",
    )
