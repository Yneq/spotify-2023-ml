import pandas as pd


def load_spotify_data(path: str) -> pd.DataFrame:
    """Load Spotify 2023 dataset from CSV."""
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    data_path = "data/raw/spotify-2023.csv"
    df = load_spotify_data(data_path)
    print(df.head())
