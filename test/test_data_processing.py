import pytest
from src.data_processing import load_data, fill_missing

def test_data_loading():
    df = load_data("../data/train.csv")
    assert not df.empty, "Data loading failed"

def test_missing_value_handling():
    df = load_data("../data/train.csv")
    df_filled = fill_missing(df)
    assert df_filled.isnull().sum().sum() == 0, "Missing values remain"

if __name__ == "__main__":
    pytest.main()