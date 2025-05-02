import pytest
from src.model_training import logistic_regression

def test_model_initialization():
    X_train, y_train = [[1, 2], [3, 4]], [0, 1]
    model = logistic_regression(X_train, y_train)
    assert hasattr(model, "fit"), "Model initialization failed"

if __name__ == "__main__":
    pytest.main()