from src.build_model import build_model

def test_model_training():
    model = build_model("data/train.csv")
    assert model is not None
