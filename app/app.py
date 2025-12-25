import pickle
from fastapi import FastAPI
import uvicorn
from typing import Dict, Any
import xgboost as xgb


app = FastAPI(title="f1_podium_prediction")

with open('model.bin', 'rb') as f_in:
    dv,model = pickle.load(f_in)


def predict_single(info):
    info_dict = [info]
    X = dv.transform(info_dict)
    features = list(dv.get_feature_names_out())
    dinfo = xgb.DMatrix(X, feature_names=features)
    result = model.predict(dinfo)[0]
    return float(result)


@app.post("/predict")
def predict(info: Dict[str, Any]):
    prob = predict_single(info)

    return {
        "podium probability": prob,
        "podium": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)