import joblib

def load_model(path="models/rul_model.pkl"):
    return joblib.load(path)

def predict_rul(model, data):
    features = data.drop(['engine_id', 'cycle', 'RUL'], axis=1, errors='ignore')
    return model.predict(features)
