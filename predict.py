from myconfig import *
from utils import *

def predict_incident(df, model, features):
    x_test, processed_features = create_features(df, n_before = 3, cols = features)
    return model.predict(x_test[processed_features])

def main():
    df = pd.read_pickle(output + '/df.pkl')
    loaded_model = pickle.load(open(model_path, "rb"))
    predict_incident(df, loaded_model)

if __name__ == '__main__':
    main()