import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
#import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
import os
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error



def preprocess_data(csv_path, encoder=None, scaler=None, fit=False, scaler_file="std_scaler.bin"):
    """ Preprocess metal complex dataset """

    df = pd.read_csv(csv_path)

    # Extract metadata
    name_w_spin = df['#name_w_spin'].values
    df = df.drop(columns=["#name_w_spin", "Unnamed: 0"], errors="ignore")

    # Split targets and spin
    y = df.pop('SSE').values
    spin = df.pop('spin').values

    # Columns to scale
    des_list = [str(i) for i in range(1, 18)]

    # Prepare encoder/scaler
    if fit:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        scaler = StandardScaler(with_mean=True, with_std=True)

        one_hot = encoder.fit_transform(df[['metal', 'Ox']])
        scaled = scaler.fit_transform(df[des_list])
        joblib.dump(scaler, scaler_file, compress=True)
    else:
        one_hot = encoder.transform(df[['metal', 'Ox']])
        scaled = scaler.transform(df[des_list])

    # Combine all processed features
    one_hot_df = pd.DataFrame(one_hot, columns=encoder.get_feature_names_out(['metal', 'Ox']))
    scaled_df = pd.DataFrame(scaled, columns=scaler.get_feature_names_out(des_list))
    df_proc = pd.concat([df.drop(['metal', 'Ox'] + des_list, axis=1), one_hot_df, scaled_df], axis=1)

    return df_proc, y, spin, name_w_spin, encoder, scaler



# Training data
x_2_train, y_train, spin_train, name_w_spin_train, encoder, scaler = preprocess_data("monometal_adiabatic_Des-d_IS-LS_train.csv", fit=True)

# Test data
x_2_test, y_test, spin_test, name_w_spin_test, _, _ = preprocess_data("monometal_adiabatic_Des-d_IS-LS_test.csv", encoder=encoder, scaler=scaler, fit=False)

print("Train, test size:", x_2_train.shape, x_2_test.shape)


def make_param_grid():
    a = np.logspace(-7, 0, num=71)  # 71 points from 1e-7 to 1
    return {'alpha': a, 'gamma': a}

def run_grid_search(x_train, y_train, x_test, y_test, out_file="GS_results.csv"):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test  = np.array(x_test)
    y_test  = np.array(y_test)

    param_grid = make_param_grid()

    krr = KernelRidge(kernel='rbf')
    grid_search = GridSearchCV(krr, param_grid,cv=5, n_jobs=-1, refit=True, return_train_score=True,verbose=1)

    # Fit once only
    grid_search.fit(x_train, y_train)

    # Save results to CSV
    results = pd.DataFrame({
        'alpha': [p['alpha'] for p in grid_search.cv_results_['params']],
        'gamma': [p['gamma'] for p in grid_search.cv_results_['params']],
        'train_mean': grid_search.cv_results_['mean_train_score'],
        'train_std': grid_search.cv_results_['std_train_score'],
        'test_mean': grid_search.cv_results_['mean_test_score'],
        'test_std': grid_search.cv_results_['std_test_score'],
    })
    results.to_csv(out_file, index=False)

    # Best model
    best_krr = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("best_params", best_params)

    # Evaluate performance
    def evaluate(model, X, y, label="Train"):
        preds = model.predict(X)
        mse = mean_squared_error(y, preds)
        mae = mean_absolute_error(y, preds)
        score = model.score(X, y)
        return preds, mae, np.sqrt(mse), score

    train_pred, train_mae, train_rmse, train_score = evaluate(best_krr, x_train, y_train, "Train")
    test_pred, test_mae, test_rmse, test_score = evaluate(best_krr, x_test, y_test, "Test")

    print(f"""
Performance Summary
-----------------------------------------
Train MAE  : {train_mae:.6f}
Test  MAE  : {test_mae:.6f}
Train RMSE : {train_rmse:.6f}
Test  RMSE : {test_rmse:.6f}
Train R2   : {train_score:.6f}
Test  R2   : {test_score:.6f}
-----------------------------------------
    """)

    return train_pred, test_pred



# Calculate the prediction for train and test data
train_pred, test_pred = run_grid_search(x_2_train, y_train, x_2_test, y_test)

train_df = pd.DataFrame({
    "name_w_spin": name_w_spin_train,
    "spin": spin_train,
    "actual_SSE": y_train,
    "predicted_SSE": train_pred
})

test_df = pd.DataFrame({
    "name_w_spin": name_w_spin_test,
    "spin": spin_test,
    "actual_SSE": y_test,
    "predicted_SSE": test_pred
})

# --- Write predicted values---
train_output_file = "monometal_adiabatic_Des-d_IS-LS_train_actVSpred.dat"
test_output_file = "monometal_adiabatic_Des-d_IS-LS_test_actVSpred.dat"

with open(train_output_file, "w") as f_out:
    f_out.write("#name_w_spin,spin,actual_SSE,predicted_SSE\n")

train_df.to_csv(train_output_file, mode="a", header=False, index=False)

with open(test_output_file, "w") as f_out:
    f_out.write("#name_w_spin,spin,actual_SSE,predicted_SSE\n")

test_df.to_csv(test_output_file, mode="a", header=False, index=False)
