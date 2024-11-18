import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic, RBF, Matern, DotProduct
from sklearn.model_selection import cross_val_score

def data_loading():
    train_df = pd.read_csv(f'task2/train.csv')
    test_df = pd.read_csv(f'task2/test.csv')

    # One-hot encoder
    encoder = OneHotEncoder(drop=None)
    train_season_encoded = encoder.fit_transform(train_df[['season']]).toarray()
    test_season_encoded = encoder.transform(test_df[['season']]).toarray()

    # Replace season column with encoded values
    train_encoded_df = pd.DataFrame(train_season_encoded, columns=encoder.get_feature_names_out(['season']))
    train_df = pd.concat([train_df.drop(columns=['season']), train_encoded_df], axis=1)

    test_encoded_df = pd.DataFrame(test_season_encoded, columns=encoder.get_feature_names_out(['season']))
    test_df = pd.concat([test_df.drop(columns=['season']), test_encoded_df], axis=1)

    # Impute missing values using KNNImputer
    imputer = KNNImputer(n_neighbors=3)
    train_df= pd.DataFrame(imputer.fit_transform(train_df), columns=train_df.columns)
    test_df = pd.DataFrame(imputer.fit_transform(test_df), columns=test_df.columns)

    print("Head of encoded training data:")
    print(train_df.head(10))

    print("Head of encoded test data:")
    print(test_df.head(10))


    # data seperation -> following is using Model imputation from Kaggle documentation
    target_column = 'price_CHF'
    X_train = train_df.drop(columns=[target_column], axis = 1)
    y_train = train_df[target_column]
    X_test = test_df

    # Ensure no missing values
    assert not X_train.isnull().values.any(), "Missing values in X_train after imputation"
    assert not X_test.isnull().values.any(), "Missing values in X_test after imputation"
    assert not y_train.isnull().values.any(), "Missing values in y_train after imputation"

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):

    y_pred=np.zeros(X_test.shape[0])

    # List of Kernels and evaluation
    kernels = [DotProduct(), RBF(), Matern(), RationalQuadratic(), RationalQuadratic(alpha=2, length_scale=1.1), RationalQuadratic(alpha=1.5, length_scale=1.5)]
    best_kernel = None
    best_score = float('-inf')

    # going through all kernels and evaluate them
    for Mykernel in kernels:
        gpr = GaussianProcessRegressor(kernel = Mykernel)
        scores = cross_val_score(gpr, X_train, y_train, cv=5, scoring='r2')
        mean_score = np.mean(scores)
        print(f"Kernel: {Mykernel}, Mean R-squared score: {mean_score}")

        # Update the best kernel if the current one has a higher score
        if mean_score > best_score:
            best_score = mean_score
            best_kernel = Mykernel

    print(f"Best Kernel: {best_kernel}, Mean R-squared score: {best_score}")

    # Fit the model using the best kernel
    gpr_best = GaussianProcessRegressor(kernel=best_kernel)
    gpr_best.fit(X_train, y_train)

    # Use the fitted model to make predictions on test data
    y_pred = gpr_best.predict(X_test)


    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    X_train, y_train, X_test = data_loading()
    y_pred = modeling_and_prediction(X_train, y_train, X_test)
    dt = pd.DataFrame(y_pred, columns=['price_CHF'])
    dt.to_csv('task2/results.csv', index=False)
    print("\nResults file successfully generated!")
