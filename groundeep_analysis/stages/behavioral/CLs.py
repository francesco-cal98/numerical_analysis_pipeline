import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    PoissonRegressor,
    SGDRegressor,
    SGDClassifier,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, zscore
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NumerosityDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Softmax(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = nn.Linear(n_inputs, n_outputs)

    def forward(self, x):
        return self.linear(x)


def Logistic_regression_multiclass(Xtrain, Xtest, Ytrain, Ytest, labels="int", scale=False):
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()
    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = LogisticRegression(max_iter=100, multi_class="multinomial", solver="saga", penalty="l2")

    _Ytrain = Ytrain
    _Ytest = Ytest

    if scale:
        scaler = StandardScaler()
        _Ytrain = scaler.fit_transform(_Ytrain.reshape(-1, 1)).ravel()
        _Ytest = scaler.transform(_Ytest.reshape(-1, 1)).ravel()

    model.fit(Xtrain, _Ytrain)

    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    if scale:
        predicted_train_classes = (predicted_train * scaler.scale_ + scaler.mean_ + 0.5).astype(int)
        predicted_test_classes = (predicted_test * scaler.scale_ + scaler.mean_ + 0.5).astype(int)
    else:
        predicted_train_classes = predicted_train
        predicted_test_classes = predicted_test

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    prob_train = model.predict_proba(Xtrain)
    prob_test = model.predict_proba(Xtest)
    probTR = np.argmax(prob_train, axis=1)
    probTE = np.argmax(prob_test, axis=1)

    return accuracy_train, predicted_train_classes, predicted_train, probTR, accuracy_test, predicted_test_classes, predicted_test, probTE


def Ridge_regression(Xtrain, Xtest, Ytrain, Ytest, labels="int", scale=False):
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()

    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = Ridge(alpha=53)

    if labels == "log":
        _Ytrain = np.log(Ytrain)
        _Ytest = np.log(Ytest)
    else:
        _Ytrain = Ytrain
        _Ytest = Ytest

    if scale:
        scaler = StandardScaler()
        _Ytrain = scaler.fit_transform(_Ytrain.reshape(-1, 1)).ravel()
        _Ytest = scaler.transform(_Ytest.reshape(-1, 1)).ravel()

    model.fit(Xtrain, _Ytrain)
    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    if labels == "log":
        if scale:
            predicted_train_classes = (np.exp(predicted_train * scaler.scale_ + scaler.mean_) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test * scaler.scale_ + scaler.mean_) + 0.5).astype(int)
        else:
            predicted_train_classes = (np.exp(predicted_train) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test) + 0.5).astype(int)
    else:
        if scale:
            predicted_train_classes = (predicted_train * scaler.scale_ + scaler.mean_ + 0.5).astype(int)
            predicted_test_classes = (predicted_test * scaler.scale_ + scaler.mean_ + 0.5).astype(int)
        else:
            predicted_train_classes = (predicted_train + 0.5).astype(int)
            predicted_test_classes = (predicted_test + 0.5).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    return accuracy_train, predicted_train_classes, predicted_train, False, accuracy_test, predicted_test_classes, predicted_test, False


def SGD_regression(Xtrain, Xtest, Ytrain, Ytest, labels="int", scale=False):
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()

    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = SGDRegressor(loss="squared_error", penalty="l2", random_state=42)

    if labels == "log":
        _Ytrain = np.log(Ytrain)
        _Ytest = np.log(Ytest)
    else:
        _Ytrain = Ytrain
        _Ytest = Ytest

    if scale:
        scaler = StandardScaler()
        _Ytrain = scaler.fit_transform(_Ytrain.reshape(-1, 1)).ravel()
        _Ytest = scaler.transform(_Ytest.reshape(-1, 1)).ravel()

    model.fit(Xtrain, _Ytrain)

    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    if labels == "log":
        if scale:
            predicted_train_classes = (np.exp(predicted_train * scaler.scale_ + scaler.mean_) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test * scaler.scale_ + scaler.mean_) + 0.5).astype(int)
        else:
            predicted_train_classes = (np.exp(predicted_train) + 0.5).astype(int)
            predicted_test_classes = (np.exp(predicted_test) + 0.5).astype(int)
    else:
        if scale:
            predicted_train_classes = (predicted_train * scaler.scale_ + scaler.mean_ + 0.5).astype(int)
            predicted_test_classes = (predicted_test * scaler.scale_ + scaler.mean_ + 0.5).astype(int)
        else:
            predicted_train_classes = (predicted_train + 0.5).astype(int)
            predicted_test_classes = (predicted_test + 0.5).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    return accuracy_train, predicted_train_classes, predicted_train, False, accuracy_test, predicted_test_classes, predicted_test, False


def Linear_regression(Xtrain, Xtest, Ytrain, Ytest):
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()

    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = LinearRegression()
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    predicted_train_classes = np.round(predicted_train).astype(int)
    predicted_test_classes = np.round(predicted_test).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    return accuracy_train, predicted_train_classes, predicted_train, False, accuracy_test, predicted_test_classes, predicted_test, False


def Lasso_regression(Xtrain, Xtest, Ytrain, Ytest):
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()

    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = Lasso(alpha=0.1)
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    predicted_train_classes = np.round(predicted_train).astype(int)
    predicted_test_classes = np.round(predicted_test).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    return accuracy_train, predicted_train_classes, predicted_train, False, accuracy_test, predicted_test_classes, predicted_test, False


def Elastic_net_regression(Xtrain, Xtest, Ytrain, Ytest):
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()

    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    predicted_train_classes = np.round(predicted_train).astype(int)
    predicted_test_classes = np.round(predicted_test).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    return accuracy_train, predicted_train_classes, predicted_train, False, accuracy_test, predicted_test_classes, predicted_test, False


def Poisson_regression(Xtrain, Xtest, Ytrain, Ytest):
    if Xtrain.ndim > 2:
        Xtrain = Xtrain.view(-1, Xtrain.shape[-1]).detach().cpu().numpy()
        Xtest = Xtest.view(-1, Xtest.shape[-1]).detach().cpu().numpy()

    Ytrain = Ytrain.detach().cpu().numpy().ravel()
    Ytest = Ytest.detach().cpu().numpy().ravel()

    model = PoissonRegressor(alpha=0.1, max_iter=1000)
    model.fit(Xtrain, Ytrain)

    predicted_train = model.predict(Xtrain)
    predicted_test = model.predict(Xtest)

    predicted_train_classes = np.round(predicted_train).astype(int)
    predicted_test_classes = np.round(predicted_test).astype(int)

    accuracy_train = accuracy_score(Ytrain, predicted_train_classes)
    accuracy_test = accuracy_score(Ytest, predicted_test_classes)

    return accuracy_train, predicted_train_classes, predicted_train, False, accuracy_test, predicted_test_classes, predicted_test, False


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def SGD_class_fixed(Xtrain, Xtest, Ytrain, Ytest):
    Xtrain = Xtrain.view(-1, Xtrain.shape[2]).detach().cpu().numpy()
    Xtest = Xtest.view(-1, Xtest.shape[2]).detach().cpu().numpy()

    Ytrain = Ytrain.view(-1, Ytrain.shape[-1]).detach().cpu().numpy()
    Ytest = Ytest.view(-1, Ytest.shape[-1]).detach().cpu().numpy()

    y_train = np.argmax(Ytrain, axis=1)
    y_test = np.argmax(Ytest, axis=1)

    unique_classes = np.unique(y_train)
    if unique_classes.size < 2:
        majority_class = int(unique_classes[0])
        pred_train = np.full_like(y_train, majority_class)
        pred_test = np.full_like(y_test, majority_class)
        acc_train = accuracy_score(y_train, pred_train)
        acc_test = accuracy_score(y_test, pred_test)
        print("[Behavioral] Fixed reference dataset contains a single class; returning majority-class baseline.")
        return acc_train, pred_train, acc_test, pred_test

    clf = SGDClassifier(loss="log_loss", penalty="l2", random_state=42)
    clf.fit(Xtrain, y_train)

    pred_train = clf.predict(Xtrain)
    pred_test = clf.predict(Xtest)

    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)

    return acc_train, pred_train, acc_test, pred_test


def beta_extraction_ref_z(choice, idxs_test, N_list, TSA_list, FA_list, guessRate=0.01, ref_num=0):
    """Fit numerosity/size/spacing model for fixed-reference datasets."""
    if isinstance(choice, torch.Tensor):
        choice = choice.detach().cpu().numpy()
    choice = np.asarray(choice)
    if choice.ndim == 2 and choice.shape[1] == 2:
        choice = np.argmax(choice, axis=1)
    choice = choice.reshape(-1)

    idxs_np = idxs_test.detach().cpu().numpy().astype(int).reshape(-1)

    N_list = np.asarray(N_list).astype(float).squeeze()
    TSA_list = np.asarray(TSA_list).astype(float).squeeze()
    FA_list = np.asarray(FA_list).astype(float).squeeze()

    n_lookup = len(N_list)
    if np.any(idxs_np < 0) or np.any(idxs_np >= n_lookup):
        raise ValueError(
            f"Fixed reference indices out of bounds (min={idxs_np.min()}, max={idxs_np.max()}, n={n_lookup})"
        )

    num = N_list[idxs_np]
    safe_num = np.clip(num, 1e-8, None)
    isa = TSA_list[idxs_np] / safe_num
    fa = FA_list[idxs_np]

    intercept, betas, weber, X, _, model, numZ, sizeZ, spaceZ, _, _, _ = _num_size_spacing_model_ref_z(
        choice, num, isa, fa, guessRate
    )

    num_ratios = num / ref_num if ref_num else np.ones_like(num, dtype=float)

    return (
        intercept,
        betas,
        weber,
        X,
        num_ratios,
        None,
        model,
        numZ,
        sizeZ,
        spaceZ,
        num,
        None,
        None,
        None,
    )


def _num_size_spacing_model_ref_z(choice, num, isa, fa, guessRate=0.01):
    tsa = isa * num
    size = isa * tsa
    spar = fa / np.clip(num, 1e-8, None)
    space = spar * fa

    numZ, sizeZ, spaceZ = _zscores_unique(num, size, space)
    X = np.column_stack((numZ, sizeZ, spaceZ))

    intercept, betas, weber = _run_irls(choice, X, guessRate)
    model = {"intercept": intercept, "betas": betas, "weber": weber}
    return intercept, betas, weber, X, None, model, numZ, sizeZ, spaceZ, None, None, None


def _zscores_unique(num, size, space):
    def _zmap(values):
        values = np.asarray(values, dtype=float)
        uniques = np.unique(values)
        if uniques.size <= 1:
            return np.zeros_like(values, dtype=float)
        safe = np.log2(np.clip(uniques, 1e-8, None))
        z_vals = zscore(safe)
        if np.isnan(z_vals).any():
            z_vals = np.zeros_like(z_vals)
        mapping = {val: z for val, z in zip(uniques, z_vals)}
        return np.array([mapping[val] for val in values], dtype=float)

    return _zmap(num), _zmap(size), _zmap(space)


def _run_irls(choice, X, guessRate=0.01, max_iter=5000, tol=1e-12):
    response_rate = 1 - guessRate
    X = np.asarray(X, dtype=float)
    choice = np.asarray(choice, dtype=float).reshape(-1)
    n_obs, n_features = X.shape
    beta = np.zeros(n_features + 1, dtype=float)
    X_design = np.column_stack((np.ones(n_obs), X))

    for _ in range(max_iter):
        linear_combination = np.dot(X_design, beta)
        prob = response_rate * (norm.cdf(linear_combination) - 0.5) + 0.5
        prob = np.clip(prob, 1e-15, 1 - 1e-15)

        pdf_vals = norm.pdf(linear_combination)
        denom = response_rate * pdf_vals
        W = (denom**2) / prob / (1 - prob)
        z = linear_combination + (choice - prob) / np.clip(denom, 1e-12, None)

        WX = W[:, np.newaxis] * X_design
        lhs = WX.T @ X_design
        rhs = WX.T @ z
        try:
            beta_new = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    if len(beta) > 1:
        denom = np.sqrt(2) * beta[1]
        if np.isclose(denom, 0.0):
            weber = np.inf
        else:
            weber = 1 / denom
    else:
        weber = np.nan
    return beta[0], beta[1:], weber
