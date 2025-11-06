import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import io
from scipy.optimize import curve_fit
from scipy.stats import norm
import torch
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forwardrbm(self, v):
    p_h = torch.sigmoid(torch.matmul(v.float(), self.W) + self.hid_bias)
    h = (p_h > torch.rand_like(p_h)).float()
    return p_h, h


def forwardDBN(self, X):
    for rbm in self.layers:
        _X = torch.zeros([X.shape[0], X.shape[1], rbm.num_hidden], device=DEVICE)
        for n in range(X.shape[0]):
            Xtorch = torch.Tensor(X[n, :, :]).to(DEVICE)
            _X[n, :, :] = forwardrbm(rbm, Xtorch.clone())[0].clone()
        X = _X.clone()
        del _X
    return X


def classifier(Xtrain, Xtest, Ytrain, Ytest):
    Xtrain = Xtrain.view(-1, Xtrain.shape[2]).detach().cpu().numpy()
    Xtest = Xtest.view(-1, Xtest.shape[2]).detach().cpu().numpy()

    Ytrain = Ytrain.view(-1, Ytrain.shape[-1]).detach().cpu().numpy()
    Ytest = Ytest.view(-1, Ytest.shape[-1]).detach().cpu().numpy()

    Xtrain = np.hstack([Xtrain, np.ones((Xtrain.shape[0], 1))])
    Xtest = np.hstack([Xtest, np.ones((Xtest.shape[0], 1))])

    weights = np.linalg.pinv(Xtrain) @ Ytrain
    pred_train = Xtrain @ weights
    pred_test = Xtest @ weights

    predicted_train = np.argmax(pred_train, axis=1)
    predicted_test = np.argmax(pred_test, axis=1)
    Ytrain_labels = np.argmax(Ytrain, axis=1)
    Ytest_labels = np.argmax(Ytest, axis=1)

    accuracy_train = accuracy_score(Ytrain_labels, predicted_train)
    accuracy_test = accuracy_score(Ytest_labels, predicted_test)

    return accuracy_train, pred_train, accuracy_test, pred_test, weights


def irls_fit(choice, X, guessRate=0.01, max_iter=5000, tol=1e-12):
    response_rate = 1 - guessRate
    n_obs, n_features = X.shape
    beta = np.zeros(n_features + 1)
    X_design = np.column_stack((np.ones(n_obs), X))

    for _ in range(max_iter):
        linear_combination = np.dot(X_design, beta)
        prob = response_rate * (norm.cdf(linear_combination) - 0.5) + 0.5
        prob = np.clip(prob, 1e-15, 1 - 1e-15)

        W = (response_rate * norm.pdf(linear_combination)) ** 2 / prob / (1 - prob)
        z = linear_combination + (choice - prob) / (response_rate * norm.pdf(linear_combination))

        WX = W[:, np.newaxis] * X_design
        lhs = WX.T @ X_design
        rhs = WX.T @ z

        reg = 1e-6 * np.eye(lhs.shape[0], dtype=lhs.dtype)
        lhs = lhs + reg
        try:
            beta_new = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    else:
        raise ValueError("IRLS failed to converge within the maximum number of iterations")

    weber = 1 / (np.sqrt(2) * beta[1])
    return beta[0], beta[1:], weber


def num_size_spacing_model(choice, numLeft, numRight, isaLeft, isaRight, faLeft, faRight, guessRate=0.01):
    tsaLeft = isaLeft * numLeft
    sizeLeft = isaLeft * tsaLeft
    sparLeft = faLeft / numLeft
    spaceLeft = sparLeft * faLeft

    tsaRight = isaRight * numRight
    sizeRight = isaRight * tsaRight
    sparRight = faRight / numRight
    spaceRight = sparRight * faRight

    numRatio = numRight / numLeft
    sizeRatio = sizeRight / sizeLeft
    spaceRatio = spaceRight / spaceLeft

    X = np.column_stack((np.log2(numRatio), np.log2(sizeRatio), np.log2(spaceRatio)))
    choice = np.array(choice)

    intercept, betas, weber = irls_fit(choice, X, guessRate)

    return intercept, betas, weber, X


def beta_extraction(choice, idxs, N_list, TSA_list, FA_list, guessRate=0.01):
    N_list = np.squeeze(N_list)
    TSA_list = np.squeeze(TSA_list)
    FA_list = np.squeeze(FA_list)

    idxs_flat = np.asarray(idxs.cpu().detach()).reshape(-1, 2)

    max_idx = np.max(idxs_flat)
    if max_idx >= len(N_list):
        raise ValueError(f"Index {max_idx} out of bounds for N_list of length {len(N_list)}")

    numLeft = []
    numRight = []
    isaLeft = []
    isaRight = []
    faLeft = []
    faRight = []
    filtered_choices = []

    for i, (idx_left, idx_right) in enumerate(idxs_flat):
        idx_left = int(idx_left)
        idx_right = int(idx_right)

        if idx_left >= len(N_list) or idx_right >= len(N_list):
            continue

        if N_list[idx_left] > 14 or N_list[idx_right] > 14:
            continue

        numLeft.append(N_list[idx_left])
        numRight.append(N_list[idx_right])
        isaLeft.append(TSA_list[idx_left] / N_list[idx_left])
        isaRight.append(TSA_list[idx_right] / N_list[idx_right])
        faLeft.append(FA_list[idx_left])
        faRight.append(FA_list[idx_right])
        filtered_choices.append(choice[i])

    numLeft = np.array(numLeft)
    numRight = np.array(numRight)
    isaLeft = np.array(isaLeft)
    isaRight = np.array(isaRight)
    faLeft = np.array(faLeft)
    faRight = np.array(faRight)
    filtered_choices = np.array(filtered_choices)

    if len(numLeft) == 0:
        raise ValueError("No valid pairs remained after filtering.")

    intercept, betas, weber = num_size_spacing_model(filtered_choices, numLeft, numRight, isaLeft, isaRight, faLeft, faRight, guessRate)
    return intercept, betas, weber, filtered_choices


def compute_weber(choice, idxs_test, N_list_test):
    if isinstance(choice, torch.Tensor):
        choice = choice.detach().cpu().numpy()
    if isinstance(idxs_test, torch.Tensor):
        idxs_test = idxs_test.detach().cpu().numpy()
    if isinstance(N_list_test, torch.Tensor):
        N_list_test = N_list_test.detach().cpu().numpy()

    if choice.ndim == 2 and choice.shape[1] == 2:
        choice = np.argmax(choice, axis=1)
    else:
        choice = choice.flatten()

    N1 = N_list_test[idxs_test[:, 0]].flatten()
    N2 = N_list_test[idxs_test[:, 1]].flatten()

    true_choice = (N2 > N1).astype(int)

    ratios = np.maximum(N1, N2) / np.minimum(N1, N2)

    x_vals = []
    y_vals = []

    for r in np.unique(ratios):
        mask = np.isclose(ratios, r)
        if np.sum(mask) == 0:
            continue
        acc = np.mean(choice[mask] == true_choice[mask])
        x_vals.append(r)
        y_vals.append(acc)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    def psychometric(x, w):
        return norm.cdf((x - 1) / (w * x))

    try:
        popt, _ = curve_fit(psychometric, x_vals, y_vals, p0=[0.3])
        weber = popt[0]
    except Exception:
        weber = np.nan

    overall_acc = np.mean(choice == true_choice)
    return weber, overall_acc
