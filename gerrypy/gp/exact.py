import gpytorch
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from gerrypy.gp.preprocess import preprocess_input


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel  # gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_mixtures=4):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=n_mixtures,
                                                                   ard_num_dims=train_x.shape[1])
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(model,
          likelihood,
          train_x,
          train_y,
          lr=.1,
          training_iterations=50):
    if torch.cuda.is_available():
        train_x = train_x.cuda().float()
        train_y = train_y.cuda().float()
        model = model.cuda().float()
        likelihood = likelihood.cuda().float()
    else:
        train_x = train_x.float()
        train_y = train_y.float()
        model = model.float()
        likelihood = likelihood.float()

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    return model, likelihood


def evaluate(model, likelihood, test_X):
    if torch.cuda.is_available():
        test_X = test_X.cuda().float()
    else:
        test_X = test_X.cuda().float()

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_X))
        mean = observed_pred.mean
        std = observed_pred.stddev

    mean = mean.cpu().numpy()
    std = std.cpu().numpy()

    return mean, std


def run_experiment(df, kernel,
                   n_splits=10,
                   n_training_iterations=25,
                   normalize_per_year=False,
                   normalize_labels=True,
                   use_boxcox=False,
                   dim=None,
                   lr=.1):
    kf = KFold(n_splits=n_splits, shuffle=True)
    results = {}
    for k, (train_index, test_index) in enumerate(kf.split(df)):
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

        prepro = preprocess_input(df_train, df_test,
                                  normalize_per_year=normalize_per_year,
                                  normalize_labels=normalize_labels,
                                  use_boxcox=use_boxcox,
                                  dim=dim)
        train_x, test_x, train_y, test_y, label_scaler = prepro

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x,
                             train_y,
                             likelihood,
                             kernel)
        model, likelihood = train(model,
                                  likelihood,
                                  train_x,
                                  train_y,
                                  lr=lr,
                                  training_iterations=n_training_iterations)
        mean, std = evaluate(model, likelihood, test_x)

        ub = mean + std
        lb = mean - std
        if normalize_labels:
            pred = label_scaler.inverse_transform(mean.reshape(-1, 1)).flatten()
            pred_ub = label_scaler.inverse_transform(ub.reshape(-1, 1)).flatten()
            pred_lb = label_scaler.inverse_transform(lb.reshape(-1, 1)).flatten()
        else:
            pred = mean
            pred_ub = ub
            pred_lb = lb

        test_y = test_y.numpy()
        l1_error = pd.Series(np.abs(test_y - pred)).describe()
        std_spread = pd.Series((pred_ub - pred_lb)/2).describe()
        pred_above_lb = (pred_lb < test_y)
        pred_below_ub = (test_y < pred_ub)

        results[k] = {
            'l1_mean_error': l1_error['mean'],
            'l1_median_error': l1_error['50%'],
            'l1_error_std': l1_error['std'],
            'median_std': std_spread['50%'],
            'std_std': std_spread['std'],
            'percent_below_std_ub': sum(pred_below_ub) / len(pred_below_ub),
            'percent_above_std_lb': sum(pred_above_lb) / len(pred_above_lb),
        }
    return results
