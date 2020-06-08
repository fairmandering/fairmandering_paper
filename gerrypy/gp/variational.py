import numpy as np
import pandas as pd
import torch
import random
import tqdm
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gerrypy.gp.preprocess import preprocess_input
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader

class ApproxGPModel(ApproximateGP):
    def __init__(self, inducing_points, train_x, train_y):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super(ApproxGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,
        #                                                            ard_num_dims=train_x.shape[1])
        # self.covar_module.initialize_from_data(train_x.cpu(), train_y.cpu())
        self.covar_module = gpytorch.kernels.RBFKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def evaluate(model, likelihood, test_loader):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    means = torch.tensor([0.])
    stds = torch.tensor([0.])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
            stds = torch.cat([stds, preds.stddev.cpu()])
    means = means[1:]
    stds = stds[1:]

    return means.numpy(), stds.numpy()


def run_experiment(df,
                   n_splits=10,
                   n_epochs=25,
                   n_inducing_points=500,
                   batchsize=1024,
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

        ixs = [i for i in range(len(train_x))]
        random.shuffle(ixs)
        inducing_points = train_x[ixs[:n_inducing_points], :]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ApproxGPModel(inducing_points,
                              train_x,
                              train_y)

        ############## Training ###################
        if torch.cuda.is_available():
            train_x = train_x.cuda().float()
            train_y = train_y.cuda().float()
            test_x = test_x.cuda().float()
            test_y = test_y.cuda().float()
            model = model.cuda().float()
            likelihood = likelihood.cuda().float()
        else:
            train_x = train_x.float()
            train_y = train_y.float()
            model = model.float()
            likelihood = likelihood.float()

        train_dataset = TensorDataset(train_x, train_y)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # We use SGD here, rather than Adam. Emperically, we find that SGD is better for variational regression
        optimizer = torch.optim.SGD([
            {'params': model.parameters()},
            {'params': likelihood.parameters()},
        ], lr=lr)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

        epochs_iter = tqdm.tqdm(range(n_epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                minibatch_iter.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()

        ################ End Training #################

        mean, std = evaluate(model, likelihood, test_loader)

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

        test_y = test_y.cpu().numpy()
        l1_error = pd.Series(np.abs(test_y - pred)).describe()
        std_spread = pd.Series(pred_ub - pred_lb).describe()
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
