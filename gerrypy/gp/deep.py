import torch
import tqdm
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from gerrypy.gp.preprocess import preprocess_input


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

        self.linear_layer = Linear(input_dims, 1)

    def forward(self, x):
        mean_x = self.mean_module(x) # self.linear_layer(x).squeeze(-1)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DeepGP(DeepGP):
    def __init__(self, train_x_shape, num_output_dims):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self.forward(x_batch))
                mus.append(preds.mean.mean(axis=0))
                variances.append(preds.variance.mean(axis=0))
                lls.append(self.likelihood.log_marginal(y_batch, self.forward(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


def run_experiment(df,
                   n_splits=10,
                   n_epochs=25,
                   n_inducing_points=500,
                   n_likelihood_samples=10,
                   batchsize=1024,
                   normalize_per_year=False,
                   normalize_labels=True,
                   use_boxcox=False,
                   dim=None,
                   lr=.01):
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
        model = DeepGP(train_x.shape, num_output_dims=1)


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
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
        ], lr=lr)
        mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

        epochs_iter = tqdm.tqdm(range(n_epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                with gpytorch.settings.num_likelihood_samples(n_likelihood_samples):
                    optimizer.zero_grad()
                    output = model(x_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()

                    minibatch_iter.set_postfix(loss=loss.item())

        ################ End Training #################

        mean, std, lls = model.predict(test_loader)

        if torch.cuda.is_available():
            mean = mean.cpu().numpy()
            std = std.cpu().numpy()

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


if __name__ == '__main__':
    import os
    df = pd.read_csv(os.path.join('training_data', 'counties.csv'))
    df['year'] = df['year'].astype(str)
    df['GEOID'] = df['GEOID'].astype(str).apply(lambda x: x.zfill(5))
    df = df.set_index(['year', 'GEOID'])
    df[df < 0] = 0
    df = df.rename(columns={'percent': 'label'})
    run_experiment(df, n_splits=4, n_epochs=2, batchsize=256,
                   normalize_per_year=False,
                   normalize_labels=True,
                   use_boxcox=False,
                   dim=None)