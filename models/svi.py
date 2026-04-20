from typing import Any, Callable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.posteriors import Posterior
from torch import Tensor, tensor

from .model import Model
from .utils import RegNet, BNN, bnn_param_site_names, flatten_bnn_sample
from pyro.infer import Predictive

from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal, AutoIAFNormal
from tqdm.auto import trange
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn

class SVIPosterior(Posterior):
    def __init__(self, X, model, output_dim, guide, inter_model, mean, std, n_posterior_samples, param_site_names):
        super().__init__()
        self.model = model
        self.output_dim = output_dim
        self.X = X
        self.preds = None
        self.guide = guide
        self.inter_model = inter_model
        self.y_mean = mean
        self.y_std = std
        self.n_posterior_samples = n_posterior_samples
        self.param_site_names = param_site_names

    def predict_model(self):
        preds = []
        for _ in range(self.n_posterior_samples):
            sample_dict = self.guide()
            param_vec = flatten_bnn_sample(sample_dict, self.param_site_names)
            torch.nn.utils.vector_to_parameters(param_vec, self.inter_model.parameters())
            output = self.inter_model(self.X.to(param_vec))[..., :self.output_dim]
            output = (output * self.y_std) + self.y_mean
            preds.append(output)
        self.preds = torch.stack(preds)

    def rsample(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        if sample_shape is None:
            sample_shape = torch.Size([1])
        if self.preds is None:
            self.predict_model()
        idx = np.random.randint(len(self.preds), size=tuple(sample_shape))
        return self.preds[idx]

    @property
    def mean(self) -> Tensor:
        if self.preds is None:
            self.predict_model()
        return self.preds.mean(axis=0)

    @property
    def variance(self) -> Tensor:
        if self.preds is None:
            self.predict_model()
        return self.preds.var(axis=0)

    @property
    def device(self) -> torch.device:
        if self.preds is None:
            self.predict_model()
        return self.preds.device

    @property
    def dtype(self) -> torch.dtype:
        if self.preds is None:
            self.predict_model()
        return self.preds.dtype


class MySVI(Model):
    def __init__(self, args, input_dim, output_dim, device):
        super().__init__()

        # problem dimensions
        self.input_dim = input_dim
        self.problem_output_dim = output_dim
        self.n_posterior_samples = args.get("n_posterior_samples", 500)
        self.param_site_names = bnn_param_site_names(self.model)
        # architecture
        self.regnet_dims = args["regnet_dims"]
        self.regnet_activation = args["regnet_activation"]
        self.noise_var = tensor(args["noise_var"])
        self.prior_var = args["prior_var"]
        self.adapt_noise = args["adapt_noise"]
        self.num_epochs = args["num_epochs"]
        # optional dimensions for noise estimate
        self.standardize_y = args["standardize_y"]
        self.mean = 0.0
        self.std = 1.0
        if self.adapt_noise:
            self.network_output_dim = 2 * output_dim
        else:
            self.network_output_dim = output_dim



        self.model = BNN(dimensions=self.regnet_dims,
                        activation=self.regnet_activation,
                        input_dim=self.input_dim,
                        output_dim=self.network_output_dim,
                        prior_var=self.prior_var,
                        dtype=torch.float64,
                        device=device)
        self.inter_model = RegNet(dimensions=self.regnet_dims,
                activation=self.regnet_activation,
                input_dim=self.input_dim,
                output_dim=self.network_output_dim,
                dtype=torch.float64,
                device=device)

    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return SVIPosterior(
    X, self.model, self.problem_output_dim, self.guide, self.inter_model,
    self.mean, self.std, self.n_posterior_samples, self.param_site_names
)

    @property
    def num_outputs(self) -> int:
        return self.problem_output_dim
    def fit_and_save(self, train_x, original_train_y,save_dir):
        if self.standardize_y:
            self.mean = original_train_y.mean(dim=0)
            if len(original_train_y) > 1:
                self.std = original_train_y.std(dim=0)
            else:
                self.std = 1.0
            train_y = (original_train_y - self.mean) / self.std
        else:
            train_y = original_train_y

        #train_x = train_x.squeeze()
        #train_y = train_y.squeeze()
        self.guide = AutoIAFNormal(self.model)
        optimizer = pyro.optim.Adam({"lr": 0.01})

        svi = SVI(self.model, self.guide, optimizer, loss=Trace_ELBO())
        pyro.clear_param_store()

        progress_bar = trange(self.num_epochs)

        for epoch in progress_bar:
            loss = svi.step(train_x, train_y)
            progress_bar.set_postfix(loss=f"{loss / train_x.shape[0]:.3f}")




