from typing import Any, Callable, List, Optional

import botorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import Posterior
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from torch import Tensor
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from .model import Model

def build_covar_module(model_args, input_dim, batch_shape=torch.Size()):
    kernel_name = model_args.get("kernel", "rbf").lower()

    if kernel_name == "rbf" or kernel_name == "se":
        base_kernel = RBFKernel(
            ard_num_dims=input_dim,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
    elif kernel_name == "matern":
        nu = model_args.get("nu", 2.5)
        base_kernel = MaternKernel(
            nu=nu,
            ard_num_dims=input_dim,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )
    else:
        raise ValueError(f"Unknown GP kernel: {kernel_name}")

    return ScaleKernel(
        base_kernel,
        batch_shape=batch_shape,
        outputscale_prior=GammaPrior(2.0, 0.15),
    )


class SingleTaskGP(Model):

    def __init__(self, model_args, input_dim, output_dim):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.nu = model_args["nu"] if "nu" in model_args else 2.5
        self.model_args = model_args
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs

    def fit_and_save(self, train_x, train_y, save_dir):
        if self.output_dim > 1:
            raise RuntimeError(
                "SingleTaskGP does not fit tasks with multiple objectives")

        covar_module = build_covar_module(self.model_args, train_x.shape[-1])
        self.gp = botorch.models.SingleTaskGP(
            train_x,
            train_y,
            covar_module=covar_module,
            outcome_transform=Standardize(m=1),
        ).to(train_x)
        mll = ExactMarginalLogLikelihood(
            self.gp.likelihood, self.gp).to(train_x)
        fit_gpytorch_mll(mll)


class MultiTaskGP(Model):

    def __init__(self, model_args, input_dim, output_dim):
        super().__init__()
        self.gp = None
        self.output_dim = output_dim
        self.nu = model_args["nu"] if "nu" in model_args else 2.5
        self.model_args = model_args
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform: Optional[Callable[[Posterior], Posterior]] = None,
        **kwargs: Any,
    ) -> Posterior:
        return self.gp.posterior(X, output_indices, observation_noise, posterior_transform, **kwargs)

    @property
    def batch_shape(self) -> torch.Size:
        return self.gp.batch_shape

    @property
    def num_outputs(self) -> int:
        return self.gp.num_outputs

    def fit_and_save(self, train_x, train_y, save_dir):
        models = []
        for d in range(self.output_dim):
            covar_module = build_covar_module(self.model_args, train_x.shape[-1])
            models.append(
                botorch.models.SingleTaskGP(
                    train_x,
                    train_y[:, d].unsqueeze(-1),
                    covar_module=covar_module,
                    outcome_transform=Standardize(m=1),
                ).to(train_x)
            )
        self.gp = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(self.gp.likelihood, self.gp).to(train_x)
        fit_gpytorch_mll(mll)