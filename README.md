# Bayesian Neural Network Surrogates for Bayesian Optimization of CCS Operations

## Installation
Create a new conda environment:
````
conda env create -f environment.yml
````

Install the project:
````
pip install -e .
````

## Running experiments
Each experiment requires a config json, and example config files are provided in `config`.

To use the config file `config/<name>.json`, run the following command from the root folder:
````
python main.py --config <name>
````

You can also include the `--bg` flag if you would like to redirect stderr and stdout to a file and save the outputs:
````
python main.py --config <name> --bg
````

## Case studies
The CCS case studies used in the manuscript are organized as follows:

- `oil_v1`: Case Study 1, Variation 1
- `oil_v2`: Case Study 1, Variation 2
- `oil`: Case Study 2

## Code Organization
The Bayesian optimization loop is in `main.py`.

`models`: the model code for each of the surrogate models considered.

`test_functions`: CCS case studies and benchmark objective functions.

`config`: configuration files for the reported experiments.

`experiment_results`: archived outputs for selected experiments reported in the manuscript.

## Notes on reproducibility
This repository contains the Bayesian optimization framework, surrogate-model implementations, selected experiment configurations, and archived outputs corresponding to the manuscript.

Reservoir simulation deck files (`.DATA` and related simulator files) are not distributed due to confidentiality restrictions.
