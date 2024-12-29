# robustness_informed


## setup

Install `miniforge`

## Experiment

The `.env` file used:

```
IVAE_ENV_FOLDER=./.venvs/ivae
BINN_ENV_FOLDER=./.venvs/binn
N_GPU=3
SEED_START=0
SEED_STOP=50
SEED_STEP=1
DEBUG=0
FRAC_START=0.05
FRAC_STOP=0.85
FRAC_STEP=0.2
RESULTS_FOLDER="path"
```

It was run using:
```
screen -d -m make
```
