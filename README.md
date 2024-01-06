# robustness_informed


## setup

Install `miniforge`

## Experiment

The `.env` file used:

```
IVAE_ENV_FOLDER=./.venvs/ivae
BINN_ENV_FOLDER=./.venvs/binn
N_GPU=3
FRAC_START=0.1
FRAC_STOP=0.9
FRAC_STEP=0.1
SEED_START=0
SEED_STOP=99
SEED_STEP=1
DEBUG=0
```

It was run using:
```
nohup make &
```
