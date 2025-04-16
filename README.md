# robustness_informed


## setup

Install `miniforge`

## Experiment

The `.env` file used:

```
RESULTS_FOLDER=/home/cloucera/mnt/output/isrobust/results
PREFECT_LOCAL_STORAGE_PATH=/home/cloucera/mnt/output/isrobust/prefect
DATA_PATH = os.getenv("DATA_PATH", "data")  # add data path

N_GPU=3
N_CPU=30
FRAC_START=0.05
FRAC_STOP=0.9
FRAC_STEP=0.05
SEED_START=0
SEED_STOP=99
SEED_STEP=1
DEBUG=0

```

It was run using:
```
screen -d -m make
```
