# A Closer Look at Invalid Action Masking in Policy Gradient Algorithms

This repo contains the source code to reproduce the results in the paper *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. 

# Steps to reproduce the experiments

Our experiments use docker containers to run and Weight and Biases (https://www.wandb.com/) to record the experiments, so the first step is to register a wandb account and get an API key, which we refer to as `YOUR_WANDB_KEY`

```bash
# build the docker container
docker build -t invalid_action_masking:latest -f sharedmemory.Dockerfile .
# build docker run commands. replace `{YOUR_WANDB_KEY}` with your own
WANDB_KEY={YOUR_WANDB_KEY} python docker.py > docker.sh
# run experiments (96 in total)
# if you have limited computational resources, consider not running all of them at a time.
# in addition, notice the commands have --cpuset-cpus="0", --cpuset-cpus="1" for different runs
# to make sure each container is only using one core. By default I assume your machine has 40 cores,
# but feel free to modify the `cores` variable in `docker.py`
bash docker.sh
```

# Steps to reproduce the figures

Record your wandb username, which we will refer to as `YOUR_WANDB_ENTITY`

```bash
cd plots
WANDB_ENTITY={YOUR_WANDB_ENTITY} python episode_reward.py
WANDB_ENTITY={YOUR_WANDB_ENTITY} python approx_kl.py
```

These command should reproduce the PDFs in plots that are attached to the repo.

# Reproduction without WANDB

Although it would be possible, it would require a significant amount of effort to properly log 
metrics and redo the plotting, so at this time we would not have intructions to do reproduction
without WANDB. Note that it is possible to use wandb locally by following https://docs.wandb.com/self-hosted/local.

# If you have an issue reproducing the results

We have tested these scripts to reproduce but it is possible that there is a bug and maybe
we are assuming something specific regarding the environment. If you couldn't reproduce our results,
please file an issue and we will address it as soon as the double-blind review is over.
