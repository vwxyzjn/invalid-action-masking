# A Closer Look at Invalid Action Masking in Policy Gradient Algorithms

This repo contains the source code to reproduce the results in the paper *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. 

## Get started
```
# if you have pyenv or poetry:
pyenv install -s $(sed "s/\/envs.*//" .python-version)
pyenv virtualenv $(sed "s/\/envs\// /" .python-version)
poetry install
# else:
# pip install -r requirements.txt

rm ~/microrts -fR && mkdir ~/microrts && \
    wget -O ~/microrts/microrts.zip http://microrts.s3.amazonaws.com/microrts/artifacts/202004222224.microrts.zip && \
    unzip ~/microrts/microrts.zip -d ~/microrts/ && \
    rm ~/microrts/microrts.zip
```

## 10x10 Experiments
```
python invalid_action_masking/ppo_10x10.py
python invalid_action_masking/ppo_no_adj_10x10.py
python invalid_action_masking/ppo_no_mask_10x10.py
```


# Reproduction without WANDB

Although it would be possible, it would require a significant amount of effort to properly log 
metrics and redo the plotting, so at this time we would not have intructions to do reproduction
without WANDB. Note that it is possible to use wandb locally by following https://docs.wandb.com/self-hosted/local.

# If you have an issue reproducing the results

We have tested these scripts to reproduce but it is possible that there is a bug and maybe
we are assuming something specific regarding the environment. If you couldn't reproduce our results,
please file an issue and we will address it as soon as the double-blind review is over.

