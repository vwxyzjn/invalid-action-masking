# A Closer Look at Invalid Action Masking in Policy Gradient Algorithms

This repo contains the source code to reproduce the results in the paper [*A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*](https://arxiv.org/abs/2006.14171). 

## Get started

If you have pyenv or poetry:
```bash
pyenv install -s $(sed "s/\/envs.*//" .python-version)
pyenv virtualenv $(sed "s/\/envs\// /" .python-version)
pyenv activate $(cat .python-version)
poetry install

rm ~/microrts -fR && mkdir ~/microrts && \
    wget -O ~/microrts/microrts.zip http://microrts.s3.amazonaws.com/microrts/artifacts/202004222224.microrts.zip && \
    unzip ~/microrts/microrts.zip -d ~/microrts/ && \
    rm ~/microrts/microrts.zip
```

Else, you can also install dependencies via `pip install -r requirements.txt`.

## 10x10 Experiments
```
python invalid_action_masking/ppo_10x10.py
python invalid_action_masking/ppo_no_adj_10x10.py
python invalid_action_masking/ppo_no_mask_10x10.py
```

# If you have an issue reproducing the results

We have tested these scripts to reproduce but it is possible that there is a bug and maybe
we are assuming something specific regarding the environment. If you couldn't reproduce our results,
please file an issue and we will address it as soon as the double-blind review is over.

