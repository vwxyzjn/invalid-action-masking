# A Closer Look at Invalid Action Masking in Policy Gradient Algorithms

This repo contains the source code to reproduce the results in the paper [*A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*](https://arxiv.org/abs/2006.14171). 

## Get started

If you have pyenv or poetry:
```bash
poetry install

rm -rf ~/microrts && mkdir ~/microrts && \
    wget -O ~/microrts/microrts.zip http://microrts.s3.amazonaws.com/microrts/artifacts/202004222224.microrts.zip && \
    unzip ~/microrts/microrts.zip -d ~/microrts/ && \
    rm ~/microrts/microrts.zip
```

Else, you can also install dependencies via `pip install -r requirements.txt`.

## 10x10 Experiments
```
poetry run python invalid_action_masking/ppo_10x10.py
poetry run python invalid_action_masking/ppo_no_adj_10x10.py
poetry run python invalid_action_masking/ppo_no_mask_10x10.py
poetry run python ppo.py # newer & recommended PPO implementation that matches implementation details in `openai/baselines`
```
