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

## Citation

```bibtex
@inproceedings{huang2020closer,
  author    = {Shengyi Huang and
               Santiago Onta{\~{n}}{\'{o}}n},
  editor    = {Roman Bart{\'{a}}k and
               Fazel Keshtkar and
               Michael Franklin},
  title     = {A Closer Look at Invalid Action Masking in Policy Gradient Algorithms},
  booktitle = {Proceedings of the Thirty-Fifth International Florida Artificial Intelligence
               Research Society Conference, {FLAIRS} 2022, Hutchinson Island, Jensen
               Beach, Florida, USA, May 15-18, 2022},
  year      = {2022},
  url       = {https://doi.org/10.32473/flairs.v35i.130584},
  doi       = {10.32473/flairs.v35i.130584},
  timestamp = {Thu, 09 Jun 2022 16:44:11 +0200},
  biburl    = {https://dblp.org/rec/conf/flairs/HuangO22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
