
import re
import time
import os

# get env variable values
wandb_key = os.environ['WANDB_KEY']
assert len(wandb_key) > 0, "set the environment variable `WANDB_KEY` to your WANDB API key, something like `export WANDB_KEY=fdsfdsfdsfads` "

# extract runs from bash scripts
final_run_cmds = []
with open("mask.sh") as f:
    strings = f.read()
runs_match = re.findall('(python)(.+)((?:\n.+)+)(seed)',strings)
for run_match in runs_match:
    run_match_str = "".join(run_match).replace("\\\n", "")
    # print(run_match_str)
    for seed in range(1,5):
        final_run_cmds += [run_match_str.replace("$seed", str(seed)).split()]

# use docker directly
cores = 40
repo = "invalid_action_masking:latest"
current_core = 0
for final_run_cmd in final_run_cmds:
    print(f'docker run -d --shm-size="500m" --cpuset-cpus="{current_core}" -e WANDB={wandb_key} {repo} ' + " ".join(final_run_cmd))
    current_core = (current_core + 1) % cores
