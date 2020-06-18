from os import path
import pickle
import wandb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
api = wandb.Api()
wandb_entity = os.environ['WANDB_ENTITY']

feature_of_interest = 'losses/approx_kl'
feature_name = feature_of_interest.replace("/", "_")
if not os.path.exists(feature_name):
    os.makedirs(feature_name)

if not path.exists(f"{feature_name}/all_df_cache.pkl"):
    # Change oreilly-class/cifar to <entity/project-name>
    runs = api.runs(f"{wandb_entity}/invalid_action_masking")
    summary_list = [] 
    config_list = [] 
    name_list = []
    envs = {}
    data = []
    rolling_average = 10
    sample_points = 500
    
    for idx, run in enumerate(runs):
        if feature_of_interest in run.summary:
            ls = run.history(keys=[feature_of_interest, 'global_step'], pandas=False)
            metrics_dataframe = pd.DataFrame(ls[0])
            if "invalid_action_penalty" in run.config:
                metrics_dataframe.insert(len(metrics_dataframe.columns), "algo", run.config['exp_name']+"-"+str(run.config['invalid_action_penalty']))
            else:
                metrics_dataframe.insert(len(metrics_dataframe.columns), "algo", run.config['exp_name'])
            metrics_dataframe.insert(len(metrics_dataframe.columns), "seed", run.config['seed'])
            # metrics_dataframe[feature_of_interest] = metrics_dataframe[feature_of_interest].rolling(rolling_average).mean()[rolling_average:]
            data += [metrics_dataframe]
            if run.config["gym_id"] not in envs:
                envs[run.config["gym_id"]] = [metrics_dataframe]
                envs[run.config["gym_id"]+"total_timesteps"] = run.config["total_timesteps"]
            else:
                envs[run.config["gym_id"]] += [metrics_dataframe]

            # run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files 
            summary_list.append(run.summary._json_dict) 
        
            # run.config is the input metrics.  We remove special values that start with _.
            config_list.append({k:v for k,v in run.config.items() if not k.startswith('_')}) 
        
            # run.name is the name of the run.
            name_list.append(run.name)       
    
    
    summary_df = pd.DataFrame.from_records(summary_list) 
    config_df = pd.DataFrame.from_records(config_list) 
    name_df = pd.DataFrame({'name': name_list}) 
    all_df = pd.concat([name_df, config_df,summary_df], axis=1)
    data = pd.concat(data, ignore_index=True)
    
    
    with open(f'{feature_name}/all_df_cache.pkl', 'wb') as handle:
        pickle.dump(all_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{feature_name}/envs_cache.pkl', 'wb') as handle:
        pickle.dump(envs, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(f'{feature_name}/all_df_cache.pkl', 'rb') as handle:
        all_df = pickle.load(handle)
    with open(f'{feature_name}/envs_cache.pkl', 'rb') as handle:
        envs = pickle.load(handle)
print("data loaded")

#smoothing
rolling_average = 20
for env in envs:
    if not env.endswith("total_timesteps"):
        for idx, metrics_dataframe in enumerate(envs[env]):
            envs[env][idx] = metrics_dataframe.dropna(subset=[feature_of_interest])
            envs[env][idx][feature_of_interest] = metrics_dataframe[feature_of_interest].rolling(rolling_average).mean()[rolling_average:]
        

sns.set(style="darkgrid")
def get_df_for_env(gym_id):
    env_total_timesteps = envs[gym_id+"total_timesteps"]
    env_increment = env_total_timesteps / 500
    envs_same_x_axis = []
    for sampled_run in envs[gym_id]:
        df = pd.DataFrame(columns=sampled_run.columns)
        x_axis = [i*env_increment for i in range(500-2)]
        current_row = 0
        for timestep in x_axis:
            while sampled_run.iloc[current_row]["global_step"] < timestep:
                current_row += 1
                if current_row > len(sampled_run)-2:
                    break
            if current_row > len(sampled_run)-2:
                break
            temp_row = sampled_run.iloc[current_row].copy()
            temp_row["global_step"] = timestep
            df = df.append(temp_row)
        
        envs_same_x_axis += [df]
    return pd.concat(envs_same_x_axis, ignore_index=True)

# uncommenet the following to generate all figures
# for env in set(all_df["gym_id"]):
#     data = get_df_for_env(env)
#     sns.lineplot(data=data, x="global_step", y=feature_of_interest, hue="algo", ci='sd')
#     plt.legend(fontsize=6)
#     plt.title(env)
#     plt.savefig(f"{env}.svg")
#     plt.clf()

# color palette settings
current_palette = sns.color_palette(n_colors=7)
exp_names = ['ppo_no_mask-0',
  'ppo_no_adj',
  'ppo',
  'ppo-maskrm',
  'ppo_no_mask--0.1',
  'ppo_no_mask--0.01',
  'ppo_no_mask--1']
real_exp_names = ['Invalid action penalty, $r_{\\text{invalid}}=0$',
  'Naive invalid action masking',
  'Invalid action masking',
  'Masking removed',
  'Invalid action penalty, $r_{\\text{invalid}}=-0.1$',
  'Invalid action penalty, $r_{\\text{invalid}}=-0.01$',
  'Invalid action penalty, $r_{\\text{invalid}}=-1$']
current_palette_dict = dict(zip(exp_names, current_palette))
exp_convert_dict = dict(zip(exp_names, real_exp_names))

interested_exp_names = ['ppo_no_mask-0',
  'ppo_no_adj',
  'ppo',
  'ppo-maskrm',
  'ppo_no_mask--0.1',
  'ppo_no_mask--0.01',
  'ppo_no_mask--1']

hue_order = [
  'ppo',
  'ppo-maskrm',
  'ppo_no_adj',
  'ppo_no_mask-0',
  'ppo_no_mask--0.1',
  'ppo_no_mask--0.01',
  'ppo_no_mask--1']

# export legend
env = "MicrortsMining10x10F9-v0"
data = get_df_for_env(env)
data[feature_of_interest] = data[feature_of_interest].astype(np.float32)

ax = sns.lineplot(data=data.loc[data['algo'].isin(interested_exp_names)], x="global_step", y=feature_of_interest, hue="algo", ci='sd', palette=current_palette_dict, hue_order=hue_order)
ax.legend()
plt.title(env)
plt.savefig(f"{env}.svg")

def export_legend(ax, filename="legend.pdf"):
    import matplotlib as mpl
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    handles, labels = ax.get_legend_handles_labels()

    legend = ax2.legend(handles=handles[1:], labels=labels[1:], frameon=False, loc='lower center', ncol=2, fontsize=20, handlelength=1)
    for text in legend.get_texts():
        text.set_text(exp_convert_dict[text.get_text()])
    for line in legend.get_lines():
        line.set_linewidth(4.0)
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    mpl.rcParams['text.usetex'] = False
export_legend(ax, "legend1.pdf")

sns.set_style("white")
# save plots
for env in set(all_df["gym_id"]):
    data = get_df_for_env(env)
    data[feature_of_interest] = data[feature_of_interest].astype(np.float32)
    ax = sns.lineplot(data=data.loc[data['algo'].isin(interested_exp_names)], x="global_step", y=feature_of_interest, hue="algo", ci='sd', palette=current_palette_dict, hue_order=hue_order)
    ax.legend().remove()
    plt.tight_layout()
    ax.set(xlabel='Time Steps', ylabel='Average KL Divergence')
    ax.set_ylim(0, 0.07)
    plt.savefig(f"{feature_name}-{env}.pdf")
    plt.clf()
