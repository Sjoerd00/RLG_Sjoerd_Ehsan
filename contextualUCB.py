import numpy as np
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

class batchUCB:
    def __init__(self, num_arms, reward_values, arms, contexts, batch_size=None, c=1, sensitive_features=None, min_arm_pulls=1, contextual=True):
        """
        Batch UCB algorithm for multi-armed bandit problem.

        Parameters:
        num_arms (int): The number of arms (actions) available to the algorithm.
        reward_values (list): A list of the reward values for each arm.
        batch_size (int): The batch size for each iteration. If None, use the entire dataset.
        """
        self.num_arms = num_arms
        self.reward_values = reward_values
        self.thetas = np.zeros((num_arms, contexts.max()+1))
        self.arm_counts = np.zeros((num_arms, contexts.max()+1))
        self.remaining_reward_values = reward_values
        self.arms = arms
        self.contexts = contexts
        self.batch_size = batch_size
        self.c = c
        self.sf = sensitive_features
        self.min_arm_pulls = min_arm_pulls
        self.contextual = contextual

    def select_arm(self, context):
        ucb_values = np.zeros(self.num_arms)
        # zero_ix = (self.arm_counts[:, context] == 0).ravel()
        zero_ix = (self.arm_counts[:, context] < self.min_arm_pulls).ravel()

        ucb_values[zero_ix] = np.inf
        ucb_values[~zero_ix] = self.thetas[~zero_ix, context] / self.arm_counts[~zero_ix, context] \
            + self.c * np.sqrt(2 * np.log(np.sum(self.arm_counts[:, context])) / self.arm_counts[~zero_ix, context])
        selected_arm = np.argmax(ucb_values)
        return selected_arm

    def update(self, arm, context, reward, ix):
        if self.contextual:
            for a, c, r in zip (arm, context, reward):
                self.arm_counts[a, c] += 1
                self.thetas[a, c] += r
        else:
            for a, r in zip (arm, context, reward):
                self.arm_counts[a, :] += 1
                self.thetas[a, :] += r

    def run(self, num_iterations=100):
        rewards = []
        arm_positions = []
        context_tr = []
        batch_size = self.batch_size if self.batch_size is not None else num_iterations

        for batch in range(0, num_iterations, batch_size):
            batch_rewards = []
            batch_arm_positions = []
            batch_context = []
            end_ix = min(batch + batch_size, num_iterations)

            for i in range(batch, end_ix):
                contex = np.random.choice(np.unique(self.contexts), size=1)[0]
                arm = self.select_arm(contex)
                indices = np.where((self.arms == arm) & (self.contexts == contex))[0]
                arm_ix = np.random.choice(indices)

                reward = self.remaining_reward_values[arm_ix]

                batch_rewards.append(reward)
                batch_arm_positions.append(arm)
                batch_context.append(contex)

            self.update(batch_arm_positions, batch_context, batch_rewards, arm_ix)
            rewards.extend(batch_rewards)
            arm_positions.extend(batch_arm_positions)
            context_tr.extend(batch_context)

        return rewards, arm_positions, context_tr



def run_experiment(c_values, batch_sizes, num_iterations, num_arms, reward_values, arms, contexts, opt_arms, contextual=True):
    metrics = {"Avg Rewards": np.zeros((len(c_values), len(batch_sizes))),
               "Optimal Percent": np.zeros((len(c_values), len(batch_sizes))),
               "Total Regret": np.zeros((len(c_values), len(batch_sizes))),
               "Cumulative Regrets": np.zeros((len(c_values), len(batch_sizes), num_iterations))}

    dftest = pd.DataFrame(reward_values)
    dftest['context'] = contexts
    dftest['arm'] = arms
    optimal_actions = []
    optimal_cumulative_rewards = []

    for i, c in tqdm(enumerate(c_values)):
        for j, batch_size in enumerate(batch_sizes):
            bandit = batchUCB(num_arms, reward_values, arms, contexts, batch_size=batch_size, c=c, contextual=contextual)
            rewards, arm_positions, context_tr = bandit.run(num_iterations)
            for contex in range(max(contexts)):
                optimal_actions.extend( (np.array(arm_positions)[np.array(context_tr)==contex] == opt_arms.loc[contex, 'arm']).tolist())

            cumulative_rewards = np.cumsum(rewards)
            optimal_cumulative_rewards = np.cumsum([opt_arms.loc[k, 'r'] for k in context_tr])

            metrics["Total Regret"][i, j] = optimal_cumulative_rewards[-1] - cumulative_rewards[-1]
            metrics["Avg Rewards"][i, j] = cumulative_rewards[-1] / num_iterations
            metrics["Optimal Percent"][i, j] = 100 * np.mean(optimal_actions)
            metrics["Cumulative Regrets"][i, j, :cumulative_rewards.shape[0]] = optimal_cumulative_rewards[:cumulative_rewards.shape[0]] - cumulative_rewards

    return metrics

#%%
cols = ['item_id', 'click', 'class_3','class_4','class_10', 'user_feature_0', 'user_feature_1', 'user_feature_2', 'user_feature_3']
data = pd.read_csv("dataProcessed1.csv", usecols=cols)
data.item_id = data.item_id - 1

for cluster in [3, 4, 10]:

    ros = RandomOverSampler(random_state=42)
    X_rs, reward_rs = ros.fit_resample(data.drop(columns=['click']).values, data.click.values)
    X_rs = pd.DataFrame(X_rs, columns=['item_id', 'user_feature_0', 'user_feature_1', 'user_feature_2',
                                       'user_feature_3', 'class_3', 'class_4', 'class_10'])
    X_rs['r'] = reward_rs

    X_rs.rename(columns={'item_id':'arm',  f'class_{cluster}':'context'}, inplace=True)

    arms = X_rs.loc[:, 'arm'].values.astype(np.int8)
    contexts = X_rs.loc[:, 'context'].values.astype(np.int8)
    # sensitive = X_rs[:, 2:].astype(np.int8)


    optimal_arms = (X_rs.groupby(['context', 'arm'])
                         .mean()
                         .reset_index()
                         .sort_values(by='r', ascending=False)
                         .groupby('context')
                         .apply(lambda x: x.loc[x['r'].idxmax()][['context', 'arm', 'r']])
                         .reset_index(drop=True))


    c_values = [0.01, 0.1, 1, 10] # c values
    batch_sizes = [1, 10, 100]
    num_iterations = 10000
    num_arms = X_rs.arm.unique().shape[0]

    metrics = run_experiment(c_values, batch_sizes, num_iterations, num_arms, reward_rs, arms, contexts, optimal_arms)

    fig, axs = plt.subplots(3, figsize=(8,12))
    for i, metric in enumerate(["Avg Rewards", "Optimal Percent", "Total Regret"]):
        ax = axs[i]
        ax.set_title(metric)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("C")
        im = ax.imshow(metrics[metric], cmap="viridis")
        ax.set_xticks(np.arange(len(batch_sizes)))
        ax.set_yticks(np.arange(len(c_values)))
        ax.set_xticklabels(batch_sizes)
        ax.set_yticklabels(c_values)
        for i in range(len(c_values)):
            for j in range(len(batch_sizes)):
                text = ax.text(j, i, "{:.2f}".format(metrics[metric][i, j]), ha="center", va="center", color="w")
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'figs\\UCB_heat_{cluster}.png')
    # plt.show()

    fig, axs = plt.subplots(len(c_values), len(batch_sizes), figsize=(10, 8))
    # Find the maximum cumulative regret across all subplots
    max_regret = np.max(metrics["Cumulative Regrets"])

    for i, c in enumerate(c_values):
        for j, batch_size in enumerate(batch_sizes):
            ax = axs[i, j]
            ax.set_title("c={}, batch_size={}".format(c, batch_size))
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cumulative Regret")
            cumulative_regret = metrics["Cumulative Regrets"][i, j]
            ax.plot(np.arange(num_iterations), cumulative_regret)
            # Set the same y-limits for all subplots
            ax.set_ylim([0, max_regret])

    plt.tight_layout()
    plt.savefig(f'figs\\UCB_cum_reg_{cluster}.png')
    # plt.show()


#%%
cols = ['item_id', 'click', 'class_3','class_4','class_10', 'user_feature_0', 'user_feature_1', 'user_feature_2', 'user_feature_3']
data = pd.read_csv("dataProcessed1.csv", usecols=cols)

data.rename(columns={'item_id':'arm', 'click':'reward', 'class_4':'context'}, inplace=True)
data.arm = data.arm - 1

ros = RandomOverSampler(random_state=42)
X_rs, reward_rs = ros.fit_resample(data.drop(columns=['reward']).values, data.reward.values)
X_rs = pd.DataFrame(X_rs, columns=['arm', 'user_feature_0', 'user_feature_1', 'user_feature_2',
                                   'user_feature_3', 'class_3', 'context', 'class_10'])
X_rs['r'] = reward_rs

c_values = [0.01, 0.1]
batch_sizes = [10, 100]
num_iterations = 5000
num_arms = data.arm.unique().shape[0]

arms = X_rs.loc[:, 'arm'].values.astype(np.int8)
contexts = X_rs.loc[:, 'context'].values.astype(np.int8)
# sensitive = X_rs[:, 2:].astype(np.int8)


optimal_arms = (X_rs.groupby(['context', 'arm'])
                     .mean()
                     .reset_index()
                     .sort_values(by='r', ascending=False)
                     .groupby('context')
                     .apply(lambda x: x.loc[x['r'].idxmax()][['context', 'arm', 'r']])
                     .reset_index(drop=True))


#%% Summary Stats

ss= data.describe()
arm_dist_h = data.groupby('arm')['reward'].count()
arm_dist_r = data.groupby('arm')['reward'].mean()

context_dist = pd.DataFrame(index=[i for i in range(data.iloc[:, -4:].max().max()+1)])
context_count = pd.DataFrame(index=[i for i in range(data.iloc[:, -4:].max().max()+1)])

context_dist.loc[data.groupby('context')['reward'].mean().index, 'cluster4'] = data.groupby('context')['reward'].mean()
context_count.loc[data.groupby('context')['reward'].mean().index, 'cluster4'] = data.groupby('context')['reward'].count()

for i in range(4):
    context_dist.loc[data.groupby(f'user_feature_{i}')['reward'].count().index.to_numpy(), f'u{i}'] = data.groupby(f'user_feature_{i}')['reward'].mean()
    context_count.loc[data.groupby(f'user_feature_{i}')['reward'].count().index, f'u{i}'] = data.groupby(f'user_feature_{i}')['reward'].count()

# after resampling
arm_dist_h_re = X_rs.groupby('arm')['r'].count()
arm_dist_r_re = X_rs.groupby(['arm'])['r'].mean()

context_dist_re = pd.DataFrame(index=[i for i in range(X_rs.iloc[:, -4:].max().max()+1)])
context_count_re = pd.DataFrame(index=[i for i in range(X_rs.iloc[:, -4:].max().max()+1)])

context_dist_re.loc[X_rs.groupby('context')['r'].mean().index, 'cluster4'] = X_rs.groupby('context')['r'].mean()
context_count_re.loc[X_rs.groupby('context')['r'].mean().index, 'cluster4'] = X_rs.groupby('context')['r'].count()

for i in range(4):
    context_dist_re.loc[X_rs.groupby(f'user_feature_{i}')['r'].count().index.to_numpy(), f'u{i}'] = X_rs.groupby(f'user_feature_{i}')['r'].mean()
    context_count_re.loc[X_rs.groupby(f'user_feature_{i}')['r'].count().index, f'u{i}'] = X_rs.groupby(f'user_feature_{i}')['r'].count()

#%%
metrics = run_experiment(c_values, batch_sizes, num_iterations, num_arms, reward_rs, arms, contexts, optimal_arms)

fig, axs = plt.subplots(3, figsize=(8,12))
for i, metric in enumerate(["avg_rewards", "optimal_percents", "regrets"]):
    ax = axs[i]
    ax.set_title(metric)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("C")
    im = ax.imshow(metrics[metric], cmap="viridis")
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_yticks(np.arange(len(c_values)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels(c_values)
    for i in range(len(c_values)):
        for j in range(len(batch_sizes)):
            text = ax.text(j, i, "{:.2f}".format(metrics[metric][i, j]), ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(len(c_values), len(batch_sizes), figsize=(10, 8))

# Find the maximum cumulative regret across all subplots
max_regret = np.max(metrics["cumulative_regrets"])

for i, c in enumerate(c_values):
    for j, batch_size in enumerate(batch_sizes):
        ax = axs[i, j]
        ax.set_title("c={}, batch_size={}".format(c, batch_size))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cumulative Regret")
        cumulative_regret = metrics["cumulative_regrets"][i, j]
        ax.plot(np.arange(num_iterations), cumulative_regret)
        # Set the same y-limits for all subplots
        ax.set_ylim([0, max_regret])

plt.tight_layout()
plt.show()


#%% fairness

num_iterations = 10000

bandit = batchUCB(num_arms, reward_rs, arms, contexts, batch_size=100, c=0.1, min_arm_pulls=1)
rewards, arm_positions, context_tr = bandit.run(1000)

result = pd.DataFrame()
result['r'] = rewards
result['a'] = arm_positions
result['c'] = context_tr

t = result.groupby(['c', 'a']).count().reset_index()
plt.plot(t[t.c==2].a,t[t.c==3].r)

# Demographic parity score
fair_contexts = np.unique(context_tr)
fair_rewards = np.zeros(len(fair_contexts))
np.mean(fair_rewards)

for i, c in enumerate(fair_contexts):
    fair_rewards[i] = np.mean(np.array(rewards)[context_tr == c])

# calculate demographic parity score
demographic_parity_score = np.abs(fair_rewards.max() - fair_rewards.min())

fair_arms = np.unique(arm_positions)
fair_rewards = np.zeros(len(fair_arms))

for i, c in enumerate(fair_arms):
    fair_rewards[i] = np.mean(np.array(rewards)[arm_positions == c])

# calculate demographic parity score
demographic_parity_score = np.abs(fair_rewards.max() - fair_rewards.min())

#%% Calculate nonContextual UCB

c_values = [0.01, 0.1]
batch_sizes = [100]
num_iterations = 5000

NonMetrics = run_experiment(c_values, batch_sizes, num_iterations, num_arms, reward_rs, arms, contexts, optimal_arms, contextual = False)

fig, axs = plt.subplots(3, figsize=(8,12))
for i, metric in enumerate(["Avg Rewards", "Optimal Percent", "Total Regret"]):
    ax = axs[i]
    ax.set_title(metric)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("C")
    im = ax.imshow(NonMetrics[metric], cmap="viridis")
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_yticks(np.arange(len(c_values)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels(c_values)
    for i in range(len(c_values)):
        for j in range(len(batch_sizes)):
            text = ax.text(j, i, "{:.2f}".format(NonMetrics[metric][i, j]), ha="center", va="center", color="w")
    fig.colorbar(im, ax=ax)
plt.tight_layout()
# plt.savefig(f'figs\\UCB_heat_{cluster}.png')
plt.show()

fig, axs = plt.subplots(len(c_values), len(batch_sizes)+1, figsize=(10, 8))
# Find the maximum cumulative regret across all subplots
max_regret = np.max(NonMetrics["Cumulative Regrets"])

for i, c in enumerate(c_values):
    for j, batch_size in enumerate(batch_sizes):
        ax = axs[i, j]
        ax.set_title("c={}, batch_size={}".format(c, batch_size))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cumulative Regret")
        cumulative_regret = NonMetrics["Cumulative Regrets"][i, j]
        ax.plot(np.arange(num_iterations), cumulative_regret)
        # Set the same y-limits for all subplots
        ax.set_ylim([0, max_regret])

plt.tight_layout()
# plt.savefig(f'figs\\UCB_cum_reg_{cluster}.png')
plt.show()
