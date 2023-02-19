import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler

class contextualTS:
    def __init__(self, n_arms, arms, rewards, contexts, batch_size=1, prior_strength=0.1, min_pulls=1, contextual=True):
        self.n_arms = n_arms
        self.n_contexts = contexts.max()+1
        self.contexts = contexts
        self.arms = arms
        self.batch_size = batch_size
        self.remaining_reward_values = rewards
        self.alpha = np.ones((n_arms, self.n_contexts))
        self.beta = np.ones((n_arms, self.n_contexts))
        self.arm_counts = np.zeros((n_arms, self.n_contexts.max()+1))
        self.prior_strength = prior_strength
        self.contextual = contextual

    def select_arm(self, context):
        theta = np.random.beta(self.alpha[:, context], self.beta[:, context])
        weights = theta + np.random.normal(scale=self.prior_strength, size=self.n_arms)
        return np.argmax(weights)

    def update(self, arm, context, reward):
        if self.contextual:
            for a,c,r in zip(arm, context, reward):
                self.arm_counts[a, c] += 1
                self.alpha[a, c] += r
                self.beta[a, c] += (1 - r)

        else:
            for a,r in zip(arm, reward):
                self.arm_counts[a, :] += 1
                self.alpha[a, :] += r
                self.beta[a, :] += (1 - r)

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

            self.update(batch_arm_positions, batch_context, batch_rewards)
            rewards.extend(batch_rewards)
            arm_positions.extend(batch_arm_positions)
            context_tr.extend(batch_context)

        return rewards, arm_positions, context_tr

def run_experiment(prior_values, batch_sizes, num_iterations, num_arms, reward_values, arms, contexts, opt_arms, contextual=True):
    metrics = {"Avg Rewards": np.zeros((len(prior_values), len(batch_sizes))),
               "Optimal Percent": np.zeros((len(prior_values), len(batch_sizes))),
               "Total Regret": np.zeros((len(prior_values), len(batch_sizes))),
               "Cumulative Regrets": np.zeros((len(prior_values), len(batch_sizes), num_iterations))}

    dftest = pd.DataFrame(reward_values)
    dftest['context'] = contexts
    dftest['arm'] = arms
    optimal_actions = []
    optimal_cumulative_rewards = []

    for i, c in tqdm(enumerate(prior_values)):
        for j, batch_size in enumerate(batch_sizes):
            bandit = contextualTS(num_arms, arms, reward_values, contexts, batch_size=batch_size, prior_strength=c, contextual=contextual)
            rewards, arm_positions, context_tr = bandit.run(num_iterations)
            for contex in range(max(contexts)):
                optimal_actions.extend( (np.array(arm_positions)[np.array(context_tr)==contex] == opt_arms.loc[contex, 'arm']).tolist())

            cumulative_rewards = np.cumsum(rewards)
            optimal_cumulative_rewards = np.cumsum([opt_arms.loc[k, 'r'] for k in context_tr])

            metrics["Total Regret"][i, j] = optimal_cumulative_rewards[-1] - cumulative_rewards[-1]
            metrics["Avg Rewards"][i, j] = cumulative_rewards[-1] / num_iterations
            metrics["Optimal Percent"][i, j] = 100 * np.mean(optimal_actions)
            metrics["Cumulative Regrets"][i, j, :] = optimal_cumulative_rewards - cumulative_rewards

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


    c_values = [0.001, 0.01, 0.1] # prior strength
    batch_sizes = [1, 10, 100]
    num_iterations = 10000
    num_arms = X_rs.arm.unique().shape[0]

    metrics = run_experiment(c_values, batch_sizes, num_iterations, num_arms, reward_rs, arms, contexts, optimal_arms)

    fig, axs = plt.subplots(3, figsize=(8,12))
    for i, metric in enumerate(["Avg Rewards", "Optimal Percent", "Total Regret"]):
        ax = axs[i]
        ax.set_title(metric)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Prior Weight")
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
    plt.savefig(f'figs\\TS_heat_{cluster}.png')
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
    plt.savefig(f'figs\\TS_cum_reg_{cluster}.png')
    # plt.show()

#%%

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