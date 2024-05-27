import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Max_Median:
    def __init__(self, narms, epsilon=1, T=200):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.intial_steps = 1
        self.random_init = False
        self.epsilon = epsilon
        self.policy = np.zeros(narms)

    def play(self, context=None):
        pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)
        if np.any(pulled_arms < self.intial_steps):
            if self.random_init:
                self.selected_arm = np.random.choice(
                    np.where(pulled_arms < self.intial_steps)[0]
                )
            else:
                self.selected_arm = (self.t - 1) % self.narms
        else:
            if np.random.uniform() > self.epsilon:
                N_k = [
                    np.sum(np.asarray(self.pulled_arms) == x) for x in range(self.narms)
                ]
                m = np.min(N_k)
                upsilon = np.ceil(N_k / m).astype(int)
                self.policy = [
                    np.sort(
                        np.asarray(self.rewards[np.asarray(self.pulled_arms) == x])
                    )[-upsilon[x]]
                    for x in range(self.narms)
                ]
                self.selected_arm = np.argmax(self.policy)
            else:
                self.selected_arm = np.random.randint(0, self.narms)
        return self.selected_arm

    def update(self, reward, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.rewards.append(reward)
        self.t += 1

    def update_cost(self, cost, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform(
            np.asarray(self.raw_rewards).reshape(-1, 1)
        )
        self.epsilon = 1 / (self.t)
        self.t += 1
