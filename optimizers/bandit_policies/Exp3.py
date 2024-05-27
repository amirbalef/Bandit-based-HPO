import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Exp3:
    def __init__(self, narms, gamma=0.1):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        self.random_init = False

        self.gamma = gamma
        self.p_t = np.zeros(self.narms)
        self.w_t = np.ones(self.narms)

    def calculate_p(self, w_t):
        return (1 - self.gamma) * (w_t / np.sum(w_t)) + (self.gamma / self.narms)

    def play(self, context=None):
        pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)
        if np.any(pulled_arms == 0):
            self.selected_arm = (self.t - 1) % self.narms
        else:
            self.p_t = self.calculate_p(self.w_t)
            self.selected_arm = np.random.choice(self.narms, p=self.p_t)
        return self.selected_arm

    def update_cost(self, cost, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform(
            np.asarray(self.raw_rewards).reshape(-1, 1)
        )

        self.w_t = np.ones(self.narms)
        for i in range(len(self.pulled_arms)):
            self.p_t = self.calculate_p(self.w_t)
            xhat_t = np.zeros(self.narms)
            xhat_t[self.pulled_arms[i]] = (
                self.rewards[i] / self.p_t[self.pulled_arms[i]]
            )
            self.w_t *= np.exp(self.gamma * xhat_t / self.narms)
        self.t += 1


class Exp3_TB:
    def __init__(self, narms, T=200):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        self.random_init = False
        self.eta = np.sqrt(np.log(self.narms) / (200 * self.narms))  # eta
        self.P_t = np.zeros(self.narms)
        self.S_t = np.zeros(self.narms)
        self.epsilon = 1e-10

    def calculate_P(self, S):
        x = self.eta * S
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        return numerator / denominator

    def play(self, context=None):
        pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)
        if np.any(pulled_arms == 0):
            self.selected_arm = (self.t - 1) % self.narms
        else:
            self.P_t = self.calculate_P(self.S_t)
            self.selected_arm = np.random.choice(self.narms, p=self.P_t)
        return self.selected_arm

    def update_cost(self, cost, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform(
            np.asarray(self.raw_rewards).reshape(-1, 1)
        )
        self.S_t = np.zeros(self.narms)
        for i in range(len(self.pulled_arms)):
            self.P_t = self.calculate_P(self.S_t)
            A_t = np.zeros(self.narms)
            A_t[self.pulled_arms[i]] = 1
            self.P_t[self.P_t == 0] = self.epsilon
            self.S_t += (
                1 - (A_t * (1 - self.rewards[i])) / self.P_t[self.pulled_arms[i]]
            )
        self.t += 1
