import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ER_UCB_S:
    def __init__(self, narms, betha=0.6, tetha=0.01, gamma=20):
        self.narms = narms
        self.betha = betha
        self.tetha = tetha
        self.gamma = gamma
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.random_init = False

    def play(self, context=None):
        pulled_arms = np.bincount(self.pulled_arms, minlength=self.narms)
        if np.any(pulled_arms == 0):
            if self.random_init:
                self.selected_arm = np.random.choice(np.where(pulled_arms == 0)[0])
            else:
                self.selected_arm = (self.t - 1) % self.narms
        else:
            mean_rewards = np.asarray(
                [
                    np.mean(
                        np.asarray(self.rewards[np.asarray(self.pulled_arms) == x])
                        - self.betha
                    )
                    for x in range(self.narms)
                ]
            )
            mean_rewards_2 = np.asarray(
                [
                    np.mean(
                        (
                            np.asarray(self.rewards[np.asarray(self.pulled_arms) == x])
                            - self.betha
                        )
                        ** 2
                    )
                    for x in range(self.narms)
                ]
            )
            mean = self.gamma * mean_rewards + np.sqrt(mean_rewards_2 / self.tetha)
            g = np.sqrt(2 * (np.log(self.t)) / pulled_arms) + np.sqrt(
                (1 / self.tetha) * np.sqrt(2 * (np.log(self.t)) / pulled_arms)
            )
            ucb_values = mean + g
            self.selected_arm = np.argmax(ucb_values)
        return self.selected_arm

    def update_cost(self, cost, arm=None, context=None):
        if arm == None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform(
            np.asarray(self.raw_rewards).reshape(-1, 1)
        )
        self.t += 1
