import numpy as np
from sklearn.preprocessing import MinMaxScaler


class ER_UCB_N:
    def __init__(self, narms, alpha=1.0, theta=0.01, gamma=20):
        self.narms = narms
        self.alpha = alpha
        # print(self.alpha)
        self.theta = theta
        self.gamma = gamma
        self.t = 1
        self.T = 200
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        self.random_init = False
        self.g_name = "sigmoid"
        self.t_axis_scale = 0.025
        self.curve_param_list = {}
        self.ucb_values = [0 for _ in range(self.narms)]
        self.intial_steps = 2
        # self.A_m = 1
        # self.A_n = 1

    # y = g^{-1}(x)
    def _g_inverse(self, x):
        if self.g_name == "linear":
            y = x
        elif self.g_name == "ln":
            y = np.exp(x) - 1
        elif self.g_name == "sigmoid":
            if x >= 1:
                print("error x: {}".format(x))
                x = 0.999999999999
            y = np.log((1 + x) / abs(1 - x)) / self.t_axis_scale
        else:
            y = 0
        return y

    # z = g(y)
    def _g(self, y):
        if self.g_name == "linear":
            z = y
        elif self.g_name == "ln":
            z = np.log(y + 1)
        elif self.g_name == "sigmoid":
            z = 2.0 / (1 + np.exp(-self.t_axis_scale * y)) - 1
        else:
            z = 0
        return z

    # Delta_{T}(x)
    def _func_delta_t(self, rewards, sigma, x):
        tn = len(rewards)
        y_list = [self._g_inverse(reward) for reward in rewards]
        this_ts = np.array([_ + 1.0 for _ in range(len(y_list))])
        AT = np.dot(this_ts - np.mean(this_ts), (this_ts - np.mean(this_ts)).T)
        # part_1 = -(x * sigma / np.sqrt(AT)) * norm.ppf(abs(np.power(self.t, -self.alpha) - self.A_m * np.power(tn, -0.5)))
        # part_2 = -sigma * np.sqrt(1.0 / tn + np.power(np.mean(this_ts), 2) / AT) * norm.ppf(abs(np.power(self.t, -self.alpha) - self.A_n * np.power(tn, -0.5)))
        part_1 = x * sigma / np.sqrt(AT)
        part_2 = sigma * np.sqrt(1.0 / tn + np.power(np.mean(this_ts), 2) / AT)
        r_value = part_1 + part_2
        return r_value

    # linear regression based on ys
    def _curve_update(self, rewards):
        y_list = [self._g_inverse(reward) for reward in rewards]
        this_ys = np.asarray(y_list, dtype=float)
        this_zs = np.array(rewards)
        this_ts = np.array([_ + 1.0 for _ in range(len(y_list))])
        AT = np.dot(this_ts - np.mean(this_ts), (this_ts - np.mean(this_ts)).T)
        # AT2 = (np.max(this_ts) ** 3 - np.max(this_ts)) / 12.0
        beta_1 = np.sum((this_ts - np.mean(this_ts)) * this_ys) / AT
        beta_0 = np.mean(this_ys) - beta_1 * np.mean(this_ts)
        cal_ys = np.array(
            [self._g(beta_1 * this_ts[_] + beta_0) for _ in range(this_ts.shape[0])]
        )
        sigma = np.sqrt(np.mean((this_zs - cal_ys) * (this_zs - cal_ys)))
        return beta_1, beta_0, sigma

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
            for x in range(self.narms):
                rewards = self.rewards[np.asarray(self.pulled_arms) == x]
                beta_1, beta_0, sigma = self._curve_update(rewards)
                exploitation_item = self._g(beta_1 * self.T + beta_0) + np.sqrt(
                    (sigma**2) / self.theta
                )

                exploration_item = self._func_delta_t(rewards, sigma, self.t) + np.sqrt(
                    (self._func_delta_t(rewards, sigma, len(rewards)) + 1)
                    * np.sqrt(self.alpha * np.log(self.t) / (2 * len(rewards)))
                )

                self.ucb_values[x] = self.gamma * exploitation_item + exploration_item

            self.selected_arm = np.argmax(self.ucb_values)
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
