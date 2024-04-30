import numpy as np 
from sklearn.preprocessing import MinMaxScaler

class Q_BayesUCB():
    def __init__(self, narms, alpha= 1.0, beta= 0.2 , tau = 0.95, T = 200):
        self.narms = narms
        self.tau = tau
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.intial_steps = 1
        self.random_init = False
        self.M  = 1000
        self.prior_alpha = alpha
        self.prior_beta = beta

    def policy_func(self, seq ):
        data = [np.quantile(seq[:i+1], self.tau) for i in range(len(seq))]
        list_q = [ (i+1)*data[i] - (i)*data[i-1] if i > 0 else data[i]  for i in range(len(data)) ]
        q = np.quantile(seq, self.tau)

        if(len(list_q)>0):
            # Update the prior with the data to get posterior parameters
            posterior_alpha = self.prior_alpha + len(list_q)/2
            posterior_beta = self.prior_beta + 0.5 * np.sum((list_q - q)**2)
            # Calculate posterior variance
            posterior_variance = posterior_beta/(posterior_alpha-1)
        else:
            posterior_variance =  self.prior_beta /(self.prior_alpha-1)

        samples = np.random.normal(q, posterior_variance, self.M)
        
        return np.quantile(samples, q = 1 - 1/self.t)

    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        if np.any(pulled_arms<self.intial_steps):
            if(self.random_init):
                self.selected_arm =  np.random.choice(np.where(pulled_arms<self.intial_steps)[0])
            else:
                self.selected_arm =  (self.t -1)%self.narms
        else:
            policy = np.zeros(self.narms)
            for i  in range(self.narms):
                reward_seq =  np.asarray(self.rewards[ np.asarray(self.pulled_arms) == i ])[:,0]
                policy[i] =  self.policy_func(reward_seq)
            self.selected_arm =  np.argmax(policy)  
        return self.selected_arm
    

    def update_cost(self, cost, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform( np.asarray(self.raw_rewards).reshape(-1,1))
        self.t += 1

