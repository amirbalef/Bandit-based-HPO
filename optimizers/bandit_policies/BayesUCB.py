import numpy as np 
from sklearn.preprocessing import MinMaxScaler


class BayesUCB():
    def __init__(self, narms, tau = 0.95, T = 200):
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
    
    def policy_func(self, seq ):

        mean=np.mean(seq)
        std = np.std(seq)
        
        if(len(seq)==1):
            std = 1.0
        samples = np.random.normal(mean, std, self.M)
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

