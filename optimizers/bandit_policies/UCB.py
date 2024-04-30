import numpy as np 
from sklearn.preprocessing import MinMaxScaler

class UCB():
    def __init__(self, narms, alpha = 0.5, tuned=False):
        self.narms = narms
        self.alpha = alpha
        self.tuned = tuned
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.random_init = False

    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        if np.any(pulled_arms==0):
            if(self.random_init):
                self.selected_arm =  np.random.choice(np.where(pulled_arms==0)[0])
            else:
                self.selected_arm =  (self.t -1)%self.narms
    
        else:
            if(self.tuned):
                mean_rewards = np.asarray([ np.mean( self.rewards[ np.asarray(self.pulled_arms) == x ] ) for x  in range(self.narms)])
                variance_rewards = np.asarray([ np.var( self.rewards[ np.asarray(self.pulled_arms) == x ] ) for x  in range(self.narms)])
                tuned_padding = variance_rewards + 2*(np.log(self.t)) / pulled_arms
                tuned_padding[tuned_padding<0.25] = 0.25
                ucb_values= mean_rewards +  np.sqrt( np.log(self.t) / pulled_arms * tuned_padding)
            else:
                mean_rewards = np.asarray([ np.mean( self.rewards[ np.asarray(self.pulled_arms) == x ] ) for x  in range(self.narms)])
                ucb_values=  mean_rewards + np.sqrt(self.alpha *(np.log(self.t)) / pulled_arms)
            self.selected_arm =  np.argmax(ucb_values)  
        return self.selected_arm
    def update(self, reward, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.rewards.append(reward)
        self.t += 1

    def update_cost(self, cost, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform( np.asarray(self.raw_rewards).reshape(-1,1))
        self.t += 1

Policy = UCB
