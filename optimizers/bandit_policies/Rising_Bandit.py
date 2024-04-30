import numpy as np 
from sklearn.preprocessing import MinMaxScaler
 
class Rising_Bandit():
    def __init__(self, narms, T = 200):
        self.narms = narms
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler(feature_range=(0.01, 0.99))
        self.S_cand = np.ones(self.narms, dtype=int)
        self.u_values = np.ones(self.narms)
        self.l_values = np.zeros(self.narms)
        self.C = 7
        self.T = T
        self.S_cand_arms_to_be_played = np.ones(self.narms, dtype=int)

    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        if np.any(pulled_arms==0):
            self.selected_arm =  (self.t -1)%self.narms
        else:
            self.selected_arm = np.where(self.S_cand_arms_to_be_played==1)[0][0]
        #print(self.selected_arm, self.removed_arms)
        return self.selected_arm
    
    def update(self, reward, arm = None, context=None):
        return NotImplemented

    def update_cost(self, cost, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.S_cand_arms_to_be_played[self.selected_arm] = 0

        if sum(self.S_cand_arms_to_be_played) == 0:
            if sum(self.S_cand)>1:
                self.rewards = self.scaler.fit_transform( np.asarray(self.raw_rewards).reshape(-1,1))
                for x in range(self.narms):
                    if(self.S_cand[x]):
                        indx = (np.asarray(self.pulled_arms) == x)        
                        rewards = np.maximum.accumulate(self.rewards[indx])[:,0]
                        #print(self.t, x, rewards)
                        if(len(rewards) > 0):
                            y_t = rewards[-1] 
                            y_t_C = rewards[-self.C -1] if(len(rewards)>self.C) else -1
                            #print(self.t, x,  y_t, y_t_C, rewards)
                            self.l_values[x] = y_t
                            omega = (y_t - y_t_C)/self.C 
                            self.u_values[x] = min(1, y_t+ omega*(self.T-self.t))
                #print(self.t, self.l_values,   self.u_values)
                if sum(self.S_cand ) > 0:
                    for i in range(self.narms):
                        if(self.S_cand[i]):
                            for j in range(self.narms):
                                if(self.S_cand[j] and i!=j):
                                    if( self.l_values[i]>= self.u_values[j]):
                                        self.S_cand[j] = 0
                #print(self.l_values,self.u_values,self.S_cand)
                #print(self.t, self.S_cand)
            self.S_cand_arms_to_be_played = np.array((self.S_cand == 1),dtype=int)
            
            #print(self.t, "(self.S_cand)", (self.S_cand))
        
        self.t += 1