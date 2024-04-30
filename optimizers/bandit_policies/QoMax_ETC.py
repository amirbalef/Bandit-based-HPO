import numpy as np 
from sklearn.preprocessing import MinMaxScaler
   
def QoMax(X,batch = 8, n=8, q= 0.5):
    #print(len(X))
    X_max_b = np.zeros(batch)
    for i in range(batch):
        X_max_b = np.max(X[n*(i):n*(i+1)])
    return np.quantile(X_max_b, q)


class QoMax_ETC():
    def __init__(self, narms, batch = 4, n=3,  T = 200):
        self.narms = narms
        self.batch = batch
        self.n = n
        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()
        self.intial_steps = (self.batch * self.n)
        self.random_init = False
        self.explore_phase = True
        self.policy = None

    def play(self, context=None):
        pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.narms)
        if np.any(pulled_arms<self.intial_steps):
            if(self.random_init):
                self.selected_arm =  np.random.choice(np.where(pulled_arms<self.intial_steps)[0])
            else:
                self.selected_arm =  (self.t -1)%self.narms
          
        else:
            if(self.explore_phase == True):
                self.policy = np.asarray([ QoMax(self.rewards[ np.asarray(self.pulled_arms) == x ], batch = self.batch, n = self.n) for x  in range(self.narms)])
                self.explore_phase = False
                self.selected_arm =  np.argmax(self.policy) 
            else:
                self.selected_arm =  np.argmax(self.policy)  
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