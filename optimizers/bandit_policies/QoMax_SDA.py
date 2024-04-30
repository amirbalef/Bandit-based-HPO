import numpy as np 
from sklearn.preprocessing import MinMaxScaler

def QoMax(X,batch = 8, n=8, q= 0.5):
    X_max_b = np.zeros(batch)
    for i in range(batch):
        X_max_b = np.max(X[n*(i):n*(i+1)])
    return np.quantile(X_max_b, q)


class QoMax_SDA():
    def __init__(self, narms, T = 200, q = 0.5):
        self.nb_arms = narms

        self.policy = None
        self.chosen_arms = []
        self.l_prev = None
        self.qomax = np.inf * np.ones(self.nb_arms)


        self.t = 1
        self.pulled_arms = []
        self.rewards = []
        self.raw_rewards = []
        self.selected_arm = 0
        self.scaler = MinMaxScaler()

        self.nb_k = np.zeros(self.nb_arms, dtype=int )#n_k_r
        self.nb_batch = np.zeros(self.nb_arms, dtype=int ) #b_k_r
        self.arms_to_be_played = list(range(self.nb_arms))
        self.r = 1
        self.q = q
        self.gamma  = 2.0/3.0

    def QoMax_duel(self, l, k):
        if k == l:
            return k
        if self.nb_k[k] <= self.f(self.r):
            return k
        # Compute challenger's QoMax
        self.qomax[k] = QoMax(self.rewards[ np.asarray(self.pulled_arms) == k], batch=self.nb_batch[k], n = self.nb_k[k], q = self.q)
        # Compute leader's QoMax (on Last Block subsample)
        Y_l = self.rewards[ np.asarray(self.pulled_arms) == l][self.nb_k[l]-self.nb_k[k]:self.nb_k[l]*self.nb_batch[k]]
        #logging.debug(repr("k,l", k,l))
        #logging.debug(repr("nb_batch   ",self.nb_batch))
        #logging.debug(repr("nb_k       ",self.nb_k))
        #pulled_arms = np.bincount(  self.pulled_arms,  minlength=self.nb_arms)
        #logging.debug(repr("pulled_arms", pulled_arms))
        #logging.debug(repr(len(self.rewards[ np.asarray(self.pulled_arms) == k])))
        #logging.debug(repr(len(self.rewards[ np.asarray(self.pulled_arms) == l])))
        #logging.debug(repr(self.t, "QoMax_duel", len(Y_l), self.nb_k[l]-self.nb_k[k], (self.nb_k[l])*self.nb_batch[k]))
        #logging.debug(repr(self.nb_k[l],self.nb_k[k],self.nb_batch[k], self.nb_batch[l] ))
        self.qomax[l] = QoMax(Y_l, batch=self.nb_batch[k], n = self.nb_k[k], q = self.q)

        if self.qomax[l]  <= self.qomax[k]:
            return k
        else:
            return l
    
    def compute_QoMAX_SDA(self):
        self.r  = self.r + 1   #each round

        if self.chosen_arms == [self.l_prev]:
                self.l = self.l_prev
        else:
            self.l = self.get_leader(self.nb_k, self.qomax)  # Compute_leader
            self.l_prev = self.l
        
        self.chosen_arms = []
        for k in range(self.nb_arms):  # Duel step
            if self.QoMax_duel( self.l, k) == k and k != self.l:
                self.chosen_arms.append(k)

        if self.nb_k[self.l] <= self.f(self.r):
            self.chosen_arms.append(self.l)

        if len(self.chosen_arms) == 0:
            self.chosen_arms = [self.l]

        return [x for x in self.chosen_arms]

    def colletect_data(self,chosen_arms):
        arms_to_be_played = []
        for arm in chosen_arms:
            if self.nb_batch[arm] > 0:
                arms_to_be_played.extend([arm for _ in range(self.nb_k[arm]+1)])
            while self.nb_batch[arm] < self.B(self.nb_k[arm]+1):
                arms_to_be_played.extend([arm for _ in range(self.nb_k[arm]+1)])
                self.nb_batch[arm] +=1
            self.nb_k[arm] += 1
        return arms_to_be_played

    def play(self, context=None):
        if(len(self.arms_to_be_played)==0):
            chosen_arms = self.compute_QoMAX_SDA() 
            #logging.debug(repr(self.t, "chosen_arms",chosen_arms))
            self.arms_to_be_played = self.colletect_data(chosen_arms)
            #logging.debug(repr(self.t, "self.arms_to_be_played",self.arms_to_be_played))
        self.selected_arm = self.arms_to_be_played.pop(0)

        #print(self.t, "self.selected_arm",self.selected_arm)
        return self.selected_arm
    
    def update_cost(self, cost, arm = None, context=None):
        if arm ==None:
            arm = self.selected_arm
        self.pulled_arms.append(arm)
        self.raw_rewards.append(-cost)
        self.rewards = self.scaler.fit_transform( np.asarray(self.raw_rewards).reshape(-1,1))
        self.t += 1

    def get_leader(self, n,X):
        m = np.amax(n)
        n_argmax = np.nonzero(n == m)[0]
        if n_argmax.shape[0] == 1:
            return n_argmax[0]
        else:
            maximomax = X[n_argmax].max()
            s_argmax = np.nonzero(X[n_argmax] == maximomax)[0]
        return n_argmax[np.random.choice(s_argmax)]

    def f(self,x):
        return np.log(x)**(1/self.gamma)
    
    def B(self,n):
        return n**(self.gamma)
