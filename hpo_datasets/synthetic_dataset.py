from __future__ import annotations
import numpy as np
from scipy import stats
from amltk.pipeline import Choice, Component

class Hosaki:
    def __init__(self, dim):
        assert (type(dim) == int and dim >=2)
        self.x_best = (4,2)
        self.f_best = -2.345811576101292
        self.f_worst = 0.5
        self.bounds = (0,5,0,5)

    def __call__(self, x):
        return  (1 + x[0]*(-8 + x[0]*(7 + x[0]*(-7.0/3.0 + x[0] *1.0/4.0))))*x[1]*x[1] * np.exp(-x[1])

class Classifier:
    def __init__(self, model, **space ): 
        self.model = model
        self.X = np.array(list(space.values()))
    def fit(self, X, y):
        pass
    def predict(self, X):
        cost = (self.model['func'](self.X) - self.model['func'].f_best)/self.model['scale_factor'] + self.model['mean']
        return cost

class Synthetic_dataset():
    def __init__(self ):
        list_offsets = np.random.uniform(0.05, 0.15, 7)
        list_offsets.sort()#list_offsets[::-1].sort()
        self.models = []
        for i, mean in enumerate(list_offsets):
            dim = 2
            model = {"mean": mean, "name": "model"+str(i), "num_parmas":dim}
            func = Hosaki(dim)
            model['func'] = func
            config = {}
            for p in range(dim):
                c = func.x_best[p]
                l,u,_,_ = func.bounds
                L = stats.truncnorm.rvs(l, c, loc = (l+c)/2, scale = 0.2, size = 1)[0]
                U = stats.truncnorm.rvs(c, u, loc = (u+c)/2, scale = 0.2, size = 1)[0]
                config["p"+str(p)] = (L, U)
            model['space']= config
            if hasattr(func, 'f_worst'):
                 f_worst = func.f_worst
            else:
                X = np.array(list(config.values()))
                f_worst = max( func(X[:,0]), func(X[:,1]) ) 
            model['scale_factor'] = f_worst - func.f_best
            self.models.append(model)

    def get_instances_list(self):
            return ["non_linear"]

    def get_pipeline(self, instance):
        routes=[]
        for i, model in enumerate(self.models):
            item =Component(
                Classifier,
                config={'model':model,},
                space= model['space'],
                name = model['name']
                )
            routes.append(item)
        return Choice(*routes, name="methods")
