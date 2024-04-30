from __future__ import annotations
import numpy as np
import pandas as pd

from amltk.pipeline import Choice, Component
import ConfigSpace

from tabrepo import load_repository, get_context, EvaluationRepository
from autogluon.common import space as ag_space
import copy
import importlib

class Classifier:
    def __init__(self, scores, hyp_configs, **config ): 
        indx = hyp_configs.index(config)
        self.result = scores[indx]
    def fit(self, X, y):
        pass
    def predict(self, X):
        return self.result


class TabRepo():
    def __init__(self, context_name = "D244_F3_C1416_30" ): 
        self.context = get_context(name=context_name)
        self.config_hyperparameters = self.context.load_configs_hyperparameters()
        self.repo: EvaluationRepository = load_repository(context_name, cache=True)
        self.methods = self.get_methods()
        self.module_names = {'CatBoost': "catboost",'ExtraTrees':"extra_trees", 'NeuralNetFastAI':"fastai",'LightGBM':"lightgbm", 'NeuralNetTorch':"nn_torch",'RandomForest' :"random_forest", 'XGBoost' :"xgboost"}

    def get_instances_list(self):
            return self.repo.datasets()
    
    def get_methods(self):
        configs = self.repo.configs()
        methods = {}
        for item in configs:
            method, key = item.split('_')[0:2]
            if(method not in methods.keys()):
                methods[method] = [item]
            else:
                methods[method].append(item)
        return methods
      
    def get_raw_configs_and_scores(self, methods, dataset_name,  fold = 0 ):
        configurations = {}
        scores = {}
        for method in methods.keys():
            list_configs=methods[method]
            configs = []
            for item in list_configs:
                method, key_id = item.split('_')[0:2]
                config_key = method + '_' + key_id
                if("c" in key_id): # Skip defualt configuratiobs
                    continue
                item_hyperparameters = copy.deepcopy(self.config_hyperparameters[config_key]["hyperparameters"])
                if("ag_args" in item_hyperparameters):
                    del item_hyperparameters["ag_args"]
                if(len(item_hyperparameters)>0): #Droping missing value
                    if(method not in configurations.keys()):
                        configurations[method] = [item_hyperparameters]
                        configs.append(item)
                    else:
                        configurations[method].append(item_hyperparameters)
                        configs.append(item)
                else:
                    print("error",config_key)
            scores[method] = np.asarray(self.repo.metrics(datasets=[dataset_name], configs=configs, folds=[fold] )['metric_error_val'].tolist())
            #print("configurations, scores[method]", len(configurations[method]), len(scores[method]), len(configs))
        return configurations,scores

    def get_subspace(self, method_id):
        subspace = ConfigSpace.configuration_space.ConfigurationSpace(name=method_id)
        generate = importlib.import_module("tabrepo.models."+self.module_names[method_id]+".generate")
        for key,value in generate.search_space.items():
            name = key
            if isinstance(value, ag_space.Real):
                hyp = ConfigSpace.api.types.float.Float(name = method_id +"_"+ key, bounds = [value.lower, value.upper], default=value.default,log=value.log)
            elif isinstance(value, ag_space.Int):
                hyp = ConfigSpace.api.types.integer.Integer(name = method_id +"_"+ key, bounds = [value.lower, value.upper])
            elif isinstance(value, ag_space.Categorical):
                datas = [str(data) for data in value.data]
                hyp = ConfigSpace.api.types.categorical.Categorical(name = method_id +"_"+ key, items = datas, default=datas[0])
            else:
                print("oh no",name, value) 
                NotImplementedError
            subspace.add_hyperparameter(hyp)
        return subspace
    
    def get_subspace_and_configs_from_raw(self, method_id, subspace_hyperparameters):
        subspace = ConfigSpace.configuration_space.ConfigurationSpace(name=method_id)
        df = pd.DataFrame.from_dict(subspace_hyperparameters)
        for col in df:
            col_data = (df[col])
            datas = col_data.dropna().drop_duplicates()#.dropna().unique()
            type_col = type(col_data.dtype)
            if(type_col == np.dtypes.Float64DType):
                    hyp = ConfigSpace.api.types.float.Float(name = method_id +"_"+ col, bounds = [min(datas), max(datas)] )
            elif(type_col == np.dtypes.Int64DType):
                    hyp = ConfigSpace.api.types.integer.Integer(name = method_id +"_"+col, bounds = [min(datas), max(datas)] )
            elif( type_col== np.dtypes.ObjectDType):
                    datas = datas.astype(str)
                    hyp = ConfigSpace.api.types.categorical.Categorical(name = method_id +"_"+ col, items = datas )
            elif( type_col== np.dtypes.BoolDType):
                    hyp = ConfigSpace.api.types.integer.Integer(name = method_id +"_"+col, bounds = [min(datas), max(datas)] )
            else:
                print(type_col)
                NotImplementedError
            subspace.add_hyperparameter(hyp)
        configs = []
        for config in subspace_hyperparameters:
            new_config ={}
            for k_old in config:                       
                name = method_id+"_"+k_old
                hyp_type = subspace.get_hyperparameter(name)
                value = config[k_old]
                if isinstance(hyp_type, (ConfigSpace.hyperparameters.CategoricalHyperparameter)):
                    value = str(value)
                new_config[name] = value
            configs.append(new_config)
        return subspace, configs

    def get_subspace_configs_and_scores(self, subspace, method_id, dataset_name,  fold = 0 ):
        configs_for_subpace = []
        configs_for_metric = []
        list_configs = self.methods[method_id]
        for item in list_configs:
            method_id, key_id = item.split('_')[0:2]
            config_key = method_id + '_' + key_id
            item_hyperparameters = copy.deepcopy(self.config_hyperparameters[config_key]["hyperparameters"])
            if("c" in key_id): # default configurations
                default_config = subspace.get_default_configuration()
                for k,v in default_config.items():
                    hyp_name = k[len(method_id+"_"):]        
                    if(hyp_name not in item_hyperparameters):
                        item_hyperparameters[hyp_name] = v       
                                 
            if("ag_args" in item_hyperparameters):
                del item_hyperparameters["ag_args"]
            new_config ={}
            for k_old in item_hyperparameters:                       
                name = method_id+"_"+k_old
                hyp_type = subspace.get_hyperparameter(name)
                value = item_hyperparameters[k_old]
                if isinstance(hyp_type, (ConfigSpace.hyperparameters.CategoricalHyperparameter)):
                    value = str(value)
                new_config[name] = value
            configs_for_subpace.append(new_config)
            configs_for_metric.append(item)
        scores= np.asarray(self.repo.metrics(datasets=[dataset_name], configs=configs_for_metric, folds=[fold] )['metric_error_val'].tolist())
        #print("configurations, scores[method]", len(configs_for_subpace), len(scores), len(configs_for_metric))
 
        return configs_for_subpace, scores
         
    def get_valid_configs(self, space, configs):
        valid_configs = []
        for i, method in enumerate(self.methods.keys()):
            for config in configs[method]:
                new_config = {'methods:__choice__': method}
                for k_old in config:          
                    name = "methods:"+ method+":"+k_old             
                    new_config[name] = config[k_old]
                valid_configs.append(ConfigSpace.configuration_space.Configuration(space, values=new_config))
        return valid_configs

    def get_pipeline(self, instance,  fold = 0, return_valid_configs = True ):
        #hyp_configs, scores = self.get_raw_configs_and_scores(self.methods, dataset_name = instance, fold = fold)
        routes = []
        configs = {}
        for i, method in enumerate(self.methods.keys()):
                sub_space = self.get_subspace(method)
                its_configs, its_scores = self.get_subspace_configs_and_scores(sub_space, method,  dataset_name = instance, fold = fold)
                configs[method]= its_configs
                #print(len(its_scores))
                item =Component(
                    Classifier,
                    config={'scores':its_scores,
                            'hyp_configs':its_configs,
                            },
                    space=sub_space,
                    name =method
                    )
                routes.append(item)
        pipeline = Choice(*routes, name="methods")
        if(return_valid_configs):
            valid_configs =  self.get_valid_configs(pipeline.search_space("configspace"), configs)
            return pipeline, valid_configs
        else:
            return pipeline
    
def test():
    dataset = TabRepo(context_name = "D244_F3_C1416_30")
    instance_names = dataset.get_instances_list()
    instance = instance_names[0]
    pipeline, valid_configs  = dataset.get_pipeline(instance=instance )

    config = pipeline.search_space("configspace").get_default_configuration()
    print(config)
    configured_pipeline = pipeline.configure(config)
    sklearn_pipeline = configured_pipeline.build("sklearn")
    sklearn_pipeline.fit(None, None)
    print("default_configuration ps", sklearn_pipeline.predict(None))
    for i in range(10):
        config = valid_configs[i]
        print(config)
        configured_pipeline = pipeline.configure(config)
        sklearn_pipeline = configured_pipeline.build("sklearn")
        sklearn_pipeline.fit(None, None)
        print("ps", sklearn_pipeline.predict(None))

#test()