from hyperopt import hp, Trials, fmin, tpe, partial, space_eval
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from analyser.MLP import MLP
from analyser.utils import k_fold_cross_validation


class TPEOptimiser:
    def __init__(self, target_column, with_soil=True):
        self.target_column = target_column
        self.with_soil = with_soil
        self.type_model_map = {
            'mlp': MLP,
            'svm': SVR,
            'random_forest': RandomForestRegressor,
            'linear_regression': LinearRegression
        }
        self.space = hp.choice('classifier_type', [
                      {
               'type': 'random_forest',
               'n_estimators': 100,
                'max_depth': hp.choice('random_forest_max_depth',
                                      [None]),
               'min_samples_split': 3,  
                'min_samples_leaf': 1,
                'random_state': 1,# 1-10 Run x10 times 
            },
        ])

    def objective(self, params, x, y, k_fold):
        model_type = params.pop('type')
        print("Attempting model {}, parameters are {}".format(model_type, params))
        model_cls = self.type_model_map[model_type]

        mean_mse, mean_r2 = k_fold_cross_validation(model_cls, params, x, y, k_fold)
        print(
            "Finished model {}, parameters are {}, mean of MSEs is {}, mean of R2 is {}".format(
                model_type,
                params,
                mean_mse,
                mean_r2
            )
        )
        return mean_mse

    def optimise(self, x, y, num_rounds, k_fold=5):
        trials = Trials()
        best = fmin(
            partial(self.objective, x=x, y=y, k_fold=k_fold),
            space=self.space,
            algo=tpe.suggest,
            max_evals=num_rounds,
            trials=trials
        )
        best_params = space_eval(self.space, best)
        best_model_type = best_params.pop('type')
        print("Finished tuning, best model found is {}, best parameters are {}".format(best_model_type, best_params))
        best_model_cls = self.type_model_map[best_model_type]
        return best_model_cls, best_params
