from .trainers import *
models = ['DAGCT_BLS']
trainer_dict = {}

def get_trainer(model):
    if model == 'DAGCT_BLS':
        return DAGCT_BLS_Trainer
    else:
        raise ValueError

for model in models:
    trainer_dict[model] = get_trainer(model)
