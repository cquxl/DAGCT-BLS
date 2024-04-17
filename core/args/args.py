from .dgact_bls_args import DAGCT_BLS_GenArgs


def gen_model_args(model):
    if model == 'DAGCT_BLS':
        return DAGCT_BLS_GenArgs
    else:
        raise ValueError

gen_args = {}
models = ['DAGCT_BLS']
for model in models:
    gen_args[model] = gen_model_args(model)