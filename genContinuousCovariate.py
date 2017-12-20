import pandas as pd

def generate_quantile(dataset, group_key, calcu_name, quant=[0.5]):
    return dataset.groupby(group_key)[calcu_name].quantile(quant).unstack()