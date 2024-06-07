import os
import pandas as pd
import sys

data_path = sys.argv[1]
metrics_path = sys.argv[2]

for target in os.listdir(data_path):
    if target.startswith('T'):
        target_path = os.path.join(data_path, target)
        if target + '.csv' not in os.listdir(metrics_path):
            continue
        print(target)
        metrics_df = pd.read_csv(os.path.join(metrics_path, target + '.csv'), sep='\t')
        available_models = set(metrics_df['model'])
        model_to_cad = dict(metrics_df[['model', 'cad']].values)
        for model in os.listdir(target_path):
            if model in available_models:
                print('\t', model)
                with open(os.path.join(target_path, model, 'y_global.txt'), 'w') as f:
                    f.write(str(model_to_cad[model]))
