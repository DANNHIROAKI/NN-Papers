# Biathlon
Harnessing Model Resilience for Accelerating ML Inference Pipelines

## Reproduce the results in the paper

### Environment setup
1. Setup Python Env (python>3.8)
2. Download Biathlon and set PYTHONPATH to include Biathlon
3. Install and setup [Clickhouse](https://clickhouse.com/docs/en/install). 
4. Download Datasets into the data dir of Clichouse (e.g., /var/lib/clickhouse/user_files/)
   1. [Trip-Fare](https://clickhouse.com/docs/en/getting-started/example-datasets/nyc-taxi)
   2. [Tick-Price](https://www.kaggle.com/datasets/joseserrat/forex-april-2020-to-june-2021-tick-data)
   3. [Battery](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset)
   4. [Turbofan](https://www.kaggle.com/datasets/msafi04/predict-rul-of-airplane-turbofan-dataset)
   5. [Fraud-Detection](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data)
   6. [Bearing-Imbalance](https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset)
   7. [Student-QA](https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data)

### Reproduce results of real pipelines
```bash
cd apxinfer/examples
python evaluate_all.py --interpreter ****/home/ckchang/anaconda3/envs/apx/bin/python --exp Trip-Fare --model lgbm --seed 0 --phase setup  # setup the database, train the model, run offline for profiling and analysis
python evaluate_all.py --interpreter /home/ckchang/anaconda3/envs/apx/bin/python --exp Trip-Fare --model lgbm --seed 0 --phase baseline  # run the pipeline in exact-manner
python evaluate_all.py --interpreter /home/ckchang/anaconda3/envs/apx/bin/python --exp Trip-Fare --model lgbm --seed 0 --phase ralf --ralf_budget 1.0  # run RALF with sepcified budget
python evaluate_all.py --interpreter /home/ckchang/anaconda3/envs/apx/bin/python --exp Trip-Fare --model lgbm --seed 0 --phase biathlon --default_only  ## run Biathlon with default configuration only
python evaluate_all.py --interpreter /home/ckchang/anaconda3/envs/apx/bin/python --exp Trip-Fare --model lgbm --seed 0 --phase biathlon  ## run Biathlon with all configurations (for plotting)
```

Note:
1. Remeber to use the correct python interpreter with Biathlon in PYTHONPATH
2. You can change --exp and --model to reproduce results for other real pipelines
  - `--exp Trip-Fare --model lgbm` for Trip-Fare pipeline
  - `--exp Tick-Price --model lr` for Tick-Price pipeline
  - `--exp Battery --model lgbm` for Battery pipeline
  - `--exp Turbofan --model rf` for Turbofan pipeline
  - `--exp Fraud-Detection --model xgb` for Fraud-Detection pipeline
  - `--exp Bearing-Imbalance --model mlp` for Bearing-Imbalance pipeline
  - `--exp Student-QA --model rf` for Student-QA pipeline
3. You are recommended to run multiple times with different seeds by specifing `--seed` in the command. In the paper, we run with five seeds from 0 to 4. Results with different seeds will be saved in different directories.
4. Evaluation results will be saved to `/home/ckchang/.cache/biathlon/vldb2024/final` by default, you can change this default path by changing the value of `EXP_HOME` in `apxinfer/core/config.py`.

### Reproduce the results for synthetic pipelines
We replaced the Average operators in real pipelines with Median Operators, to reproduce them, just add `-Median` after the pipeline name. For example, to reproduce the results for the Trip-Fare pipeline with Median operators, run the following command:
```bash
cd apxinfer/examples
python evaluate_all.py --interpreter /home/ckchang/anaconda3/envs/apx/bin/python --exp Trip-Fare-Median --model lgbm --seed 0 --phase setup
python evaluate_all.py --interpreter /home/ckchang/anaconda3/envs/apx/bin/python --exp Trip-Fare-Median --model lgbm --seed 0 --phase baseline 
python evaluate_all.py --interpreter /home/ckchang/anaconda3/envs/apx/bin/python --exp Trip-Fare-Median --model lgbm --seed 0 --phase biathlon --default_only
```

### Collect results
```bash
cd plotting
python vldb_collect.py --data_dir /home/ckchang/.cache/biathlon/vldb2024/final --out_dir ./cache --filename avg_final.csv
```
Notes:
1. You can change `--data_dir` if you changed the `EXP_HOME` when reproducing results.
 
### Plotting for real pipelines
```bash
cd plotting
python vldb_plotting.py --home_dir ./cache --filename avg_final.csv --score_type accuracy --cls_score f1 --reg_score r2
python vldb_plotting.py --home_dir ./cache --filename avg_final.csv --score_type similarity --cls_score f1 --reg_score r2
```
1. The accuracy metric is specified with options `--score_type accuracy --cls_score f1 --reg_score r2`, where the default one is also the default one in paper, i.e. measure accuracy based on true label. You can change them to other metrics like `--score_type similarity --cls_score f1 --reg_score r2` to plot for figures "vary confidence level" and "vary error bound", where we measure accuracy based on baseline's output.
2. The plotted figures with be saved in `figs_{score_type}_{cls_score}_{reg_score}`, you can also specify your own dir to save by `--plot_dir {your_path}`


### Plotting for Experiments on Median (Appendix)

Figure 11: Example Error Distribution of Median Feature
```bash
cd revision
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks tripsralfv3 tripsralfv3median --metric r2 --erid 20 --phase collect_erros
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks tickralfv2 tickralfv2median --metric r2 --erid 20 --phase collect_errors
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks batteryv2 batteryv2median --metric r2 --erid 20 --phase collect_errors
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks turbofan turbofanmedian --metric r2 --erid 20 --phase collect_errors
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks machineryralf machineryralfmedian --erid 20 --phase collect_errors
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks tdfraudralf2d tdfraudralf2dv2median --erid 20 --phase collect_errors
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks studentqnov2subset studentqnov2subsetmedian --erid 20 --phase collect_errors
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --phase final_errors
```

Figure 12: Original pipelines vs Pipelines with operators substituted by MEDIAN
```bash
cd revision
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks tripsralfv3 tripsralfv3median --metric r2 --erid 20 --phase e2e
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks tickralfv2 tickralfv2median --metric r2 --erid 20 --phase e2e
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks batteryv2 batteryv2median --metric r2 --erid 20 --phase e2e
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks turbofan turbofanmedian --metric r2 --erid 20 --phase e2e
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks machineryralf machineryralfmedian --erid 20 --phase e2e
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks tdfraudralf2d tdfraudralf2dv2median --erid 20 --phase e2e
sudo /home/ckchang/anaconda3/envs/apx/bin/python R2-W1-F3.py --tasks studentqnov2subset studentqnov2subsetmedian --erid 20 --phase e2e
```

Figure 13: Varying imbalance ratio for the median feature in Bearing-Imbalance pipeline.
```bash
cd revision
sudo /home/ckchang/anaconda3/envs/apx/bin/python R3-W2-F2.py --task_name machineryralf
```

Figure 14: Varying imbalance ratio for the median feature in Tick-Price pipeline.

```bash
cd revision
sudo /home/ckchang/anaconda3/envs/apx/bin/python R3-W2-F2.py --task_name tickralfv2 --selected_qid 6
```
