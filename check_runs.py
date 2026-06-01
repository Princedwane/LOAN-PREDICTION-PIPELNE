import mlflow

runs = mlflow.search_runs(experiment_names=["Loan_Pipeline"])

print(runs[[
    'tags.mlflow.runName',
    'start_time',
    'metrics.accuracy',
    'metrics.rows_before',
    'metrics.rows_after',
    'params.model_type',
    'params.test_size'
]].to_string())

#python check_runs.py to see the metrics