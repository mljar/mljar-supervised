# API

```python
automl = AutoML()
automl.fit(X, y)
predictions = automl.predict(X)
```


The parameters that you can use to control the training process are:

- **total_time_limit** - it is a total time limit that AutoML can spend for searching to the best ML model. It is in seconds. _Default is set to 3600 seconds._
- **learner_time_limit** - the time limit for training single model, in case of `k`-fold cross validation, the time spend on training is `k*learner_time_limit`. This parameter is only considered when `total_time_limit` is set to None. _Default is set to 120 seconds_.
- **algorithms** - the list of algorithms that will be checked. _Default is set to ["CatBoost", "Xgboost", "RF", "LightGBM", "NN"]_.
- **start_random_models** - the number of models to check with _not so random_ algorithm. _Default is set to 10_.
- **hill_climbing_steps** - number of hill climbing steps used in models tuning. _Default is set to 3_.
- **top_models_to_improve** - number of models considered for improvement in each hill climbing step. _Default is set to 5_.
- **train_ensemble** - decides if ensemble model is trained at the end of AutoML fit procedure. _Default is set to True_.
- **verbose** - controls printouts, _Default is set to True_.

