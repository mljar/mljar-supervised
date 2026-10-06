# Universal Features Selection Procedure

Feature selection in `AutoML` is performed in two steps during fitting:

- `random_feature` - in which `AutoML` decides which features are important and which should be dropped.
- `features_selection` - training of the new models on selected features.

## `random_feature` step

In this step the following actions are performed:

- Select the best model so far and save its hyperparameters.
- Insert a `radnom_feature` to the dataset. The feature has uniform distribution from `0` to `1` range. 
- Train the model with the best hyperparameters on extended dataset. 
- Compute permutation-based feature importance for the new model. Because of using permutation-based feature importance this procedure can be applied to any Machine Learning algorithm.
- For each feature count how many times it has smaller importance than `random_feature`.
- If feature was less important at half of learners or more, then drop this feature. 

!!! note Importance counter
    There can be several learners in the model. For the example in 10-fold cross-validation, there will be 10 learners. So feature to be dropped must be less important for half or more of the learners (at least `5` times to be dropped).


## `features_selection` step

- If there are no features to be dropped then this step is skipped.
- There is selected the best model for each of the algorithm type: [`Xgboost`, `LightGBM`, `CatBoost`, `Neural Network`, `Random Forest`, `Extra Trees`]. The hyperparameters of the best models are copied and reused with selected features.
