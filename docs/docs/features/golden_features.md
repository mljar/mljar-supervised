# Golden Features

Golden Features are the features which have great predictive power. They can be constructed based on the original features. The common way to construct them is to try features differences or ratios. The [`mljar-superverised`](https://github.com/mljar/mljar-supervised) package has built-in step that performs Golden Features search. 

The procedure to find Golden Features:

- Generate all possible unique pairs of original features.
- If there is more than `250,000` pairs then subsample them randomly to `250,000`.
- For each pair of features construct a new feature with substract or division operators. 
- Based on the new feature train a `Decision Tree` with `max_depth = 3` (using only one feature). 
- For training there are used up to `2,500` samples randomly selected from the dataset. The same for testing, also up to `2,500` samples randomly selected.
- There is computed a score on test samples for each feature. The score is `logloss` metric for classification tasks, and `mean squared error` for regression tasks.
- Newly generated features are sorted based on the score (the lower score the better).
- As a Golden Features there are selected new features with smallest score values and are inserted into the training data.
- The number of Golden Features selected depends on number of original features. It is `5%` of original features number, but not less than `5` features and not more than `50` features.
- The results of Golden Features search is saved into `golden_features.json` file.
