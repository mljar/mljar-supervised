---
title: "Golden Features for Automated Feature Engineering"
description: "Learn how MLJAR AutoML generates golden features from differences and ratios of input columns, scores candidates, and adds selected features to training."
---
# Golden Features

Golden features are new input columns constructed from pairs of existing numeric
features. MLJAR AutoML tests differences and ratios, scores the candidates, and
adds selected features to the training data.

The separate [feature discovery step](feature_discovery.md) can run afterward,
sharing this step's time allocation and testing up to three Goldmine feature sets
on the best current models.

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
