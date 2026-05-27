# Automated Exploratory Data Analysis

AutoML provides automatic exploratory data analysis for the train data frame which is used to initialize AutoML. The EDA is done only if the `explain_level` parameter is set to a value of `2`. By default, it is set to 2 and will execute automatic eda if not modified.

For each variable in the input data frame the automated EDA finds the following:

- If numeric data, the result’s index will include count, mean, std, min, max as well as lower, 50, and upper percentiles. By default, the lower percentile is 25 and the upper percentile is 75. The 50 percentile is the same as the median. The probability distribution of the variable is also provided.

- If object data (e.g. strings or timestamps), the result’s index will include count, unique, top, and freq. The top is the most common value. The freq is the most common value’s frequency. Timestamps also include the first and last items. A bar graph depicting the number of samples in each category is provided for categorical data type and a word cloud is provided for the text data type.

All the results are saved into `README.md` file inside `EDA` folder.

## Extended Exploratory Data Analysis

 mljar additionally provides an `extended_eda` method that is capable of doing bivariate analysis. It takes a data frame and a target feature and provides the bivariate analysis of each feature in the data frame against the target variable.

- mljar additionally provides an extended_eda method that is capable of doing bivariate analysis. It takes a data frame and a target feature and provides the bivariate analysis of each feature in the data frame against the target variable.

- It also provides a heatmap depicting the Pearson correlation of continuous variables in the data frame.

All the results are saved into `Extensive_EDA.md` file inside the specified folder.

