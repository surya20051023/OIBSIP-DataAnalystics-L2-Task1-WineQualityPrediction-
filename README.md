# OIBSIP-DataAnalystics-L2-Task1-WineQualityPrediction-
# Wine Quality Prediction Using Machine Learning

## Project Overview

This project uses machine learning classification algorithms to predict wine quality based on its physicochemical properties. The dataset contains measurements such as fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, sulphates, alcohol, density, and pH.

The original wine quality scores are grouped into three categories: Low, Medium, and High. Three machine learning algorithms—Random Forest, Stochastic Gradient Descent (SGD), and Support Vector Classifier (SVC)—are trained and evaluated to identify a suitable model for wine quality classification.

## Objectives

- Analyze the Wine Quality dataset using exploratory data analysis (EDA).
- Identify relationships between chemical properties and wine quality.
- Visualize quality score distributions and feature correlations.
- Categorize wine quality into Low, Medium, and High classes.
- Train and compare three machine learning classification models.
- Evaluate model performance using standard classification metrics.
- Identify important features influencing wine quality predictions.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook

## Dataset

The project uses the Wine Quality dataset, which contains physicochemical measurements and quality ratings for wine samples.

**Dataset source:** [UCI Machine Learning Repository – Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

The red wine dataset typically contains 1,599 samples and 12 columns, including 11 input features and the original quality score.

## Machine Learning Algorithms

### 1. Random Forest Classifier
Random Forest combines multiple decision trees to classify wine quality. It can capture nonlinear relationships and provides feature importance scores.

### 2. Stochastic Gradient Descent (SGD) Classifier
SGD trains a classification model using iterative optimization. Feature scaling is applied to help the model learn effectively.

### 3. Support Vector Classifier (SVC)
SVC identifies decision boundaries between wine quality categories. Feature scaling helps ensure that differences in feature magnitudes do not dominate the model.

## Project Workflow

1. Load the Wine Quality dataset.
2. Inspect the dataset structure, data types, and missing values.
3. Perform exploratory data analysis.
4. Visualize the quality score distribution and feature correlations.
5. Convert numerical quality scores into Low, Medium, and High categories.
6. Split the dataset into training and testing sets using stratified sampling.
7. Train the Random Forest, SGD, and SVC models.
8. Evaluate each model using accuracy, precision, recall, F1-score, and confusion matrices.
9. Analyze Random Forest feature importance.
10. Compare model performance and identify the best-performing classifier.

## Wine Quality Categories

The original quality scores are grouped as follows:

- **Low Quality:** Score less than or equal to 4
- **Medium Quality:** Scores 5 and 6
- **High Quality:** Score greater than or equal to 7

These categories simplify the original numerical ratings into a three-class classification problem.

## Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy:** Proportion of correctly classified samples.
- **Precision:** Proportion of predicted positive-class samples that are correct, calculated per class.
- **Recall:** Proportion of actual samples correctly identified for each class.
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect predictions for each quality category.

Macro-averaged precision, recall, and F1-score are useful for comparing performance across categories, especially when the class distribution is imbalanced.

## Installation

Install the required Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## How to Run

1. Download the Wine Quality dataset from the UCI Machine Learning Repository.
2. Place the red wine CSV file in your project directory.
3. Open Jupyter Notebook.
4. Load the dataset with the correct separator for your file. For a comma-separated CSV, use:

```python
import pandas as pd

df = pd.read_csv("winequality-red.csv", sep=",")
```

For the original UCI file, which uses semicolons, use `sep=";"` instead.

5. Execute the notebook cells in order to perform data analysis, train the models, and compare their results.

## Expected Results

The project produces:

- Wine quality distribution plots.
- Feature distribution charts and a correlation heatmap.
- Classification reports and confusion matrices.
- A comparison table of the three trained models.
- Random Forest feature importance rankings.
- A selection of the best-performing model based on evaluation results.

Actual performance values should be reported from the executed notebook rather than assumed in advance.

## Applications

- Educational demonstrations of machine learning classification.
- Analysis of relationships between wine chemistry and quality ratings.
- Decision-support research for wine quality assessment.
- Comparison of classification algorithms on structured datasets.

## Conclusion

This project demonstrates how machine learning can classify wine quality using physicochemical measurements. By comparing Random Forest, SGD, and SVC, the project evaluates different approaches to a multiclass classification problem. Exploratory data analysis and feature importance provide additional insight into the dataset. The final model should be selected using the measured test-set results, with particular attention to macro F1-score and performance across all quality categories.

## References

- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/186/wine+quality
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Pandas documentation: https://pandas.pydata.org/docs/
- Seaborn documentation: https://seaborn.pydata.org/
