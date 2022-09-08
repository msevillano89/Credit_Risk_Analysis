# Credit Risk Analysis
## Purpose
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. The purpose of this analysis is to build and evaluate different machine learning models to predict credit risk. We will use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.
## Analysis Overview
1. Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we will oversample the data using the **RandomOverSampler** and **SMOTE algorithms**, and undersample the data using the **ClusterCentroids algorithm**.
2. We will use a combinatorial approach of over- and undersampling using the **SMOTEENN algorithm**.
3. We will compare two new machine learning models that reduce bias, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**, to predict credit risk. O
4. Finally, we will evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.
## Resampling Models to Predict Credit Risk
### Results
#### Naive Random Oversampling
- Balanced Accuracy Score
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred

Output: 0.6471469838161886
```
- Classification Report:

#### SMOTE Oversampling
- Balanced Accuracy Score
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred

Output: 0.653361454568895
```
- Classification Report:

#### Cluster Centroids Undersampling
- Balanced Accuracy Score
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred

Output: 0.5158443824004577
```
- Classification Report:

#### SMOTEENN Combination (Over and Under) Sampling
- Balanced Accuracy Score
```
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test)
balanced_accuracy_score(y_test, y_pred

Output: 0.6375533316412246
```
- Classification Report:

### Summary
- The SMOTE Oversampling method had the best balanced accuracy score at 65.34%, and recall score at 88%.
- The SMOTEENN Combination method had the best geometric mean score at 79%.

## Ensemble Classifiers to Predict Credit Risk
### Results
#### Balanced Random Forest Classifier
- Balanced Accuracy Score
```
# Calculated the balanced accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
predictions = model.predict(X_test_scaled)
accuracy_score(y_test, predictions)

Output: 0.9254867770996803
```
- Classification Report:

#### Easy Ensemble AdaBoost Classifier
- Balanced Accuracy Score
```
# Calculated the balanced accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
predictions = model.predict(X_test_scaled)
accuracy_score(y_test, predictions)

Output: 0.9426910781749491
```
- Classification Report:

### Summary
Overall the Easy Ensemble AdaBoost Classifier outperformed the Balanced Random Forest Classifier and previously mentioned resampling methods:
- It had the best balanced accuracy score at 94%.
- Best recall score at 94%.
- And, best geometric mean score at 93%
