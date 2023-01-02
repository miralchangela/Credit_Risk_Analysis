# Credit Risk Analysis

## Overview of the analysis:

- The credit risk analysis using machine learning algorithms to identify credit card risk using a dataset from LendingClub.

- The purpose of this analysis is to understand how to utilize machine learning algorithms to make predictions based on data patterns given.

- Here, we focused on the supervised learning  using LeadingClub dataset.

- For this analysis, we use different machine learning techniques to train and test the data unbalanced classes.

- The dataset has unbalanced classification problem due to the number of good loans outweighing the amount of risks loans.

- To balance out the classifications and improve the accuracy score, we needed different machine learning algorithms to re sample the data. 

- In this analysis, we used randomOversample , SMOTE , ClusterCentroid , SMOTEENN , BalancedRandomForestClassifier and EasyEnsembleclassifier.

## Results:

### Resampling  Models to Predict Credit Risk:

- Using the knowledge of the imbalanced-learn and scikit-learn libraries to evaluate three machine learning models by using resampling to determine which is better at predicting credit risk.

1.Random Over Sampler
    
RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.

!(naive_random_sampling)[https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/naive_random_sampling.png]

- Balanced accuracy score: 65.20%.

![random_oversampling_balanced_accuracy](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/random_oversampling_balanced_accuracy.png)

- confusion matrix

![random_oversampling_confusion_matrix](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/random_oversampling_confusion_matrix.png)

- The "High Risk" precision rate was only 1% with the recall at 75% giving this model an F1 score of 2%.
- "Low Risk" had a precision rate of 100% and recall at 55%.

![random_oversampling_imbance_classifier_report](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/random_oversampling_imbance_classifier_report.png)


2.SMOTE 

 SMOTE (Synthetic Minority Oversampling Technique) Model, like RandomOverSampler increases the size of the minority class by creating new values based on the value of the closest neighbours to the minority class instead of random selection.

![smote_oversampling_model](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/smote_oversampling_model.png)

- The balanced accuracy score improved slightly to 65.81%.

![smote_oversampling_balanced_accuracy](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/smote_oversampling_balanced_accuracy.png)

- Confusion Metrix:

![smote_oversampling_confusion_matrix](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/smote_oversampling_confusion_matrix.png)

- Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 63% giving this model an F1 score of 2%.
- "Low Risk" had a precision rate of 100% and an improved recall at 68%.

![smote_oversampling_imbalanced_classifier_report](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/smote_oversampling_imbalanced_classifier_report.png)

3.Clustered Centroids

ClusterCentroids Model, an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk.

 ![undersampling_model](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/undersampling_model.png)

- Balanced accuracy score was lower than the oversampling models at 54.47%.

![undersampling_model_balanced_accuracy](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/undersampling_model_balanced_accuracy.png)

- Confusion Metrix:

![undersampling_model_confusion_matrix](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/undersampling_model_confusion_matrix.png)

- The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
- "Low Risk" had a precision rate of 100% and with a lower recall at 40% compared to the oversampling models.

![undersampling_model_imbalanced_classification_report](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/undersampling_model_imbalanced_classification_report.png)

4.SMOTEENN

SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model combines aspects of both oversampling and undersampling. The model classified 68,460 records as High Risk and 62,011 as Low Risk.

![smoteenn_model](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/smoteenn_model.png)

- The balanced accuracy score improved to 64.5% when using a combined sampling model.

![smoteenn_balanced_accuracy](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/smoteenn_balanced_accuracy.png)

- Confusion Metrix:

![smoteenn_confusion_metrix](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/smoteenn_confusion_metrix.png)

- The "High Risk" precision rate did not improve was only 1%, however the recall increased to 72% giving this model an F1 score of 2%.
- "Low Risk" still showed a precision rate of 100% with the recall at 57%.

![smoteenn_imbalanced_classification_report](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/smoteenn_imbalanced_classification_report.png)


### Ensemble Classifiers to Predict Credit Risk

Compare two new Machine Learning models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.

1. Balanced Random Forest Classifier

BalancedRandomForestClassifier Model, two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.

![balanced_random_forest_classifier_model](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/balanced_random_forest_classifier_model.png)

- The balanced accuracy score increased to 78.85% for this model.

![BRFC_balanced_accuracy](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/BRFC_balanced_accuracy.png)

- Confusion Metrix:

![BRFC_confusion_matrix](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/BRFC_confusion_matrix.png)

- The "High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
- "Low Risk" still had a precision rate of 100% with the recall at 87%.

![BRFC_imbalanced_classifier_report](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/BRFC_imbalanced_classifier_report.png)

2. Easy Ensemble Classifier

EasyEnsembleClassifier Model, a set of classifiers where individual decisions are combined to classify new examples.

![easy_ensemble_adaboost_classifier_model](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/easy_ensemble_adaboost_classifier_model.png)

- The balanced accuracy score increased to 92.01% with this model.

![EAC_balanced_accuracy](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/EAC_balanced_accuracy.png)

- Confusion Metrix:

![EAC_confusion_matrix](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/EAC_confusion_matrix.png)

- The "High Risk precision rate increased to 9% with the recall at 89% giving this model an F1 score of 17%.
- "Low Risk" still had a precision rate of 100% with the recall now at 95%.

![EAC_imbalanced_classifier_report](https://github.com/miralchangela/Credit_Risk_Analysis/blob/main/resources/images/EAC_imbalanced_classifier_report.png)

## Summary:

 The EasyEnsembleClassifer model yielded the best results with an accuracy rate of 92.01% and a 9% precision rate when predicting "High Risk" candidates. The sensitivity rate (aka recall) was also the highest at 89% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 95% and an F1 score of 97%. Therefore, if a model needed to perform this type of analysis, then
 Easy Ensemble Classifier model would be the perfect choice.

Ranking of models in descending order based on "High Risk" results:

- EasyEnsembleClassifer: 92.01% accuracy, 9% precision, 89% recall, and 17% F1 Score
- BalancedRandomForestClassifer: 78.85% accuracy, 3% precision, 70% recall and 6% F1 Score
- SMOTE: 65.81% accuracy, 1% precision, 63% recall and 2% F1 Score
- RandomOverSampler: 65.20% accuracy, 1% precision, 75% recall and 2% F1 Score
- SMOTEENN: 64.49% accuracy, 1% precision, 72% recall and 2% F1 Score
- ClusterCentroids: 54.47% accuracy, 1% precision, 69% recall and 1% F1 Score

