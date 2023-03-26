
# UCI Heart Disease Prediction

### Software-Hackathon
Code and relevant report submission for the software hackathon in AI/ML at IIT Ropar's ZEITGEIST.

### Overview
- Cardiovascular disease (CVD) is the leading cause of death worldwide. Early detection and accurate prediction of CVD can help prevent serious complications such as heart attacks and strokes.

- We have developed a machine learning model that can predict the likelihood of a heart patient experiencing a heart attack.

- The developed machine learning model is evaluated using appropriate performance metrics, such as accuracy, sensitivity, specificity, and area under the receiver operating characteristic (ROC) curve.

- The model is able to take in a set of input features for a given patient and output a probability or score indicating the likelihood of a heart attack occurring within a specific time frame.

### Dataset
- The **dataset.csv** dataset from UCI contains various features related to the health of individuals. The dataset has a total of 303 instances, with 13 input features including age, sex, chest pain type, resting blood pressure, serum cholesterol level, fasting blood sugar level, electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise relative to rest, slope of the peak exercise ST segment, number of major vessels colored by fluoroscopy, and thallium stress test results.

- The target attribute of the dataset is "num," where the dataset is classified into 5 classes, namely 0, 1, 2, 3, and 4, based on the severity of getting heart disease. However, since the size of the dataset is not large enough to cluster data into 5 classes with good accuracy, it has been converted into binary classification 0 and 1, indicating the absence or presence of heart disease, respectively.

- The source of the dataset is the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease).

*We have provided the updated dataset for reference which is a combination of the datasets referenced above.*

### Models

We have used following models for clustering of target variable with their corresponding Accuracy score, Sensitivity score, Specificity score and ROC-AUC score:

| Models            | Accuracy | Sensitivity | Specificity |  ROC AUC |
| :--------------   | :------- | :-------    |:----------  |:-------|
| **Random Forest**    | **0.95** |   **0.93** | **0.96** | **0.97**|
| Gradient Boost    | 0.90 |   0.93  | 0.85 | 0.95|
|Adaboost           | 0.90 |   0.93  | 0.85 | 0.92|
|Extra Trees Classifier| 0.90 | 0.96 | 0.82 | 0.97|
|Decision Trees     | 0.81 | 0.93 | 0.67 | 0.85|
|Gaussian Naive Bayes| 0.86 | 0.93 | 0.78 | 0.91 |
|Support Vector Machine| 0.91 | 0.93 | 0.89 | 0.94 |


### Final Function
- The function heart_disease_classification() takes in two arguments `x_test` and `model`.

- `x_test` is a list of 13 parameters representing the input features for predicting the presence or absence of heart disease.

- `model` is a machine learning model, specifically a RandomForestClassifier model, trained on a dataset of input features and their corresponding output labels.

- If the length of `x_test` is not equal to 13, the function prints a message indicating that the list should contain 13 parameters and returns.

- If the length of `x_test` is equal to 13, the function uses the `predict_proba()` method of the model object to obtain the probability of the input features belonging to each class - presence or absence of heart disease.

- The probabilities are printed as output, along with a message indicating whether the probabilities correspond to the presence or absence of heart disease.

**Note that the function does not return the predicted label for the input features, only the probabilities.**

#### Overall, the function provides a quick way to obtain the probability of heart disease given a set of input features using a pre-trained machine learning model.

### Conclusion
The developed machine learning model provides a promising approach to predict the likelihood of a heart patient experiencing a heart attack. The model is trained and evaluated using appropriate performance metrics, achieving high accuracy, sensitivity, specificity, and ROC AUC scores. The model takes in a set of input features for a given patient and outputs a probability or score indicating the likelihood of a heart attack occurring within a specific time frame. The heart_disease_classification() function provides a convenient way to obtain the probability of heart disease given a set of input features using a pre-trained machine learning model. This model and function can potentially be used as a tool to assist healthcare professionals in early detection and prevention of cardiovascular disease, leading to improved patient outcomes and quality of life.

**Note:** The finalized model is **RandomForestClassifier** which is selected on the basis of it's accuracy score.
