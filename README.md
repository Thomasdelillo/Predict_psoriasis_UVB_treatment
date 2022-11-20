# Predicting Psoriasis UVB Phototherapy Treatment Trajectories with Machine Learning

  ## Introduction
- Psoriasis is a skin disease that leaves the skin red, flaky, and patchy.  UVB phototherapy treatment can reduce symptoms greatly, but there are no clinical approaches to estimate its success before starting treatment.  This means the general course of UVB phototherapy treatment usually commences with a "wait and see" approach, leading patients and practitioners through some uncertainty.
- Machine learning and artificial intelligence can facilitate treatment plans through highly accurate and confident models, where implementation of these algorithms can predict the chances of treatment success through patient baseline measurements and early treatment responses.  
- The goal of these models is to predict how well treatment will work for patients before they start any treatment.
 
 ## The Dataset <br />
 ```python
 import pandas as pd
 df = pd.read_csv('psoriasis_data.csv')
 df
 ```
 
 > <img src="/blurred_dataset.png" width="500">
 - The <b>confidential</b> dataset includes ≈200 psoriasis patients undergoing UVB phototherapy in Tyne and Wear, England.
 - The baseline features include biological metrics including age, sex, gender, baseline MED, skin type, presence of psoriasis family history, and more.
 - The time-series features include weekly PASI scores after every week of treatment. PASI = psoriasis area and severity index; an evaluation of psoriasis severity ranging from 0 (no disease) to 72 (severe disease).
 - In the classification models, the target feature is a class-label indicative of treatment success (computed by other machine learning models).
 - In the regression models, the target feature is the end of treatment PASI score.
 
 ## Data Preprocessing and Feature Selection <br />
 - Commenced with a <b>statistical analysis</b> and <b>biological query</b> of the features.  The former spanned from measures of central tendency to construction of pairplots to view the interactions between PASI scores throughout treatment. <br />
 ```python
 import seaborn as sns
g = sns.pairplot(df)
g.fig.suptitle("        Pairplot PASI Weeks 0-7")
```
> <img src="/pairplot.jpg" width = "300"> <br />
- This stage provided a set of significant features to integrate into the models, which <i>should</i> hypothetically yield the strongest results.

## Model Engineering & Design
- Because the goal was to predict how well treatment would work before treatment, only significant baseline features were used initially.
- After engineering models with only baseline features and assessing their performance, I started to incorporate the time-series PASI scores to possibly enhance the models.
- To yield the best results, "insignificant" features from the feature selection stage were implemented through the Wrapper Method, where model error analyses drive feature selection.  In other words, different combinations of features were added/removed on a trial and error basis.

## Model Execution
- The features went through the Random Forest, Support Vector Machine, and XGBoost algorithms using 5-fold cross validation and `GridSearchCV()` for hyperparameter tuning.  

### GridSearchCV()
  <p align = "center">
 <img src="/5-fold_cross_validation.png" width="500">
  </p>
 
 ```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
 df = pd.read_csv('psoriasis_data.csv')
 df = df.dropna()
X = df.drop(columns = ['CLASS'])
y = df['CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
```
 
```python

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 250, num = 10)]
max_features = ['auto', 'sqrt']
min_samples_split = [2, 5]
min_samples_leaf = [1, 2]
max_depth = [2, 4, 8, 16, 32, None]
bootstrap = [True, False]
```
```python

param_grid = {'n_estimators': n_estimators,
             'max_features': max_features,
             'min_samples_split': min_samples_split,
             'min_samples_leaf': [1, 2],
              'max_depth': max_depth,
              'bootstrap': bootstrap}
 ```
 ```python     
 model = RandomForestClassifier()
BestModel = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, verbose = 2, n_jobs = 4)
BestModel.fit(X_train, y_train)
print(BestModel.score(X_train, y_train))
> 1.0
```
- The perfect score of 1.0 is indicative of overfitting, possibly as a result from tuning too many parameters.
- Further GridSearchCV() methods only tuned 1 or 2 parameters with more appropriate results.  

### 5-Fold Cross Validation with the SVM Algorithm: <br />
<p align = "center">
<img src="/svm.png" width="500"> <br />
  </p>
```python
from sklearn import svm
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, RepeatedKFold
model2 = svm.SVC(kernel = 'linear')
kf = RepeatedKFold(n_splits = 5, n_repeats = 10000)
b = 0, d = 0, f = 0, h = 0
for train_index, test_index in kf.split(X, y):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    a = accuracy_score(y_test, predictions)
    b += a
    c = recall_score(y_test, predictions, average = 'macro')
    d+=c
    e = precision_score(y_test, predictions, average = 'macro')
    f += e
    g = f1_score(y_test, predictions, average = 'macro')
    h += g
print("The average accuracy across the 5 folds is " + str(b/50000))
print('The average recall across the iterations is ' + str(d/50000))
print('The average precision across the iterations is ' + str(f/50000))
print('The average f1 score across the iterations is ' + str(h/50000))
> The average accuracy across the 5 folds is 0.84325
> The average recall across the iterations is 0.7949863414363416
> The average precision across the iterations is 0.8073479825729826
> The average f1 score across the iterations is 0.7815214681103407
```

## Results
- Each classification model was analyzed using accuracy, precision, recall, and F1 scores.  The last 3 were especially important as the classes were imbalanced as seen below: <br />
<p align = "center">
<img src="/Imbalanced%20Classes.png" width="250"> <br />
</p> 
<br />
- The <b>baseline features</b> alone did not generate suitable models to predict UVB phototherapy outcomes: the accuracies reached 60-70%, but their complementary confidence metrics (precision, recall, and F1 scores) fell between 40 and 50%, indicating that model performance was compromised by the dataset's imbalanced classes.
- Surprisingly, the best models across all algorithms did not include any baseline biological features, but rather only the PASI treatment scores, where even addition of the first week PASI scores increased performance considerably.  The graph below demonstrates this, where the x-axis is the successive week by week inclusion of PASI treatment scores.  0 = week 0, 1 = week 0 + week 1, 2 = week 0 + week 1 + week 2, etc.
<p align = "center">
<img src="/graph1.png" width="500"> <br />
<img src="/graph2.png" width="500"> <br />
  
 </p>
 <br />
 
 - The figure below displays a similar trend with another partition of the dataset, however the accuracy remains fairly consistent while the confidence metrics increase with the addition of more treatment weeks:
 <p align = "center">
 <img src="/graph3.png" width="500"> <br />
  </p> 
  <br />
  
 - The regressor models did not perform as well, even with the addition of PASI treatment weeks:
  


              
              


  
 
 
 
 
