# Predicting Psoriasis UVB Phototherapy Treatment Outcomes

  **Introduction**
- Psoriasis is a skin disease that leaves the skin red, flaky, and patchy.  UVB phototherapy treatment can reduce symptoms greatly, but there are no clinical approaches to estimate its success before starting treatment.  This means the general course of UVB phototherapy treatment usually commences with a "wait and see" approach, leading patients and practitioners through some uncertainty.
- Machine learning and artificial intelligence can facilitate treatment plans through highly accurate and confident models, where implementation of these algorithms can predict the chances of treatment success through patient baseline measurements and early treatment responses.  
- The goal of these models is to predict how well treatment will work for patients before they start any treatment.
 
 **The Dataset** <br />
 `import pandas as pd` <br />
 `df = pd.read_csv('psoriasis_data.csv')` <br />
 `df` <br />
 <img src="/blurred_dataset.png" width="500">
 - The <b>confidential</b> dataset includes ≈200 psoriasis patients undergoing UVB phototherapy in Tyne and Wear, England.
 - The baseline features include biological metrics including age, sex, gender, baseline MED, skin type, presence of psoriasis family history, and more.
 - The time-series features include weekly PASI scores after every week of treatment. PASI = psoriasis area and severity index; an evaluation of psoriasis severity ranging from 0 (no disease) to 72 (severe disease).
 - In the classification models, the target feature is a class-label indicative of treatment success (computed by other machine learning models).
 - In the regression models, the target feature is the end of treatment PASI score.
 
 **Data Preprocessing and Feature Selection** <br />
 - Commenced with a <b>statistical analysis</b> and <b>biological query</b> of the features.  The former spanned from measures of central tendency to construction of pairplots to view the interactions between PASI scores throughout treatment. <br />
 `import seaborn as sns` <br />
`g = sns.pairplot(df)` <br />
`g.fig.suptitle("        Pairplot PASI Weeks 0-7")` <br />
<img src="/pairplot.jpg" width = "300"> <br />
- This stage provided a set of significant features to integrate into the models, which <i>should</i> hypothetically yield the strongest results.

**Model Engineering & Design**
- Because the goal was to predict how well treatment would work before treatment, only significant baseline features were used initially.
- After engineering models with only baseline features and assessing their performance, I started to incorporate the time-series PASI scores to possibly enhance the models.
- To yield the best results, "insignificant" features from the feature selection stage were implemented through the Wrapper Method, where model error analyses drive feature selection.  In other words, different combinations of features were added/removed on a trial and error basis.

**Model Execution**
- The features went through the Random Forest, Support Vector Machine, and XGBoost algorithms using 5-fold cross validation and `GridSearchCV()` for hyperparameter tuning.  Below is an example of GridSearchCV(): <br />
     <br />
     
`import pandas as pd` <br />
`from sklearn.ensemble import RandomForestClassifier` <br />
`from sklearn.model_selection import train_test_split` <br />
`from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score` <br />
`from sklearn.model_selection import GridSearchCV` <br/>
 `df = pd.read_csv('psoriasis_data.csv')` <br />
 `df = df.dropna()` <br />
`X = df.drop(columns = ['CLASS'])` <br />
`y = df['CLASS']` <br />
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)` <br />

`from sklearn.ensemble import RandomForestClassifier` <br />

`n_estimators = [int(x) for x in np.linspace(start = 10, stop = 250, num = 10)]`<br />
`max_features = ['auto', 'sqrt']`<br />
`max_depth = [2, 4, 8, 16, 32, None]`<br />
`bootstrap = [True, False]`<br />

`param_grid = {'n_estimators': n_estimators,
             'max_features': max_features,
              'max_depth': max_depth,
              'bootstrap': bootstrap}` <br />
              
 `model = RandomForestClassifier()` <br />
 
`BestModel = GridSearchCV(estimator = model, param_grid = param_grid, cv = 5, verbose = 2, n_jobs = 4)` <br/>

`BestModel.fit(X_train, y_train)` <br/>

`print(BestModel.score(X_train, y_train))` <br/>
> 1.0
              
              


  
 
 
 
 
