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
 This stage included a <b>statistical analysis</b> and <b>biological query</b> of the features.  The former included evaluation of variances `df.var()` and generation of pairplots to see the interactions between all PASI scores throughout treatment. <br />
 `import seaborn as sns` <br />
`g = sns.pairplot(df)` <br />
`g.fig.suptitle("        Pairplot PASI Weeks 0-7")` <br />
 The latter included psorasis research to infer the features that held significant biological direction. 
 
 
 
 
