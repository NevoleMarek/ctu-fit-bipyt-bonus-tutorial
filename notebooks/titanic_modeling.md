---
jupyter:
  jupytext:
    cell_metadata_filter: -all,tags
    notebook_metadata_filter: kernelspec,jupytext
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: bonus-tutorial
    language: python
    name: bonus-tutorial
---

```python
import pickle
import warnings

import pandas as pd
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
```

```python
# https://www.kaggle.com/datasets/yasserh/titanic-dataset
df = pd.read_csv("../data/titanic.csv")

df_train, df_test = train_test_split(df, test_size=0.3, random_state=41)
```

```python
df.info()
```

```python
df.head()
```

# Preprocessing

```python
# get rid of PassengerId as it is just arbitrary
df = df.drop(columns=["PassengerId"])
```

## Converting categorical variables

```python
# replace "male" and "female" strings to numbers
df = df.assign(Sex=lambda df_: df_["Sex"].replace({"male": 0, "female": 1}))
```

```python
# replace name with name length
df = df.assign(NameLength=lambda df_: df_["Name"].str.len()).drop(columns=["Name"])
```

```python
# one hot encode Embarked, why?
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')
```

```python
# get rid of ticket column as it would be difficult to get meaningful information
df = df.drop(columns=["Ticket"])
```

```python
df.head()
```

## Handle missing values

```python
df.isna().sum()
```

```python
# drop Cabin column due to high number of missing values
df = df.drop(columns=["Cabin"])
```

```python
fig = px.histogram(df, x="Age")
fig.update_layout(title="Passengers' age distribution")
fig.show()
```

```python
# fillna of Age and Embarked with averages, discuss other options?
df = df.fillna({"Age": df["Age"].mean()})
```

```python
df.info()
```

## We were cheating a bit

```python
X_train, y_train = df_train.drop(columns=["Survived"]), df_train["Survived"]
X_test, y_test = df_test.drop(columns=["Survived"]), df_test["Survived"]
```

```python
class TitanicDataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Calculate mean age during fitting
        self._age_mean = X['Age'].mean()
        return self

    def transform(self, X):
        return (
            X.drop(columns=["PassengerId"])
            .assign(Sex=lambda df_: df_["Sex"].replace({"male": 0, "female": 1}))
            .assign(NameLength=lambda df_: df_["Name"].str.len())
            .drop(columns=["Name"])
            .pipe(pd.get_dummies, columns=['Embarked'], prefix='Embarked')
            .drop(columns=["Ticket"])
            .drop(columns=["Cabin"])
            .fillna({"Age": self._age_mean})
        )
```

```python
titanic_data_preprocessor = TitanicDataPreprocessor().fit(X_train)
X_train = titanic_data_preprocessor.transform(X_train)
X_test = titanic_data_preprocessor.transform(X_test)
```

```python
X_train.head()
```

# Feature selection

```python
df_train_prep = titanic_data_preprocessor.transform(df_train)

for feature in df_train_prep.drop(columns=["Survived"]).columns:
    fig = px.histogram(
        df_train_prep,
        x=feature,
        color="Survived",
        color_discrete_map={0: 'orange', 1: 'blue'},
        barnorm='percent',
        barmode='group',
        opacity=0.7,
        title=f'Histogram of {feature.capitalize()} colored by Survival',
    )
    fig.update_layout(bargap=0.05)
    fig.show()
```

```python
correlations = X_train.corrwith(y_train)
correlations
```

```python
class TitanicDataFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        correlations = X.corrwith(y)
        self._selected_feature_columns = list(correlations.abs().nlargest(4).index)
        return self

    def transform(self, X):
        return X[self._selected_feature_columns]
```

```python
titanic_data_feature_selector = TitanicDataFeatureSelector().fit(X_train, y_train)
```

```python
X_train = titanic_data_feature_selector.transform(X_train)
X_test = titanic_data_feature_selector.transform(X_test)
```

```python
X_train.head()
```

# Modeling

```python
clf = DecisionTreeClassifier().fit(X_train, y_train)
```

```python
accuracy_score(clf.predict(X_test), y_test)
```

```python
from sklearn import tree
from matplotlib import pyplot as plt

plt.figure(figsize=(60, 60))
           
tree.plot_tree(clf)
plt.savefig("../models/tree.png")
```

```python
df = pd.read_csv("../data/titanic.csv")

df_train, df_test = train_test_split(df, test_size=0.3, random_state=41)
X_train, y_train = df_train.drop(columns=["Survived"]), df_train["Survived"]
X_test, y_test = df_test.drop(columns=["Survived"]), df_test["Survived"]

pipeline = Pipeline(
    [
        ('preprocessor', TitanicDataPreprocessor()),
        ('feature_selector', TitanicDataFeatureSelector()),
        ('classifier', DecisionTreeClassifier()),
    ]
)

titanic_classification_pipeline = pipeline.fit(X_train, y_train)
```

```python
def save_model(model, path: str):
    """Saves model as pickle file"""
    with open(path, 'wb') as file:
        pickle.dump(model, file)
```

```python
save_model(titanic_classification_pipeline, "../models/titanic_classification_pipeline.pkl")
```

```python

```
