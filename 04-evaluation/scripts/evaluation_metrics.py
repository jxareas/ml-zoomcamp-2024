# # 4. Evaluation Metrics for Classification
#
# In the previous session we trained a model for predicting churn. How do we know if it's good?
#
#
# ## 4.1 Evaluation metrics: session overview
#
# * Dataset: https://www.kaggle.com/blastchar/telco-customer-churn
# * https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv
#
#
# *Metric* - function that compares the predictions with the actual values and outputs a single number that tells how good the predictions are

# %% Importing libraries


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from fontTools.ttLib.tables.otBase import CountReference
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

# In[5]:

data_path = './04-evaluation/data/data-week-3.csv'
df = pd.read_csv(data_path)

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

# In[6]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

# In[7]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

# In[8]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)

# In[9]:


val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()

# %% 4.2 Accuracy and dummy model

# Total number of validation records
len(y_val)

# Accuracy
accuracy = (y_val == churn_decision).mean()
print(f"{accuracy=}")

accuracy_scores = []

# Changing the threshold
thresholds = np.linspace(start=0, stop=1, num=21)
for t in thresholds:
    accuracy = accuracy_score(y_val, y_pred >= t)
    print("%.2f threshold : %.3f accuracy" % (t, accuracy))
    accuracy_scores.append(accuracy)

# Decision Threshold vs Accuracy
plt.style.use('ggplot')
plt.plot(thresholds, accuracy_scores)
plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=1)
plt.xticks(np.arange(0, 1.1, 0.1))
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.title("Decision Threshold vs Accuracy Score", fontdict={'fontweight': 'bold'})
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.savefig('./m3_threshold_v_accuracy.png', format='png')
plt.show()

# Count all the values whose prediction is greater than 1, which turns out to be 0
Counter(y_pred >= 1.0)

# %% 4.3 Confusion table

# Implementing the confusion table (TP, TN, FP, FN) manually

actual_positive = y_val == 1
actual_negative = y_val == 0

t = 0.5
predict_positive = y_pred >= t
predict_negative = y_pred < t

tp = sum(predict_positive & actual_positive)
tn = sum(predict_negative & actual_negative)
fp = sum(predict_positive & actual_negative)
fn = sum(predict_negative & actual_positive)

confusion_table = np.array([
    [tn, fp],
    [fn, tp],
])
print(confusion_table)

confusion_percentages = confusion_table / confusion_table.sum()

print(confusion_percentages.round(2))

# %% 4.4. Precision and Recall

# Printing the accuracy -> the percentage of correct predictions
print((tp + tn) / (tp + tn + fp + fn))

# Printing the precision -> the percentage of correct positive predictions AKA positive predictive value
precision = tp / (tp + fp)
print(precision)

# Printing the recall -> percentage of correctly identified positive observations AKA true positive rate
recall = tp / (tp + fn)
print(recall)
