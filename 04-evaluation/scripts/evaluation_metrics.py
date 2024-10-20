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


import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from tqdm.auto import tqdm

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

# %% ROC Curves
tpr = tp / (tp + fn)
print(f"False positive rate {tpr=}")

fpr = fp / (fp + tn)
print(f"False positive rate {fpr=}")

actual_positive = y_val == 1
actual_negative = y_val == 0

thresholds = np.linspace(0, 1, 101)
scores = []

for t in thresholds:
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = sum(predict_positive & actual_positive)
    tn = sum(predict_negative & actual_negative)
    fp = sum(predict_positive & actual_negative)
    fn = sum(predict_negative & actual_positive)
    scores.append((t, tp, fp, fn, tn))

print(scores)

columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['tpr'] = df_scores['tp'] / (df_scores['tp'] + df_scores['fn'])
df_scores['fpr'] = df_scores['fp'] / (df_scores['fp'] + df_scores['tn'])

# %% Plotting
plt.style.use('ggplot')
plt.plot(df_scores['threshold'], df_scores['tpr'], label='TPR', color='blue')
plt.plot(df_scores['threshold'], df_scores['fpr'], label='FPR', color='orange')
plt.legend()
plt.show()

# %% Random model

np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))

# Accuracy of the random model
(y_val == (y_rand >= 0.5)).mean()  # Approx 50% accuracy


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []
    thresholds = np.linspace(0, 1, 101)
    actual_positive = y_val == 1
    actual_negative = y_val == 0

    for t in thresholds:
        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = sum(predict_positive & actual_positive)
        tn = sum(predict_negative & actual_negative)
        fp = sum(predict_positive & actual_negative)
        fn = sum(predict_negative & actual_positive)
        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores['tp'] / (df_scores['tp'] + df_scores['fn'])
    df_scores['fpr'] = df_scores['fp'] / (df_scores['fp'] + df_scores['tn'])

    return df_scores


df_random = tpr_fpr_dataframe(y_val, y_rand)
print(df_random[::10])

plt.style.use('ggplot')
plt.plot(df_random['threshold'], df_random['tpr'], label='TPR', color='blue')
plt.plot(df_random['threshold'], df_random['fpr'], label='FPR', color='orange')
plt.legend()
plt.show()

# %% Ideal Model

num_neg = sum(y_val == 0)
num_pos = sum(y_val == 1)
print(f"{num_neg=}, {num_pos=}")

y_ideal = np.repeat(a=[0, 1], repeats=[num_neg, num_pos])
y_ideal_pred = np.linspace(0, 1, len(y_val))

((y_ideal_pred >= 0.726) == y_ideal).mean()
df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
print(df_ideal[::10])

plt.style.use('ggplot')
plt.plot(df_ideal['threshold'], df_ideal['tpr'], label='TPR', color='blue')
plt.plot(df_ideal['threshold'], df_ideal['fpr'], label='FPR', color='orange')
plt.legend()
plt.show()

# %% Putting everything together

plt.style.use('ggplot')
plt.plot(df_scores['threshold'], df_scores['tpr'], label='TPR', color='limegreen')
plt.plot(df_scores['threshold'], df_scores['fpr'], label='FPR', color='red')

# plt.plot(df_random['threshold'], df_random['tpr'], label='TPR')
# plt.plot(df_random['threshold'], df_random['fpr'], label='FPR')

plt.plot(df_ideal['threshold'], df_ideal['tpr'], label='TPR Ideal', color='blue', linestyle='--')
plt.plot(df_ideal['threshold'], df_ideal['fpr'], label='FPR Ideal', color='blue', linestyle='--')

plt.legend()
plt.show()

# %% False positive rate vs True Positive Rate

plt.style.use('ggplot')
plt.figure(figsize=(5, 5))
plt.plot(df_scores['fpr'], df_scores['tpr'], label='model', color='orange')
plt.plot(df_random['fpr'], df_random['tpr'], label='random', color='violet')
plt.plot(df_ideal['fpr'], df_ideal['tpr'], label='ideal', color='limegreen')

plt.title("FPR vs TPR", fontdict={"fontweight": "bold"})
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

plt.show()

# %% Plotting the ROC curve with scikit-learn
fpr, tpr, thresholds = roc_curve(y_true=y_val, y_score=y_pred)

plt.style.use('ggplot')
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label='model', color='orange')
plt.plot([0, 1], [0, 1], label='random', color='gray', linestyle='--')
plt.title("FPR vs TPR", fontdict={"fontweight": "bold"})
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()

plt.show()

# %% 4.6 ROC AUC

# Calculating the AUC for our logistic regression model
auc(fpr, tpr)  # AUC = 0.84 -> good performance

# Calculating the AUC for the ideal model
auc(df_ideal['fpr'], df_ideal['tpr'])  # AUC = 1 for perfect classifiers

# Calculating the AUC for the random model
auc(df_random['fpr'], df_random['tpr'])  # AUC = 0.5 for random change models

# fpr, tpr, thresholds = roc_curve(y_val, y_pred)
# auc(fpr, tpr)
roc_auc_score(y_val, y_pred)  # shortcut for the code shown above

# ROC AUC represents the probability that a randomly selected positive example
# is higher than a randomly selected negative example
neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]

n = 100_000
success = 0
for i in range(n):
    pos_idx = random.randint(a=0, b=len(pos) - 1)
    neg_idx = random.randint(a=0, b=len(neg) - 1)
    if pos[pos_idx] > neg[neg_idx]:
        success += 1

success_rate = success / n
print(f"{success_rate=}")

# Running same comparison but vectorized
n = 500_000
np.random.seed(1)
pos_idx = np.random.randint(0, len(pos), size=n)
neg_idx = np.random.randint(0, len(neg), size=n)
np.mean(pos[pos_idx] > neg[neg_idx])


# %% Cross-Validation

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=5_000)
    model.fit(X_train, y_train)

    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


df, model = train(df_train, y_train)
y_pred = predict(df_val, dv, model)

# %% Running the k-fold cross validation with k=5
# Creating the k-fold
n_splits = 5

for c in tqdm([0.001, 0.01, 0.1, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    auc_scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train['churn'].values
        y_val = df_val['churn'].values

        dv, model = train(df_train, y_train, C=c)
        y_pred = predict(df_val, dv, model)

        auc_score = roc_auc_score(y_val, y_pred)
        rounded_auc = auc_score.round(3)
        auc_scores.append(auc_score)
    print('\n C=%s : %.3f +- %.3f' % (c, np.mean(auc_scores), np.std(auc_scores)))

# %% Training final model
y_full_train = df_full_train['churn'].values
# Fitting a model to the full train dataset with C=1.0 (default C)
dv, model = train(df_full_train, y_full_train)
y_pred = predict(df_test, dv, model)

auc_score = roc_auc_score(y_test, y_pred)
print(auc_score)


# %% Computing and plotting the ROC Curve for our final model

# Computes the ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

random_auc = 0.5

plt.figure(figsize=(8, 6))
plt.style.use('bmh')
plt.plot(fpr, tpr, label=f'Logistic Classifier')
plt.fill_between(fpr, tpr, color='blue', alpha=0.2)
plt.plot([0, 1], [0, 1], label='Random Classifier', color='orange')
plt.fill_between([0, 1], [0, 1], color='orange', alpha=0.2)
plt.text(0.35, 0.8, f'Logistic AUC: {roc_auc:.2f}', fontsize=12, fontweight='bold')
plt.text(0.5, 0.25, f'Random AUC: {random_auc:.2f}', fontsize=12, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('./charts/m3_roc_curve')
plt.show()
