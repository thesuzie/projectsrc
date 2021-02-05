from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,
                                    KMeansSMOTE)
from imblearn.over_sampling import RandomOverSampler
from imblearn.base import BaseSampler
from sklearn.linear_model import LogisticRegression


def plot_resampling(X, y, sampling, ax):
    X_res, y_res = sampling.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor='k')
    # make nice plotting
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    return Counter(y_res)


def plot_decision_function(X, y, clf, ax):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')
    return None


training = pd.read_csv('./test_train/train_IM.csv', sep=',')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

X_text = training["Content Cleaned"]
vec = CountVectorizer(stop_words=stopwords.words('english'))
X = vec.fit_transform(X_text)

y = training["Label"]
clf = LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg')
clf.fit(X, y)
plot_decision_function(X, y, clf, ax1)
ax1.set_title('LogReg with y={}'.format(Counter(y)))

pipe = make_pipeline(RandomOverSampler(random_state=0), LogisticRegression(random_state=0, multi_class='multinomial', penalty='l2', solver='newton-cg'))
pipe.fit(X, y)
plot_decision_function(X, y, pipe, ax2)
ax2.set_title('Decision function for RandomOverSampler')
fig.tight_layout()
