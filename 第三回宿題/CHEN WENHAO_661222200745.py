import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# ガウスナイーブベイズによる計算
clf = GaussianNB()
clf.fit(X_train, y_train)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# アセスメント
y_pred = clf.predict(X_test)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))
print(f1_score(y_test, y_pred, average=None))

# 予測
y_proba = clf.predict_proba(X_test[:1])
acc = np.sum(y_test == y_pred) / X_test.shape[0]
print("判別精度: %.3f" % acc)
print(clf.predict(X_test[:1]))
print("期待される確率の値:", y_proba)