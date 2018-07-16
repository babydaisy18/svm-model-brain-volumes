# Scikit Learn: Receiver Operating Characteristic with Cross Validation
import pandas as pd
import numpy as np
from scipy import interp
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)

d = pd.read_csv('Temp_Asym_Data.csv')
target = d['MTLLat']
Y = []
for val in target:
    if(val == 'L'):
        Y.append(0)
    else:
        Y.append(1)
d = d.drop(['MTLLat'], axis=1)
X = d.values.tolist()
X = np.array(X)
Y = label_binarize(Y, classes=[0, 1])
Y = Y.ravel()
# print(X)
# print(Y)


def f_imp(coef, names):
    imp = coef
    imp = imp.ravel()
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.title('Variable Importance: Weights with Correlation')
    plt.xlabel('Weight of Coefficient with Correlation')
    plt.ylabel('Attribute')
    plt.tight_layout()
    plt.show()
    imp = abs(imp)
    print(imp)
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.title('Variable Importance: Weights of Attributes')
    plt.xlabel('Weight of Coefficient')
    plt.ylabel('Attribute')
    plt.tight_layout()
    plt.show()


cv = StratifiedKFold(n_splits=5)
cv.get_n_splits(X, Y)
clf = SVC(kernel='linear', probability=True)
features_names = [
    'Vol_CA1_asym',
    'Vol_CA2_asym',
    'Vol_CA3_asym',
    'Vol_DG_asym',
    'Vol_SUB_asym']
clf.fit(X, Y)
f_imp(clf.coef_, features_names)
# print(clf._get_coef_)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X, Y):
    # print(X[test])
    # print(Y[train])
    x_train, x_test = X[train], X[test]
    y_train, y_test = Y[train], Y[test]
    proba = clf.fit(x_train, y_train).predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, proba[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=4, color='black',
         label='Luck', alpha=0.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='red',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=3, alpha=0.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve: CrossVal for MTLLat vs. Asym Indices')
plt.legend(loc="lower right")
plt.show()
