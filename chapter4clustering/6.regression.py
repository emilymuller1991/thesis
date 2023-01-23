# regression model for perceptions

import pandas as pd
from patsy import dmatrices
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import sys
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
import shap 

perception = 'clusters'
# read in features file
census = pd.read_csv('/home/emily/phd/2_interpretability/2018_all_variables.csv')
columns = list(census.columns)
new_columns = [feature.replace(' ', '_') for feature in columns]
census.columns = new_columns

# DATA FORMATTING
# features 94 in total
# features = list(census.columns[22:-22])
features = ['building_seg',
 'vegetation_seg',
 'car_seg',
 'road_seg',
 'sky_seg',
 'sidewalk_seg',
 'fence_seg',
 'wall_seg',
 'pole_seg',
 'traffic_sign_seg',
 'terrain_seg',
 'car',
 'person_seg',
 'traffic_light_seg',
 'truck_seg',
 'bicycle_seg',
 'bus_seg',
 'train_seg',
 'motorcycle_seg',
 'person',
 'truck']
 # bus
equation = ''
for f in features:
    equation = equation + f + ' + ' 
equation_ = perception + '~ ' + equation[:-3]
# get features and predictors
y, X = dmatrices(equation_, data=census, return_type='dataframe')

# sample for balanced classes
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
rus = RandomUnderSampler(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=13
)

X_train, y_train = rus.fit_resample(X_train, y_train)
X_test, y_test= rus.fit_resample(X_test, y_test)
print(sorted(Counter(y_test).items()))


# MODELS
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
#from sklearn import datasets, ensemble

# def run_model_with_cross_validation(X, y, n_splits, classifiers, names):
#     random_state = np.random.RandomState(0)
#     cv = StratifiedKFold(n_splits, random_state=random_state, shuffle=True)

#     accuracies = {}
#     n = 0
#     for classifier in classifiers:
#         accuracies[names[n]] = []
#         for train, test in cv.split(X,y):
#             model = classifier.fit(X.iloc[train], y.iloc[train])
#             #return model
#             y_hat = model.predict(X.iloc[test])
#             acc = accuracy_score(y.iloc[test], y_hat)
#             # ham_l = hamming_loss(y.iloc[test], y_hat) 
#             # ham_s = hamming_score(y.iloc[test], y_hat)
#             accuracies[names[n]].append([acc])
#             print ('Completed one split')
#         print ('Completed one model')
#         m = np.array(accuracies[names[n]])
#         accuracies[names[n]] = m.mean(axis=0) 
#         n =+ 1
#     return accuracies

# n_splits = 5
# names = ["RF"]
# classifiers = [ RandomForestClassifier()  ]
# for i in range(1):
#     start_time = time.time()
#     try:
#         acc = run_model_with_cross_validation(X_resampled, y_resampled, n_splits, [classifiers[i]], [names[i]])
#         end_time = time.time()
#         print ('Completed model %s in %s seconds' % (names[i], start_time-end_time))
#         print (acc)
#     except:
#         print ('Model not completed')

def run_model_with_cross_validation_model(X_train, y_train, X_test, y_test, n_splits, classifiers, names):
    random_state = np.random.RandomState(0)
    # cv = StratifiedKFold(n_splits, random_state=random_state, shuffle=True)

    accuracies = {}
    n = 0
    for classifier in classifiers:
        accuracies[names[n]] = []
        #for train, test in cv.split(X,y):
        model = classifier.fit(X_train, y_train)
        y_hat = model.predict(X_test)
        acc = accuracy_score(y_test, y_hat)
        return acc, model

n_splits = 5
names = ["RF"]
classifiers = [ RandomForestClassifier()  ]
for i in range(1):
    start_time = time.time()
    try:
        acc, model = run_model_with_cross_validation_model(X_train, y_train, X_test, y_test, n_splits, [classifiers[i]], [names[i]])
        end_time = time.time()
        print ('Completed model %s in %s seconds' % (names[i], start_time-end_time))
        print (acc)# 0.47703813585512084
    except:
        print ('Model not completed')

# random forest features
# importances = list(model.feature_importances_)
# feature_list = list(X.columns) 
# # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# sub_features = [feature[0] for feature in feature_importances[:21]]

# DATA FORMATTING
# features 94 in total
# features = sub_features
# equation = ''
# for f in features:
#     equation = equation + f + ' + ' 
# equation_ = perception + '~ ' + equation[:-3]
# # get features and predictors
# y, X = dmatrices(equation_, data=census, return_type='dataframe')

# rus = RandomUnderSampler(random_state=0)
# X_resampled, y_resampled = rus.fit_resample(X, y)
# print(sorted(Counter(y_resampled).items()))

# X_train, X_test, y_train, y_test = train_test_split(
#     X_resampled, y_resampled, test_size=0.3, random_state=13
# )

# n_splits = 5
# names = ["RF"]
# classifiers = [ RandomForestClassifier()  ]
# for i in range(1):
#     start_time = time.time()
#     try:
#         acc, model = run_model_with_cross_validation_model(X_resampled, y_resampled, n_splits, [classifiers[i]], [names[i]])
#         end_time = time.time()
#         print ('Completed model %s in %s seconds' % (names[i], start_time-end_time))
#         print (acc)# 0.477239013536065
#     except:
#         print ('Model not completed')


# FEATURE IMPORTANCES
# importances = model.feature_importances_
# std = np.std([tree.feature_importances_ for tree in model.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
# feature_list = [X.columns[indices[f]] for f in range(X.shape[1])]  #names of features.
# ff = np.array(feature_list)
from sklearn.inspection import permutation_importance

r = permutation_importance(model, X_test, y_test,
                           n_repeats=10,
                           random_state=0)
perm = pd.DataFrame(columns=['AVG_Importance', 'STD_Importance'], index=[i for i in X_train.columns])
perm['AVG_Importance'] = r.importances_mean


# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f) name: %s" % (f + 1, indices[f], importances[indices[f]], ff[indices[f]]))

# Plot the feature importances of the forest
import matplotlib as mpl
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

plt.figure()
plt.rcParams['figure.figsize'] = [16, 6]
#plt.title("Feature importances")
indices = np.argsort(perm['AVG_Importance'][:20])[::-1]
ff = np.array(perm['AVG_Importance'][:20].index)
plt.bar(range(20), perm['AVG_Importance'][indices], align="center")
plt.xticks(range(20), ff[indices], rotation=90)
plt.xlim([-1, 20])
plt.yticks(rotation=90)
plt.savefig('/home/emily/phd/2_interpretability/features/outputs/RF_feature_importance_20_permutation.png',  bbox_inches='tight'  )
#plt.show()


# import shap


# shap_values = shap.TreeExplainer(model)
# shap.summary_plot(shap_values,shap.sample(X_test, 1000))

sample = shap.sample(X_train, 1000)

# explainer = shap.KernelExplainer(reg_xgb.predict, sample)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
shap.summary_plot(shap_values, sample, plot_type="bar", class_names=model.predict(sample))
for i in range(len(shap_values)):
    shap.summary_plot(shap_values[i], sample, show=False)
    plt.savefig('/home/emily/phd/2_interpretability/features/outputs/shap_values_cluster_%s.png'% str(i), bbox_inches='tight')
    plt.clf()

# get mean(|shap|) values per cluster
fi = {}
for i in range(20):
    fip = np.abs(shap_values[i]).mean(axis=0)
    fi[i] = fip
    # shap_values[i][shap_values[i] > 0] = 1
    # shap_values[i][shap_values[i] <= 0] = 0
    # v = sample.to_numpy()*shap_values[i]
    # fi[i] = v.mean(axis=0)  #- sample.to_numpy().mean(axis=0)
fi_df = pd.DataFrame(fi).T
fi_df.to_csv('/home/emily/phd/2_interpretability/features/outputs/RF_shap_class_fi.csv', index=False)

sign = {}
for i in range(20):
    shap_sign = shap_values[i].copy()
    shap_sign[shap_sign > 0] = 1
    shap_sign[shap_sign <= 0] = np.nan
    mul = sample.to_numpy()*shap_sign
    pos_mean = np.nanmean(mul,axis=0)

    shap_sign = shap_values[i].copy()
    shap_sign[shap_sign > 0] = np.nan
    shap_sign[shap_sign <= 0] = 1
    mul = sample.to_numpy()*shap_sign
    neg_mean = np.nanmean(mul,axis=0)

    s = []
    for j in range(pos_mean.shape[0]):
        if pos_mean[j] > neg_mean[j]:
            s.append(1)
        elif pos_mean[j] <= neg_mean[j]:
            s.append(-1)
        else:
            s.append(0)
    sign[i] = s
sign_df = pd.DataFrame(sign).T
sign_df.to_csv('/home/emily/phd/2_interpretability/features/outputs/RF_shap_class_sign.csv', index=False)

# 1 if mean of feature values above SHAP > 0 is greater than overall mean
# 0 if mean of feature values above SHAP > 0 is less than overall mean.
# Create two different types of plots = Radar + Bar.



# I don't think this is the right way of thinking about feature importances
## The new additions to get feature importance to classes: 
# from sklearn.preprocessing import scale
# import json
# # To get the importance according to each class:
# def class_feature_importance(X, Y, feature_importances):
#     N, M = X.shape
#     X = scale(X)

#     out = {}
#     for c in set(Y['clusters']):
#         out[c] = dict(
#             zip(range(N), np.mean(X[list(Y[Y['clusters']==c].index), :], axis=0)*feature_importances)
#         )

#     return out

# for test_sample in range(len(X_test)):
#     prediction, bias, contributions = ti.predict(model[0], X_test.iloc[test_sample].to_numpy().reshape(1,22))
#     print ("Class Prediction", prediction)
#     print ("Bias (trainset prior)", bias)

#     # now extract contributions for each instance
#     for c, feature in zip(contributions[0], features):
#         print (feature, c)

#     print ('\n')

# result = class_feature_importance(X, y, importances)
# print (json.dumps(result,indent=4))

# class_importance = pd.DataFrame(result)
# class_importance.to_csv('/home/emily/phd/2_interpretability/features/outputs/RF_class_fi.csv')
# Plot the feature importances of the forest

titles = [str(i) for i in range(20)]
for t, i in zip(titles, range(len(result))):
    plt.figure()
    plt.rcParams['figure.figsize'] = [16, 6]
    plt.title(t)
    plt.bar(range(len(result[i])), result[i].values(),
           color="r", align="center")
    plt.xticks(range(len(result[i])), ff[list(result[i].keys())], rotation=90)
    plt.xlim([-1, len(result[i])])
    plt.show()
plt.savefig('/home/emily/phd/2_interpretability/features/outputs/RF_feature_importance_20.png'  )













# XBG regression
# from sklearn import datasets, ensemble
# params = {
#     "n_estimators": 20,
#     "max_depth": 4,
#     "min_samples_split": 5,
#     "learning_rate": 0.01
# }
# reg_xgb = ensemble.GradientBoostingClassifier(**params, verbose=True)
# reg_xgb.fit(X_train, y_train)
# acc = accuracy_score(y_test, reg_xgb.predict(X_test))
# print("The accuracy on test set: {:.4f}".format(acc))
# The accuracy on test set:  0.3989

# Plotting Deviance
# import matplotlib.pyplot as plt 

# test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
# for i, y_pred in enumerate(reg_xgb.staged_predict(X_test)):
#     test_score[i] = reg_xgb.loss_(np.squeeze(y_test), y_pred)

# fig = plt.figure(figsize=(6, 6))
# plt.subplot(1, 1, 1)
# plt.title("XGB %s Regression" % str(perception))
# plt.plot(
#     np.arange(params["n_estimators"]) + 1,
#     reg_xgb.train_score_,
#     "b-",
#     label="Training Set MSE",
# )
# plt.plot(
#     np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set MSE"
# )
# plt.legend(loc="upper right")
# plt.xlabel("Boosting Iterations")
# plt.ylabel("MSE")
# fig.tight_layout()
# plt.savefig('/home/emily/phd/2_interpretability/features/outputs/XGB_%s_mse_iterations.png' % str(perception) )
# plt.clf() 

# Subselect only features with feature importance
# feature_importance = reg_xgb.feature_importances_
# # features 94 in total
# sub_features = []
# for feature, importance in zip(X.columns, feature_importance):
#     if importance != 0:
#         print (feature, importance)
#         sub_features.append(feature)

# equation = ''
# for f in sub_features:
#     equation = equation + f + ' + ' 

# equation_ = perception + '~ ' + equation[:-3]
# # get features and predictors
# y, X = dmatrices(equation_, data=census, return_type='dataframe')

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=13
# )

# params = {
#     "n_estimators": 20,
#     "max_depth": 4,
#     "min_samples_split": 5,
#     "learning_rate": 0.01,
#     "loss": "ls",
# }
# reg_xgb = ensemble.GradientBoostingClassifier(**params, verbose=True)
# reg_xgb.fit(X_train, y_train)
# acc = accuracy_score(y_test, reg_xgb.predict(X_test))
# print("The accuracy on test set: {:.4f}".format(acc))
# The accuracy on test set:  0.3989

# Plotting Deviance
# test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
# for i, y_pred in enumerate(reg_xgb.staged_predict(X_test)):
#     test_score[i] = reg_xgb.loss_(np.squeeze(y_test), y_pred)

# fig = plt.figure(figsize=(6, 6))
# plt.subplot(1, 1, 1)
# plt.title("XGB %s Regression Sub Features" % str(perception))
# plt.plot(
#     np.arange(params["n_estimators"]) + 1,
#     reg_xgb.train_score_,
#     "b-",
#     label="Training Set MSE",
# )
# plt.plot(
#     np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set MSE"
# )
# plt.legend(loc="upper right")
# plt.xlabel("Boosting Iterations")
# plt.ylabel("MSE")
# fig.tight_layout()
# plt.savefig('/home/emily/phd/2_interpretability/pp/outputs/XGB_%s_mse_iterations_sub.png' % str(perception) )
# plt.clf() 

# Feature Importance
# importance_type = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
#feature_importance = reg_xgb.get_booster().get_score(importance_type='gain')
# feature_importance = reg_xgb.feature_importances_
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + 0.5
# fig = plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.barh(pos, feature_importance[sorted_idx], align="center")
# plt.yticks(pos, np.array(X.columns)[sorted_idx])
# plt.title("Feature Importance (MDI)")
# plt.savefig('/home/emily/phd/2_interpretability/features/outputs/XGB_%s_feature_importance_sub.png' % str(perception) )


# result = permutation_importance(
#     reg_xgb, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
# )
# sorted_idx = result.importances_mean.argsort()
# plt.subplot(1, 2, 2)
# plt.boxplot(
#     result.importances[sorted_idx].T,
#     vert=False,
#     labels=np.array(X.columns)[sorted_idx],
# )
# plt.title("Permutation Importance (test set)")
# fig.tight_layout()
# plt.savefig('/home/emily/phd/2_interpretability/pp/outputs/XGB_%s_feature_importance_sub.png' % str(perception) )

# import shap


# shap_values = shap.KernelExplainer(reg_xgb.predict, X_test)
# shap.summary_plot(shap_values,shap.sample(X_test, 1000))

# sample = shap.sample(X_test, 1000)

# explainer = shap.KernelExplainer(reg_xgb.predict, sample)
# shap_values = explainer.shap_values(sample,approximate=True)
# shap.summary_plot(shap_values, sample, plot_type="bar", class_names=reg_xgb.predict(sample))


# I think I need a tree model to be able to build class specific feature importances 



# explainer = shap.KernelExplainer(reg_xgb.predict, sample)
# shap_values_ = explainer.shap_values(sample)

# shap.summary_plot(shap_values_,features=sample, feature_names=list(X.columns), plot_size=(7,7))
# plt.savefig('/home/emily/phd/2_interpretability/pp/outputs/%s_summary_plot_sub_1000.png' % str(perception))


# def prediction_hist(y, y_true, perception):
#     left, width = 0.1, 0.65
#     bottom, height = 0.1, 0.65
#     spacing = 0.005

#     rect_scatter = [left, bottom, width, height]
#     rect_histx = [left, bottom + height + spacing, width, 0.2]
#     rect_histy = [left + width + spacing, bottom, 0.2, height]

#     # start with a rectangular Figure
#     plt.figure(figsize=(8, 8))

#     ax_scatter = plt.axes(rect_scatter)
#     ax_scatter.tick_params(direction='in', top=True, right=True)
#     ax_histx = plt.axes(rect_histx)
#     ax_histx.tick_params(direction='in', labelbottom=False)
#     ax_histy = plt.axes(rect_histy)
#     ax_histy.tick_params(direction='in', labelleft=False)

#     # the scatter plot:
#     ax_scatter.scatter(y, y_true)

#     # now determine nice limits by hand:
#     binwidth = 0.25
#     lim = np.ceil(np.abs([y, y_true]).max() / binwidth) * binwidth
#     ax_scatter.set_xlim((-lim, lim))
#     ax_scatter.set_ylim((-lim, lim))
#     ax_scatter.set_xlabel('Prediction')
#     ax_scatter.set_ylabel('Ground Truth')

#     bins = np.arange(-lim, lim + binwidth, binwidth)
#     ax_histx.hist(y, bins=bins)
#     ax_histy.hist(y_true, bins=bins, orientation='horizontal')

#     ax_histx.set_xlim(ax_scatter.get_xlim())
#     ax_histy.set_ylim(ax_scatter.get_ylim())

#     plt.show()
#     plt.savefig('/home/emily/phd/2_interpretability/pp/outputs/%s_scatterplot.png' % str(perception))
#     print ('Plotted predictions')
#     plt.clf()