
"""
This is a full content of all classification modules
"""

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                            roc_curve, auc, precision_recall_curve, log_loss, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier


# load the data extracted by mlp
mlp_features = np.load(r'D:\research\research\others\hantong\mlp_features.npy')
data = pd.read_excel(r'D:\research\research\others\figures\data_with_pnatietID_retained.xlsx')
y = data['病理结果编码'].values

# confirm the amount of data
print("Features shape:", mlp_features.shape)
print("Labels shape:", y.shape)

# check if sizes are matched
if mlp_features.shape[0] != y.shape[0]:
    raise ValueError(f"Feature and label sample sizes are inconsistent: {mlp_features.shape[0]} features, {y.shape[0]} labels.")


X_train, X_test, Y_train, Y_test = train_test_split(mlp_features, y, test_size=0.3, random_state=42)

# check the shape of the features
print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

def evaluate_model(model, x_test, y_test, model_name):
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    # targets
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    logloss = log_loss(y_test, y_prob)
    error_rate = 1 - accuracy
    specificity = recall_score(y_test, y_pred, pos_label=0, average='weighted')
    roc_auc = roc_auc_score(y_test, y_prob)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Matrix of confusion:%s', cm)
    (tn,fp,fn,tp) = cm.ravel()
    print('tn=',tn)
    print('fp=',fp)
    print('fn=',fn)
    print('tp=',tp)
    print('------------------------')
    sensitivity_new = (tp/(tp+fn))*100
    specificity_new = (tn/(fp+tn))*100
    PPV=tp/(tp+fp)*100
    NPV=tn/(fn+tn)*100
    print(f'PPV = {"%.1f"%PPV}\n({tp}/{(tp+fp)})')
    print(f'NPV = {"%.1f"%NPV}\n({tn}/{(fn+tn)})')
    print(f'sensitivity = {"%.1f"%sensitivity_new}\n({tp}/{(tp+fn)})')
    print(f'specificity_new = {"%.1f"%specificity_new}\n({tn}/{(fp+tn)})')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc='best')
    plt.show()

    # PR curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall_vals, precision_vals, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend(loc='best')
    plt.show()

    print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f},\
        Log Loss: {logloss:.4f}, Error Rate: {error_rate:.4f}, Specificity: {specificity:.4f}")


# define baseline model
base_estimator = DecisionTreeClassifier(random_state=42, ccp_alpha=0.0)

# BaggingClassifier
bagging_model = BaggingClassifier(
    estimator=base_estimator,
    n_estimators=30,
    random_state=42
)

# define pipelines for PCA analysis

pipeline_bagging_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=64)),
    ('bagging', bagging_model)
])


pipeline_svm_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=64)),
    ('svm', SVC(class_weight='balanced', probability=True))
])

pipeline_xgb_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=64)),
    ('xgb', XGBClassifier(eval_metric="logloss"))
])

pipeline_rf_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=64)),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

pipeline_lgbm_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=64)),
    ('lgbm', LGBMClassifier())
])



# define the grid of each classifier's hyperparameters
# all the parameters are tested to make sure they are the best choices
"""
    bagging - Accuracy: 0.7200, Precision: 0.7294, Recall: 0.7200, F1: 0.7210,
    Log Loss: 0.5991, Error Rate: 0.2800, Specificity: 0.7200
    AUC = 0.81
"""
param_grid_bagging_pca = {
    'bagging__n_estimators': [30],
    'bagging__estimator': [ ExtraTreesClassifier()],
    'bagging__max_samples': [0.8],
    'bagging__max_features': [0.8],
    'bagging__bootstrap_features': [True, False]
}
# bagging - Accuracy: 0.7200, Precision: 0.7294, Recall: 0.7200, F1: 0.7210,
#           Log Loss: 0.5991, Error Rate: 0.2800, Specificity: 0.7200

"""
    svm - when C = 0.01, the best hyperparameters are as follows:
        svm - Accuracy: 0.7520, Precision: 0.7540, Recall: 0.7520, F1: 0.7526,
        Log Loss: 0.5711, Error Rate: 0.2480, Specificity: 0.7520
        AUC = 0.84
    when C = 0.01, the best hyperparameters are as follows:
        svm - Accuracy: 0.7840, Precision: 0.7837, Recall: 0.7840, F1: 0.7826,
        Log Loss: 0.5245, Error Rate: 0.2160, Specificity: 0.7840
        AUC = 0.82
    when C = 0.001, the best hyperparameters are as follows:
        svm - Accuracy: 0.7680, Precision: 0.7745, Recall: 0.7680, F1: 0.7623,
        Log Loss: 0.5256, Error Rate: 0.2320, Specificity: 0.7680
        AUC = 0.80
"""
param_grid_svm_pca = {
    'svm__C': [0.01],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto'],
    'svm__degree': [3],  #  only for ploy kernel
    'svm__coef0': [0.0]  # only for poly and sigmoid
}




"""
    when lr = 0.1, the best hyperparameters are as follows:
        xgb - Accuracy: 0.7520, Precision: 0.7540, Recall: 0.7520, F1: 0.7526,
        Log Loss: 0.5711, Error Rate: 0.2480, Specificity: 0.7520
        AUC = 0.83
    when lr = 0.01, the best hyperparameters are as follows:
        xgb - Accuracy: 0.7600, Precision: 0.7612, Recall: 0.7600, F1: 0.7604,
        Log Loss: 0.5302, Error Rate: 0.2400, Specificity: 0.7600
        AUC = 0.82
"""
param_grid_xgb_pca = {
    'xgb__max_depth': [7],
    # 'xgb__learning_rate': [0.01,0.1],
    'xgb__learning_rate': [0.1],
    'xgb__n_estimators': [200],
    'xgb__colsample_bytree': [0.6],
    'xgb__subsample': [0.6, 0.8],
    'xgb__gamma': [0.1],
    'xgb__reg_alpha': [0, 0.1],
    'xgb__reg_lambda': [0, 0.1]
}




""" rf - Accuracy: 0.7600, Precision: 0.7629, Recall: 0.7600, F1: 0.7607,
    Log Loss: 0.5753, Error Rate: 0.2400, Specificity: 0.7600
    AUC = 0.80
"""
param_grid_rf_pca = {
    'rf__n_estimators': [200],
    'rf__max_depth': [7],
    'rf__min_samples_split': [5],
    'rf__min_samples_leaf': [7],
    'rf__max_features': ['auto', 'sqrt'],
    'rf__bootstrap': [True]
}



"""
    when lr = 0.1, the best hyperparameters are as follows:
        lgbm - Accuracy: 0.7200, Precision: 0.7206, Recall: 0.7200, F1: 0.7203,
        Log Loss: 0.6695, Error Rate: 0.2800, Specificity: 0.7200
        AUC = 0.81
    when lr = 0.01, the best hyperparameters are as follows:
       lgbm - Accuracy: 0.7360, Precision: 0.7401, Recall: 0.7360, F1: 0.7369,
       Log Loss: 0.5405, Error Rate: 0.2640, Specificity: 0.7360
       AUC = 0.80
"""
param_grid_lgbm_pca = {
    'lgbm__num_leaves': [30, 50],
    # 'lgbm__learning_rate': [0.01,0.1],
    'lgbm__learning_rate': [0.01],
    'lgbm__n_estimators': [200],
    'lgbm__max_depth': [10],
    'lgbm__min_child_samples': [30],
    'lgbm__subsample': [0.8],
    'lgbm__colsample_bytree': [0.8]
}


# use grid search to find the best hyperparameters for each classifier

grid_bagging_pca = GridSearchCV(pipeline_bagging_pca,param_grid_bagging_pca,cv = 3, scoring='accuracy', n_jobs=-1)
grid_bagging_pca.fit(X_train, Y_train)
print(grid_bagging_pca.best_estimator_)

grid_svm_pca = GridSearchCV(pipeline_svm_pca, param_grid_svm_pca, cv=3, scoring='accuracy', n_jobs=-1)
grid_svm_pca.fit(X_train, Y_train)
print(grid_svm_pca.best_estimator_)

grid_xgb_pca = GridSearchCV(pipeline_xgb_pca, param_grid_xgb_pca, cv=3, scoring='accuracy', n_jobs=-1)
grid_xgb_pca.fit(X_train, Y_train)
print(grid_xgb_pca.best_estimator_)

grid_rf_pca = GridSearchCV(pipeline_rf_pca, param_grid_rf_pca, cv=3, scoring='accuracy', n_jobs=-1)
grid_rf_pca.fit(X_train, Y_train)
print(grid_rf_pca.best_estimator_)


grid_lgbm_pca = GridSearchCV(pipeline_lgbm_pca, param_grid_lgbm_pca, cv=3, scoring='accuracy', n_jobs=-1)
grid_lgbm_pca.fit(X_train, Y_train)
print(grid_lgbm_pca.best_estimator_)


# weighted voting for classification
weighted_voting_model = VotingClassifier(
    estimators=[
        ('bagging', grid_bagging_pca.best_estimator_),
        ('svm', grid_svm_pca.best_estimator_),
        ('xgb', grid_xgb_pca.best_estimator_),
        ('rf', grid_rf_pca.best_estimator_),
        ('lgbm', grid_lgbm_pca.best_estimator_)
    ],
    voting='soft',
    weights=[0,1.0,0.8,1.0,0]  # weights based on their performance
)

# stacking classifier
stacking_model = StackingClassifier(
    estimators=[
        ('svm', grid_svm_pca.best_estimator_),
        ('xgb', grid_xgb_pca.best_estimator_),
        ('rf', grid_rf_pca.best_estimator_),
        ('lgbm', grid_lgbm_pca.best_estimator_)
    ],
    final_estimator=LogisticRegression(random_state=42)
)

# training and evaluation
weighted_voting_model.fit(X_train, Y_train)
weighted_voting_accuracy = accuracy_score(Y_test, weighted_voting_model.predict(X_test))

stacking_model.fit(X_train, Y_train)
stacking_accuracy = accuracy_score(Y_test, stacking_model.predict(X_test))



# evaluate models
evaluate_model(grid_bagging_pca.best_estimator_,X_test,Y_test,"bagging")
evaluate_model(grid_svm_pca.best_estimator_, X_test, Y_test, "svm")
evaluate_model(grid_xgb_pca.best_estimator_, X_test, Y_test, "xgb")
evaluate_model(grid_rf_pca.best_estimator_, X_test, Y_test, "rf")
evaluate_model(grid_lgbm_pca.best_estimator_, X_test, Y_test, "lgbm")
evaluate_model(weighted_voting_model, X_test, Y_test, "Weighted Voting Model")
evaluate_model(stacking_model, X_test, Y_test, "Stacking Model")


# outputs
print("Weighted Voting Model Accuracy with MLP Features:", weighted_voting_accuracy)
print("Stacking Model Accuracy with MLP Features:", stacking_accuracy)

# Final results

# total 5

# Weighted Voting Model - Accuracy: 0.7680, Precision: 0.7685, Recall: 0.7680, F1: 0.7682,
# Log Loss: 0.5215, Error Rate: 0.2320, Specificity: 0.7680


# Stacking Model - Accuracy: 0.7680, Precision: 0.7685, Recall: 0.7680, F1: 0.7682,
# Log Loss: 0.5066, Error Rate: 0.2320, Specificity: 0.7680


#only 3
# Weighted Voting Model Accuracy with MLP Features: 0.768
# Stacking Model Accuracy with MLP Features: 0.776
