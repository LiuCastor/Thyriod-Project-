from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                            roc_curve, auc, precision_recall_curve, log_loss, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os


# load the data extracted by mlp
mlp_features = np.load(r'D:\research\research\others\hantong\mlp_features.npy')
data = pd.read_excel(r'D:\research\research\others\figures\data_with_pnatietID_retained.xlsx')
y = data['病理结果编码'].values

# # confirm the amount of data
# print("Features shape:", mlp_features.shape)
# print("Labels shape:", y.shape)

# check if sizes are matched
if mlp_features.shape[0] != y.shape[0]:
    raise ValueError(f"Feature and label sample sizes are inconsistent: {mlp_features.shape[0]} features, {y.shape[0]} labels.")

X_train, X_test, Y_train, Y_test = train_test_split(mlp_features, y, test_size=0.3, random_state=42)

# # check the shape of the features
# print("X_train shape:", X_train.shape)
# print("Y_train shape:", Y_train.shape)
# print("X_test shape:", X_test.shape)
# print("Y_test shape:", Y_test.shape)

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

    # Calculate SHAP values
    shap.initjs()

    # Get the correct model from pipeline and create explainer
    if model_name == "rf":
        model_to_explain = model.named_steps['rf']
        explainer = shap.TreeExplainer(model_to_explain)
    elif model_name == "svm":
        model_to_explain = model.named_steps['svm']
        explainer = shap.KernelExplainer(model_to_explain.predict, X_train)
    elif model_name == "xgb":
        model_to_explain = model.named_steps['xgb']
        explainer = shap.TreeExplainer(model_to_explain)
    elif model_name == "lgbm":
        model_to_explain = model.named_steps['lgbm']
        explainer = shap.TreeExplainer(model_to_explain)
    elif model_name == "bagging":
        explainer = shap.KernelExplainer(model.predict_proba, x_test)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Calculate SHAP values
    shap_values = explainer.shap_values(x_test, check_additivity=False)

    # Create feature names and DataFrame
    feature_names = [f"Feature {i}" for i in range(x_test.shape[1])]
    X_test_df = pd.DataFrame(x_test, columns=feature_names)

    sample_index = 1  # select the first sample
    print(shap_values.shape)
    #!!! the shape of shap value has changed from rf （125，64，2）to xgb to (125,64)
    #xgb
    shap_values_for_sample = shap_values[0]
    # rf remove the first dimension-features after transpose
    # shap_values_for_sample = shap_values.transpose(2, 0, 1)[0]

    #!!! xgb(shap_values,shape_values_for_sample)
    shap_values_explanation = shap.Explanation(values=shap_values,\
        base_values=explainer.expected_value, data=X_test_df)


    shap_values_explanation_for_sample = shap_values_explanation[0]

    # Create directory for saving plots
    save_path = f'shap_plots/{model_name}'
    os.makedirs(save_path, exist_ok=True)

    try:

        # 1. decision_plots
        # xgb
        shap.decision_plot(explainer.expected_value, shap_values)
        # After checking all features，we only need one figure for estimation and comparison
        n_features = x_test.shape[1]
        n_cols = 4
        features_per_plot = 12
        n_plots = (n_features + features_per_plot - 1) // features_per_plot
        path_for_decision = os.path.join(save_path, 'decision_plots')
        os.makedirs(path_for_decision, exist_ok=True)

        for plot_index in range(n_plots):
            start_feature = plot_index * features_per_plot
            end_feature = min(start_feature + features_per_plot, n_features)
            current_features = end_feature - start_feature
            n_rows = (current_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
            axes = axes.ravel()

            for i in range(current_features):
                feature_index = start_feature + i
                plt.sca(axes[i])
                feature_name = X_test_df.columns[feature_index]
                shap.decision_plot(
                    explainer.expected_value,
                    shap_values=shap_values_for_sample,
                    features=x_test,
                    # feature_names=X_test_df.columns.tolist(),
                    show=False,
                )
                # hhh
                # if we need to check the feature name, we can add it to the title
                # but it will be extremely massive and not readable
                # try to change the number of images in one PNG
                # axes[i].set_title(f'{feature_name}')
                plt.gca().set_yticklabels([])

            for i in range(current_features, len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(f'{model_name} - SHAP Decision Plots (Features {start_feature} to {end_feature})', y=1.02)

            plt.tight_layout()
            plt.savefig(os.path.join(path_for_decision, \
                                    f'decision_plots_features_{start_feature}_to_{end_feature}.png'), \
                                        dpi=300, bbox_inches='tight')
            plt.close()


        # 2. Summary Plots (containing all features)
        #xgb
        shap.summary_plot(shap_values, X_test_df)
        # After checking all features，we only need one figure for estimation and comparison

        n_features = x_test.shape[1]
        n_cols = 1
        features_per_plot = 9
        n_plots = (n_features + features_per_plot - 1) // features_per_plot
        path_for_summary = os.path.join(save_path, 'summary_plots')
        os.makedirs(path_for_summary, exist_ok=True)

        for plot_index in range(n_plots):
            start_feature = plot_index * features_per_plot
            end_feature = min(start_feature + features_per_plot, n_features)
            current_features = end_feature - start_feature
            n_rows = (current_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
            axes = axes.ravel()

            for i in range(current_features):
                feature_index = start_feature + i
                plt.sca(axes[i])
                feature_name = X_test_df.columns[feature_index]
                shap.summary_plot(
                    shap_values=shap_values,
                    features=x_test,
                    feature_names=X_test_df.columns.tolist(),
                    show=False,
                )
                # hhh
                # if we need to check the feature name, we can add it to the title
                # but it will be extremely massive and not readable
                # try to change the number of images in one PNG
                # axes[i].set_title(f'{feature_name}')
                # plt.gca().set_yticklabels([])

            for i in range(current_features, len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(f'{model_name} - SHAP Summary Plots (Features {start_feature} to {end_feature})', y=1.02)

            plt.tight_layout()
            plt.savefig(os.path.join(path_for_summary, \
                                    f'summary_plots_features_{start_feature}_to_{end_feature}.png'), \
                                        dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Dependence Plots for all features

        # normal
        shap.dependence_plot(
            ind = 0,
            shap_values=shap_values,
            features=x_test
            )

        # set the number of features per plot
        n_features = 64
        n_cols = 8
        n_rows = (n_features + n_cols - 1) // n_cols
        # create a figure with multiple subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 3*n_rows))
        # Fallen
        axes = axes.ravel()
        for i in range(n_features):
            plt.sca(axes[i])
            feature_name = X_test_df.columns[i]
            shap.dependence_plot(
            ind=i,
            shap_values=shap_values,
            features=x_test,
            feature_names=X_test_df.columns.tolist(),
            show=False,
            ax=axes[i]
            )
            axes[i].set_title(f'{feature_name}')

        # if there are more axes than features （because n_rows * n_cols maybe larger than n_features），hide them
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        # layout
        plt.tight_layout()
        # save the figure
        plt.savefig(os.path.join(save_path, 'dependence_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()



        # 4. Bar Plot (all features)
        shap.plots.bar(shap_values_explanation)
        # shap.plots.bar(shap_values)
        # After checking all features，we only need one figure for estimation and comparison


        # 5. Waterfall Plots (containing all features) (same)
        shap.plots.waterfall(shap_values_explanation_for_sample)

        # 4. Violin Plots

        # xgb
        shap.plots.violin(shap_values)

        # 9. Force Plots (每个特征一个)
        # xgb
        print(shap_values_for_sample.shape)
        print(x_test[1].shape)
        # [0,1] [0,1]
        shap.force_plot(explainer.expected_value, shap_values[0], x_test[1],matplotlib=True)

        # After checking all features，we only need one figure for estimation and comparison

        #!!! completely no record
        # n_features = x_test.shape[1]
        # n_cols = 1
        # features_per_plot = 4
        # n_plots = (n_features + features_per_plot - 1) // features_per_plot
        # path_for_force = os.path.join(save_path, 'force_plots')
        # os.makedirs(path_for_force, exist_ok=True)

        # for plot_index in range(n_plots):
        #     start_feature = plot_index * features_per_plot
        #     end_feature = min(start_feature + features_per_plot, n_features)
        #     current_features = end_feature - start_feature
        #     n_rows = (current_features + n_cols - 1) // n_cols

        #     fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6*n_rows))
        #     axes = axes.ravel()

        #     for i in range(current_features):
        #         feature_index = start_feature + i
        #         plt.sca(axes[i])
        #         feature_name = X_test_df.columns[feature_index]
        #         shap.force_plot(
        #             explainer.expected_value,
        #             shap_values=shap_values[1],
        #             features=x_test[1],
        #             feature_names=X_test_df.columns.tolist(),
        #             show=False,
        #         )
        #         # hhh
        #         # if we need to check the feature name, we can add it to the title
        #         # but it will be extremely massive and not readable
        #         # try to change the number of images in one PNG
        #         # axes[i].set_title(f'{feature_name}')
        #         # plt.gca().set_yticklabels([])

        #     for i in range(current_features, len(axes)):
        #         axes[i].set_visible(False)

        #     plt.suptitle(f'{model_name} - SHAP Force Plots (Features {start_feature} to {end_feature})', y=1.02)

        #     plt.tight_layout()
        #     plt.savefig(os.path.join(path_for_force, \
        #                             f'force_plots_features_{start_feature}_to_{end_feature}.png'), \
        #                                 dpi=300, bbox_inches='tight')
        #     plt.close()

    except Exception as e:
        print(f"Error generating plots for {model_name}: {str(e)}")
        print(f"Error occurred in plot generation: {e.__class__.__name__}")
        import traceback
        traceback.print_exc()



pipeline_xgb_pca = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=64)),
    ('xgb', XGBClassifier(eval_metric="logloss"))
])


# define the grid of each classifier's hyperparameters
# all the parameters are tested to make sure they are the best choices

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


# use grid search to find the best hyperparameters for each classifier

grid_xgb_pca = GridSearchCV(pipeline_xgb_pca, param_grid_xgb_pca, cv=3, scoring='accuracy', n_jobs=-1)
grid_xgb_pca.fit(X_train, Y_train)
print(grid_xgb_pca.best_estimator_)


evaluate_model(grid_xgb_pca.best_estimator_, X_test, Y_test, "xgb")



