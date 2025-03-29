import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, roc_auc_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import lightgbm as lgb

class PermutationFeatureSelector:
    def __init__(self, model, X_test, y_test, metric='rmse', n_repeats=30, random_state=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.use_wrapper = isinstance(self.model, lgb.Booster)
        self.metric_funcs = self._init_metric_funcs()
        self.base_score = self._calculate_base_score()
        self.perm_importance = None
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _init_metric_funcs(self):
        return {
            'rmse': lambda y, y_pred: -np.sqrt(mean_squared_error(y, y_pred)),
            'mae': lambda y, y_pred: -mean_absolute_error(y, y_pred),
            'r2': r2_score,
            'mape': lambda y, y_pred: -mean_absolute_percentage_error(y, y_pred),
            'auc': roc_auc_score,
            'accuracy': accuracy_score,
            'f1': f1_score
        }

    class ModelWrapper:
        def __init__(self, model, use_wrapper, metric_funcs):
            self.model = model
            self.use_wrapper = use_wrapper
            self.metric_funcs = metric_funcs

        def predict(self, X):
            if self.use_wrapper:
                return self.model.predict(X, num_iteration=-1)  # 修正: `best_iteration` ではなく `-1` をデフォルト
            else:
                return self.model.predict(X)

        def score(self, X, y, metric):
            if metric not in self.metric_funcs:
                raise ValueError(f"Unsupported metric: {metric}")
            preds = self.predict(X)
            if metric in ['accuracy', 'f1'] and self.use_wrapper:
                preds = (preds > 0.5).astype(int)
            return self.metric_funcs[metric](y, preds)

    def _calculate_base_score(self):
        wrapped_model = self.ModelWrapper(self.model, self.use_wrapper, self.metric_funcs)
        return wrapped_model.score(self.X_test, self.y_test, self.metric)

    def calculate_permutation_importance(self):
        wrapped_model = self.ModelWrapper(self.model, self.use_wrapper, self.metric_funcs)
        feature_importances = np.zeros(self.X_test.shape[1])

        self.X_test = self.X_test.replace({pd.NA: np.nan})

        for col in range(self.X_test.shape[1]):
            scores = np.zeros(self.n_repeats)
            for n in range(self.n_repeats):
                X_permuted = self.X_test.copy()
                X_permuted.iloc[:, col] = np.random.permutation(X_permuted.iloc[:, col])
                permuted_score = wrapped_model.score(X_permuted.values, self.y_test, self.metric)

                permuted_score = wrapped_model.score(X_permuted, self.y_test, self.metric)
                scores[n] = permuted_score
            
            if np.isnan(scores).all():
                feature_importances[col] = 0  
            else:
                feature_importances[col] = self.base_score - np.nanmean(scores) 
        
        self.perm_importance = feature_importances
        return feature_importances

    def plot_permutation_importance(self, figsize=(10, 8), positive_color='blue', negative_color='red'):
        if self.perm_importance is None:
            perm_importance = self.calculate_permutation_importance()
        else:
            perm_importance = self.perm_importance
        
        perm_importance_df = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Importance': perm_importance
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=figsize)
        colors = perm_importance_df['Importance'].apply(lambda x: negative_color if x < 0 else positive_color)
        plt.barh(perm_importance_df['Feature'], perm_importance_df['Importance'], color=colors)
        plt.xlabel('Mean Accuracy Decrease')
        plt.ylabel('Feature')
        plt.title('Permutation Importance')
        plt.gca().invert_yaxis()
        plt.show()

    def choose_feat(self, threshold_method='mean', threshold_value=1.0):
        if self.perm_importance is None:
            perm_importance = self.calculate_permutation_importance()
        else:
            perm_importance = self.perm_importance

        if threshold_method == 'mean':
            threshold = np.mean(perm_importance) * threshold_value
        elif threshold_method == 'median':
            threshold = np.median(perm_importance) * threshold_value
        elif threshold_method == 'value':
            threshold = threshold_value
        else:
            raise ValueError(f"Unsupported threshold method: {threshold_method}")

        chosen_features = self.X_test.columns[perm_importance > threshold].tolist()
        chosen_features_df = pd.DataFrame({
            'Feature': self.X_test.columns[perm_importance > threshold],
            'Importance': perm_importance[perm_importance > threshold]
        }).sort_values(by='Importance', ascending=False)

        return chosen_features, chosen_features_df

#permutation_importance = PermutationFeatureSelector(model, X_test, y_test, metric='rmse', random_state=42)
#permutation_importance.plot_permutation_importance(figsize=(12, 10), positive_color='blue', negative_color='red')
#selected_features, selected_features_df = permutation_importance.choose_feat(threshold_method='value', threshold_value=1) 
#display(len(selected_features))
#display(selected_features_df)
