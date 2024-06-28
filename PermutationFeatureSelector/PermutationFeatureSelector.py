import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, roc_auc_score, accuracy_score
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
        self.use_wrapper = self._should_use_wrapper()
        self.base_score = self._calculate_base_score()
        self.perm_importance = None
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _should_use_wrapper(self):
        return isinstance(self.model, lgb.Booster)

    class ModelWrapper:
        def __init__(self, model, use_wrapper):
            self.model = model
            self.use_wrapper = use_wrapper

        def predict(self, X):
            if self.use_wrapper:
                return self.model.predict(X, num_iteration=self.model.best_iteration)
            else:
                return self.model.predict(X)

        def score(self, X, y, metric):
            preds = self.predict(X)
            if metric == 'rmse':
                return -np.sqrt(mean_squared_error(y, preds))
            elif metric == 'mae':
                return -mean_absolute_error(y, preds)
            elif metric == 'r2':
                return r2_score(y, preds)
            elif metric == 'mape':
                return -mean_absolute_percentage_error(y, preds)
            elif metric == 'auc':
                return roc_auc_score(y, preds)
            elif metric == 'accuracy':
                if self.use_wrapper:
                    preds = (preds > 0.5).astype(int)
                return accuracy_score(y, preds)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

    def _calculate_base_score(self):
        wrapped_model = self.ModelWrapper(self.model, self.use_wrapper)
        return wrapped_model.score(self.X_test, self.y_test, self.metric)

    def calculate_permutation_importance(self):
        wrapped_model = self.ModelWrapper(self.model, self.use_wrapper)
        feature_importances = np.zeros(self.X_test.shape[1])

        for col in range(self.X_test.shape[1]):
            scores = np.zeros(self.n_repeats)
            for n in range(self.n_repeats):
                X_permuted = self.X_test.copy()
                X_permuted.iloc[:, col] = np.random.permutation(X_permuted.iloc[:, col])
                permuted_score = wrapped_model.score(X_permuted.values, self.y_test, self.metric)
                scores[n] = permuted_score
            feature_importances[col] = self.base_score - np.mean(scores)

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