import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import lightgbm as lgb
from permutation_feature_selector import PermutationFeatureSelector

class Test_PermutationFeatureSelector:

    # 初期化テスト
    def test_initialization(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        model = lgb.LGBMClassifier()
        selector = PermutationFeatureSelector(model, X, y)
        assert selector.model == model
        assert selector.X_test.shape[0] == X.shape[0]
        assert selector.y_test.shape[0] == y.shape[0]

    # 基本スコア計算テスト
    def test_calculate_base_score(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
        selector = PermutationFeatureSelector(model, X_test, y_test, metric='accuracy')
        base_score = selector._calculate_base_score()
        assert isinstance(base_score, float)

    # パーミュテーション重要度計算テスト
    def test_calculate_permutation_importance(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
        selector = PermutationFeatureSelector(model, X_test, y_test, metric='accuracy')
        importance = selector.calculate_permutation_importance()
        assert len(importance) == X.shape[1]

    # プロット生成テスト
    def test_plot_permutation_importance(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
        selector = PermutationFeatureSelector(model, X_test, y_test, metric='accuracy')
        selector.calculate_permutation_importance()
        selector.plot_permutation_importance()

    # 重要な特徴量の選択テスト
    def test_choose_feat(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
        selector = PermutationFeatureSelector(model, X_test, y_test, metric='accuracy')
        chosen_features, chosen_features_df = selector.choose_feat()
        assert isinstance(chosen_features, list)
        assert not chosen_features_df.empty

    # 不正なパラメータの取り扱いテスト
    def test_invalid_metric(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        model = lgb.LGBMClassifier()
        with pytest.raises(ValueError):
            selector = PermutationFeatureSelector(model, X, y, metric='invalid_metric')

    def test_invalid_threshold_method(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        model = lgb.LGBMClassifier()
        selector = PermutationFeatureSelector(model, X, y)
        with pytest.raises(ValueError):
            selector.choose_feat(threshold_method='invalid_method')

    # データの不整合テスト
    def test_data_mismatch(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        model = lgb.LGBMClassifier()
        with pytest.raises(ValueError):
            selector = PermutationFeatureSelector(model, X[:100], y)

    # 結果の妥当性テスト
    def test_result_validity_iris(self):
        X, y = load_iris(return_X_y=True)
        X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMClassifier()
        model.fit(X_train, y_train)
        selector = PermutationFeatureSelector(model, X_test, y_test, metric='accuracy')
        importance = selector.calculate_permutation_importance()
        assert len(importance) == X.shape[1]

    def test_result_validity_boston(self):
        data = load_boston()
        X, y = data.data, data.target
        X = pd.DataFrame(X, columns=data.feature_names)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = lgb.LGBMRegressor()
        model.fit(X_train, y_train)
        selector = PermutationFeatureSelector(model, X_test, y_test, metric='rmse')
        importance = selector.calculate_permutation_importance()
        assert len(importance) == X.shape[1]
