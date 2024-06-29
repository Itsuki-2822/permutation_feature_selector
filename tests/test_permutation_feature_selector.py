import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from permutation_feature_selector import PermutationFeatureSelector

@pytest.fixture(scope='module')
def iris_data():
    X, y = load_iris(return_X_y=True)
    X = pd.DataFrame(X, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X, y, X_train, X_test, y_train, y_test

@pytest.fixture(scope='module')
def trained_model(iris_data):
    X_train, y_train = iris_data[2], iris_data[4]
    model = lgb.LGBMClassifier()
    model.fit(X_train, y_train)
    return model

class Test_PermutationFeatureSelector:
    def test_initialization(self, iris_data, trained_model):
        X, y, _, X_test, _, y_test = iris_data
        selector = PermutationFeatureSelector(trained_model, X_test, y_test)
        assert selector.model == trained_model
        assert selector.X_test.shape[0] == X_test.shape[0]
        assert selector.y_test.shape[0] == y_test.shape[0]

    def test_calculate_base_score(self, iris_data, trained_model):
        _, _, _, X_test, _, y_test = iris_data
        selector = PermutationFeatureSelector(trained_model, X_test, y_test, metric='accuracy')
        base_score = selector._calculate_base_score()
        assert isinstance(base_score, float)

    def test_calculate_permutation_importance(self, iris_data, trained_model):
        _, _, _, X_test, _, y_test = iris_data
        selector = PermutationFeatureSelector(trained_model, X_test, y_test, metric='accuracy')
        importance = selector.calculate_permutation_importance()
        assert len(importance) == X_test.shape[1]

    def test_plot_permutation_importance(self, iris_data, trained_model):
        _, _, _, X_test, _, y_test = iris_data
        selector = PermutationFeatureSelector(trained_model, X_test, y_test, metric='accuracy')
        selector.calculate_permutation_importance()
        selector.plot_permutation_importance()

    def test_choose_feat(self, iris_data, trained_model):
        _, _, _, X_test, _, y_test = iris_data
        selector = PermutationFeatureSelector(trained_model, X_test, y_test, metric='accuracy')
        chosen_features, chosen_features_df = selector.choose_feat()
        assert isinstance(chosen_features, list)
        assert not chosen_features_df.empty

    def test_invalid_metric(self, iris_data):
        X, y = iris_data[0], iris_data[1]
        model = lgb.LGBMClassifier()
        with pytest.raises(ValueError):
            selector = PermutationFeatureSelector(model, X, y, metric='invalid_metric')

    def test_invalid_threshold_method(self, iris_data, trained_model):
        _, _, _, X_test, _, y_test = iris_data
        selector = PermutationFeatureSelector(trained_model, X_test, y_test)
        with pytest.raises(ValueError):
            selector.choose_feat(threshold_method='invalid_method')

    def test_data_mismatch(self, iris_data):
        X, y = iris_data[0], iris_data[1]
        model = lgb.LGBMClassifier()
        with pytest.raises(ValueError):
            selector = PermutationFeatureSelector(model, X[:100], y)

    def test_result_validity_iris(self, iris_data, trained_model):
        X, _, _, X_test, _, y_test = iris_data
        selector = PermutationFeatureSelector(trained_model, X_test, y_test, metric='accuracy')
        importance = selector.calculate_permutation_importance()
        assert len(importance) == X_test.shape[1]
