import numpy as np
import lightgbm as lgb
from sklearn.base import BaseEstimator
from LightGBMConverter import LightGBM

from sklearn.metrics import accuracy_score, r2_score
import os
import tempfile
from sklearn.model_selection import train_test_split

class LightGBMRandomizedSearch(BaseEstimator):
    """
    A custom estimator for LightGBM that includes all optimization techniques. 
    Specifically designed for use with scikit-learn's RandomizedSearchCV.

    Attributes:
        enable_prepruning (bool): Toggles pre-pruning optimization technique.
        enable_pruning (bool): Toggles post-pruning optimization technique.
        pruning_percentile (float): Bottom percentile of trees to prune.
        enable_cegb (bool): Toggles CEGB optimization technique.
        enable_merging (bool): Toggles threshold sharing optimization technique.
        num_clusters (int): Number of clusters to merge thresholds into.
        enable_quantization (bool): Toggles PTQ optimization technique.
        quantization_type (str): Type of quantization scheme to apply.
        quantization_diff (str): Model part to apply quantization to.
        quantization_bits (int): Number of bits to quantize to.
        estimate_size (bool): Toggles between estimating model size and measuring it after compilation.

        Other hyperparameters are the same as those in LightGBM.
    """

    def __init__(self,
                 ### SET HYPERPARAMETERS ###
                 objective='multiclass',
                 num_class=None,
                 metric=None,
                 boosting_type='gbdt',
                 verbosity=-1,
                 device_type="cpu",
                 num_threads = 0,
                 seed = 42,
                 deterministic = False, #Not used
                 force_row_wise = False, #Not used
                 estimate_size = False, #True, False

                ### FIXED HYPERPARAMETERS ### 
                 learning_rate=0.1, 
                 n_estimators=500,
                 lambda_l1 = 0.0,
                 lambda_l2 = 0.0,
                 stopping_rounds = None,

                 ### PREPRUNING HYPERPARAMETERS ###
                 enable_prepruning=False, #True, False
                 max_depth=11,
                 num_leaves=2047,
                 min_gain_to_split=0.0,
                 min_data_in_leaf=20,

                 ### POSTPRUNING HYPERPARAMETERS ###
                 enable_pruning=False, #True, False
                 pruning_percentile=None, #0.0 to 1.0

                 ### CEGB HYPERPARAMETERS ###
                 enable_cegb=False, #True, False
                 cegb_penalty_split = 0.0,
            
                 ### THRESHOLD SHARING HYPERPARAMETERS ###
                 enable_merging=False, #True, False
                 num_clusters=None, #Int
                 
                 ### QUANTIZATION HYPERPARAMETERS ###
                 enable_quantization=False, #True, False
                 quantization_type=None, #affine, scale
                 quantization_diff=None, #leafs, thresholds, both
                 quantization_bits=None, #8, 16
                 ):
        ### SET HYPERPARAMETERS ###
        self.objective = objective
        self.num_class = num_class
        self.metric = metric
        self.boosting_type = boosting_type
        self.verbosity = verbosity
        self.device_type = device_type
        self.num_threads = num_threads
        self.seed = seed
        self.deterministic = deterministic
        self.force_row_wise = force_row_wise
        self.estimate_size = estimate_size

        ### FIXED HYPERPARAMETERS ###
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.stopping_rounds = stopping_rounds

        ### PREPRUNING HYPERPARAMETERS ###
        self.enable_prepruning = enable_prepruning
        self.max_depth = max_depth
        self.min_gain_to_split = min_gain_to_split
        self.min_data_in_leaf = min_data_in_leaf

        ### POSTPRUNING HYPERPARAMETERS ###
        self.enable_pruning = enable_pruning
        self.pruning_percentile = pruning_percentile

        ### CEGB HYPERPARAMETERS ###
        self.enable_cegb = enable_cegb
        self.cegb_penalty_split = cegb_penalty_split

        ### THRESHOLD SHARING HYPERPARAMETERS ###
        self.enable_merging = enable_merging
        self.num_clusters = num_clusters

        ### QUANTIZATION HYPERPARAMETERS ###
        self.enable_quantization = enable_quantization
        self.quantization_type = quantization_type
        self.quantization_diff = quantization_diff
        self.quantization_bits = quantization_bits
        
        ### INTERNAL VALUES ###
        self.lgb = None
        self.model = None
        self.model_size = []
        self.dynamic_size = []
        self.features = None
        self.thresholds = None
    
    def fit(self, X, y):
        """
        Fits the LightGBM model to the training data and applies optimization techniques.

        Args:
            X: array-like of shape (n_samples, n_features), training data.
            y: array-like of shape (n_samples,), target values.
        """
        ### Set optimization technique hyperparameters to default if not enabled. ###
        if not self.enable_cegb:
            self.cegb_penalty_split = 0.0
        if not self.enable_prepruning:
            self.max_depth = 11
            self.num_leaves = 2047
            self.min_data_in_leaf = 20
            self.min_gain_to_split = 0.0
        ### As per LightGBM Documentation, set num_leaves to 2^max_depth - 1 if num_leaves is greater than 2^max_depth - 1. ###
        if self.max_depth > 0:
            self.num_leaves = (2 ** self.max_depth)-1 if self.num_leaves >= (2 ** self.max_depth) else self.num_leaves
        ### Create LightGBM model according to task type. ###
        if self.objective in ['regression', 'regression_l1', 'huber', 'quantile', 'mape', 'poisson']:
             self.lgb = lgb.LGBMRegressor(
                ### SET HYPERPARAMETERS ###
                objective=self.objective,
                metric=self.metric,
                verbosity=self.verbosity,
                boosting_type=self.boosting_type,
                device_type=self.device_type,
                num_threads=self.num_threads,
                seed=self.seed,
                deterministic=self.deterministic,
                force_row_wise=self.force_row_wise,


                ### NATIVE HYPERPARAMETERS ###
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                min_gain_to_split=self.min_gain_to_split,
                cegb_penalty_split=self.cegb_penalty_split,
                min_data_in_leaf=self.min_data_in_leaf,
                lambda_l1=self.lambda_l1,
                lambda_l2=self.lambda_l2,
            )
        else:
            self.lgb = lgb.LGBMClassifier(
                ### SET HYPERPARAMETERS ###
                objective=self.objective,
                num_class = self.num_class,
                metric = self.metric,
                verbosity = self.verbosity,
                boosting_type = self.boosting_type,
                device_type=self.device_type,
                num_threads=self.num_threads,
                seed=self.seed,
                deterministic=self.deterministic,
                force_row_wise=self.force_row_wise,

                ### NATIVE HYPERPARAMETERS ###
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                min_gain_to_split = self.min_gain_to_split,
                cegb_penalty_split = self.cegb_penalty_split,
                min_data_in_leaf = self.min_data_in_leaf,
                lambda_l1 = self.lambda_l1,
                lambda_l2 = self.lambda_l2

            )
            
        callbacks = []
        if self.stopping_rounds is not None:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            callbacks.append(lgb.early_stopping(self.stopping_rounds))

        ### Fit the LightGBM model to the training data. ###    
        self.lgb.fit(
            X, 
            y,
            eval_set=[(X_val, y_val)] if self.stopping_rounds is not None else None,
            callbacks=callbacks,
        )
        ### Parses the LightGBM model from the text file, building a internal representation of the model using LightGBMConverter.py ###
        with tempfile.NamedTemporaryFile(mode = "w+b", delete=False, suffix=".txt") as tmp_file:
            tmp_file.close()
            self.lgb.booster_.save_model(tmp_file.name,num_iteration=self.lgb.booster_.best_iteration)
        self.model = LightGBM()
        self.model.load(tmp_file.name)
        os.remove(tmp_file.name)

        ### Apply threshold sharing if enabled. ###
        if self.enable_merging:
            self.model.mergeThresholds(self.num_clusters)

        ### Apply post-pruning if enabled. ###
        if self.enable_pruning:
            self.model.killTrees(self.pruning_percentile)

        ### Apply quantization if enabled. ###
        if self.enable_quantization:
            self.model.setBits(self.quantization_bits)
            self.model.setQuant(self.quantization_diff)
            self.model.setQuantType(self.quantization_type)
            if self.quantization_type == "affine":
                if self.quantization_diff == "both":
                    test_data = self.model.affineQuantization(X)
                elif self.quantization_diff == "leafs":
                    self.model.affineQuantLeafs()
                elif self.quantization_diff == "thresholds":
                    test_data = self.model.affineQuantThresholds(X)
            elif self.quantization_type == "scale":
                if self.quantization_diff == "both":
                    test_data = self.model.scaleQuantization(X)
                elif self.quantization_diff == "leafs":
                    self.model.scaleQuantLeafs()
                elif self.quantization_diff == "thresholds":
                    test_data = self.model.affineQuantThresholds(X)

        ### Read the model size after compilation or estimate it  ###
        ### Set estimate_size to True if returnMemory() does not work. ###
        if self.estimate_size:
            self.model_size[0] = self.model.estimateMemory()
        else:
            self.model_size, self.dynamic_size = self.model.returnMemory()
            
    def dequantize(self, output):
        """
        Dequantizes the output of the model.

        Args:
            output: array-like of shape (n_samples,), predicted values.

        Returns:
            array-like of shape (n_samples,), dequantized predicted values.
        """
        if self.quantization_type == "affine":
            return [(1/self.model.s) * (x - len(self.model.trees) * self.model.z) for x in output]
        elif self.quantization_type == "scale":
            return [(x/self.model.s) for x in output]
        return
            
    def predict(self, X):
        """
        Makes predictions using the optimized model.

        Args:
            X: array-like of shape (n_samples, n_features), test data.

        Returns:
            output: array-like of shape (n_samples,), predicted values.
        """
        dataCopy = X.copy()
        if self.enable_quantization and (self.quantization_diff == "thresholds" or self.quantization_diff == "both"):
            if self.quantization_type == "affine":
                s, z = self.model.affineGet(self.model.beta, self.model.alpha)
                dataCopy = np.vectorize(lambda x: int(self.model.affineFunction(x, s, z)))(dataCopy)
            elif self.quantization_type == "scale":
                s = self.model.scaleGet(self.model.alpha)
                dataCopy = np.vectorize(lambda x: int(self.model.scaleFunction(x, s)))(dataCopy)
        output = self.model.predict(dataCopy)
        if self.enable_quantization and (self.quantization_diff == "leafs" or self.quantization_diff == "both") and self.model.objective == "regression":
            output = self.dequantize(output)
        return output

class CustomScorer:
    """
    Custom scoring function that measures memory efficiency of the model
    using a custom function. Higher scores indicate better performance with smaller model size.
    """
    def __init__(self):
        pass

    def __call__(self, estimator, X, y):
        """
        Calculate custom score for model evaluation.

        Args:
            estimator: The fitted LightGBMRandomizedSearch model.
            X: array-like of shape (n_samples, n_features), test data.
            y: array-like of shape (n_samples,), true labels.

        Returns:
            Score value representing memory efficiency.
        """
        y_pred = estimator.predict(X)
        if estimator.model.objective == "regression":
            accuracy = r2_score(y, y_pred)
            accuracy = 1/(accuracy + 1e-5)
        else:
            accuracy = accuracy_score(y, y_pred)
            accuracy = np.exp(-1 / (accuracy + 1e-5)) * accuracy
        return accuracy / estimator.model_size[0]
    
class SizeMetric:
    """Metric that returns the model size in memory."""
    
    def __init__(self):
        pass

    def __call__(self, estimator, X, y):
        """
        Calculate model size.

        Args:
            estimator: The fitted LightGBMRandomizedSearch model.
            X: array-like of shape (n_samples, n_features), test data (unused).
            y: array-like of shape (n_samples,), true labels (unused).

        Returns
            Model size in memory.
        """
        return estimator.model_size[0]

    
class ThresholdMetric:
    """Metric that returns the number of thresholds in the model."""
    
    def __init__(self):
        pass

    def __call__(self, estimator, X, y):
        """
        Calculate the number of thresholds.

        Args:
            estimator: The fitted LightGBMRandomizedSearch model.
            X: array-like of shape (n_samples, n_features), test data (unused).
            y: array-like of shape (n_samples,), true labels (unused).

        Returns:
            Number of thresholds.
        """
        return len(estimator.model.thresholds)

class TreeMetric:
    """Metric that returns the number of trees in the model."""
    
    def __init__(self):
        pass

    def __call__(self, estimator, X, y):
        """
        Calculate the number of trees.

       Args:
            estimator: The fitted LightGBMRandomizedSearch model.
            X: array-like of shape (n_samples, n_features), test data (unused).
            y: array-like of shape (n_samples,), true labels (unused).

        Returns:
            Number of trees.
        """
        return len(estimator.model.trees)
   
class AccuracyImprovement:
    """
    Calculates the percentage improvement in accuracy compared to the base model.

    Attributes:
        base_score(float): Accuracy score of the base model to compare against.
    """

    def __init__(self, base_score):
        self.base_score = base_score

    def __call__(self, estimator, X, y):
        """
        Calculate accuracy improvement percentage.

         Args:
            estimator: The fitted LightGBMRandomizedSearch model.
            X: array-like of shape (n_samples, n_features), test data.
            y: array-like of shape (n_samples,), true labels.

        Returns:
            Percentage improvement in accuracy.
        """
        y_pred = estimator.predict(X)
        if estimator.model.objective == "regression":
            accuracy = r2_score(y, y_pred)
        else:
            accuracy = accuracy_score(y, y_pred)
        percent_change = ((accuracy - self.base_score) / self.base_score) * 100
        return percent_change

class SizeImprovement:
    """
    Calculates the percentage improvement in model size compared to the base model.

    Attributes:
        base_size(float): Model size of the base model to compare against.
    """

    def __init__(self, base_size):
        self.base_size = base_size

    def __call__(self, estimator, X, y):
        """
        Calculate model size improvement percentage.

        Args:
            estimator: The fitted LightGBMRandomizedSearch model.
            X: array-like of shape (n_samples, n_features), test data (unused).
            y: array-like of shape (n_samples,), true labels (unused).

        Returns:
            Percentage improvement in model size.
        """
        percent_change = ((estimator.model_size[0]- self.base_size) / self.base_size) * 100
        return percent_change
