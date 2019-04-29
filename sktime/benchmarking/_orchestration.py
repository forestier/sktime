"""Classes for handling input and output of benchmarking experiments training and evalaluating one or multiple prediction strategies on
one or multiple datasets.
"""

from ..benchmarking._io import Results
from ..model_selection import PresplitFilesCV

from sklearn.model_selection import KFold

__all__ = ["Orchestrator"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

# TODO add logging for more detailed feedback
# TODO add separate predict method to Orchestrator for already fitted and saved strategies
# TODO take and save fit/predict timings
# TODO optionally skip strategies if fitted strategies/results already exist
# TODO expand supported backends for data/results classes, including in-memory benchmarking,
#  making the _load_data() and _save_results() methods abstract
# TODO improve user feedback


class Orchestrator:
    """Orchestrator for benchmarking one or more strategies on one or more datasets.
    """
    def __init__(self,
                 data,
                 results,
                 tasks,
                 strategies,
                 cv=None):

        # TODO add input checks for tasks and data
        if len(tasks) != len(data.names):
            raise ValueError('Inconsistent number of datasets and tasks, '
                             'there must be one task for each dataset')
        self.data = data
        self.tasks = tasks

        # if pre-split files cv is passed, use predefined train/test files
        self._validate_cv(cv)
        self.cv = KFold(n_splits=10) if cv is None else cv

        self._validate_strategies(strategies)
        self.strategies = strategies

        if not isinstance(results, Results):
            # TODO replace with more specific checks, making sure classes are easily extendible
            raise ValueError('Passed results object is unknown')
        self.results = results

    def fit_predict(self, save_fitted_strategies=False, save_training_predictions=False, verbose=True):
        """Main method for benchmarking which iterates though all datasets and strategies.
        """

        for task, (dataset_name, dataset) in zip(self.tasks, self.data.load()):
            if verbose:
                print(f'Running strategies on {dataset_name}')

            for strategy_name, strategy in self.strategies:
                for fold, (train_idx, test_idx) in enumerate(self.cv.split(dataset)):

                    # fitting
                    train = dataset.iloc[train_idx]
                    strategy.fit(task, train)
                    if save_fitted_strategies:
                        self.results.save_fitted_strategy(strategy, strategy_name=strategy_name,
                                                          dataset_name=dataset_name, fold=fold)
                    # predict on training set
                    if save_training_predictions:
                        y_pred = strategy.predict(train)
                        y_train = train[task.target]
                        self.results.save_predictions(y_train, y_pred, train_idx, dataset_name=dataset_name,
                                                      strategy_name=strategy_name, train_or_test='train', fold=fold)

                    # predict on test set
                    test = dataset.iloc[test_idx]
                    y_pred = strategy.predict(test)
                    y_test = test[task.target]
                    self.results.save_predictions(y_test, y_pred, test_idx, dataset_name=dataset_name,
                                                  strategy_name=strategy_name, train_or_test='test', fold=fold)

        return self.results

    def predict(self, save_training_predictions=False, verbose=True):
        """Predict from fitted strategies"""
        # self.strategy = self.results.load_fitted_strategy()
        raise NotImplementedError()

    def fit(self, overwrite_fitted_strategies=True):
        """Fit strategies without predicting on train or test set"""
        raise NotImplementedError()

    def _validate_cv(self, cv):
        if isinstance(cv, PresplitFilesCV):
            if self.data.train_test_exists:
                if not hasattr(self.data, '_use_presplit_files'):
                    raise ValueError('Data does not know how to load pre-split train/test files')
                # set flag for loading pre-split files in data object to True
                self.data._use_presplit_files = True

            else:
                raise ValueError('PresplitFilesCV specified, but data does not know about pre-split train/test files')

    @staticmethod
    def _validate_strategies(strategies):
        """Helper function to check validity of passed name-strategy tuples"""
        names, strategies = zip(*strategies)

        # TODO add checks for strategies: all strategies of same type (i.e classifiers or regressors)

        # Unique names
        if len(set(names)) != len(names):
            raise ValueError('Names provided are not unique: '
                             '{0!r}'.format(list(names)))

        # Conflicts with constructor arguments
        # TODO add checks for conflicts with strategy and passed estimator
        # invalid_names = set(names).intersection(self.get_params(deep=False))
        # if invalid_names:
        #     raise ValueError('Estimator names conflict with constructor '
        #                      'arguments: {0!r}'.format(sorted(invalid_names)))

        # Conflicts with double underscore convention
        invalid_names = [name for name in names if '__' in name]
        if invalid_names:
            raise ValueError('Estimator names must not contain __: got '
                             '{0!r}'.format(invalid_names))
