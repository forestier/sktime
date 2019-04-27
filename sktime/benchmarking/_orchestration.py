"""Classes for orchestrating and evaluating benchmarking experiments comparing the predictive performance of multiple
prediction strategies on multiple datasets.
"""

from ..utils.load_data import load_from_tsfile_to_dataframe
from ..model_selection import PresplitFilesCV

from sklearn.model_selection import KFold
import pandas as pd
import os

__all__ = ["Orchestrator", "Data", "Results"]

# TODO add logging for more detailed feedback
# TODO add separate predict method to Orchestrator for already fitted and saved strategies
# TODO take and save fit/predict timings
# TODO optionally skip strategies if fitted strategies/results already exist
# TODO expand supported backends for data/results classes, including in-memory benchmarking,
#  making the _load_data() and _save_results() methods abstract
# TODO sklearn CV iterators return arrays, not pd DataFrames, pass dataframe index instead of data?


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

        # if presplit files cv is passed, use predefined train/test splits
        if isinstance(cv, PresplitFilesCV):
            if self.data.train_test_exists:
                if not hasattr(self.data, '_use_presplit_files'):
                    raise ValueError('Data cannot know how to handle/load pre-split files')
                self.data._use_presplit_files = True
            # raise error if cv is predefined but no predefined splits exist
            else:
                raise ValueError('Predefined cv specified, but no predefined train/test splits exist')
        if cv is None:
            cv = KFold(n_splits=10)
        self.cv = cv

        self._validate_strategies(strategies)
        self.strategies = strategies

        if not isinstance(results, Results):
            # TODO replace with more specific checks, making sure classes are easily extendible
            raise ValueError('Passed results object is unknown')
        self.results = results

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

    def fit_predict(self, save_fitted_strategies=False, save_training_predictions=False, verbose=True):
        """Main method for benchmarking which iterates though all datasets and strategies.
        """

        for task, (dataset_name, dataset) in zip(self.tasks, self.data.load()):
            if verbose:
                # TODO improve user feedback
                print(f'Running strategies on {dataset_name}')

            for strategy_name, strategy in self.strategies:
                for i, (train_idx, test_idx) in enumerate(self.cv.split(dataset)):

                    # fitting
                    train = dataset.iloc[train_idx]
                    strategy.fit(task, train)
                    if save_fitted_strategies:
                        self.results.save_fitted_strategy(strategy, strategy_name=strategy_name,
                                                          dataset_name=dataset_name, fold=i)
                    if save_training_predictions:
                        y_pred = strategy.predict(train)
                        y_train = train[task.target]
                        self.results.save_predictions(y_train, y_pred, train_idx, dataset_name=dataset_name,
                                                      strategy_name=strategy_name, train_or_test='train', fold=i)

                    # prediction
                    test = dataset.iloc[test_idx]
                    y_pred = strategy.predict(test)
                    y_test = test[task.target]
                    self.results.save_predictions(y_test, y_pred, test_idx, dataset_name=dataset_name,
                                                  strategy_name=strategy_name, train_or_test='test', fold=i)

        return self.results

    def predict(self, save_training_predictions=False, verbose=True):
        """Predict from fitted strategies"""
        # self.strategy = self.results.load_fitted_strategy()
        raise NotImplementedError()


class Data:
    """Class for loading data during benchmarking.
    """
    def __init__(self, data_dir, names=None, train_test_exists=False):

        # TODO input checks
        # Input checks.
        if os.path.exists(data_dir):
            self.data_dir = data_dir
        else:
            raise ValueError(f'{data_dir} does not exist')

        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            self.names = names
        else:
            raise ValueError('All names have to be specified as strings')
        
        self.train_test_exists = train_test_exists

        # assigned via cv argument of orchestrator
        self._use_presplit_files = False

    def load(self):
        """Iterator method for loading datasets
        """
        # load from hard-drive directory

        for name in self.names:
            # create path to data files: directory + folder + file
            path = os.path.join(self.data_dir, name, name)

            if self.train_test_exists:
                if self._use_presplit_files:
                    # return train/test data separately in tuple
                    # load predefined train/test files
                    train = self._load(path, 'TRAIN')
                    test = self._load(path, 'TEST')
                    data = pd.concat([train, test], axis=0, keys=['train', 'test']).reset_index(level=1, drop=True)

                else:
                    # concatenate into a single dataframe
                    data = self._load(path, "ALL")
                    # data = shuffle(data)  # TODO necessary to reshuffle data?

            else:
                # load single file
                # TODO
                raise NotImplementedError()

            yield name, data

    @staticmethod
    def _load(path, split):
        """Helper function to load datasets.
        """
        if split in ["TRAIN", "TEST"]:
            file = path + '_' + split + '.ts'
            X, y = load_from_tsfile_to_dataframe(file)

        elif split == "ALL":
            X = pd.DataFrame()
            y = pd.Series()
            for split in ["TRAIN", "TEST"]:
                file = path + '_' + split + '.ts'
                result = load_from_tsfile_to_dataframe(file)
                X = pd.concat([X, pd.DataFrame(result[0])])
                y = pd.concat([y, pd.Series(result[1])])
        else:
            raise ValueError("Invalid split value")

        X['class_val'] = pd.Series(y)
        return X

    def save(self, dataset):
        """Method for storing in-memory datasets into database or specific format
        """
        raise NotImplementedError()


class Results:
    """Results object for storing and accessing benchmarking results.
    """
    def __init__(self, results_dir, dataset_names=None, strategies=None):

        self.results_dir = results_dir
        self.dataset_names = dataset_names if dataset_names is not None else []

        if strategies is not None:
            strategy_names, _ = zip(*strategies)
            self.strategy_names = strategy_names
        else:
            self.strategy_names = []

    def save_predictions(self, y_true, y_pred, index, strategy_name=None, dataset_name=None, train_or_test=None,
                         fold=None):
        """Save predictions"""
        filedir = self._prepare_save(strategy_name, dataset_name)

        filename = train_or_test + str(fold) + '.csv'

        results = pd.DataFrame({'index': index, 'y_true': y_true, 'y_pred': y_pred})
        results.to_csv(os.path.join(filedir, filename), index=False, header=True)

    def load_predictions(self, train_or_test='test', fold=0):
        """Load saved predictions"""

        for strategy_name in self.strategy_names:
            for dataset_name in self.dataset_names:
                filedir = os.path.join(self.results_dir, strategy_name, dataset_name)
                filename = train_or_test + str(fold) + '.csv'

                results = pd.read_csv(os.path.join(filedir, filename), header=True)
                index = results.loc[:, 'index']
                y_true = results.loc[:, 'y_true']
                y_pred = results.loc[:, 'y_pred']

                yield strategy_name, dataset_name, index, y_true, y_pred

    def save_fitted_strategy(self, strategy, strategy_name, dataset_name, fold):
        """Save fitted strategy"""
        filedir = self._prepare_save(strategy_name, dataset_name)
        filename = strategy_name + str(fold)
        strategy.save(os.path.join(filedir, filename))

    def load_fitted_strategy(self, strategy_name, dataset_name, fold):
        """Load saved (fitted) strategy"""
        filedir = self._make_dir(strategy_name, dataset_name)
        filename = strategy_name + str(fold)
        #TODO if we use strategy specific saving function, how do we know how to load them? check file endings?
        raise NotImplementedError()

    def _prepare_save(self, dataset_name, strategy_name):
        """Helper function to keep track of processed datasets and strategies during orchestration"""
        self._append_names(dataset_name, strategy_name)
        filedir = self._make_dir(dataset_name, strategy_name)
        return filedir

    def _make_dir(self, strategy_name, dataset_name):
        """Helper function to create file directories"""
        filedir = os.path.join(self.results_dir, strategy_name, dataset_name)
        if not os.path.exists(filedir):
            # recursively create directory including intermediate-level folders
            os.makedirs(filedir)
        return filedir

    def _append_names(self, dataset_name, strategy_name):
        """Helper function to append names of datasets and strategies to results objects during orchestration"""
        if dataset_name not in self.dataset_names:
            self.dataset_names.append(dataset_name)

        if strategy_name not in self.strategy_names:
            self.strategy_names.append(strategy_name)

