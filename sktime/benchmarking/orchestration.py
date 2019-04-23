"""Classes for orchestrating and evaluating benchmarking experiments comparing the predictive performance of multiple
prediction strategies on multiple datasets.
"""

from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
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
                 tasks,
                 strategies,
                 cv=None,
                 results_dir=None):

        # TODO add input checks for tasks and data
        if len(tasks) != len(data.names):
            raise ValueError('Inconsistent number of datasets and tasks, '
                             'there must be one task for each dataset')
        self.data = data
        self.tasks = tasks

        self._validate_strategies(strategies)
        self.strategies = strategies

        # use predefined train/test splits
        if cv == 'presplit':
            if self.data.train_test_exists:
                self.cv = PredefinedSplit()
                self.data._use_presplit = True
            else:
                # raise error if cv is predefined but no predefined splits exist
                raise ValueError('Predefined cv specified, but no predefined train/test splits exist')

        else:
            # set default
            if cv is None:
                self.cv = KFold(n_splits=10)
            else:
                self.cv = cv

        # TODO allow users to pass results object for whatever backend they prefer
        self.results = Results(results_dir=results_dir,
                               dataset_names=data.names,
                               strategies=self.strategies)

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

    def run(self, overwrite_fitted_strategies=False, verbose=True):
        """Main method for benchmarking which iterates though all datasets and strategies.
        """

        for task, (dataset_name, dataset) in zip(self.tasks, self.data.load()):

            if verbose:
                # TODO improve user feedback
                print(f'Running strategies on {dataset_name}')

            for strategy_name, strategy in self.strategies:
                # fit and predict strategy
                for i, (train, test) in enumerate(self.cv.split(dataset)):
                        strategy.fit(task, train)

                        if overwrite_fitted_strategies:
                            self.results.save_fitted_strategy()

                        y_pred = strategy.predict(test)

                        # save results
                        y_test = test[task.target]
                        self.results.save_predictions(
                            y_test,
                            y_pred,
                            test.index,
                            dataset_name=dataset_name,
                            strategy_name=strategy_name,
                            train_or_test='test',
                            fold=i)

        return self.results

    def run_from_fitted_strategies(self):
        # self.strategy = self.results.load_fitted_strategy()
        raise NotImplementedError()


class Data:
    """Class for loading data during benchmarking.
    """
    def __init__(self, datadir=None, names=None, train_test_exists=False):

        # TODO input checks
        # Input checks.
        # if isinstance(datasets, list) and all(isinstance(dataset, pd.DataFrame) for dataset in datasets):
        #     self.data = datasets
        #     self._is_dir = False

        if os.path.exists(datadir):
            self.datadir = datadir

        else:
            raise ValueError(f'{datadir} does not exist')

        if isinstance(names, list) and all(isinstance(name, str) for name in names):
            self.names = names
        else:
            raise ValueError()
        
        self.train_test_exists = train_test_exists

        # assigned via cv argument of orchestrator
        self._use_presplit = False

    def load(self):
        """
        Iterator method for loading datasets
        """
        # load from hard-drive directory
        return self._load_from_dir()

    def _load_from_dir(self):
        """Iterator helper function for loading datasets from directory"""

        for name in self.names:
            # create path to data files: directory + folder + file
            path = os.path.join(self.datadir, name, name)

            if self.train_test_exists:

                if self._use_presplit:
                    # return train/test data separately in tuple
                    # load predefined train/test files
                    train = self._load_dataset(path, 'TRAIN')
                    test = self._load_dataset(path, 'TEST')
                    data = pd.concat([train, test], axis=0, keys=['train', 'test']).reset_index(level=1, drop=True)

                else:
                    # concatenate into a single dataframe
                    data = self._load_dataset(path, "ALL")
                    # data = shuffle(data)  # TODO necessary to reshuffle data?

            else:
                # load single file
                # TODO
                raise NotImplementedError()

            yield name, data

    @staticmethod
    def _load_dataset(path, split):
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
        """Method for storing datasets in some database or specific format
        """
        raise NotImplementedError()


class Results:
    """Results object for storing and accessing benchmarking results.
    """
    def __init__(self, results_dir=None, dataset_names=None, strategies=None):

        self.results_dir = os.path.join(os.getcwd(), 'results') if results_dir is None else results_dir
        self.dataset_names = dataset_names

        strategy_names, _ = zip(*strategies)
        self.strategy_names = strategy_names

    def save_predictions(self, y_true, y_pred, index, strategy_name=None, dataset_name=None, train_or_test=None,
                         fold=None):

        filedir = self._make_dir(strategy_name, dataset_name)
        filename = train_or_test + str(fold) + '.csv'

        results = pd.DataFrame([index, y_true, y_pred])
        results.to_csv(os.path.join(filedir, filename), index=False, header=False)

    def load_predictions(self, train_or_test='test', fold=0):
        """Load saved results"""

        # load from hard-drive directory
        return self._load_from_dir(train_or_test, fold)

    def save_fitted_strategy(self, strategy, strategy_name=None, dataset_name=None, fold=None):
        filedir = self._make_dir(strategy_name, dataset_name)
        filename = strategy_name + str(fold)
        strategy.save(os.path.join(filedir, filename))

    def load_fitted_strategy(self, filepath):
        raise NotImplementedError()

    def _make_dir(self, strategy_name, dataset_name):
        """Helper function to create file directories"""
        filedir = os.path.join(self.results_dir, strategy_name, dataset_name)
        if not os.path.exists(filedir):
            # recursively create directory including intermediate-level folders
            os.makedirs(filedir)
        return filedir

    def _load_from_dir(self, train_or_test, fold):
        """Helper function for loading results"""

        for strategy_name in self.strategy_names:
            for dataset_name in self.dataset_names:

                path = os.path.join(self.results_dir, strategy_name, dataset_name)
                file = train_or_test + str(fold) + '.csv'

                results = pd.read_csv(os.path.join(path, file), header=None)
                idx = results.iloc[:, 0]
                y_true = results.iloc[:, 1]
                y_pred = results.loc[:, 2]

                yield strategy_name, dataset_name, idx, y_true, y_pred


class PredefinedSplit:
    """
    Helper class for iterating over predefined splits in orchestration.
    """
    def __init__(self, check_input=True):
        self.check_input = check_input

    def split(self, data):
        # Input checks.
        if self.check_input:
            if not isinstance(data, pd.DataFrame):
                raise ValueError(f'Data must be pandas dataframe, but found {type(data)}')
            if not np.all(data.index.unique().isin(['train', 'test'])):
                raise ValueError('Train-test split not properly defined in index of passed pandas dataframe')

        train = data.loc['train'].reset_index(drop=True)
        test = data.loc['test'].reset_index(drop=True)
        yield train, test

