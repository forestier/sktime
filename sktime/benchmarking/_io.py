"""Classes for handling input of data and output of results from orchestration and evaluation of
benchmarking experiments"""

import os
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe

__all__ = ["Data", "Results"]
__author__ = ["Viktor Kazakov", "Markus Löning"]


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

                results = pd.read_csv(os.path.join(filedir, filename), header=0)
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

    def _prepare_save(self, strategy_name, dataset_name):
        """Helper function to keep track of processed datasets and strategies during orchestration"""
        self._append_names(strategy_name, dataset_name)
        filedir = self._make_dir(strategy_name, dataset_name)
        return filedir

    def _make_dir(self, strategy_name, dataset_name):
        """Helper function to create file directories"""
        filedir = os.path.join(self.results_dir, strategy_name, dataset_name)
        if not os.path.exists(filedir):
            # recursively create directory including intermediate-level folders
            os.makedirs(filedir)
        return filedir

    def _append_names(self, strategy_name, dataset_name):
        """Helper function to append names of datasets and strategies to results objects during orchestration"""
        if strategy_name not in self.strategy_names:
            self.strategy_names.append(strategy_name)

        if dataset_name not in self.dataset_names:
            self.dataset_names.append(dataset_name)
