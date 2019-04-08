from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import pandas as pd
import os

__all__ = ["Orchestrator", "Data", "Results"]

# TODO add logging for more detailed feedback
# TODO add separate predict methods for already fitted and saved strategies
# TODO take fit/predict timings
# TODO optionally skip strategies if fitted strategies/results already exist
# TODO integrate data loading functions with dataset submodule
# TODO expand supported backends for data/results classes, including in-memory benchmarking
# TODO let input/output (Data/Results) objects take care of loading data/results and writing results
# TODO make sure strategy names are unique to prevent overwriting of existing results when e.g.
#  comparing different param settings of same strategy


class Orchestrator:
    """Orchestrator for benchmarking one or more strategies on one or more datasets.
    """
    def __init__(self,
                 data=None,
                 tasks=None,
                 strategies=None,
                 strategy_names=None,
                 cv=None,
                 results_dir='results'):

        if len(tasks) != len(data.names):
            raise ValueError('Inconsistent number of datasets and tasks, '
                             'there must be one task for each dataset')

        if len(strategies) != len(strategy_names):
            raise ValueError()

        self.data = data
        self.tasks = tasks
        self.strategies = strategies
        self.strategy_names = strategy_names

        self.results = Results(
            results_dir=results_dir,
            dataset_names=data.names,
            strategy_names=strategy_names
        )

        # use predefined train/test splits
        if cv == 'presplit':
            if self.data.train_test_exists:
                self.cv = _PredefinedSplit()
                self.data._use_presplit = True
            else:
                # raise error if cv is predefined but no predefined splits exist
                raise ValueError('predefined cv specified but no predefined train/test split exist')

        else:
            # set default
            if cv is None:
                self.cv = KFold(n_splits=10)
            else:
                self.cv = cv

    def fit_predict(self, verbose=True):
        """Main method for benchmarking. Iterates though all strategies and through all tasks.
        """

        for task, (dataset_name, dataset) in zip(self.tasks, self.data.load()):

            if verbose:
                print(f'Running strategies on {dataset_name}')

            for strategy_name, strategy in zip(self.strategy_names, self.strategies):

                # fit and predict strategy
                for fold, (train, test) in enumerate(self.cv.split(dataset)):

                        # fit
                        strategy.fit(task, train)

                        # predict
                        y_pred = strategy.predict(test)

                        # save results
                        y_test = test[task.target]
                        self.results.save_results(
                            y_test, y_pred, test.index,
                            dataset_name=dataset_name,
                            strategy_name=strategy_name,
                            train_or_test='test',
                            fold=fold)

        return self.results



class Data:
    """Class for accessing data in orchestration.
    """
    def __init__(self, datadir=None, names=None, train_test_exists=False):

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
                    data = (train, test)

                else:
                    # concatenate into a single dataframe
                    data = self._load_dataset(path, "ALL")
                    data = shuffle(data)  # TODO necessary to reshuffle data?

            else:
                # load single file
                raise NotImplemented

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


class Results:
    """Class for accessing benchmarking results.
    """
    def __init__(self,
                 results_dir=None,
                 dataset_names=None,
                 strategy_names=None):

        self.results_dir = results_dir
        self.dataset_names = dataset_names
        self.strategy_names = strategy_names

    def save_results(self,
                     y_true,
                     y_pred,
                     index,
                     strategy_name=None,
                     dataset_name=None,
                     train_or_test=None,
                     fold=None):

        path = os.path.join(self.results_dir, strategy_name, dataset_name)
        if not os.path.exists(path):
            # recursively create directory including intermediate-level folders
            os.makedirs(path)

        results = pd.DataFrame([index, y_true, y_pred])
        file = train_or_test + str(fold) + '.csv'
        results.to_csv(os.path.join(path, file), index=False, header=False)

    def load_results(self, train_or_test='test', fold=0):
        """Load saved results"""

        # load from hard-drive directory
        return self._load_from_dir(train_or_test, fold)

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

    def load_fitted_strategies(self):
        pass


class _PredefinedSplit:
    """
    Helper class for iterating over predefined splits in orchestration.
    """
    def __init__(self):
        pass

    @staticmethod
    def split(data):
        if isinstance(data, tuple):
            train, test = data
        else:
            raise ValueError()
        yield train, test

