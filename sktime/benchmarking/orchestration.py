import logging
import json
import numpy as np
import os
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import re
import pandas as pd
from sktime.highlevel import Task


class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.
    """
    def __init__(self,
                data_holders=None,
                data_loader=None,
                data_dir='data',
                experiments_trained_estimators_dir='trained_estimators',
                experiments_results_dir='results',
                save_results=False):
        """
        Parameters
        ----------
        data_holders: list 
            list of sktime DataHolder objects
        data_loader: sktime.data.dataloader
            DataLoader object
        data_dir: string
            data directory for saving output of orchestrator
        experiments_trained_estimators_dir: string
            path on the disk for saving the trained estimators
        experiments_results_dir: string
            path for saving json objects with results
        save_results: Boolean
             If True saves results to HDD
        """

        self._data_dir = data_dir
        self._experiments_trained_estimators_dir = experiments_trained_estimators_dir
        self._experiments_results_dir = experiments_results_dir
        self._save_results = save_results

    def _save_results(self, results, save_path, overwrite_predictions):
        """
        Saves the results of the experiments to disk
        
        Parameters
        ----------
        results: json
            json with the results of the experiment
        save_path: string
            directory where the results will be saved
        overwrite_predictions: Boolean
            If True overwrites previous saved results
        """
        save_file_exists = os.path.isfile(save_path)

        #strip the filename fron the save path
        re_srch = re.search('.*\/',save_path)
        dir_only = save_path[re_srch.span()[0]:re_srch.span()[1]]
        if save_file_exists is False or (save_file_exists is True and overwrite_predictions is True):
            if not os.path.exists(dir_only):
                os.makedirs(dir_only)
            with open(save_path, 'w') as outfile:
                json.dump(results, outfile)
            logging.warning(f'Saved predictions to: {save_path}')
        else:
            logging.warning(f'Path {save_path} exists. Set overwrite_predictions=True if you wish to overwrite it.')
    
    def fit(self, 
            data, 
            strategies, 
            resampling=None,
            overwrite_saved_estimators=False,
            verbose=True,
            predict_on_runtime=True,
            overwrite_predictions=False,
            save_resampling_splits=True):
        """
        Combines both fit from memory and from hdd

        strategies: list
            list of sktime strategies
        resampling: sktime.resampling object
            resampling strategy for the data.
        overwrite_saved_estimators:Boolean
            If True overwrites the esimators that were saved on the disk
        verbose:Boolean
            If True outputs messages during training
        predict_on_runtime:Boolean
            If True makes predictions after the estimator is trained
        overwrite_preictions:Boolean
            If True overwrites the predictions made at previous runs
        save_resampling_splits: Boolean
            If `True` saves resampling splits in database
        
        Returns
        -------
        list 
            sktime.experiments.data.Results objects 
        """

        if isinstance(data, DataLoader):
            return self.run_from_disk(data_loader=data, 
                               strategies=strategies, 
                               predict_on_runtime=predict_on_runtime)

        elif all(isinstance(d, DataHolder) for d in data):
            if resampling is None:
                raise ValueError('Specify resampling strategy')
            return self.run_from_memory(data_holders=data,
                                strategies=strategies,
                                resampling=resampling,
                                overwrite_saved_estimators=overwrite_saved_estimators,
                                verbose=verbose,
                                predict_on_runtime=predict_on_runtime,
                                overwrite_predictions=overwrite_predictions,
                                save_resampling_splits=save_resampling_splits)
        else:
            raise ValueError('Invalid data argument')
        
    def run_from_memory(self,
            data_holders,
            strategies,
            resampling,
            overwrite_saved_estimators=False,
            verbose=True,
            predict_on_runtime=True,
            overwrite_predictions=False,
            save_resampling_splits=True):
        """
        Main method for running the experiments. Iterates though all strategies and through all tasks.

        Parameters
        ----------
        data_holders: list 
            list of sktime DataHolder objects
        strategies: list
            list of sktime strategies
        resampling: sktime.resampling object
            resampling strategy for the data.
        overwrite_saved_estimators:Boolean
            If True overwrites the esimators that were saved on the disk
        verbose:Boolean
            If True outputs messages during training
        predict_on_runtime:Boolean
            If True makes predictions after the estimator is trained
        overwrite_preictions:Boolean
            If True overwrites the predictions made at previous runs
        save_resampling_splits: Boolean
            If `True` saves resampling splits in database
        
        Returns
        -------
        list 
            sktime.highlevel.results objects 
        """
        results_list = []
        for dh in data_holders:
            logging.warning(f'Training estimators on {dh.dataset_name}')
            data = dh.data
            idx = np.arange(data.shape[0])

            train_idx, test_idx = resampling.resample(idx)
            if save_resampling_splits is True:
                dh_idx = data_holders.index(dh)
                dh.set_resampling_splits(train_idx=train_idx, test_idx=test_idx)
                data_holders[dh_idx]=dh

            for strat in strategies:
                strategy = self._train_strategy(strategy=strat, 
                                              data_holder=dh, 
                                              train_idx=train_idx)
                if predict_on_runtime is True:
                    result = self.predict(strategy=strategy, data_holder=dh, test_idx=test_idx)
                    results_list.append(result)
                       
        return results_list
        

    def run_from_disk(self, data_loader, strategies, predict_on_runtime=True):
        """
        data_loader : sktime.data.DataLoader
            instance of DataLoader class
        strategies: list
            list of sktime strategies
        predict_on_runtime:Boolean
            If True makes predictions after the estimator is trained
        """

        while True:
            try:
                dh_train, dh_test = data_loader.load(load_test_train='both')
                logging.warning(f'Training estimators on {dh_train.dataset_name}')
                results_list = []
                for strat in strategies:
                    strategy = self._train_strategy(strategy=strat, 
                                              data_holder=dh_train)
                    if predict_on_runtime is True:
                        result = self.predict(strategy=strategy, data_holder=dh_test)
                        results_list.append(result)
                       
            except StopIteration:
                logging.warning('All datasets completed')
                return results_list
                

    def _train_strategy(self, strategy, data_holder, train_idx=None):
        """
        Trains the strategy and saves the results to disk

        Parameters
        ----------

        strategy : sktime.highlevel.strategy
            sktime strategy object
        data_holder : sktime.experiments.data.DataHolder
            sktime DataHolder object
        train_idx : array
            indices of the training data. Default is `None` which means that all data will be used for training. This is usefull if the dataset was pre-split.

        Returns
        -------
        sktime.highlevel.strategy
            trained sktime strategy
        """
        data = data_holder.data
        if train_idx is None:
            train_idx = np.arange(data.shape[0])
        #check whether the model was already trained
        path_to_check = os.path.join(self._data_dir, 
                                        self._experiments_results_dir,
                                        data_holder.dataset_name, 
                                        strategy.name)

        strategy_estists = os.path.isfile(path_to_check)

        if strategy_estists is True and overwrite_saved_estimators is False:
            if verbose is True:
                logging.warning(f'Estimator {strategy.name} already trained on {data_holder.dataset_name}. Skipping it.')
        
        else:
            strategy.fit(data_holder.task, data.iloc[train_idx])
            #TODO: save trained strategy to disk

            return strategy

    def predict(self, strategy, data_holder, test_idx=None):
        """
        Makes predictions on trained strategy

        Parameters
        ----------
        strategy : sktime.highlevel.strategy
            trained strategy
        data_holder : sktime.experiments.data.DataHolder
            sktime DataHolder object
        test_idx : array
            indeces of the test data. If None the entire dataset is used
        Returns
        -------
        sktime.experiments.data.Result
            prediction result
        """
        if test_idx is None:
            test_idx = np.arange(data_holder.data.shape[0])
        logging.warning(f'Making predictions for strategy: {strategy.name} on dataset: {data_holder.dataset_name}')
        predictions = strategy.predict(data_holder.data.iloc[test_idx])
        #create a results object
        y = data_holder.data[data_holder.task.target]
        result = Result(dataset_name=data_holder.dataset_name,
                        strategy_name=strategy.name,
                        true_labels=y.iloc[test_idx].tolist(),
                        predictions=predictions.tolist()) 
        
        
        if self._save_results:
            result_to_save = pd.DataFrame()
            result_to_save['TRUE_LABELS']=y.iloc[test_idx]
            result_to_save['PREDICTIONS']=predictions
            path_to_save = os.path.join(self._data_dir, 
                                        self._experiments_results_dir, 
                                        data_holder.dataset_name)
            try:
                os.makedirs(path_to_save)
                result_to_save.to_csv(f'{path_to_save}{os.sep}{strategy.name}.csv', index=False)
            except:
                #directory already exists
                result_to_save.to_csv(f'{path_to_save}{os.sep}{strategy.name}.csv', index=False)
        
        return result

    import os
    from sktime.utils.load_data import load_from_tsfile_to_dataframe
    import re
    import pandas as pd
    from sktime.highlevel import Task

    class DataHolder:
        """
        Class for holdig the data, schema, resampling splits and metadata
        """

        def __init__(self, data, task, dataset_name):
            """
            Parameters
            ----------
            data: pandas DataFrame
                dataset in pandas format
            task: sktime task
                sktime task object
            dataset_name: string
                Name of the datasets
            """

            self._data = data
            self._task = task
            self._dataset_name = dataset_name

        @property
        def data(self):
            return self._data

        @property
        def task(self):
            return self._task

        @property
        def dataset_name(self):
            return self._dataset_name

        def set_resampling_splits(self, train_idx, test_idx):
            """
            Saves the train test indices after the data is resampled

            Parameters
            -----------
            train_idx: numpy array
                array with indices of the train set
            test_idx: numpy array
                array with indices of the test set
            """
            self._train_idx = train_idx
            self._test_idx = test_idx

    class DataLoader:
        """
        Class for loading large amounts of datasets. Usefull for automation of experiments. All datasets need to be saved under the same directory
        """

        def __init__(self, dts_dir, task_types, train_test_exists=True):
            """
            Parameters
            ----------
            dts_dir : string
                root directory where the datasets are saved
            train_test_exists : Boolean
                If True the datasets are split with the suffix _TRAIN and _TEST
            task_types : string
                TSC or TSR
            """

            self._dts_dir = dts_dir
            self._train_test_exists = train_test_exists
            self._task_types = task_types

            self._all_datasets = iter(os.listdir(dts_dir))

        def _load_ts(self, load_train=None, load_test=None, task_target='target'):
            """
            Iterates and loads the next dataset

            Parameters
            ----------

            load_train : Boolean
                loads the train set
            load_test : Boolean
                loads the test set
            task_target : String
                Name of the DataFrame column containing the label
            Returns
            -------
            tuple of sktime.dataholder
                dataholder_train and dataholder_test.
            """

            if self._train_test_exists:
                if (load_train is None) and (load_test is None):
                    raise ValueError('At least the train or the test set needs to be loaded')
            # get the next dataset in the list
            dts_name = next(self._all_datasets)

            datasets = os.listdir(os.path.join(self._dts_dir, dts_name))

            dataholder_train = None
            dataholder_test = None

            if self._train_test_exists:
                re_train = re.compile('.*_TRAIN.ts')
                re_test = re.compile('.*_TEST.ts')

                # find index of test or train file
                if load_train:
                    data = pd.DataFrame()

                    for dts in datasets:
                        if re_train.match(dts):
                            loaded_dts = load_from_tsfile_to_dataframe(os.path.join(self._dts_dir, dts_name, dts))
                            if len(loaded_dts) == 2:
                                data = loaded_dts[0]
                                y = loaded_dts[1]
                                data[task_target] = y
                                task = Task(case=self._task_types, dataset_name=dts_name, data=data, target=task_target)
                                dataholder_train = DataHolder(data=data, task=task, dataset_name=dts_name)
                if load_test:
                    data = pd.DataFrame()

                    for dts in datasets:
                        if re_test.match(dts):
                            loaded_dts = load_from_tsfile_to_dataframe(os.path.join(self._dts_dir, dts_name, dts))
                            if len(loaded_dts) == 2:
                                data = loaded_dts[0]
                                y = loaded_dts[1]
                                data[task_target] = y
                                task = Task(case=self._task_types, dataset_name=dts_name, data=data, target=task_target)
                                dataholder_test = DataHolder(data=data, task=task, dataset_name=dts_name)

            return dataholder_train, dataholder_test

        def load(self, load_test_train='both', task_target='target'):
            """
            Method for loading sequentially the data irrespective of how it was supplied, i.e. ts files, csv, db.

            Parameters
            ----------

            load_test_train : String
                Acceptable valies: test, train, both. Loads either the test set, the train set or both.
            load_test : Boolean
                loads the test set
            task_target : String
                Name of the DataFrame column containing the label
            Returns
            -------
            sktime.dataholder
                dataholder ready for running experiments
            """
            # TODO expand as the list of backend functions grows

            acceptable_test_train_values = ['test', 'train', 'both']
            if load_test_train not in acceptable_test_train_values:
                raise ValueError('Acceptable values for the load_test_train parameter are: test, train or both')

            if load_test_train == 'train':
                return self._load_ts(load_train=True, task_target=task_target)
            if load_test_train == 'test':
                return self._load_ts(load_test=True, task_target=task_target)
            if load_test_train == 'both':
                return self._load_ts(load_train=True, load_test=True, task_target=task_target)

    class Result:
        """
        Class for storing the results of the orchestrator
        """

        def __init__(self, dataset_name, strategy_name, true_labels, predictions):
            """
            Parameters
            -----------
            dataset_name: string
                name of the dataset
            strategy_name: string
                name of strategy
            true_labels: array
                true labels
            predictions: array
                predictions of estimator
            """

            self._dataset_name = dataset_name
            self._strategy_name = strategy_name
            self._true_labels = true_labels
            self._predictions = predictions

        @property
        def dataset_name(self):
            return self._dataset_name

        @property
        def strategy_name(self):
            return self._strategy_name

        @property
        def true_labels(self):
            return self._true_labels

        @property
        def predictions(self):
            return self._predictions


class DataHolder:
    """
    Class for holdig the data, schema, resampling splits and metadata
    """

    def __init__(self, data, task, dataset_name):
        """
        Parameters
        ----------
        data: pandas DataFrame
            dataset in pandas format
        task: sktime task
            sktime task object
        dataset_name: string
            Name of the datasets
        """

        self._data = data
        self._task = task
        self._dataset_name = dataset_name

    @property
    def data(self):
        return self._data

    @property
    def task(self):
        return self._task

    @property
    def dataset_name(self):
        return self._dataset_name

    def set_resampling_splits(self, train_idx, test_idx):
        """
        Saves the train test indices after the data is resampled

        Parameters
        -----------
        train_idx: numpy array
            array with indices of the train set
        test_idx: numpy array
            array with indices of the test set
        """
        self._train_idx = train_idx
        self._test_idx = test_idx


class DataLoader:
    """
    Class for loading large amounts of datasets. Usefull for automation of experiments. All datasets need to be saved under the same directory
    """

    def __init__(self, dts_dir, task_types, train_test_exists=True):
        """
        Parameters
        ----------
        dts_dir : string
            root directory where the datasets are saved
        train_test_exists : Boolean
            If True the datasets are split with the suffix _TRAIN and _TEST
        task_types : string
            TSC or TSR
        """

        self._dts_dir = dts_dir
        self._train_test_exists = train_test_exists
        self._task_types = task_types

        self._all_datasets = iter(os.listdir(dts_dir))

    def _load_ts(self, load_train=None, load_test=None, task_target='target'):
        """
        Iterates and loads the next dataset

        Parameters
        ----------

        load_train : Boolean
            loads the train set
        load_test : Boolean
            loads the test set
        task_target : String
            Name of the DataFrame column containing the label
        Returns
        -------
        tuple of sktime.dataholder
            dataholder_train and dataholder_test.
        """

        if self._train_test_exists:
            if (load_train is None) and (load_test is None):
                raise ValueError('At least the train or the test set needs to be loaded')
        # get the next dataset in the list
        dts_name = next(self._all_datasets)

        datasets = os.listdir(os.path.join(self._dts_dir, dts_name))

        dataholder_train = None
        dataholder_test = None

        if self._train_test_exists:
            re_train = re.compile('.*_TRAIN.ts')
            re_test = re.compile('.*_TEST.ts')

            # find index of test or train file
            if load_train:
                data = pd.DataFrame()

                for dts in datasets:
                    if re_train.match(dts):
                        loaded_dts = load_from_tsfile_to_dataframe(os.path.join(self._dts_dir, dts_name, dts))
                        if len(loaded_dts) == 2:
                            data = loaded_dts[0]
                            y = loaded_dts[1]
                            data[task_target] = y
                            task = Task(case=self._task_types, dataset_name=dts_name, data=data, target=task_target)
                            dataholder_train = DataHolder(data=data, task=task, dataset_name=dts_name)
            if load_test:
                data = pd.DataFrame()

                for dts in datasets:
                    if re_test.match(dts):
                        loaded_dts = load_from_tsfile_to_dataframe(os.path.join(self._dts_dir, dts_name, dts))
                        if len(loaded_dts) == 2:
                            data = loaded_dts[0]
                            y = loaded_dts[1]
                            data[task_target] = y
                            task = Task(case=self._task_types, dataset_name=dts_name, data=data, target=task_target)
                            dataholder_test = DataHolder(data=data, task=task, dataset_name=dts_name)

        return dataholder_train, dataholder_test

    def load(self, load_test_train='both', task_target='target'):
        """
        Method for loading sequentially the data irrespective of how it was supplied, i.e. ts files, csv, db.

        Parameters
        ----------

        load_test_train : String
            Acceptable valies: test, train, both. Loads either the test set, the train set or both.
        load_test : Boolean
            loads the test set
        task_target : String
            Name of the DataFrame column containing the label
        Returns
        -------
        sktime.dataholder
            dataholder ready for running experiments
        """
        # TODO expand as the list of backend functions grows

        acceptable_test_train_values = ['test', 'train', 'both']
        if load_test_train not in acceptable_test_train_values:
            raise ValueError('Acceptable values for the load_test_train parameter are: test, train or both')

        if load_test_train == 'train':
            return self._load_ts(load_train=True, task_target=task_target)
        if load_test_train == 'test':
            return self._load_ts(load_test=True, task_target=task_target)
        if load_test_train == 'both':
            return self._load_ts(load_train=True, load_test=True, task_target=task_target)


class Result:
    """
    Class for storing the results of the orchestrator
    """

    def __init__(self, dataset_name, strategy_name, true_labels, predictions):
        """
        Parameters
        -----------
        dataset_name: string
            name of the dataset
        strategy_name: string
            name of strategy
        true_labels: array
            true labels
        predictions: array
            predictions of estimator
        """

        self._dataset_name = dataset_name
        self._strategy_name = strategy_name
        self._true_labels = true_labels
        self._predictions = predictions

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def strategy_name(self):
        return self._strategy_name

    @property
    def true_labels(self):
        return self._true_labels

    @property
    def predictions(self):
        return self._predictions
