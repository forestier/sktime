"""Classes for evaluating results from benchmarking experiments comparing one or multiple prediction strategies on
one or multiple datasets.
"""

import itertools
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import ranksums
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_consistent_length

__all__ = ["Evaluator"]
__author__ = ["Viktor Kazakov", "Markus Löning"]


class Evaluator:

    def __init__(self, results):
        self.results = results

    def prediction_errors(self, metric, train_or_test='test'):
        """
        Calculates the average prediction error per estimator as well as the prediction error
        achieved by each estimator on individual datasets.

        Parameters
        -----------
        metric: `sktime.analyse_results.scores`
            Error function 
        Returns
        --------
        pickle of pandas DataFrame
            ``estimator_avg_error`` represents the average error and standard deviation achieved
            by each estimator. ``estimator_avg_error_per_dataset`` represents the average error
            and standard deviation achieved by each estimator on each dataset.
        """
        calculator = MetricCalculator(metric)
        for strategy_name, dataset_name, idx, y_true, y_pred in self.results.load_predictions(
                train_or_test=train_or_test, fold=0):
            calculator.evaluate(y_true, y_pred, dataset_name, strategy_name)
        return calculator.get_losses()

    def average_and_std_error(self, scores_dict):
        """
        Calculates simple average and standard error.

        Args:
            scores_dict(dictionary): Dictionary with estimators (keys) and corresponding prediction accuracies on different datasets.
        
        Returns:
            pandas DataFrame
        """
        result = {}
        for k in scores_dict.keys():
            average = np.average(scores_dict[k])
            n = len(scores_dict[k])
            std_error = np.std(scores_dict[k]) / np.sqrt(n)
            result[k] = [average, std_error]

        res_df = pd.DataFrame.from_dict(result, orient='index')
        res_df.columns = ['avg_score', 'std_error']
        res_df = res_df.sort_values(['avg_score', 'std_error'], ascending=[1, 1])

        return res_df

    def plot_boxcharts(self, scores_dict):
        data = []
        labels = []
        avg_error = []
        for e in scores_dict.keys():
            data.append(scores_dict[e])
            avg_error.append(np.mean(scores_dict[e]))
            labels.append(e)
        # sort data and labels based on avg_error
        idx_sort = np.array(avg_error).argsort()
        data = [data[i] for i in idx_sort]
        labels = [labels[i] for i in idx_sort]
        # plot the results
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_xticklabels(labels, rotation=90)
        plt.tight_layout()

        return fig

    def ranks(self, strategy_dict, ascending=True):
        """
        Calculates the average ranks based on the performance of each estimator on each dataset

        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        ascending: boolean
            Rank the values in ascending (True) or descending (False) order

        Returns
        -------
        DataFrame
            Returns the mean peformance rank for each estimator
        """
        if not isinstance(ascending, bool):
            raise ValueError('Variable ascending needs to be boolean')

        df = pd.DataFrame(strategy_dict)
        ranked = df.rank(axis=1, ascending=ascending)
        mean_r = pd.DataFrame(ranked.mean(axis=0))
        mean_r.columns = ['avg_rank']
        mean_r = mean_r.sort_values('avg_rank', ascending=ascending)
        return mean_r

    def t_test(self, strategy_dict):
        """
        Runs t-test on all possible combinations between the estimators.

        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        Returns
        --------
        tuple 
            pandas DataFrame (Database style and MultiIndex)
        """
        t_df = pd.DataFrame()
        perms = itertools.product(strategy_dict.keys(), repeat=2)
        values = np.array([])
        for perm in perms:
            x = np.array(strategy_dict[perm[0]])
            y = np.array(strategy_dict[perm[1]])
            t_stat, p_val = ttest_ind(x, y)

            t_test = {
                'estimator_1': perm[0],
                'estimator_2': perm[1],
                't_stat': t_stat,
                'p_val': p_val
            }

            t_df = t_df.append(t_test, ignore_index=True)
            values = np.append(values, t_stat)
            values = np.append(values, p_val)

        index = t_df['estimator_1'].unique()
        values_names = ['t_stat', 'p_val']
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return t_df, values_df_multiindex

    def sign_test(self, strategy_dict):
        """
        Non-parametric test for test for consistent differences between pairs of observations.
        See `<https://en.wikipedia.org/wiki/Sign_test>`_ for details about the test
        and `<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.binom_test.html>`_ for
        details about the scipy implementation.

        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        Returns
        -------
        tuple of dataframes 
            pandas DataFrame (Database style), pivot table)
        """
        sign_df = pd.DataFrame()
        perms = itertools.product(strategy_dict.keys(), repeat=2)
        for perm in perms:
            x = np.array(strategy_dict[perm[0]])
            y = np.array(strategy_dict[perm[1]])
            signs = np.sum([i[0] > i[1] for i in zip(x, y)])
            n = len(x)
            p_val = stats.binom_test(signs, n)
            sign_test = {
                'estimator_1': perm[0],
                'estimator_2': perm[1],
                'p_val': p_val
            }

            sign_df = sign_df.append(sign_test, ignore_index=True)
            sign_df_pivot = sign_df.pivot(index='estimator_1', columns='estimator_2', values='p_val')

        return sign_df, sign_df_pivot

    def ranksum_test(self, strategy_dict):
        """
        Non-parametric test for testing consistent differences between pairs of obeservations.
        The test counts the number of observations that are greater, smaller and equal to the mean
        `<http://en.wikipedia.org/wiki/Wilcoxon_rank-sum_test>`_.

        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        Returns
        -------
        tuple of pandas DataFrame 
            Database style and MultiIndex
        """
        ranksum_df = pd.DataFrame()
        perms = itertools.product(strategy_dict.keys(), repeat=2)
        values = np.array([])
        for perm in perms:
            comb = perm[0] + ' - ' + perm[1]
            x = strategy_dict[perm[0]]
            y = strategy_dict[perm[1]]
            t_stat, p_val = ranksums(x, y)
            ranksum = {
                'estimator_1': perm[0],
                'estimator_2': perm[1],
                't_stat': t_stat,
                'p_val': p_val
            }
            ranksum_df = ranksum_df.append(ranksum, ignore_index=True)
            values = np.append(values, t_stat)
            values = np.append(values, p_val)

        index = ranksum_df['estimator_1'].unique()
        values_names = ['t_stat', 'p_val']
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return ranksum_df, values_df_multiindex

    def t_test_with_bonferroni_correction(self, strategy_dict, alpha=0.05):
        """
        correction used to counteract multiple comparissons
        https://en.wikipedia.org/wiki/Bonferroni_correction

        
        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        alpha: float
            confidence level.
        Returns
        -------
        DataFrame 
            MultiIndex DataFrame
        """
        df_t_test, _ = self.t_test(strategy_dict)
        idx_estim_1 = df_t_test['estimator_1'].unique()
        idx_estim_2 = df_t_test['estimator_2'].unique()
        estim_1 = len(idx_estim_1)
        estim_2 = len(idx_estim_2)
        critical_value = alpha / (estim_1 * estim_2)

        bonfer_test = df_t_test['p_val'] <= critical_value

        bonfer_test_reshaped = bonfer_test.values.reshape(estim_1, estim_2)

        bonfer_df = pd.DataFrame(bonfer_test_reshaped, index=idx_estim_1, columns=idx_estim_2)

        return bonfer_df

    def wilcoxon_test(self, strategy_dict):
        """http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
        `Wilcoxon signed-rank test <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`_.
        Tests whether two  related paired samples come from the same distribution. 
        In particular, it tests whether the distribution of the differences x-y is symmetric about zero

        Parameters
        ----------
        strategy_dict: dictionary
            Dictionary with errors on test sets achieved by estimators.
        Returns
        -------
        tuple 
            pandas DataFrame (Database style and MultiIndex)
        """
        wilcoxon_df = pd.DataFrame()
        values = np.array([])
        prod = itertools.product(strategy_dict.keys(), repeat=2)
        for p in prod:
            estim_1 = p[0]
            estim_2 = p[1]
            w, p_val = stats.wilcoxon(strategy_dict[p[0]],
                                      strategy_dict[p[1]])

            w_test = {
                'estimator_1': estim_1,
                'estimator_2': estim_2,
                'statistic': w,
                'p_val': p_val
            }

            wilcoxon_df = wilcoxon_df.append(w_test, ignore_index=True)
            values = np.append(values, w)
            values = np.append(values, p_val)

        index = wilcoxon_df['estimator_1'].unique()
        values_names = ['statistic', 'p_val']
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return wilcoxon_df, values_df_multiindex

    def friedman_test(self, strategy_dict):
        """
        The Friedman test is a non-parametric statistical test used to detect differences 
        in treatments across multiple test attempts. The procedure involves ranking each row (or block) together, 
        then considering the values of ranks by columns.
        Implementation used: `scipy.stats <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.friedmanchisquare.html>`_. 
        
        Parameters
        ----------
        strategy_dict: dictionary
            Dictionary with errors on test sets achieved by estimators.
        Returns
        -------
        tuple 
            dictionary, pandas DataFrame.
        
        """

        """
        use the * operator to unpack a sequence
        https://stackoverflow.com/questions/2921847/what-does-the-star-operator-mean/2921893#2921893
        """
        friedman_test = stats.friedmanchisquare(*[strategy_dict[k] for k in strategy_dict.keys()])
        values = [friedman_test[0], friedman_test[1]]
        values_df = pd.DataFrame([values], columns=['statistic', 'p_value'])

        return friedman_test, values_df

    def nemenyi(self, strategy_dict):
        """
        Post-hoc test run if the `friedman_test` reveals statistical significance.
        For more information see `Nemenyi test <https://en.wikipedia.org/wiki/Nemenyi_test>`_.
        Implementation used `scikit-posthocs <https://github.com/maximtrp/scikit-posthocs>`_.
        
        Parameters
        ----------
        strategy_dict: dictionary
            Dictionary with errors on test sets achieved by estimators.
        Returns
        -------
        pandas DataFrame
        """

        strategy_dict = pd.DataFrame(strategy_dict)
        strategy_dict = strategy_dict.melt(var_name='groups', value_name='values')
        nemenyi = sp.posthoc_nemenyi(strategy_dict, val_col='values', group_col='groups')
        return nemenyi


class MetricCalculator:
    """
    Calculates prediction losses on test datasets achieved by the trained estimators. When the class is instantiated it creates a dictionary that stores the losses.

    Args:
        metric(`mlaut.benchmarking.scores` object): score function that will be used for the estimation. Must be `mlaut.benchmarking.scores` object.
        estimators(`array of mlaut estimators`): Array of estimators on which the results will be compared.
        exact_match(Boolean): If `True` when y_pred for all estimators in the estimators array is not available no evaluation is performed on the remaining estimators.
    """

    def __init__(self, metric):
        self._losses = defaultdict(list)
        self._metric = metric
        self._losses_per_estimator = defaultdict(list)
        self._losses_per_dataset_per_estimator = defaultdict(list)

    def evaluate(self, y_true, y_pred, dataset_name, strategy_name):
        """
        Calculates the loss metrics on the test sets.

        Parameters
        ----------
        y_pred: numpy array
            Predictions of trained estimators in the form
        y_true: numpy array
            true labels of test dataset.
        dataset_name: string
            Name of the dataset
        dataset_name: string
            Name of the strategy
        """

        check_consistent_length(y_true, y_pred)

        # evaluate per estimator
        loss = self._metric.calculate(y_true, y_pred)
        self._losses_per_estimator[strategy_name].append(loss)

        # evaluate per dataset
        avg_score, std_score = self._metric.calculate_per_dataset(y_true=y_true, y_pred=y_pred)
        self._losses_per_dataset_per_estimator[dataset_name].append([strategy_name, avg_score, std_score])

    def get_losses(self):
        """
        When the MetricCalculator class is instantiated a dictionary that holds all losses is created and appended every time the evaluate() method is run. This method returns this dictionary with the losses.

        Returns
        -------
            errors_per_estimator (dictionary), errors_per_dataset_per_estimator (dictionary), errors_per_dataset_per_estimator_df (pandas DataFrame): Returns dictionaries with the errors achieved by each estimator and errors achieved by each estimator on each of the datasets.  ``errors_per_dataset_per_estimator`` and ``errors_per_dataset_per_estimator_df`` return the same results but the first object is a dictionary and the second one a pandas DataFrame. ``errors_per_dataset_per_estimator`` and ``errors_per_dataset_per_estimator_df`` contain both the mean error and deviation.
        """
        return self._losses_per_estimator, self._losses_to_dataframe(self._losses_per_dataset_per_estimator)

    def _losses_to_dataframe(self, losses):
        """
        Reformats the output of the dictionary returned by the :func:`mlaut.benchmarking.losses.MetricCalculator.get_losses` to a pandas DataFrame. This method can only be applied to reformat the output produced by :func:`mlaut.benchmarking.MetricCalculator.evaluate_per_dataset`.

        Parameters
        ----------

        losses: dictionary returned by the :func:`mlaut.benchmarking.losses.MetricCalculator.get_losses` generated by :func:`mlaut.benchmarking.losses.MetricCalculator.evaluate_per_dataset`
        """

        df = pd.DataFrame(losses)
        # unpivot the data
        df = df.melt(var_name='dts', value_name='values')
        df['classifier'] = df.apply(lambda raw: raw.values[1][0], axis=1)
        df['loss'] = df.apply(lambda raw: raw.values[1][1], axis=1)
        df['std_error'] = df.apply(lambda raw: raw.values[1][2], axis=1)
        df = df.drop('values', axis=1)

        # create multilevel index dataframe
        dts = df['dts'].unique()
        estimators_list = df['classifier'].unique()
        score = df['loss'].values
        std = df['std_error'].values

        df = df.drop('dts', axis=1)
        df = df.drop('classifier', axis=1)

        df.index = pd.MultiIndex.from_product([dts, estimators_list])

        return df
