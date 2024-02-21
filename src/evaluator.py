# third party
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import *


class Evaluator:
    def __init__(self, hardness_dict, flag_ids, p):
        """
        This is the constructor function for a class that takes in a dictionary from the HCMs, a
        list of flag IDs, and a percentage value, and sets them as attributes of the class instance.

        Args:
          hardness_dict: It is a dictionary for the HCMs.
        The keys of the dictionary are the names of the HCMs and the values are the corresponding instantiations
          flag_ids: list with flag ids
          p: proportion of data perturbed
        """
        self.hardness_dict = hardness_dict
        self.flag_ids = flag_ids
        self.prop = 100 - (p * 100)

    def binary_classification_metrics(self, pred, true):
        """
        This function computes binary classification metrics such as accuracy, precision, recall, and F1
        score.

        Args:
          pred: The predicted labels for a binary classification problem.
          true: The true labels or ground truth values for a binary classification problem. These are
        the actual labels for the data points that the model is trying to predict.

        Returns:
          a dictionary containing the accuracy, precision, recall, and F1 score metrics for a binary
        classification problem.
        """
        # Convert inputs to numpy arrays if they are not already
        pred = np.array(pred)
        true = np.array(true)

        # Compute accuracy, precision, and recall
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        return metrics

    def compute_results(self):
        """
        The function computes various metrics for different HCMs and returns the
        results along with the raw scores.

        Returns:
          a tuple containing two dictionaries: `results` and `raw_scores_dict`. The `results` dictionary
        contains the evaluation metrics for each method, while the `raw_scores_dict` dictionary contains
        the raw scores for each method.
        """

        results = {}
        raw_scores_dict = {}
        for method in self.hardness_dict.keys():

            print(f"Evaluating {method}...")

            # create a dictionary with method names as keys and corresponding score methods as values
            score_methods = {
                "aum": self.aum_scores,
                "cleanlab": self.cleanlab_scores,
                "grand": self.grand_scores,
                "el2n": self.el2n_scores,
                "dataiq": self.dataiq_scores,
                "datamaps": self.datamaps_scores,
                "forgetting": self.forgetting_scores,
                "prototypicality": self.prototypicality_scores,
                "allsh": self.allsh_scores,
                "loss": self.loss_scores,
                "conf_agree": self.conf_agree_scores,
                "vog": lambda mode="complete": self.vog_scores(mode),
                "detector": self.detector_scores,
            }

            # call the corresponding score method based on the input method name
            if method in score_methods:

                raw_scores = self.hardness_dict[method].scores

                if method == "dataiq" or method == "datamaps":
                    if method == "dataiq":
                        raw = self.hardness_dict[method].compute_scores(datamaps=False)
                    else:
                        raw = self.hardness_dict[method].compute_scores(datamaps=True)
                    _, confidence = raw
                    raw_scores = confidence

                if method == "cleanlab":
                    raw = self.hardness_dict[method].compute_scores()
                    raw_scores, _ = raw

                if np.isnan(raw_scores).any():
                    # Calculate the mean of non-NaN values
                    mean = np.nanmean(raw_scores)

                    # Replace NaN values with the mean using np.nan_to_num()
                    raw_scores = np.nan_to_num(raw_scores, nan=mean)

                threshold_flag = (~self.flag_ids.astype(bool)).astype(
                    int
                )  # since most methods select good samples, so we make good=1

                if (
                    method == "el2n"
                    or method == "grand"
                    or method == "loss"
                    or method == "forgetting"
                    or method == "detector"
                ):
                    auc_roc = roc_auc_score(y_true=self.flag_ids, y_score=raw_scores)
                else:
                    auc_roc = roc_auc_score(y_true=threshold_flag, y_score=raw_scores)

                auc_prc = average_precision_score(
                    y_true=threshold_flag, y_score=raw_scores
                )

                # compute thresholded metrics
                pred_scores = score_methods[method]()
                metrics = {}

                metrics["auc_roc"] = auc_roc
                metrics["auc_prc"] = auc_prc

                results[method] = metrics
                raw_scores_dict[method] = raw_scores

        return results, raw_scores_dict

    # Below are scoring methods for each HCM with fixed thresholds, if you'd like to use them

    def cleanlab_scores(self):
        scores, num_errors = self.hardness_dict["cleanlab"].compute_scores()

        cl_error_indices = np.argsort(scores)[:num_errors]

        label_issues_mask = np.zeros(len(scores), dtype=bool)
        for idx in cl_error_indices:
            label_issues_mask[idx] = True

        return label_issues_mask

    def aum_scores(self):
        aum_raw = self.hardness_dict["aum"].scores
        th_val = np.percentile(aum_raw, 99)  # as per the paper
        return np.array(aum_raw < th_val)

    def forgetting_scores(self):
        forgetting = self.hardness_dict["forgetting"].scores
        return np.array(forgetting > 0)

    def grand_scores(self):
        grand_raw = self.hardness_dict["grand"].scores
        # Calculate the score threshold
        threshold = np.percentile(grand_raw, self.prop)

        # Create a new array and label scores based on the threshold
        labels = np.where(grand_raw >= threshold, 1, 0)
        return labels

    def el2n_scores(self):
        el2n_raw = self.hardness_dict["el2n"].scores
        # Calculate the score threshold
        threshold = np.percentile(el2n_raw, self.prop)

        # Create a new array and label scores based on the threshold
        labels = np.where(el2n_raw >= threshold, 1, 0)
        return labels

    def vog_scores(self, mode="complete"):
        self.hardness_dict["vog"].compute_scores(mode=mode)
        vog_raw = self.hardness_dict["vog"].scores
        # Calculate the score threshold
        threshold = np.percentile(vog_raw, self.prop)

        # threshold values with high VOG as 1 and low Vog as 0
        labels = np.where(vog_raw > threshold, 1, 0)
        return labels

    def dataiq_scores(self):
        dataiq_raw = self.hardness_dict["dataiq"].compute_scores(datamaps=False)
        uncertainty, confidence = dataiq_raw
        easy_train, ambig_train, hard_train = get_groups(
            confidence=confidence,
            uncertainty=uncertainty,
            data_iq_xthresh=0.15,
            data_iq_ythresh=0.2,
        )

        labels = np.zeros(len(confidence))
        labels[hard_train] = 1

        return labels

    def datamaps_scores(self):
        datamaps_raw = self.hardness_dict["datamaps"].compute_scores(datamaps=True)
        uncertainty, confidence = datamaps_raw
        easy_train, ambig_train, hard_train = get_groups(
            confidence=confidence,
            uncertainty=uncertainty,
            data_iq_xthresh=0.15,
            data_iq_ythresh=0.2,
        )

        labels = np.zeros(len(confidence))
        labels[hard_train] = 1

        return labels

    def prototypicality_scores(self):
        prototypicality_raw = self.hardness_dict["prototypicality"].scores
        # Calculate the score threshold
        threshold = np.percentile(prototypicality_raw, self.prop)

        # Create a new array and label scores based on the threshold
        labels = np.where(prototypicality_raw >= threshold, 1, 0)
        return labels

    def allsh_scores(self):
        allsh_raw = self.hardness_dict["allsh"].scores
        # Calculate the score threshold
        threshold = np.percentile(allsh_raw, self.prop)

        # Create a new array and label scores based on the threshold
        labels = np.where(allsh_raw >= threshold, 1, 0)
        return labels

    def detector_scores(self):
        labels = self.hardness_dict["detector"].scores
        return labels

    def loss_scores(self):
        loss_raw = self.hardness_dict["loss"].scores

        # Calculate the score threshold
        threshold = np.percentile(loss_raw, self.prop)

        # Create a new array and label scores based on the threshold
        labels = np.where(loss_raw >= threshold, 1, 0)
        return labels

    def conf_agree_scores(self):
        conf_agree_raw = self.hardness_dict["conf_agree"].scores
        # Calculate the score threshold
        threshold = np.percentile(conf_agree_raw, 100 - self.prop)

        # Create a new array and label scores based on the threshold
        labels = np.where(conf_agree_raw <= threshold, 1, 0)
        return labels
