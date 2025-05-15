import numpy as np
from scipy.stats import pearsonr, spearmanr
#from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
import torch

'''
def evaluate_classification_1(predictions, ground_truths):
    results = {}

    # Pearson's correlation
    pearson_corr, _ = pearsonr(predictions, ground_truths)
    results['pearson_correlation'] = pearson_corr

    # Spearman's correlation
    spearman_corr, _ = spearmanr(predictions, ground_truths)
    results['spearman_correlation'] = spearman_corr

    # Mean Squared Error
    mse = mean_squared_error(ground_truths, predictions)
    results['mean_squared_error'] = mse

    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    results['root_mean_squared_error'] = rmse

    # Mean Absolute Error
    mae = np.mean(np.abs(np.array(predictions) - np.array(ground_truths)))
    results['mean_absolute_error'] = mae

    return results
'''

'''
def rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error using PyTorch
    """
    mse = torch.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)
'''

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def calculate_frqeuncy_based_metrics(predictions, ground_truths, mask_table):
    predict_label = []
    true_label = []
    
    drug_num = 750
    side_effect_num = 994

    for drug_index in range(drug_num):
        for side_effect_index in range(side_effect_num):
            if mask_table[drug_index, side_effect_index] == 0 and ground_truths[drug_index, side_effect_index] != 0:
                predict_label.append(predictions[drug_index][side_effect_index])
                true_label.append(ground_truths[drug_index][side_effect_index])

    scc = spearmanr(predict_label, true_label).correlation
    rmse_value = rmse(torch.tensor(predict_label), torch.tensor(true_label))
    mae = metrics.mean_absolute_error(predict_label, true_label)
    return scc, rmse_value, mae
    #fpr, tpr, thresholds = metrics.roc_curve(ys_bin, fxs, pos_label=1)
    #threshold = thresholds[np.argmax(tpr - fpr)]
    #auroc = metrics.auc(fpr, tpr)
    #auprc = metrics.average_precision_score(ys_bin, fxs)


def calculate_classification_based_metrics(predictions, ground_truths, mask_table):
    drugs_auroc = []
    drugs_aupr = []

    drug_num = 750
    side_effect_num = 994

    for drug_index in range(drug_num):
        positive_samples = []
        negative_samples = []
        for se_index in range(side_effect_num):
            #print(mask_table[drug_index][se_index], 'drug_index:', drug_index, 'se_index:', se_index)
            if int(mask_table[drug_index][se_index]) == 0 and ground_truths[drug_index][se_index] > 0:
                positive_samples.append(predictions[drug_index][se_index])
            if int(ground_truths[drug_index][se_index]) == 0:
                negative_samples.append(predictions[drug_index][se_index])

        if len(positive_samples) == 0:#drugs without any masked positive samples are not calculated
            continue
        
        #print('positive num', len(positive_samples), ' negative num', len(negative_samples))
        cur_drug_labels = [1] * len(positive_samples) + [0] * len(negative_samples)
        cur_drug_results = positive_samples + negative_samples

        #fpr, tpr, thresholds = metrics.roc_curve(cur_drug_labels, cur_drug_results, pos_label=1)
        #threshold = thresholds[np.argmax(tpr - fpr)]
        _auc = metrics.roc_auc_score(cur_drug_labels, cur_drug_results)
        _map = metrics.average_precision_score(cur_drug_labels, cur_drug_results)
        drugs_auroc.append(_auc)
        drugs_aupr.append(_map)
    return np.mean(drugs_auroc), np.mean(drugs_aupr)

'''
def calculate_classification_based_metrics(predictions, ground_truths, mask_table):
    drugs_auroc = []
    drugs_aupr = []

    drug_num = 750
    side_effect_num = 994

    for drug_index in range(drug_num):
        positive_samples = []
        negative_samples = []
        for se_index in range(side_effect_num):
            #print(mask_table[drug_index][se_index], 'drug_index:', drug_index, 'se_index:', se_index)
            if int(mask_table[drug_index][se_index]) == 0:
                positive_samples.append(predictions[drug_index][se_index])
            if int(ground_truths[drug_index][se_index]) == 0:
                negative_samples.append(predictions[drug_index][se_index])

        if len(positive_samples) == 0:
            continue
        
        #print('positive num', len(positive_samples), ' negative num', len(negative_samples))
        cur_drug_labels = [1] * len(positive_samples) + [0] * len(negative_samples)
        cur_drug_results = positive_samples + negative_samples

        fpr, tpr, thresholds = metrics.roc_curve(cur_drug_labels, cur_drug_results, pos_label=1)
        threshold = thresholds[np.argmax(tpr - fpr)]
        _auc = metrics.auc(fpr, tpr)
        _map = metrics.average_precision_score(cur_drug_labels, cur_drug_results)
        drugs_auroc.append(_auc)
        drugs_aupr.append(_map)
    return np.mean(drugs_auroc), np.mean(drugs_aupr)

'''               




def metrics_calculation(predictions, ground_truths, mask_table):
    results = {}
    scc, rmse_value, mae = calculate_frqeuncy_based_metrics(predictions, ground_truths, mask_table)

    results['scc'] = scc
    results['rmse'] = rmse_value
    results['mae'] = mae

    auroc, aupr = calculate_classification_based_metrics(predictions, ground_truths, mask_table)

    results['auroc'] = auroc
    results['aupr'] = aupr
    
    return results

def metrics_calculation_dual_embeddings(predictions, ground_truths, mask_table):
    #(all_regression_outputs, all_prediction_outputs)
    regression_predictions = predictions[0]
    classification_predictions = predictions[1]
    #print(regression_predictions.shape, classification_predictions.shape)
    results = {}
    scc, rmse_value, mae = calculate_frqeuncy_based_metrics(regression_predictions, ground_truths, mask_table)

    results['scc'] = scc
    results['rmse'] = rmse_value
    results['mae'] = mae

    auroc, aupr = calculate_classification_based_metrics(classification_predictions, ground_truths, mask_table)

    results['auroc'] = auroc
    results['aupr'] = aupr
    
    return results