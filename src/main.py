import argparse
import read_data
import model
import llm_inference
import structural_encoding
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import evaluation
import torch
import dataset
import create_similarity_network





def save_results_to_file(results, filename):
    with open(filename, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print('results have been saved to', filename)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('--run_all', default=1, type=int, help='run the experiment for ten fold validation')
    parser.add_argument('--index', default=0, type=str, help='Description for foo argument')
    parser.add_argument('--previous_embedding', default=True, type=bool, help='using previous embedding')
    parser.add_argument('--cvs', default = 1, type=int, help='cross validation setting 1 for warm start and 2 for cold start')
    parser.add_argument('--epoch',default = 30, type=int, help='number of epochs')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    return parser.parse_args()


def main():
    args = parse_arguments()
    result_all_path = f'results/average_cvs{args.cvs}_e{args.epoch}_result_st_q5_all.txt'
    if args.cvs != 1 and args.cvs != 2:
        raise Exception('cvs should be 1 or 2, however, the input is ', args.cvs)
    all_results = []
    if args.run_all == 0:
        for i in range(10):
            cur_results = single_exp(split = i, cvs = args.cvs, epoch = args.epoch, device = args.device)
            all_results.append(cur_results)
        keys = all_results[0].keys()
        avg_results = {}
        for key in keys:
            values = [result[key] for result in all_results]
            avg_results[key] = sum(values) / len(values)
        save_results_to_file(avg_results, result_all_path)
    else:
        single_exp(split = args.index, cvs = args.cvs, epoch = args.epoch, device = args.device)
        
        
def single_exp(split = 0, cvs = 1, epoch = 200, device = 'cuda:0'):
    print('start drug side effect prediction')
    print('index:', split)
    if split < 0 or split >= 10:
        raise Exception('index should be in [0, 10)')


    result_path = f'results/cvs_{cvs}_result_st_q5_split_e{epoch}_' + str(split) + '.txt'
    
    if cvs == 1:
        cur_fold_frequency_table, cur_fold_test_data, cur_fold_train_data, cur_fold_mask_table = read_data.read_cvs1_10_fold_split()
    elif cvs == 2:
        cur_fold_frequency_table, cur_fold_test_data, cur_fold_train_data, cur_fold_mask_table = read_data.read_cvs2_10_fold_split()
    else:
        raise Exception('cvs should be 1 or 2')



    drug_descrption_embeddings = llm_inference.bert_sentence_inference_s4(read_data.get_all_drug_descriptions(), device = device)
    se_descrption_embeddings = llm_inference.bert_sentence_inference_s4(read_data.get_all_side_descriptions(), device = device)
    
    se_descrption_embeddings = torch.tensor(se_descrption_embeddings, dtype=torch.float)

    drug_smiles = read_data.get_all_drug_smiles()

    drug_smiles_embeddings = llm_inference.chemberta_smiles_inference_entire_sequence(drug_smiles, device = device)

    #print(drug_smiles_embeddings.shape)

    #raise Exception('type:', type(drug_smiles_embeddings))

    drug_text_similarity = read_data.read_drug_text_smilarity()#from previous work

    drug_mfs = read_data.get_drug_morgan_fingerprints()#from previous work

    drug_target_features = read_data.read_drug_target_feature()

    Train_data = []
    Test_data = []
    for drug_index in range(750):
        input_1 = torch.tensor(cur_fold_frequency_table[drug_index], dtype=torch.float)#from previous work
        input_2 = torch.tensor(drug_text_similarity[drug_index], dtype=torch.float)#from reevious work dim 750
        input_3 = torch.tensor(structural_encoding.smile_encoding(drug_smiles[drug_index]), dtype=torch.float)# 
        input_4 = torch.tensor(drug_descrption_embeddings[drug_index], dtype=torch.float)
        input_5 = torch.tensor(drug_mfs[drug_index], dtype=torch.float)#from previous work
        input_6 = drug_smiles[drug_index]#max 1021
        input_7 = drug_target_features[drug_index]
        input_8 = torch.tensor(drug_smiles_embeddings[drug_index], dtype=torch.float)

        #please note that in dataset.py the order has been changed, thus input and prediction model input is not identical
        cur_info = (input_1, input_2, input_3, input_4, input_5, input_6, input_7, input_8)
        if drug_index in cur_fold_train_data:
            Train_data.append(cur_info)
            Test_data.append(cur_info)
            '''
            This is for known drugs that are used for training and testing
            drugs without any masked positive samples are not utlised during evaluating for AURR and AURC
            '''
        if drug_index not in cur_fold_train_data:
            Test_data.append(cur_info)
            '''
            This is for unknown drugs that are only used for testing
            '''

    Train_dataset = dataset.TrainDataset(Train_data)
    Test_dataset = dataset.TrainDataset(Test_data)

    if cvs == 1:
        side_effect_edges = create_similarity_network.create_side_effect_graph(cur_fold_frequency_table, top_k= 10)
    elif cvs == 2:
        side_effect_edges = create_similarity_network.create_side_effect_graph(cur_fold_frequency_table, top_k= 5)
    else:
        raise Exception('csv setting is not correct')

    #drug implicit graph
    #drug_smilarity_network




    model.Train_Model(Train_dataset, side_effect_edges, se_descrption_embeddings, save_path='trained_model/model_st_q5_fold_' + str(split) + '.pt', epochs=epoch, batch_size=10, lr=0.001, device_id=device)


    prediction_outputs = model.Predict(Test_dataset, side_effect_edges, se_descrption_embeddings, load_path = 'trained_model/model_st_q5_fold_' + str(split) + '.pt', device_id=device)
    
    ground_truth = read_data.read_frequency()


    evluation_results = evaluation.metrics_calculation_dual_embeddings(prediction_outputs, ground_truth, cur_fold_mask_table)
    print(evluation_results)
    save_results_to_file(evluation_results, result_path)

    return evluation_results

if __name__ == "__main__":
    main()
