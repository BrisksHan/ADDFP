import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.nn import DataParallel
from tqdm import tqdm
import numpy as np
from torch_geometric.nn import GATConv, GINConv, LEConv, FAConv, global_max_pool, GeneralConv, GENConv, SAGEConv, GCNConv




def loss_fun_regression(output, label, lamb = 0.01, eps = 0.5):#lamb == alpha in the paper
    x0 = torch.where(label == 0)
    x1 = torch.where(label != 0)

    loss = (torch.sum((output[x1] - label[x1]) ** 2) + lamb * torch.sum((output[x0] - eps) ** 2)) / (output.shape[0])
    return loss

def loss_fun_classification(output, label):
    label_binary = (label > 0).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, label_binary)
    return loss



# Define a simple neural network
class model_st(nn.Module):
    def __init__(self, side_effect_edges, se_description_embeddings, char_dim = 128, dropout_rate = 0.125, drug_kernel = [4,6,8], conv = 64, heads = 8, dropout = 0.1, drug_MAX_LENGH = 1021, embed_dim = 128, device = 'cuda:0'):
        super(model_st, self).__init__()
        self.dim = char_dim
        self.conv = conv
        self.drug_kernel = drug_kernel
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.device = device
        self.se_description_embeddings = se_description_embeddings

        side_effect_feature_paths = [
            #'data/glove_wordEmbedding.pkl',#embedding of description 300
            'data/side_effect_label_750.pkl',#meddra 243
        ]
        side_effect_features = []
        for path in side_effect_feature_paths:
            with open(path, 'rb') as f:
                side_effect_features.append(pickle.load(f))
        self.side_effects_tensor = torch.from_numpy(np.concatenate(side_effect_features, axis=1)).float().to(device)
        self.side_effects_tensor = torch.cat((self.side_effects_tensor, self.se_description_embeddings), dim=1)
        self.side_encoder_encoder = nn.Sequential(
            nn.Linear(self.side_effects_tensor.shape[1], embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
            )
        
        self.side_encoder_encoder_2 = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.Tanh()
            )
        
        self.side_encoder_encoder_2_1 = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.Tanh()
            )
        
        self.side_encoder_encoder_2_2 = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.Tanh()
            )
    
        
        self.side_encoder_encoder_2_8 = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.Tanh()
            )
        

        self.side_encoder_encoder_2_10 = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.Tanh()
            )
        

        self.drug_text_similarity_ff_classification = nn.Sequential(
            nn.Linear(750, 750),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(750, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.Tanh()
            )
        
        self.drug_text_similarity_ff_regression = nn.Sequential(
            nn.Linear(750, 750),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(750, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.Tanh()
            )
        
        self.drug_description_ff_regression = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 256),
            nn.Tanh()
        )

        self.drug_description_ff_classification = nn.Sequential(
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 256),
            nn.Tanh()
        )

        self.prediction_layer = nn.Linear(128, 994)
        self.drug_cnn_ff_regression = nn.Sequential(
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(192, 256),
            nn.ReLU(),
        )

        self.drug_cnn_ff_classification = nn.Sequential(
            nn.Linear(192, 192),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(192, 256),
            nn.ReLU(),
        )


        self.graph_encoder_1_1 = GCNConv(110, 256)
        self.graph_encoder_1_2 = GCNConv(256, 256)
        self.graph_encoder_1_3 = GCNConv(256, 256)

        self.se_graph_encoder_1 = GCNConv(627, 256)#543
        self.se_graph_encoder_2 = GCNConv(256, 256)
        self.se_graph_encoder_3 = GCNConv(256, 256)

        self.side_effect_edges = side_effect_edges


        self.relu = nn.ReLU()

        self.leakyrelu = nn.LeakyReLU()

        self.graph_feature_maxpool = nn.MaxPool1d(1021)

        self.graph_feature_ff_classification = nn.Sequential(
            nn.Linear(256, 256),
        )

        self.graph_feature_ff_regression = nn.Sequential(
            nn.Linear(256, 256),
        )



        self.drug_classification_layer = nn.Sequential(
            nn.Linear(768, 256),
        )   

        self.drug_regression_layer = nn.Sequential(
            nn.Linear(768, 256),
        )

        self.side_effect_classification_layer = nn.Sequential(
            nn.Linear(256, 256),
        )
        
        self.side_effect_regression_layer = nn.Sequential(
            nn.Linear(256, 256),
        )


        self.classification_view_mlp = nn.Sequential(
            nn.Linear(3, 1),

        )



    def process_mol_data(self, graph: torch.Tensor, batch: torch.Tensor, target_size: int) -> torch.Tensor:
        """Process molecular features tensor to have exactly target_size nodes through padding or truncation for each graph in batch."""
        unique_batch = torch.unique(batch)
        batch_size = len(unique_batch)
        num_features = graph.size(-1)
        
        # Create output tensor for all graphs in batch
        output = torch.zeros(batch_size, target_size, num_features, device=graph.device)
        
        # Process each graph in batch
        for i, b in enumerate(unique_batch):
            mask = (batch == b)
            graph_i = graph[mask]
            num_nodes = graph_i.size(0)
            
            # Copy data up to target size for this graph
            n = min(num_nodes, target_size)
            output[i, :n] = graph_i[:n]
            
        return output

    def forward(self, drug_features):
        side_effect_tensor = self.side_encoder_encoder(self.side_effects_tensor)

        molecular_graph, drug_text_similarity_tensor, drug_smiles_encoding_tensor, drug_descrption_embeddings_tensor, drug_mf, drug_target_feature, chemberta_embeddings = drug_features


        #--------------------side_effect graph-------------
        se_graph_feature = self.se_graph_encoder_1(self.side_effects_tensor, self.side_effect_edges)
        se_graph_feature = self.leakyrelu(se_graph_feature)
        se_graph_feature = self.se_graph_encoder_2(se_graph_feature, self.side_effect_edges)
        se_graph_feature = self.leakyrelu(se_graph_feature)
        se_graph_feature = self.se_graph_encoder_3(se_graph_feature, self.side_effect_edges)
        se_graph_feature = self.leakyrelu(se_graph_feature)
        #print(side_effect_tensor.shape)
        #print(se_graph_feature.shape)
        side_effect_tensor_ini = torch.cat((side_effect_tensor, se_graph_feature), dim=1)

        side_effect_tensor = self.side_encoder_encoder_2(side_effect_tensor_ini)

        side_effect_tensor_2 = self.side_encoder_encoder_2_2(side_effect_tensor_ini)
        
        side_effect_tensor_8 = self.side_encoder_encoder_2_8(side_effect_tensor_ini)

        side_effect_tensor_10 = self.side_encoder_encoder_2_10(side_effect_tensor_ini)

        #----------------------graph-----------------------
    
        graph_feature_original, edge_index, batch = molecular_graph.x.to(self.device), molecular_graph.edge_index.to(self.device), molecular_graph.batch.to(self.device)
        

        graph_feature_1 = self.graph_encoder_1_1(graph_feature_original, edge_index)
        graph_feature_1 = self.leakyrelu(graph_feature_1)
        graph_feature_1 = self.graph_encoder_1_2(graph_feature_1, edge_index)
        graph_feature_1 = self.leakyrelu(graph_feature_1)
        graph_feature_1 = self.graph_encoder_1_3(graph_feature_1, edge_index)
        graph_feature_1 = self.leakyrelu(graph_feature_1)

        graph_feature_1 = self.process_mol_data(graph_feature_1, batch, 1021)
        graph_feature_1 = graph_feature_1.permute(0, 2, 1)
        graph_feature_1 = self.graph_feature_maxpool(graph_feature_1)
        graph_feature_1 = graph_feature_1.squeeze(-1)


        graph_feature_regression = self.graph_feature_ff_regression(graph_feature_1)
        graph_feature_classification = self.graph_feature_ff_classification(graph_feature_1)

        #---------------------drug text similarity----------------
        #drug text similarity
        drug_text_similarity_tensor = drug_text_similarity_tensor.to(self.device)

        drug_text_similarity_tensor_regression = self.drug_text_similarity_ff_regression(drug_text_similarity_tensor)
        drug_text_similarity_tensor_classification = self.drug_text_similarity_ff_classification(drug_text_similarity_tensor)

        #drug_text_similarity_tensor = self.drug_text_similarity_ff_dropout(drug_text_similarity_tensor)

        #---------------------drug description-------------------
        #drug description
        drug_descrption_embeddings_tensor = drug_descrption_embeddings_tensor.to(self.device)
        drug_descrption_embeddings_tensor_classification = self.drug_description_ff_classification(drug_descrption_embeddings_tensor)
        drug_descrption_embeddings_tensor_regression = self.drug_description_ff_regression(drug_descrption_embeddings_tensor)

        side_effect_tensor_regression = self.side_effect_regression_layer(side_effect_tensor)

        prediction_classification_molecular_graph = torch.matmul(graph_feature_classification, side_effect_tensor_2.T).unsqueeze(2)

        prediction_classification_text_similarity = torch.matmul(drug_text_similarity_tensor_classification, side_effect_tensor_8.T).unsqueeze(2)

        prediction_drug_description = torch.matmul(drug_descrption_embeddings_tensor_classification, side_effect_tensor_10.T).unsqueeze(2)

        predictions_classification = torch.cat((prediction_classification_molecular_graph, prediction_classification_text_similarity, prediction_drug_description), dim=2)

        drug_feature_all_regression = torch.cat((graph_feature_regression, drug_text_similarity_tensor_regression, drug_descrption_embeddings_tensor_regression), dim=1)

        drug_feature_all_regression = self.drug_regression_layer(drug_feature_all_regression)

        predictions_classification = self.classification_view_mlp(predictions_classification)


        predictions_regression = torch.matmul(drug_feature_all_regression, side_effect_tensor_regression.T)

        return predictions_regression, predictions_classification


class DynamicWeightAverage:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_loss = None
        
    def update(self, losses):
        losses = torch.tensor(losses)
        
        if self.prev_loss is None:
            self.prev_loss = losses
            return torch.ones_like(losses) / len(losses)
            
        # Compute relative uncertainty
        loss_ratios = losses / self.prev_loss
        # Update weights inversely proportional to loss ratios
        weights = loss_ratios ** (-self.alpha)
        # Normalize weights
        weights = weights / weights.sum()
        
        self.prev_loss = losses
        return weights


def Train_Model(dataset, side_effect_edges, se_description_embeddings, epochs = 1000, batch_size = 512, device_id = 'cuda:0', save_path = 'trained_model/bilinear_model.pth', log_filename = 'log/training_log.txt', early_stop = True, early_stop_threshold=0.0001, lr = 0.0001):

    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle=True)#, num_workers=2)
    side_effect_edges = side_effect_edges.to(device_id)
    se_description_embeddings = se_description_embeddings.to(device_id)
    model = model_st(side_effect_edges, se_description_embeddings, device = device_id)

    """weight initialize"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]
    """load trained model"""
    

    model = model.to(device_id) 
    
    #criterion = nn.BCEWithLogitsLoss()
    #criterion = torch.nn.MSELoss()
    #criterion = loss_fun()
    #criterion = torch.nn.L1Loss()
    #criterion = torch.nn.SmoothL1Loss()
    #criterion = nn.CrossEntropyLoss()   maybe try CrossEntropyLoss
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    clip_value = 1.0
    #best_loss = float('inf')

    weight_scheduler = DynamicWeightAverage()
    #weight_scheduler.to(device_id)

    if True:#use a single gpu
        print("training with the following gpu:",device_id)
        for epoch in tqdm(range(epochs)):
            the_loss = 0
            #with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for i, (input1, input2, input3, input4, input5, input6, input7, labels) in enumerate(train_loader):
                
                #drug_smile, drug_embedding, target_sequence, target_embedding
                #drug_smile = drug_smile.cuda(final_device_ids[0])
                #drug_kg_embedding = drug_kg_embedding.cuda(final_device_ids[0])
                #target_sequence = target_sequence.cuda(final_device_ids[0])
                #target_kg_embedding = target_kg_embedding.cuda(final_device_ids[0])
                #drug_features = drug_features.cuda(final_device_ids[0])
                labels = labels.cuda(device_id)

                optimizer.zero_grad()

                # Forward pass
                predictions_regression, predictions_classification = model((input1, input2, input3, input4, input5, input6, input7))

                #print(predictions.shape)
                #print(labels.shape)
                #predictions = predictions.flatten()
                predictions_regression = predictions_regression.flatten()
                predictions_classification = predictions_classification.flatten()

                labels = labels.flatten()
                #print(predictions.shape)
                #print(labels.shape)
                #loss = torch.sum((predictions - labels) ** 2)

                #the_loss += loss.item()
                #print(outputs.shape)
                #print(labels.shape)
                #loss = criterion(predictions, labels)  # Remove .squeeze()
                #loss = loss_fun_2(predictions, labels)

                loss_regression = loss_fun_regression(predictions_regression, labels)
                
                loss_classification = loss_fun_classification(predictions_classification, labels)

                weights = weight_scheduler.update([loss_regression.item(), loss_classification.item()])

                the_loss = weights[0]*loss_regression.item() + weights[1]*loss_classification.item()

                loss = weights[0]*loss_regression + weights[1]*loss_classification
                # Backward pass and optimization
                loss.backward()
                
                # Clip gradients to prevent exploding gradients
                #nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                optimizer.step()
                #if (i + 1) % 1 == 0:
                #    print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
                    #logging.info(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {the_loss}')
            # Calculate average loss for the epoch
            #average_loss = calculate_average_loss(model, train_loader, criterion, device = final_device_ids[0])
    
            #print(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss}')
            #logging.info(f'Epoch [{epoch+1}/{epochs}], Loss: {average_loss}')

            # Check for early stopping
            #if average_loss < early_stop_threshold and early_stop == True:
            #    print(f'Loss is below the early stopping threshold. Training stopped.')
            #    break
            


    print('Training finished!')
    torch.save(model.state_dict(), save_path)
    print('save model succesuffly')

def Predict(Test_data, side_effect_edges, se_description_embeddings, kg_dim = 100, load_path = 'prediction_model/bilinear_MLP.pth', device_id = 'cuda:0'):
    print('start loading data')

    side_effect_edges = side_effect_edges.to(device_id)
    se_description_embeddings = se_description_embeddings.to(device_id)
    model = model_st(side_effect_edges, se_description_embeddings, device = device_id)

    model = model.to(device_id)

    state_dict = torch.load(load_path, map_location=device_id)  # Load the state dictionary

    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    

    model.eval()

    print('prediction load succesfully')

    all_regression_outputs = []
    all_prediction_outputs = []
    
    #print(type(Test_data))
    #print(len(Test_data))

    test_loader = DataLoader(Test_data, batch_size = 1, shuffle=False)#, num_workers=2)

    #frequency, drug_text_similarity, smiles_encoding, drug_description_embedding, drug_mfp, smiles_str
    with torch.no_grad():

        for i, (input1, input2, input3, input4, input5, input6, input7, labels) in enumerate(test_loader):

            #print(input1.shape)
            #print(input2.shape)
            #print(input3.shape)
            prediction_regression, prediction_classification = model((input1, input2, input3, input4, input5, input6, input7))
            #cur_predicttion = cur_predicttion.squeeze(0).cpu().numpy()
            prediction_regression = prediction_regression.squeeze(0).cpu().numpy()
            prediction_classification = prediction_classification.squeeze(0).cpu().numpy()
            #all_outputs.append(cur_predicttion)
            all_regression_outputs.append(prediction_regression)
            all_prediction_outputs.append(prediction_classification)
    
    return (all_regression_outputs, all_prediction_outputs)
if __name__ == "__main__":
    pass
    '''
    import pickle

    side_effect_feature_paths = [
            'data/glove_wordEmbedding.pkl',
            'data/side_effect_label_750.pkl',
        ]
    side_effect_features = []
    for path in side_effect_feature_paths:
        with open(path, 'rb') as f:
            cur = pickle.load(f)
            print(cur.shape)
    model = model_st()
    '''