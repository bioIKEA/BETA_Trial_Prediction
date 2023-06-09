import os
import sys
import pathlib
import pickle
#import wget
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from DeepPurpose import DTI as models

cuda = torch.cuda.is_available() # check if cuda avilabe 
device = torch.device("cuda" if cuda else "cpu")



def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)


# random_fold
def create_fold(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac = test_frac, replace = False, random_state = fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(frac = val_frac/(1-test_frac), replace = False, random_state = 1)
    train = train_val[~train_val.index.isin(val.index)]
    
    return train, val, test

# cold protein
def create_fold_setting_cold_protein(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    gene_drop = df['Target Sequence'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['Target Sequence'].isin(gene_drop)]

    train_val = df[~df['Target Sequence'].isin(gene_drop)]
    
    gene_drop_val = train_val['Target Sequence'].drop_duplicates().sample(frac = val_frac/(1-test_frac), 
                                                                          replace = False, 
                                                                          random_state = fold_seed).values
    val = train_val[train_val['Target Sequence'].isin(gene_drop_val)]
    train = train_val[~train_val['Target Sequence'].isin(gene_drop_val)]
    
    return train, val, test

# cold drug
def create_fold_setting_cold_drug(df, fold_seed, frac):
    train_frac, val_frac, test_frac = frac
    drug_drop = df['SMILES'].drop_duplicates().sample(frac = test_frac, replace = False, random_state = fold_seed).values
    
    test = df[df['SMILES'].isin(drug_drop)]

    train_val = df[~df['SMILES'].isin(drug_drop)]
    
    drug_drop_val = train_val['SMILES'].drop_duplicates().sample(frac = val_frac/(1-test_frac), 
                                                                 replace = False, 
                                                                 random_state = fold_seed).values
    val = train_val[train_val['SMILES'].isin(drug_drop_val)]
    train = train_val[~train_val['SMILES'].isin(drug_drop_val)]
    
    return train, val, test


def encode_drug(df_data, drug_encoding, column_name = 'SMILES', save_column_name = 'drug_encoding'):
    print('encoding drug...')
    print('unique drugs: ' + str(len(df_data[column_name].unique())))

    if drug_encoding == 'CNN':
        unique = pd.Series(df_data[column_name].unique()).apply(trans_drug)
        unique_dict = dict(zip(df_data[column_name].unique(), unique))
        df_data[save_column_name] = [unique_dict[i] for i in df_data[column_name]]
        # the embedding is large and not scalable but quick, so we move to encode in dataloader batch. 
    else:
        raise AttributeError("Please use the correct drug encoding available!")
    return df_data

def encode_protein(df_data, target_encoding, column_name = 'Target Sequence', save_column_name = 'target_encoding'):
    print('encoding protein...')
    print('unique target sequence: ' + str(len(df_data[column_name].unique())))

    if target_encoding == 'CNN':
        AA = pd.Series(df_data[column_name].unique()).apply(trans_protein)
        AA_dict = dict(zip(df_data[column_name].unique(), AA))
        df_data[save_column_name] = [AA_dict[i] for i in df_data[column_name]]
        # the embedding is large and not scalable but quick, so we move to encode in dataloader batch. 

    else:
        raise AttributeError("Please use the correct protein encoding available!")
    return df_data

def data_process(X_drug = None, X_target = None, y=None, drug_encoding=None, target_encoding=None, 
                 split_method = 'random', frac = [0.7, 0.1, 0.2], random_seed = 1, sample_frac = 1, mode = 'DTI', X_drug_ = None, X_target_ = None):
    
    if random_seed == 'TDC':
        random_seed = 1234
    #property_prediction_flag = X_target is None
    property_prediction_flag, function_prediction_flag, DDI_flag, PPI_flag, DTI_flag = False, False, False, False, False

    if (X_target is None) and (X_drug is not None) and (X_drug_ is None):
        property_prediction_flag = True
    elif (X_target is not None) and (X_drug is None) and (X_target_ is None):
        function_prediction_flag = True
    elif (X_drug is not None) and (X_drug_ is not None):
        DDI_flag = True
        if (X_drug is None) or (X_drug_ is None):
            raise AttributeError("Drug pair sequence should be in X_drug, X_drug_")
    elif (X_target is not None) and (X_target_ is not None):
        PPI_flag = True
        if (X_target is None) or (X_target_ is None):
            raise AttributeError("Target pair sequence should be in X_target, X_target_")
    if (X_drug is not None) and (X_target is not None):
        DTI_flag = True
        if (X_drug is None) or (X_target is None):
            raise AttributeError("Target pair sequence should be in X_target, X_drug")
    else:
        raise AttributeError("Please use the correct mode. Currently, we support DTI, DDI, PPI, Drug Property Prediction and Protein Function Prediction...")

    if split_method == 'repurposing_VS':
        y = [-1]*len(X_drug) # create temp y for compatitibility
    
    if DTI_flag:
        print('Drug Target Interaction Prediction Mode...')
        if isinstance(X_target, str):
            X_target = [X_target]
        if len(X_target) == 1:
            # one target high throughput screening setting
            X_target = np.tile(X_target, (length_func(X_drug), ))

        df_data = pd.DataFrame(zip(X_drug, X_target, y))
        # print(df_data.head(5))
        df_data.rename(columns={0:'SMILES',
                                1: 'Target Sequence',
                                2: 'Label'}, 
                                inplace=True)
        print('in total: ' + str(len(df_data)) + ' drug-target pairs')
        # print('label: ' + str(df_data.Label.values[0]))
    elif property_prediction_flag:
        print('Drug Property Prediction Mode...')
        df_data = pd.DataFrame(zip(X_drug, y))
        df_data.rename(columns={0:'SMILES',
                                1: 'Label'}, 
                                inplace=True)
        print('in total: ' + str(len(df_data)) + ' drugs')
    elif function_prediction_flag:
        print('Protein Function Prediction Mode...')
        df_data = pd.DataFrame(zip(X_target, y))
        df_data.rename(columns={0:'Target Sequence',
                                1: 'Label'}, 
                                inplace=True)
        print('in total: ' + str(len(df_data)) + ' proteins')
    elif PPI_flag:
        print('Protein Protein Interaction Prediction Mode...')

        df_data = pd.DataFrame(zip(X_target, X_target_, y))
        df_data.rename(columns={0: 'Target Sequence 1',
                                1: 'Target Sequence 2',
                                2: 'Label'}, 
                                inplace=True)
        print('in total: ' + str(len(df_data)) + ' protein-protein pairs')
    elif DDI_flag:
        print('Drug Drug Interaction Prediction Mode...')

        df_data = pd.DataFrame(zip(X_drug, X_drug_, y))
        df_data.rename(columns={0: 'SMILES 1',
                                1: 'SMILES 2',
                                2: 'Label'}, 
                                inplace=True)
        print('in total: ' + str(len(df_data)) + ' drug-drug pairs')


    if sample_frac != 1:
        df_data = df_data.sample(frac = sample_frac).reset_index(drop = True)
        print('after subsample: ' + str(len(df_data)) + ' data points...') 

    if DTI_flag:
        df_data = encode_drug(df_data, drug_encoding)
        df_data = encode_protein(df_data, target_encoding)
    elif DDI_flag:
        df_data = encode_drug(df_data, drug_encoding, 'SMILES 1', 'drug_encoding_1')
        df_data = encode_drug(df_data, drug_encoding, 'SMILES 2', 'drug_encoding_2')
    elif PPI_flag:
        df_data = encode_protein(df_data, target_encoding, 'Target Sequence 1', 'target_encoding_1')
        df_data = encode_protein(df_data, target_encoding, 'Target Sequence 2', 'target_encoding_2')
    elif property_prediction_flag:
        df_data = encode_drug(df_data, drug_encoding)
    elif function_prediction_flag:
        df_data = encode_protein(df_data, target_encoding)

    # dti split
    if DTI_flag:
        if split_method == 'repurposing_VS':
            pass
        else:
            print('splitting dataset...')

        if split_method == 'random': 
            train, val, test = create_fold(df_data, random_seed, frac)
        elif split_method == 'cold_drug':
            train, val, test = create_fold_setting_cold_drug(df_data, random_seed, frac)
        elif split_method == 'HTS':
            train, val, test = create_fold_setting_cold_drug(df_data, random_seed, frac)
            val = pd.concat([val[val.Label == 1].drop_duplicates(subset = 'SMILES'), val[val.Label == 0]])
            test = pd.concat([test[test.Label == 1].drop_duplicates(subset = 'SMILES'), test[test.Label == 0]])        
        elif split_method == 'cold_protein':
            train, val, test = create_fold_setting_cold_protein(df_data, random_seed, frac)
        elif split_method == 'repurposing_VS':
            train = df_data
            val = df_data
            test = df_data
        elif split_method == 'no_split':
            print('do not do train/test split on the data for already splitted data')
            return df_data.reset_index(drop=True)
        else:
            raise AttributeError("Please select one of the three split method: random, cold_drug, cold_target!")
    elif DDI_flag:
        if split_method == 'random': 
            train, val, test = create_fold(df_data, random_seed, frac)
        elif split_method == 'no_split':
            return df_data.reset_index(drop=True)
    elif PPI_flag:
        if split_method == 'random': 
            train, val, test = create_fold(df_data, random_seed, frac)
        elif split_method == 'no_split':
            return df_data.reset_index(drop=True)
    elif function_prediction_flag:
        if split_method == 'random': 
            train, val, test = create_fold(df_data, random_seed, frac)
        elif split_method == 'no_split':
            return df_data.reset_index(drop=True)
    elif property_prediction_flag:
        # drug property predictions
        if split_method == 'repurposing_VS':
            train = df_data
            val = df_data
            test = df_data
        elif split_method == 'no_split':
            print('do not do train/test split on the data for already splitted data')
            return df_data.reset_index(drop=True)
        else:
            train, val, test = create_fold(df_data, random_seed, frac)

    print('Done.')
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)



def data_process_(X_drug=None, X_target=None, drug_encoding=None, target_encoding=None,
                 DTI_flag=True, X_drug_=None,X_target_=None):

    if DTI_flag:
        print('Drug Target Interaction Prediction Mode...')


        df_data_drug = pd.DataFrame(zip(X_drug))
        df_data_target = pd.DataFrame(zip(X_target))

        # print(df_data.head(5))
        df_data_drug.rename(columns={0: 'SMILES'},
                       inplace=True)
        df_data_target.rename(columns={0: 'Target Sequence'},
                            inplace=True)

    if DTI_flag:
        df_data_drug = encode_drug(df_data_drug, drug_encoding)
        df_data_target = encode_protein(df_data_target, target_encoding)

    return df_data_drug, df_data_target




class drug_data_process_loader(data.Dataset):

    def __init__(self, list_IDs, df, **config):
        'Initialization'
        self.list_IDs = list_IDs     
        self.df = df                      
        self.config = config

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['drug_encoding']        
        if self.config['drug_encoding'] == 'CNN' or self.config['drug_encoding'] == 'CNN_RNN':
            v_d = drug_2_embed(v_d) 

        return v_d
   
class protein_data_process_loader(data.Dataset):

    def __init__(self, list_IDs, df, **config):
        'Initialization'
        self.list_IDs = list_IDs      
        self.df = df                   
        self.config = config

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.list_IDs[index]
        
        v_p = self.df.iloc[index]['target_encoding']
        if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
            v_p = protein_2_embed(v_p)

        return v_p





def data_encoder(train, config):

    D_BATCH_SIZE = config['drug_batch_size']
    P_BATCH_SIZE = config['protein_batch_size']

    drug_params = {'batch_size': D_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': config['num_workers']}
                   
    protein_params = {'batch_size': P_BATCH_SIZE,
                      'shuffle': False,
                      'num_workers': config['num_workers']}

    model_drug = models.model_initialize_drug(**config)
    model_protein = models.model_initialize_protein(**config)
    
    training_generator_drug = data.DataLoader(drug_data_process_loader(train.index.values, train, **config), **drug_params)
    training_generator_protein = data.DataLoader(protein_data_process_loader(train.index.values, train, **config), **protein_params)
    
    
    model_drug = model_drug.to(device)
    for i, v_d in enumerate(training_generator_drug):
        # print('i: ',i, ' v_d: ',v_d.shape)
        if i == 1:
            break # break here
        v_d = v_d.float().to(device)
        r_D = model_drug(v_d)

    model_protein = model_protein.to(device)  
    for j, v_p in enumerate(training_generator_protein):
        if j == 1:
            break # break here
        v_p = v_p.float().to(device)
        r_P = model_protein(v_p)
   
    return r_D, r_P


def data_encoder_(X_drugs,X_targets, config):
    D_BATCH_SIZE = len(X_drugs)
    P_BATCH_SIZE = len(X_targets)

    drug_params = {'batch_size': D_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': config['num_workers']}

    protein_params = {'batch_size': P_BATCH_SIZE,
                      'shuffle': False,
                      'num_workers': config['num_workers']}

    model_drug = models.model_initialize_drug(**config)
    model_protein = models.model_initialize_protein(**config)

    training_generator_drug = data.DataLoader(drug_data_process_loader(X_drugs.index.values, X_drugs, **config),
                                              **drug_params)
    training_generator_protein = data.DataLoader(protein_data_process_loader(X_targets.index.values, X_targets, **config),
                                                 **protein_params)

    model_drug = model_drug.to(device)
    for i, v_d in enumerate(training_generator_drug):
        # print('i: ',i, ' v_d: ',v_d.shape)
        if i == 1:
            break  # break here
        v_d = v_d.float().to(device)
        r_D = model_drug(v_d)

    model_protein = model_protein.to(device)
    for j, v_p in enumerate(training_generator_protein):
        if j == 1:
            break  # break here
        v_p = v_p.float().to(device)
        r_P = model_protein(v_p)

    return r_D, r_P
def generate_config(drug_encoding = None, target_encoding = None, 
                    input_dim_drug = 1024, 
                    input_dim_protein = 8420,
                    hidden_dim_drug = 256, 
                    hidden_dim_protein = 256,
                    cls_hidden_dims = [1024, 1024, 512], 
                    drug_batch_size = 256, 
                    protein_batch_size = 256, 
                    cnn_drug_filters = [32,64,96],
                    cnn_drug_kernels = [4,6,8],
                    cnn_target_filters = [32,64,96],
                    cnn_target_kernels = [4,8,12],
                    num_workers = 0,
                    cuda_id = None, 
                    ):

    base_config = {'input_dim_drug': input_dim_drug,
                    'input_dim_protein': input_dim_protein,
                    'hidden_dim_drug': hidden_dim_drug, # hidden dim of drug
                    'hidden_dim_protein': hidden_dim_protein, # hidden dim of protein
                    'cls_hidden_dims' : cls_hidden_dims, # decoder classifier dim 1
                    'drug_batch_size': drug_batch_size, 
                    'protein_batch_size': protein_batch_size, 
                    'drug_encoding': drug_encoding,
                    'target_encoding': target_encoding, 
                    'binary': False,
                    'num_workers': num_workers,
                    'cuda_id': cuda_id                 
    }
                 
    if drug_encoding == 'CNN':
        base_config['cnn_drug_filters'] = cnn_drug_filters
        base_config['cnn_drug_kernels'] = cnn_drug_kernels
    elif drug_encoding is None:
        pass
    else:
        raise AttributeError("Please use the correct drug encoding available!")

            
    if target_encoding == 'CNN':
        base_config['cnn_target_filters'] = cnn_target_filters
        base_config['cnn_target_kernels'] = cnn_target_kernels
    elif target_encoding is None:
        pass
    else:
        raise AttributeError("Please use the correct protein encoding available!")

    return base_config


# '?' padding
amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']

from sklearn.preprocessing import OneHotEncoder
enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100


def trans_protein(x):
    temp = list(x.upper())
    temp = [i if i in amino_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_PROTEIN:
        temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
    else:
        temp = temp [:MAX_SEQ_PROTEIN]
    return temp

def protein_2_embed(x):
    return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

def trans_drug(x):
    temp = list(x)
    temp = [i if i in smiles_char else '?' for i in temp]
    if len(temp) < MAX_SEQ_DRUG:
        temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
    else:
        temp = temp [:MAX_SEQ_DRUG]
    return temp

def drug_2_embed(x):
    return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T    
