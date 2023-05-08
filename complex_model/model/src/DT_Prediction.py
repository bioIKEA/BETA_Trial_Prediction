# -*- coding: utf-8 -*-
"""
Created on Sun May 08 16:02:48 2022

@author: Shibo Zhou

"""
# import os
# import numpy as np
# import pickle
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import average_precision_score
# from sklearn.model_selection import KFold
# from .utils import *
# from torch import nn
# import torch
# import torch.nn.functional as F
# from sklearn.model_selection import train_test_split,StratifiedKFold
# import sys
# from optparse import OptionParser
# from .DeepPurpose.utils import *
# from .DeepPurpose import dataset
# from .DeepPurpose import DTI as models
# from . import DataGenerator

from optparse import OptionParser
import DataGenerator
import numpy as np
import torch
import torch.nn.functional as F
from DeepPurpose import dataset
from DeepPurpose.utils import *
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from torch import nn
from utils import *
import sys

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

import random
random.seed(0)
np.random.seed(0)

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def row_normalize(a_matrix, substract_self_loop):
    if substract_self_loop == True:
        np.fill_diagonal(a_matrix, 0)
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1) + 1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # features
        self.drug_embedding = weight_variable([num_drug, dim_drug])
        print('self.drug_embedding.shape', self.drug_embedding.shape)
        self.protein_embedding = weight_variable([num_protein, dim_protein])
        print('self.protein_embedding.shape', self.protein_embedding.shape)
        self.disease_embedding = weight_variable([num_disease, dim_disease])
        print('self.disease_embedding.shape', self.disease_embedding.shape)
        self.sideeffect_embedding = weight_variable([num_sideeffect, dim_sideeffect])
        print('self.sideeffect_embedding.shape', self.sideeffect_embedding.shape)
        # feature passing weights (maybe different types of nodes can use different weights)
        self.W0 = weight_variable([dim_pass + dim_drug, dim_drug])
        print('self.W0.shape', self.W0.shape)
        self.W0_2 = weight_variable([dim_drug, dim_drug])
        self.b0 = bias_variable([dim_drug])
        print('self.b0.shape', self.b0.shape)
        # passing 1 times (can be easily extended to multiple passes)
        self.ddn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        print('self.ddn_de_w', self.ddn_de_w)
        print('len(self.ddn_de_w)', len(self.ddn_de_w))
        print('self.ddn_de_w[0].shape', self.ddn_de_w[0].shape)
        print('self.ddn_de_w[1].shape', self.ddn_de_w[1].shape)
        self.dcn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        print('len(self.dcn_de_w)', len(self.dcn_de_w))
        print('self.dcn_de_w[0].shape', self.dcn_de_w[0].shape)
        print('self.dcn_de_w[1].shape', self.dcn_de_w[1].shape)
        self.ddin_die_w = a_layer([num_disease, dim_disease], dim_pass)
        print('len(self.ddin_die_w)', len(self.ddin_die_w))
        print('self.ddin_die_w[0].shape', self.ddin_die_w[0].shape)
        print('self.ddin_die_w[1].shape', self.ddin_die_w[1].shape)
        self.dsn_se_w = a_layer([num_sideeffect, dim_sideeffect], dim_pass)
        print('len(self.dsn_se_w)', len(self.dsn_se_w))
        self.dpn_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        print('len(self.dpn_pe_w)', len(self.dpn_pe_w))
        self.dpan_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        print('len(self.dpan_pe_w)', len(self.dpan_pe_w))
        self.sv_sdn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        print('len(self.sv_sdn_de_w)', len(self.sv_sdn_de_w))
        self.pv_ppn_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        print('len(self.pv_ppn_pe_w)',  len(self.pv_ppn_pe_w))
        self.pv_psn_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        print('len(self.pv_psn_pe_w)', len(self.pv_psn_pe_w))
        self.pv_pdin_die_w = a_layer([num_disease, dim_disease], dim_pass)
        print('len(self.pv_pdin_die_w)', len(self.pv_pdin_die_w))
        self.pv_pdn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        print('len(self.pv_pdn_de_w)', len(self.pv_pdn_de_w))
        self.pv_pdan_de_w = a_layer([num_drug, dim_drug], dim_pass)
        print('len(self.pv_pdan_de_w)', len(self.pv_pdan_de_w))
        self.div_didn_de_w = a_layer([num_drug, dim_drug], dim_pass)
        print('len(self.div_didn_de_w)', len(self.div_didn_de_w))
        self.div_dipn_pe_w = a_layer([num_protein, dim_protein], dim_pass)
        print('len(self.div_dipn_pe_w)', len(self.div_dipn_pe_w))
        # bi weight
        self.ddr = bi_layer(dim_drug, dim_drug, sym=True, dim_pred=dim_pred)
        print('len(self.ddr)', len(self.ddr))
        print('self.ddr[0].shape', self.ddr[0].shape)
        print('self.ddr[1].shape', self.ddr[1].shape)
        self.dcr = bi_layer(dim_drug, dim_drug, sym=True, dim_pred=dim_pred)
        print('len(self.dcr)', len(self.dcr))
        print('self.dcr[0].shape', self.dcr[0].shape)
        print('self.dcr[1].shape', self.dcr[1].shape)
        self.ddir = bi_layer(dim_drug, dim_drug, sym=False, dim_pred=dim_pred)
        print('len(self.ddir)', len(self.ddir))
        self.dsr = bi_layer(dim_drug, dim_drug, sym=False, dim_pred=dim_pred)
        print('len(self.dsr)', len(self.dsr))
        self.ppr = bi_layer(dim_drug, dim_drug, sym=True, dim_pred=dim_pred)
        print('len(self.ppr)', len(self.ppr))
        self.psr = bi_layer(dim_drug, dim_drug, sym=True, dim_pred=dim_pred)
        print('len(self.psr)', len(self.psr))
        self.pdir = bi_layer(dim_drug, dim_drug, sym=False, dim_pred=dim_pred)
        print('len(self.pdir)', len(self.pdir))
        self.dpr = bi_layer(dim_drug, dim_drug, sym=False, dim_pred=dim_pred)
        print('len(self.dpr)', len(self.dpr))
        print('self.dpr[0].shape', self.dpr[0].shape)
        print('self.dpr[1].shape', self.dpr[1].shape)
        #sys.exit()

    def forward(self, drug_drug, drug_drug_normalize,
                drug_chemical, drug_chemical_normalize,
                drug_disease, drug_disease_normalize,
                drug_sideeffect, drug_sideeffect_normalize,
                protein_protein, protein_protein_normalize,
                protein_sequence, protein_sequence_normalize,
                protein_disease, protein_disease_normalize,
                drug_encoder_data, protein_encoder_data,
                disease_drug, disease_drug_normalize,
                disease_protein, disease_protein_normalize,
                sideeffect_drug, sideeffect_drug_normalize,
                drug_protein, drug_protein_normalize,
                protein_drug, protein_drug_normalize,
                drug_protein_mask,
                ):
        l2_loss = cuda(torch.tensor([0], dtype=torch.float64))
        print('l2_loss', l2_loss)



        drug_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(drug_drug_normalize, a_cul(self.drug_embedding, *self.ddn_de_w)) + \
                       torch.matmul(drug_chemical_normalize, a_cul(self.drug_embedding, *self.dcn_de_w)) + \
                       torch.matmul(drug_disease_normalize, a_cul(self.disease_embedding, *self.ddin_die_w)) + \
                       torch.matmul(drug_sideeffect_normalize, a_cul(self.sideeffect_embedding, *self.dsn_se_w)) + \
                       torch.matmul(drug_protein_normalize,
                                    a_cul(self.protein_embedding, *self.dpn_pe_w))], axis=1), self.W0_2) + self.b0), dim=1)



        tmp_just = torch.matmul(drug_drug_normalize, a_cul(self.drug_embedding, *self.ddn_de_w)) + \
                       torch.matmul(drug_chemical_normalize, a_cul(self.drug_embedding, *self.dcn_de_w)) + \
                       torch.matmul(drug_disease_normalize, a_cul(self.disease_embedding, *self.ddin_die_w)) + \
                       torch.matmul(drug_sideeffect_normalize, a_cul(self.sideeffect_embedding, *self.dsn_se_w)) + \
                       torch.matmul(drug_protein_normalize,
                                    a_cul(self.protein_embedding, *self.dpn_pe_w))
        tmp_just_2 = torch.cat([torch.matmul(drug_drug_normalize, a_cul(self.drug_embedding, *self.ddn_de_w)) + \
                       torch.matmul(drug_chemical_normalize, a_cul(self.drug_embedding, *self.dcn_de_w)) + \
                       torch.matmul(drug_disease_normalize, a_cul(self.disease_embedding, *self.ddin_die_w)) + \
                       torch.matmul(drug_sideeffect_normalize, a_cul(self.sideeffect_embedding, *self.dsn_se_w)) + \
                       torch.matmul(drug_protein_normalize,
                                    a_cul(self.protein_embedding, *self.dpn_pe_w)), \
                       self.drug_embedding], axis=1)
        print('tmp_just.shape', tmp_just.shape)
        print('tmp_just_2.shape', tmp_just_2.shape)
        print('self.W0.shape', self.W0.shape)
        print('self.b0.shape', self.b0.shape)
        print('self.protein_embedding.shape', self.protein_embedding.shape)



        tmp_just_3 = torch.cat([torch.matmul(drug_drug_normalize, a_cul(self.drug_embedding, *self.ddn_de_w))], axis=1)
        print('tmp_just_3.shape', tmp_just_3.shape)



        tmp_just_4 = torch.cat([torch.matmul(drug_drug_normalize, a_cul(self.drug_embedding, *self.ddn_de_w)) + drug_encoder_data], axis= 1)
        print('tmp_just_4', tmp_just_4)



        print('drug_vector1.shape', drug_vector1.shape)


        sideeffect_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(sideeffect_drug_normalize, a_cul(self.drug_embedding, *self.sv_sdn_de_w)), \
                       self.sideeffect_embedding], axis=1), self.W0) + self.b0), dim=1)
        print('sideeffect_vector1.shape', sideeffect_vector1.shape)


        protein_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(protein_protein_normalize, a_cul(self.protein_embedding, *self.pv_ppn_pe_w)) + \
                       torch.matmul(protein_sequence_normalize, a_cul(self.protein_embedding, *self.pv_psn_pe_w)) + \
                       torch.matmul(protein_disease_normalize, a_cul(self.disease_embedding, *self.pv_pdin_die_w)) + \
                       torch.matmul(protein_drug_normalize, a_cul(self.drug_embedding, *self.pv_pdn_de_w))], axis=1), self.W0_2) + self.b0), dim=1)


        print('protein_vector1.shape', protein_vector1.shape)

        #sys.exit()

        disease_vector1 = torch.nn.functional.normalize(F.relu(torch.matmul(
            torch.cat([torch.matmul(disease_drug_normalize, a_cul(self.drug_embedding, *self.div_didn_de_w)) + \
                       torch.matmul(disease_protein_normalize, a_cul(self.protein_embedding, *self.div_dipn_pe_w)), \
                       self.disease_embedding], axis=1), self.W0) + self.b0), dim=1)
        print('disease_vector1.shape', disease_vector1.shape)

        drug_representation = drug_vector1
        protein_representation = protein_vector1
        disease_representation = disease_vector1
        sideeffect_representation = sideeffect_vector1

        # reconstructing networks
        drug_drug_reconstruct = bi_cul(drug_representation, drug_representation, *self.ddr)
        print('drug_drug_reconstruct.shape', drug_drug_reconstruct.shape)
        drug_drug_reconstruct_loss = torch.sum(
            torch.multiply((drug_drug_reconstruct - drug_drug), (drug_drug_reconstruct - drug_drug)))
        print('drug_drug_reconstruct_loss.shape', drug_drug_reconstruct_loss.shape)

        drug_chemical_reconstruct = bi_cul(drug_representation, drug_representation, *self.dcr)
        print('drug_chemical_reconstruct.shape', drug_chemical_reconstruct.shape)
        drug_chemical_reconstruct_loss = torch.sum(
            torch.multiply((drug_chemical_reconstruct - drug_chemical), (drug_chemical_reconstruct - drug_chemical)))
        print('drug_chemical_reconstruct_loss.shape', drug_chemical_reconstruct_loss.shape)

        drug_disease_reconstruct = bi_cul(drug_representation, disease_representation, *self.ddir)
        print('drug_disease_reconstruct.shape', drug_disease_reconstruct.shape)
        drug_disease_reconstruct_loss = torch.sum(
            torch.multiply((drug_disease_reconstruct - drug_disease), (drug_disease_reconstruct - drug_disease)))
        print('drug_disease_reconstruct_loss.shape', drug_disease_reconstruct_loss.shape)

        drug_sideeffect_reconstruct = bi_cul(drug_representation, sideeffect_representation, *self.dsr)
        print('drug_sideeffect_reconstruct.shape', drug_sideeffect_reconstruct.shape)
        drug_sideeffect_reconstruct_loss = torch.sum(torch.multiply((drug_sideeffect_reconstruct - drug_sideeffect),
                                                                    (drug_sideeffect_reconstruct - drug_sideeffect)))
        print('drug_sideeffect_reconstruct_loss.shape', drug_sideeffect_reconstruct_loss.shape)

        protein_protein_reconstruct = bi_cul(protein_representation, protein_representation, *self.ppr)
        print('protein_protein_reconstruct.shape', protein_protein_reconstruct.shape)
        protein_protein_reconstruct_loss = torch.sum(torch.multiply((protein_protein_reconstruct - protein_protein),
                                                                    (protein_protein_reconstruct - protein_protein)))
        print('protein_protein_reconstruct_loss.shape', protein_protein_reconstruct_loss.shape)

        protein_sequence_reconstruct = bi_cul(protein_representation, protein_representation, *self.psr)
        print('protein_sequence_reconstruct.shape', protein_sequence_reconstruct.shape)
        protein_sequence_reconstruct_loss = torch.sum(torch.multiply((protein_sequence_reconstruct - protein_sequence),
                                                                     (protein_sequence_reconstruct - protein_sequence)))
        print('protein_sequence_reconstruct_loss.shape', protein_sequence_reconstruct_loss.shape)

        protein_disease_reconstruct = bi_cul(protein_representation, disease_representation, *self.pdir)
        print('protein_disesase_recontruct.shape', protein_disease_reconstruct.shape)
        protein_disease_reconstruct_loss = torch.sum(torch.multiply((protein_disease_reconstruct - protein_disease),
                                                                    (protein_disease_reconstruct - protein_disease)))
        print('protein_disease_reconstruct_loss.shape', protein_disease_reconstruct_loss.shape)

        drug_protein_reconstruct = bi_cul(drug_representation, protein_representation, *self.dpr)
        print('drug_protein_reconstruct.shape', drug_protein_reconstruct.shape)
        tmp = torch.multiply(drug_protein_mask, (drug_protein_reconstruct - drug_protein))
        print('tmp.shape', tmp.shape)
        drug_protein_reconstruct_loss = torch.sum(torch.multiply(tmp, tmp))  # / (torch.sum(self.drug_protein_mask)
        print('drug_protein_reconstruct_loss.shape', drug_protein_reconstruct_loss.shape)

        for param in model.parameters():
            l2_loss += torch.norm(param, 2)


        loss = drug_protein_reconstruct_loss + 0.0 * (drug_drug_reconstruct_loss + drug_chemical_reconstruct_loss +
                                                      drug_disease_reconstruct_loss + drug_sideeffect_reconstruct_loss +                                                      protein_protein_reconstruct_loss + protein_sequence_reconstruct_loss + protein_disease_reconstruct_loss) 

        print('loss.shape', loss.shape)
        #sys.exit()
        return loss, drug_protein_reconstruct_loss, drug_protein_reconstruct


def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x


def train_and_evaluate_(optimizer, DTItrain, DTItest, num_steps=4000):
    drug_protein = np.zeros((num_drug, num_protein))
    mask = np.zeros((num_drug, num_protein))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein, False)
    protein_drug_normalize = row_normalize(protein_drug, False)

    drug_protein_test = np.zeros((num_drug, num_protein))
    mask_test = np.zeros((num_drug, num_protein))
    for ele in DTItest:
        drug_protein_test[ele[0], ele[1]] = ele[2]
        mask_test[ele[0], ele[1]] = 1
    protein_drug_test = drug_protein_test.T

    drug_protein_normalize_test = row_normalize(drug_protein_test, False)
    protein_drug_normalize_test = row_normalize(protein_drug_test, False)


    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0
    train_aupr = 0
    train_auc = 0

    #model.zero_grad()
    print('model', model)

    for i in range(num_steps):
        model.zero_grad()
        print('doing round %s step %s / %s' % (round, i, num_steps))
        model.train()
        mask_ = cuda(torch.tensor(mask))
        protein_drug_normalize_ = cuda(torch.tensor(protein_drug_normalize))
        protein_drug_ = cuda(torch.tensor(protein_drug))
        drug_protein_normalize_ = cuda(torch.tensor(drug_protein_normalize))
        drug_protein_ = cuda(torch.tensor(drug_protein))
        global sideeffect_drug_normalize
        sideeffect_drug_normalize_ = cuda(torch.tensor(sideeffect_drug_normalize))
        global sideeffect_drug
        sideeffect_drug_ = cuda(torch.tensor(sideeffect_drug))
        global disease_protein_normalize
        disease_protein_normalize_ = cuda(torch.tensor(disease_protein_normalize))
        global disease_protein
        disease_protein_ = cuda(torch.tensor(disease_protein))
        global disease_drug_normalize
        disease_drug_normalize_ = cuda(torch.tensor(disease_drug_normalize))
        global disease_drug
        disease_drug_ = cuda(torch.tensor(disease_drug))
        global protein_disease_normalize
        protein_disease_normalize_ = cuda(torch.tensor(protein_disease_normalize))
        global protein_disease
        protein_disease_ = cuda(torch.tensor(protein_disease))
        global protein_sequence_normalize
        protein_sequence_normalize_ = cuda(torch.tensor(protein_sequence_normalize))
        global protein_sequence
        protein_sequence_ = cuda(torch.tensor(protein_sequence))
        global protein_protein_normalize
        protein_protein_normalize_ = cuda(torch.tensor(protein_protein_normalize))
        global protein_protein
        protein_protein_ = cuda(torch.tensor(protein_protein))
        global drug_sideeffect_normalize
        drug_sideeffect_normalize_ = cuda(torch.tensor(drug_sideeffect_normalize))
        global drug_sideeffect
        drug_sideeffect_ = cuda(torch.tensor(drug_sideeffect))
        global drug_disease_normalize
        drug_disease_normalize_ = cuda(torch.tensor(drug_disease_normalize))
        global drug_disease
        drug_disease_ = cuda(torch.tensor(drug_disease))
        global drug_chemical_normalize
        drug_chemical_normalize_ = cuda(torch.tensor(drug_chemical_normalize))
        global drug_chemical
        drug_chemical_ = cuda(torch.tensor(drug_chemical))
        global drug_drug_normalize
        drug_drug_normalize_ = cuda(torch.tensor(drug_drug_normalize))
        global drug_drug
        drug_drug_ = cuda(torch.tensor(drug_drug))
        global drug_encoder_data
        drug_encoder_data_ = cuda(drug_encoder_data)
        global protein_encoder_data
        protein_encoder_data_ = cuda(protein_encoder_data)

        tloss_train, dtiloss_train, results_train = model(
            drug_drug_, drug_drug_normalize_, \
            drug_chemical_, drug_chemical_normalize_, \
            drug_disease_, drug_disease_normalize_, \
            drug_sideeffect_, drug_sideeffect_normalize_, \
            protein_protein_, protein_protein_normalize_, \
            protein_sequence_, protein_sequence_normalize_, \
            protein_disease_, protein_disease_normalize_, \
            drug_encoder_data_, protein_encoder_data_, \
            disease_drug_, disease_drug_normalize_, \
            disease_protein_, disease_protein_normalize_, \
            sideeffect_drug_, sideeffect_drug_normalize_, \
            drug_protein_, drug_protein_normalize_, \
            protein_drug_, protein_drug_normalize_, \
            mask_)
        tloss_train.backward(retain_graph=True)
        optimizer.step()

        pred_list_train = []
        ground_truth_train = []
        ii_train = 1
        for ele in DTItrain:
            # print('Testing %s / %s' % (ii, len(DTItest)))
            pred_list_train.append(results_train[ele[0], ele[1]].cpu().detach().numpy())
            ground_truth_train.append(ele[2])
            # pred_list.append(results[ele[0], ele[1]])
            print(results_train[ele[0], ele[1]], "-> ", ele[2])
            ii_train += 1
        train_auc = roc_auc_score(ground_truth_train, pred_list_train)
        train_aupr = average_precision_score(ground_truth_train, pred_list_train)
        print('EPOCH', i)
        print('train_auc: ',train_auc)
        print('train_aupr: ', train_aupr)

        #test for epoch starts here
        mask_test = cuda(torch.tensor(mask_test))
        protein_drug_normalize_test = cuda(torch.tensor(protein_drug_normalize_test))
        protein_drug_test = cuda(torch.tensor(protein_drug_test))
        drug_protein_normalize_test = cuda(torch.tensor(drug_protein_normalize_test))
        drug_protein_test = cuda(torch.tensor(drug_protein_test))

      
        
    # every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
    with torch.no_grad():
        model.eval()

        tloss, dtiloss, results_test = model(
            drug_drug_, drug_drug_normalize_, \
            drug_chemical_, drug_chemical_normalize_, \
            drug_disease_, drug_disease_normalize_, \
            drug_sideeffect_, drug_sideeffect_normalize_, \
            protein_protein_, protein_protein_normalize_, \
            protein_sequence_, protein_sequence_normalize_, \
            protein_disease_, protein_disease_normalize_, \
            drug_encoder_data_, protein_encoder_data_, \
            disease_drug_, disease_drug_normalize_, \
            disease_protein_, disease_protein_normalize_, \
            sideeffect_drug_, sideeffect_drug_normalize_, \
            drug_protein_test, drug_protein_normalize_test, \
            protein_drug_test, protein_drug_normalize_test, \
            mask_)
        pred_list = []
        ground_truth = []
        ii = 1
        for ele in DTItest:
            # print('Testing %s / %s' % (ii, len(DTItest)))
            pred_list.append(results_test[ele[0], ele[1]].cpu().detach().numpy())
            ground_truth.append(ele[2])
            # pred_list.append(results[ele[0], ele[1]])
            print(results_test[ele[0], ele[1]], "-> ", ele[2])
            ii += 1
        test_auc = roc_auc_score(ground_truth, pred_list)
        test_aupr = average_precision_score(ground_truth, pred_list)

        print('test_auc: ',test_auc)
        print('test_aupr: ', test_aupr)
    return test_auc, test_aupr


def train_and_evaluate(DTItrain, DTIvalid, DTItest, round, verbose=True, num_steps=4000):
    drug_protein = np.zeros((num_drug, num_protein))
    mask = np.zeros((num_drug, num_protein))
    for ele in DTItrain:
        drug_protein[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1
    protein_drug = drug_protein.T

    drug_protein_normalize = row_normalize(drug_protein, False)
    protein_drug_normalize = row_normalize(protein_drug, False)

    best_valid_aupr = 0
    best_valid_auc = 0
    test_aupr = 0
    test_auc = 0

    model.zero_grad()

    for i in range(num_steps):
        print('doing round %s step %s / %s' % (round, i, num_steps))
        model.train()
        mask_ = cuda(torch.tensor(mask))
        protein_drug_normalize_ = cuda(torch.tensor(protein_drug_normalize))
        protein_drug_ = cuda(torch.tensor(protein_drug))
        drug_protein_normalize_ = cuda(torch.tensor(drug_protein_normalize))
        drug_protein_ = cuda(torch.tensor(drug_protein))
        global sideeffect_drug_normalize
        sideeffect_drug_normalize_ = cuda(torch.tensor(sideeffect_drug_normalize))
        global sideeffect_drug
        sideeffect_drug_ = cuda(torch.tensor(sideeffect_drug))
        global disease_protein_normalize
        disease_protein_normalize_ = cuda(torch.tensor(disease_protein_normalize))
        global disease_protein
        disease_protein_ = cuda(torch.tensor(disease_protein))
        global disease_drug_normalize
        disease_drug_normalize_ = cuda(torch.tensor(disease_drug_normalize))
        global disease_drug
        disease_drug_ = cuda(torch.tensor(disease_drug))
        global protein_disease_normalize
        protein_disease_normalize_ = cuda(torch.tensor(protein_disease_normalize))
        global protein_disease
        protein_disease_ = cuda(torch.tensor(protein_disease))
        global protein_sequence_normalize
        protein_sequence_normalize_ = cuda(torch.tensor(protein_sequence_normalize))
        global protein_sequence
        protein_sequence_ = cuda(torch.tensor(protein_sequence))
        global protein_protein_normalize
        protein_protein_normalize_ = cuda(torch.tensor(protein_protein_normalize))
        global protein_protein
        protein_protein_ = cuda(torch.tensor(protein_protein))
        global drug_sideeffect_normalize
        drug_sideeffect_normalize_ = cuda(torch.tensor(drug_sideeffect_normalize))
        global drug_sideeffect
        drug_sideeffect_ = cuda(torch.tensor(drug_sideeffect))
        global drug_disease_normalize
        drug_disease_normalize_ = cuda(torch.tensor(drug_disease_normalize))
        global drug_disease
        drug_disease_ = cuda(torch.tensor(drug_disease))
        global drug_chemical_normalize
        drug_chemical_normalize_ = cuda(torch.tensor(drug_chemical_normalize))
        global drug_chemical
        drug_chemical_ = cuda(torch.tensor(drug_chemical))
        global drug_drug_normalize
        drug_drug_normalize_ = cuda(torch.tensor(drug_drug_normalize))
        global drug_drug
        drug_drug_ = cuda(torch.tensor(drug_drug))
        global drug_encoder_data
        drug_encoder_data_ = cuda(drug_encoder_data)
        global protein_encoder_data
        protein_encoder_data_ = cuda(protein_encoder_data)

        tloss, dtiloss, results = model(
            drug_drug_, drug_drug_normalize_, \
            drug_chemical_, drug_chemical_normalize_, \
            drug_disease_, drug_disease_normalize_, \
            drug_sideeffect_, drug_sideeffect_normalize_, \
            protein_protein_, protein_protein_normalize_, \
            protein_sequence_, protein_sequence_normalize_, \
            protein_disease_, protein_disease_normalize_, \
            drug_encoder_data_, protein_encoder_data_, \
            disease_drug_, disease_drug_normalize_, \
            disease_protein_, disease_protein_normalize_, \
            sideeffect_drug_, sideeffect_drug_normalize_, \
            drug_protein_, drug_protein_normalize_, \
            protein_drug_, protein_drug_normalize_, \
            mask_)
        tloss.backward(retain_graph=True)
        # every 25 steps of gradient descent, evaluate the performance, other choices of this number are possible
        if i % 25 == 0 and verbose == True:
            # if i == 0:
            print('step', i, 'total and dtiloss', tloss, dtiloss)

            pred_list = []
            ground_truth = []
            ii = 1
            for ele in DTIvalid:
                print('Evaluating %s / %s' % (ii, len(DTIvalid)))
                pred_list.append(results[ele[0], ele[1]].cpu().detach().numpy())
                ground_truth.append(ele[2])
                ii += 1
            valid_auc = roc_auc_score(ground_truth, pred_list)
            valid_aupr = average_precision_score(ground_truth, pred_list)
            if valid_aupr >= best_valid_aupr:
                torch.save(model, 'model_round_%s.pkl' % (round))
                best_valid_aupr = valid_aupr
                best_valid_auc = valid_auc
                pred_list = []
                ground_truth = []
                ii = 1
                for ele in DTItest:
                    print('Testing %s / %s' % (ii, len(DTItest)))
                    pred_list.append(results[ele[0], ele[1]].cpu().detach().numpy())
                    ground_truth.append(ele[2])
                    ii += 1
                test_auc = roc_auc_score(ground_truth, pred_list)
                test_aupr = average_precision_score(ground_truth, pred_list)
            print('valid auc aupr,', valid_auc, valid_aupr, 'test auc aupr', test_auc, test_aupr)

    return best_valid_auc, best_valid_aupr, test_auc, test_aupr


parser = OptionParser()
parser.add_option("-d", "--d", default=1024, help="The embedding dimension d")
parser.add_option("-n", "--n", default=1, help="global norm to be clipped")
parser.add_option("-k", "--k", default=512, help="The dimension of project matrices k")
parser.add_option("-t", "--t", default="o", help="Test scenario")
parser.add_option("-r", "--r", default="ten", help="positive negative ratio")

(opts, args) = parser.parse_args()

# network encoder data


network_path = 'complex_model/model/src/benchmark_2.0'

opts_d=1024
opts_k=256
opts_n=2


result_file = network_path + '/newmodel_.txt'


if os.path.exists(result_file):
    os.remove(result_file)



drug_idx_file = network_path + '/drug.idx'
target_idx_file = network_path + '/target.idx'

smile_file = network_path + '/smile.txt'
sequence_file = network_path + '/sequence.txt'

print('drug_idx_file', drug_idx_file)
drugs = DataGenerator.readIdx(drug_idx_file)
print('drugs', drugs)
print('type(drugs)', type(drugs))
#sys.exit()
print('len(drugs)', len(drugs))
targets = DataGenerator.readIdx(target_idx_file)
print('targets', targets)
print('len(targets)', len(targets))
#sys.exit()

print('len(drugs): ', len(drugs))
print('len(targets): ', len(targets))

X_drugs = dataset.read_drugfile(path=smile_file, drugs=drugs)
print('X_drugs', X_drugs)
print('\n')
X_targets = dataset.read_targetfile(path=sequence_file, targets=targets)
print('X_targets', X_targets)
print('X_drugs.shape: ', X_drugs.shape)
print('X_targets.shape: ', X_targets.shape)
#sys.exit()
#print(X_drugs)
#print(X_targets)

drug_encoding, target_encoding = 'CNN', 'CNN'

df_data_drug, df_data_target = data_process_(X_drug=X_drugs, X_target=X_targets, drug_encoding=drug_encoding,
                                             target_encoding=target_encoding)

print('df_data_drug', df_data_drug)
print('type(df_data_drug)', type(df_data_drug))
print('\n')
print('df_data_target', df_data_target)
print('type(df_data_target)', type(df_data_target))

#sys.exit()

config = generate_config(drug_encoding=drug_encoding,
                         target_encoding=target_encoding,
                         drug_batch_size=708,
                         protein_batch_size=1512,
                         hidden_dim_drug=1024,
                         hidden_dim_protein=1024,
                         )

drug_encoder_data, protein_encoder_data = data_encoder_(df_data_drug, df_data_target, config)
print('drug_encoder_data: ', drug_encoder_data.shape)
print('protein_encoder_data: ', protein_encoder_data.shape)
print('type(drug_encoder_data)', type(drug_encoder_data))
print('type(protein_encoder_data', type(protein_encoder_data))

print('New data encoder finish')

# load network

drug_drug_file = network_path + '/mat_drug_drug.txt'
drug_chemical_file = network_path + '/Similarity_Matrix_Drugs.txt'
drug_disease_file = network_path + '/mat_drug_disease.txt'
drug_sideeffect_file = network_path + '/mat_drug_sider.txt'
protein_protein_file = network_path + '/mat_target_target.txt'
protein_sequence_file = network_path + '/Similarity_Matrix_Targets.txt'
protein_disease_file = network_path + '/mat_target_disease.txt'

drug_drug = np.loadtxt(drug_drug_file)
print('drug_drug', drug_drug)
print('drug_drug.shape', drug_drug.shape)
#sys.exit()
drug_chemical = np.loadtxt(drug_chemical_file)
print('drug_chemical', drug_chemical)
print('drug_chemical.shape', drug_chemical.shape)
drug_disease = np.loadtxt(drug_disease_file)
print('drug_disease', drug_disease)
print('drug_disease.shape', drug_disease.shape)
drug_sideeffect = np.loadtxt(drug_sideeffect_file)
print('drug_sideeffect', drug_sideeffect)
print('drug_sideeffect.shape', drug_sideeffect.shape)
protein_protein = np.loadtxt(protein_protein_file)
print('protein_protein', protein_protein)
print('protein_protein.shape', protein_protein.shape)
protein_sequence = np.loadtxt(protein_sequence_file)
print('protein_sequence', protein_sequence)
print('protein_sequence.shape', protein_sequence.shape)
protein_disease = np.loadtxt(protein_disease_file)
print('protein_disease', protein_disease)
print('protein_disease.shape', protein_disease.shape)

#sys.exit()

disease_drug = drug_disease.T
sideeffect_drug = drug_sideeffect.T
disease_protein = protein_disease.T

drug_drug_normalize = row_normalize(drug_drug, True)
drug_chemical_normalize = row_normalize(drug_chemical, True)
drug_disease_normalize = row_normalize(drug_disease, False)
drug_sideeffect_normalize = row_normalize(drug_sideeffect, False)
protein_protein_normalize = row_normalize(protein_protein, True)
protein_sequence_normalize = row_normalize(protein_sequence, True)
protein_disease_normalize = row_normalize(protein_disease, False)
disease_drug_normalize = row_normalize(disease_drug, False)
disease_protein_normalize = row_normalize(disease_protein, False)
sideeffect_drug_normalize = row_normalize(sideeffect_drug, False)

num_drug = len(drug_drug_normalize)
num_protein = len(protein_protein_normalize)
num_disease = len(disease_protein_normalize)
num_sideeffect = len(sideeffect_drug_normalize)
dim_drug = int(opts_d)
dim_protein = int(opts_d)
dim_disease = int(opts_d)
dim_sideeffect = int(opts_d)
dim_pred = int(opts_k)
dim_pass = int(opts_d)

print('dim_drug', dim_drug)
print('dim_protein', dim_protein)
print('dim_disease', dim_disease)
print('dim_sideeffect', dim_sideeffect)
print('dim_pred', dim_pred)
print('dim_pass', dim_pass)
print('num_drug: ', num_drug)
print('num_protein: ', num_protein)
print('num_disease: ', num_disease)
print('num_sideeffect: ', num_sideeffect)
print('drug_drug_normalize: ', drug_drug_normalize.shape)
print('drug_chemical_normalize: ', drug_chemical_normalize.shape)
print('drug_disease_normalize: ', drug_disease_normalize.shape)
print('drug_sideeffect_normalize: ', drug_sideeffect_normalize.shape)
print('protein_protein_normalize: ', protein_protein_normalize.shape)
print('protein_sequence_normalize: ', protein_sequence_normalize.shape)
print('protein_disease_normalize: ', protein_disease_normalize.shape)
print('disease_drug_normalize: ', disease_drug_normalize.shape)
print('disease_protein_normalize: ', disease_protein_normalize.shape)
print('sideeffect_drug_normalize: ', sideeffect_drug_normalize.shape)

#sys.exit()

lr = 0.001
model = cuda(Model())
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#NEW DATA
train_file = 'new_data/train.nt'

test_file = 'new_data/test.nt'



DTItrain = DataGenerator.readTrain(train_file, drug_idx_file, target_idx_file)
print('DTItrain', DTItrain)
print('DTItrain.shape', DTItrain.shape)
DTItest = DataGenerator.readTest(test_file, drug_idx_file, target_idx_file)
print('DTItest', DTItest)
print('DTItest.shape', DTItest.shape)

train_and_evaluate_(optimizer, DTItrain=DTItrain, DTItest=DTItest, num_steps=100)
