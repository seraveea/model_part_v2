import datetime
import sys
sys.path.insert(0, sys.path[0]+"/../")
from utils.dataloader import create_test_loaders
from utils.utils import DotDict
import numpy as np
import pandas as pd
import torch
import argparse
from tqdm import tqdm
from models.model import MLP, HIST, GRU, LSTM, GAT, ALSTM, SFM, RSR, relation_GATs, relation_GATs_3heads
from qlib.contrib.model.pytorch_transformer import Transformer
from models.DLinear import DLinear_model
from models.Autoformer import Model as autoformer
from models.Crossformer import Model as crossformer
from models.ETSformer import Model as ETSformer
from models.FEDformer import Model as FEDformer
from models.FiLM import Model as FiLM
from models.Informer import Model as Informer
from models.PatchTST import Model as PatchTST
import json


time_series_library = [
    'DLinear',
    'Autoformer',
    'Crossformer',
    'ETSformer',
    'FEDformer',
    'FiLM',
    'Informer',
    'PatchTST'
]

relation_model_dict = [
    'RSR',
    'relation_GATs',
    'relation_GATs_3heads'
]


def get_model(model_name):
    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTM

    if model_name.upper() == 'GRU':
        return GRU

    if model_name.upper() == 'TRANSFORMER':
        return Transformer

    if model_name.upper() == 'GATS':
        return GAT

    if model_name.upper() == 'SFM':
        return SFM

    if model_name.upper() == 'ALSTM':
        return ALSTM

    if model_name.upper() == 'HIST':
        return HIST

    if model_name.upper() == 'RSR':
        return RSR

    if model_name.upper() == 'PATCHTST':
        return PatchTST

    raise ValueError('unknown model name `%s`' % model_name)


def inference(model, data_loader, stock2concept_matrix=None, stock2stock_matrix=None, model_name=''):
    model.eval()
    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
        # 当日切片
        feature, label, market_value, stock_index, index, mask = data_loader.get(slc)
        # feature, label, index = data_loader.get(slc)
        with torch.no_grad():
            if model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            elif model_name in relation_model_dict:
                pred = model(feature, stock2stock_matrix[stock_index][:, stock_index])
            elif model_name in time_series_library:
                pred = model(feature, mask)
            else:
                pred = model(feature)
            preds.append(
                pd.DataFrame({'pred_score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def _prediction(param_dict, test_loader, device):
    """
    test single model first, load model from folder and do prediction
    """
    # test_loader = create_test_loaders(args, for_individual=for_individual)
    stock2concept_matrix = param_dict['stock2concept_matrix']
    stock2stock_matrix = param_dict['stock2stock_matrix']
    print('load model ', param_dict['model_name'])
    if param_dict['model_name'] == 'SFM':
        model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], output_dim=32, freq_dim=25,
                                           hidden_size=param_dict['hidden_size'],
                                           dropout_W=0.5, dropout_U=0.5, device=device)
    elif param_dict['model_name'] == 'ALSTM':
        model = get_model(param_dict['model_name'])(param_dict['d_feat'], param_dict['hidden_size'],
                                                    param_dict['num_layers'], param_dict['dropout'], 'LSTM')
    elif param_dict['model_name'] == 'Transformer':
        model = get_model(param_dict['model_name'])(param_dict['d_feat'], param_dict['hidden_size'],
                                                    param_dict['num_layers'], dropout=0.5)
    elif param_dict['model_name'] == 'HIST':
        # HIST need stock2concept matrix, send it to device
        stock2concept_matrix = torch.Tensor(np.load(stock2concept_matrix)).to(device)
        model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers']
                                                    , K=param_dict['K'])
    elif param_dict['model_name'] in relation_model_dict:
        stock2stock_matrix = torch.Tensor(np.load(stock2stock_matrix)).to(device)
        num_relation = stock2stock_matrix.shape[2]  # the number of relations
        model = get_model(param_dict['model_name'])(num_relation=num_relation, d_feat=param_dict['d_feat'],
                                                    num_layers=param_dict['num_layers'])
    elif param_dict['model_name'] in time_series_library:
        model = get_model(param_dict['model_name'])(DotDict(param_dict))
    else:
        model = get_model(param_dict['model_name'])(d_feat=param_dict['d_feat'], num_layers=param_dict['num_layers'])
    model.to(device)
    model.load_state_dict(torch.load(param_dict['model_dir'] + '/model.bin', map_location=device))
    print('predict in ', param_dict['model_name'])
    pred = inference(model, test_loader, stock2concept_matrix, stock2stock_matrix, param_dict['model_name'])
    return pred


def prediction(args, model_path, device):
    param_dict = json.load(open(model_path+'/info.json'))['config']
    param_dict['model_dir'] = model_path
    test_loader = create_test_loaders(args, param_dict, device=device)
    pred = _prediction(param_dict, test_loader, device)
    return pred


def main(args, device):
    model_path = args.model_path
    pd.to_pickle(prediction(args, model_path, device), args.pkl_path)


def parse_args():
    """
    deliver arguments from json file to program
    :param param_dict: a dict that contains model info
    :return: no return
    """
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('--test_start_date', default='2022-06-01')
    parser.add_argument('--test_end_date', default='2023-06-01')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--model_path', default='./output/for_platform/LSTM')
    parser.add_argument('--pkl_path', default='./pred_output/csi_300_lstm.pkl',
                        help='location to save the pred dictionary file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    main(args, device)
