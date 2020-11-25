import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import transformer.Constants as Constants
import Utils

from preprocess.Dataset import get_dataloader
from transformer.Models import Transformer
from tqdm import tqdm
import os
from data_foursquare import foursquare
from data_earthquakes import Earthquakes
from data_nyc import  NycTaxi

def prepare_dataloader(opt, dataobj):
    """ Load data and prepare dataloader. """

    def load_data(name):
        with open(name, 'rb') as f:
            data = pickle.load(f)
            num_types = 1 # There is no event type prediction, hence using a dummy value, this will basically be a constant value field
            return data, num_types

    print('[Info] Loading train data...')
    train_data, num_types = load_data(opt.data + 'train_ny.pkl')
    print('[Info] Loading dev data...')
    val_data, _ = load_data(opt.data + 'val_ny.pkl')
    print('[Info] Loading test data...')
    test_data, _ = load_data(opt.data + 'test_ny.pkl')

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    validationloader = get_dataloader(val_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, validationloader, testloader, num_types


def train_epoch(model, training_data, optimizer, pred_loss_func, opt):
    """ Epoch operation in training phase. """

    model.train()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_time_error = 0  # cumulative time prediction squared-error
    total_time_latitude = 0  # cumulative latitude prediction squared-error
    total_time_longitude = 0  # cumulative longitude prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type, latitude, longitude = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        enc_out, prediction = model(event_type, event_time, latitude, longitude) # change the event_time to time gap

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, event_time, event_type, latitude, longitude) # change the event_time to time gap
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = Utils.type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = Utils.time_loss(prediction[1], event_time) # change the event_time to time gap

        # latitude prediction
        le = Utils.time_loss(prediction[2], latitude)

        # longitude prediction
        ge = Utils.time_loss(prediction[3], longitude)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = event_loss + pred_loss + se / scale_time_loss + le / scale_time_loss + ge / scale_time_loss
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se +=  se.item() + le.item() + ge.item()
        total_time_error += se.item()
        total_time_latitude += le.item()
        total_time_longitude += ge.item()

        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(Constants.PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    total_rmse = np.sqrt(total_time_se / total_num_pred)
    time_rmse = np.sqrt(total_time_error / total_num_pred)
    latitude_rmse = np.sqrt(total_time_latitude / total_num_pred)
    longitude_rmse = np.sqrt(total_time_longitude / total_num_pred)
    print('Time: {:5f} Latitude: {:5f} Longitude: {:5f} Overall: {:5f} '.format(time_rmse, latitude_rmse, longitude_rmse, total_rmse))

def eval_epoch(model, validation_data, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_time_error = 0  # cumulative time prediction squared-error
    total_time_latitude = 0  # cumulative latitude prediction squared-error
    total_time_longitude = 0  # cumulative longitude prediction squared-error

    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, time_gap, event_type, latitude, longitude = map(lambda x: x.to(opt.device), batch)

            """ forward """
            enc_out, prediction = model(event_type, event_time, latitude, longitude) # change the event_time to time gap

            """ compute loss """
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, time_gap, event_type, latitude, longitude) # change the event_time to time gap
            event_loss = -torch.sum(event_ll - non_event_ll)

            _, pred_num = Utils.type_loss(prediction[0], event_type, pred_loss_func)

            se = Utils.time_loss(prediction[1], event_time) # change the event_time to time gap
            le = Utils.time_loss(prediction[2], latitude)
            ge = Utils.time_loss(prediction[3], longitude)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item() + le.item() + ge.item()
            total_time_error += se.item()
            total_time_latitude += le.item()
            total_time_longitude += ge.item()

            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(Constants.PAD).sum().item()
            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0] # change the event_time to time gap

    test_rmse = np.sqrt(total_time_se / total_num_pred)
    test_time_rmse = np.sqrt(total_time_error / total_num_pred)
    test_latitude_rmse = np.sqrt(total_time_latitude / total_num_pred)
    test_longitude_rmse = np.sqrt(total_time_longitude / total_num_pred)

    return total_event_ll / total_num_event, total_event_rate / total_num_pred, test_time_rmse, test_latitude_rmse, test_longitude_rmse, test_rmse


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_epoch(model, training_data, optimizer, pred_loss_func, opt)

        start = time.time()
        val_event, val_type, val_time_rmse, val_latitude_rmse, val_longitude_rmse, val_total_rmse = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('Time: {:5f} Latitude: {:5f} Longitude: {:5f} Overall: {:5f} '.format(val_time_rmse, val_latitude_rmse, val_longitude_rmse, val_total_rmse))

        valid_event_losses += [val_event]
        valid_pred_losses += [val_type]
        valid_rmse += [val_total_rmse]

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=val_event, acc=val_type, rmse=val_total_rmse))

        scheduler.step()

def evaluate(model, test_data, pred_loss_func, dataobj, opt):
    model.eval()
    outputs = []
    targets = []
    total_num_pred = 0

    with torch.no_grad():
        for batch in tqdm(test_data, mininterval=2,
                          desc='  - (Testing) ', leave=False):
            event_time, time_gap, event_type, latitude, longitude = map(lambda x: x.to(opt.device), batch)
            enc_out, prediction = model(event_type, event_time, latitude, longitude)
            event_ll, non_event_ll = Utils.log_likelihood(model, enc_out, time_gap, event_type, latitude, longitude)
            event_loss = -torch.sum(event_ll - non_event_ll)

            #p0 = prediction[0].detach().flatten().unsqueeze(-1)
            p1 = prediction[1].detach().flatten().unsqueeze(-1)
            p2 = prediction[2].detach().flatten().unsqueeze(-1)
            p3 = prediction[3].detach().flatten().unsqueeze(-1)

            output_temp = torch.cat((p1, p2, p3), axis=1)
            outputs_inv = dataobj.scaler.inverse_transform(output_temp)
            outputs.append(outputs_inv)

            #o0 = latitude.flatten().unsqueeze(-1)
            o1 = latitude.flatten().unsqueeze(-1)
            o2 = longitude.flatten().unsqueeze(-1)
            o3 = event_time.flatten().unsqueeze(-1)

            target_temp = torch.cat((o1, o2, o3), axis=1)
            targets_inv = dataobj.scaler.inverse_transform(target_temp)
            targets.append(targets_inv)

            total_num_pred += event_type.ne(Constants.PAD).sum().item() - event_time.shape[0]

    t1 = np.sum((np.asarray(outputs).reshape(-1, 3)[:, 0] - np.asarray(targets).reshape(-1, 3)[:, 0]))
    t2 = np.sum((np.asarray(outputs).reshape(-1, 3)[:, 1] - np.asarray(targets).reshape(-1, 3)[:, 1]))
    t3 = np.sum((np.asarray(outputs).reshape(-1, 3)[:, 2] - np.asarray(targets).reshape(-1, 3)[:, 2]))

    test_latitude_rmse = np.sqrt(t1 * t1) / total_num_pred
    test_longitude_rmse = np.sqrt(t2 * t2) / total_num_pred
    test_time_rmse = np.sqrt(t3 * t3) / total_num_pred

    print(test_latitude_rmse, test_longitude_rmse, test_time_rmse)

def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', default='C:\\Users\\Manjit\\Downloads\\THP\\Transformer-Hawkes-Process-master4-wc\\data\\')
    parser.add_argument('-dataset', type=str, default="earthquake")

    parser.add_argument('-epoch', type=int, default=50)
    parser.add_argument('-batch_size', type=int, default=16)

    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4) # changing from 4 to 8
    parser.add_argument('-n_layers', type=int, default=4) # changing from 4 to 6

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='log.txt')

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cpu') # changed to cpu

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    dataobj = foursquare()
    #dataobj = Earthquakes()
    #dataobj = NycTaxi()

    """ prepare dataloader """
    trainloader, validationloader, testloader, num_types = prepare_dataloader(opt, dataobj)

    """ prepare model """
    model = Transformer(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, validationloader, optimizer, scheduler, pred_loss_func, opt)
    evaluate(model, testloader, pred_loss_func, dataobj, opt)

if __name__ == '__main__':
    main()
