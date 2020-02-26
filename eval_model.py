"""Training script to train a language autoencoder model.
"""

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import argparse
import h5py
import json
import logging
import random
import time
import torch
import os

from models import SleepPredictionMLP
from models import SleepPredictionCNN
from models import SleepPredictionSeq
from dataloader import get_data_loader

class Dict2Obj(dict):
    """Converts dicts to objects.

    """
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def merge(self, other, overwrite=True):
        for name in other:
            if overwrite or name not in self:
                self[name] = other[name]


def evaluate(model, data_loader, criterion, epoch, args):
    """Calculates vqg average loss on data_loader.

    Args:
        model: model.
        data_loader: Iterator for the data.
        criterion: The criterion function used to evaluate the loss.
        args: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    gts, preds = [], []
    model.eval()
    total_loss = 0.0
    iterations = 0
    acc = 0.0
    count = 0
    total_steps = len(data_loader)

    start_time = time.time()
    for i, (conditions, eegs, labels) in enumerate(data_loader):

        if torch.cuda.is_available():
            eegs = eegs.cuda()
            conditions = conditions.cuda()
            labels = labels.cuda()            
        
        # Forward.
        out = model(eegs, conditions)
        
        # Calculate the loss.
        loss = criterion(out, labels)

        # Backprop and optimize.
        total_loss += loss.item()
        iterations += 1
        _, preds = torch.max(out.data, 1)
        acc += (preds == labels).sum().item()
        count += eegs.shape[0]

    return total_loss / iterations, acc/float(count)

def run_eval(model, data_loader, criterion, args, epoch, scheduler):
    start_time = time.time()
    val_loss, val_acc = evaluate(model, data_loader, criterion, epoch, args)
    delta_time = time.time() - start_time
    scheduler.step(val_loss)
    logging.info('Time: %.4f, Epoch [%d/%d], Val loss: %.4f, Val accuracy: %.4f' % (
        delta_time, epoch, args.num_epochs, val_loss, val_acc))
    logging.info('=' * 80)
    scheduler.step(val_loss)
    
def get_dataset_size(dataset):
    annos = h5py.File(dataset, 'r')
    size = annos['mlp'].shape[0]
    annos.close()
    return size

def main(args):
    # Load the arguments.
    model_dir = os.path.dirname(args.model_path)
    params = Dict2Obj(json.load(
            open(os.path.join(model_dir, "args.json"), "r")))

    # Config logging
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(model_dir, 'eval.log')
    if os.path.exists(logfile):
        os.remove(logfile)
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Build data loader.
    logging.info("Building data loader...")
    data_loader = get_data_loader(args.dataset, params.labels_data, params.batch_size, shuffle=False,
                                  num_workers=params.num_workers, model_type=params.mode)
    logging.info("Done")

    # Build the models
    logging.info("Building Sleep Stage Predictor...")
    if params.mode == 'mlp':
        model = SleepPredictionMLP(params.input_dim, 256, num_classes=params.output_size,
                                 num_layers=1, dropout_p=0.0, w_norm=False)
    elif params.mode == 'cnn':
        model = SleepPredictionCNN(num_classes=params.output_size)
    else:
        model = SleepPredictionSeq(num_classes=params.output_size)
        
    logging.info("Done")

    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(args.model_path))

    preds = []
    model.eval()
    total_steps = len(data_loader)

    start_time = time.time()
    for i, (conditions, eegs, labels) in enumerate(data_loader):

        if torch.cuda.is_available():
            eegs = eegs.cuda()
            conditions = conditions.cuda()
            labels = labels.cuda()            
        
        # Forward.
        out = model(eegs, conditions)
        
        _, pred = torch.max(out.data, 1)
        preds.extend(pred.tolist())
        # Evaluation and learning rate updates.
        logging.info('Step [%d/%d], batch accuracy: %.4f' % (
                    i, total_steps, (pred == labels).sum().item() / params.batch_size))
    with open("y_benchmark.csv", "w") as f:
        f.write("".join(["id,label\n"] + ["{},{}\n".format(i, y) for i, y in enumerate(preds)]))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Location parameters.
    parser.add_argument('--model-path', type=str,
                        default='weights/mlp/model.pkl',
                        help='Path for saving trained models.')
    parser.add_argument('--dataset', type=str,
                        default='data/processed_test_dataset.hdf5',
                        help='Train features.')
    parser.add_argument('--labels-data', type=str,
                        default='data/y_train_2.csv',
                        help='Train labels.')

    # Session parameters.
    parser.add_argument('--log-step', type=int , default=10,
                        help='Step size for printing log info.')
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    main(args)