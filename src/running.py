import logging
import sys
import os
import traceback
import json
from datetime import datetime
import string
import random
from collections import OrderedDict
import time
import pickle
from functools import partial

import ipdb
import torch
from torch.utils.data import DataLoader
import numpy as np
import sklearn

from utils import utils
from utils.analysis import Analyzer
from models.loss import l2_reg_loss
from datasets.dataset import ImputationDataset, DynamicImputationDataset, DynamicClassDataset, ClassiregressionDataset, collate_unsuperv, collate_superv
from utils.sampler import create_balanced_sampler
from utils.utils import calculate_epoch_balance


logger = logging.getLogger('__main__')

NEG_METRICS = {'loss'}  # metrics for which "better" is less

val_times = {"total_time": 0, "count": 0}


def pipeline_factory(config):
    """Returns the corresponding combination of Dataset class, collate function and Runner class."""
    
    task = config['task']
    
    # Set up dataset class and collate function based on task
    if task == "imputation":
        dataset_class = partial(ImputationDataset, 
                              mean_mask_length=config.get('mean_mask_length', 10),
                              masking_ratio=config.get('masking_ratio', 0.15),
                              mode=config.get('mask_mode', 'random'),
                              distribution=config.get('mask_distribution', 'geometric'),
                              exclude_feats=config.get('exclude_feats', None))
        collate_fn = collate_unsuperv
        runner_class = UnsupervisedRunner
    elif task == "dynamic_imputation": # added this for compatibility with DynamicSPESData
        dataset_class = partial(DynamicImputationDataset,
                                mean_mask_length=config.get('mean_mask_length', 10),
                                masking_ratio=config.get('masking_ratio', 0.15),
                                mode=config.get('mask_mode', 'random'),
                                distribution=config.get('mask_distribution', 'geometric'),
                                exclude_feats=config.get('exclude_feats', None))
        collate_fn = collate_unsuperv
        runner_class = UnsupervisedRunner
    elif task == "dynamic_classification": # added this for compatibility with DynamicSPESData
        dataset_class = DynamicClassDataset
        collate_fn = collate_superv
        runner_class = SupervisedRunner
    elif (task == "classification") or (task == "regression"):
        dataset_class = ClassiregressionDataset
        collate_fn = collate_superv
        runner_class = SupervisedRunner
    else:
        raise NotImplementedError(f"Task '{task}' not implemented")

    # Return the raw components - let main.py handle DataLoader creation
    return dataset_class, collate_fn, runner_class


def setup(args):
    """Prepare training session: read configuration from file (takes precedence), create directories.
    Input:
        args: arguments object from argparse
    Returns:
        config: configuration dictionary
    """

    config = args.__dict__  # configuration dictionary

    if args.config_filepath is not None:
        logger.info("Reading configuration ...")
        try:  # dictionary containing the entire configuration settings in a hierarchical fashion
            config.update(utils.load_config(args.config_filepath))
        except:
            logger.critical("Failed to load configuration file. Check JSON syntax and verify that files exist")
            traceback.print_exc()
            sys.exit(1)

    # Create output directory
    initial_timestamp = datetime.now()
    output_dir = config['output_dir']
    if not os.path.isdir(output_dir):
        raise IOError(
            "Root directory '{}', where the directory of the experiment will be created, must exist".format(output_dir))

    output_dir = os.path.join(output_dir, config['experiment_name'])

    formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    config['initial_timestamp'] = formatted_timestamp
    if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
    config['output_dir'] = output_dir
    config['save_dir'] = os.path.join(output_dir, 'checkpoints')
    config['pred_dir'] = os.path.join(output_dir, 'predictions')
    config['tensorboard_dir'] = os.path.join(output_dir, 'tb_summaries')
    utils.create_dirs([config['save_dir'], config['pred_dir'], config['tensorboard_dir']])

    # Save configuration as a (pretty) json file
    with open(os.path.join(output_dir, 'configuration.json'), 'w') as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(output_dir))

    return config


def fold_evaluate(dataset, model, device, loss_module, target_feats, config, dataset_name):

    allfolds = {'target_feats': target_feats,  # list of len(num_folds), each element: list of target feature integer indices
                'predictions': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) prediction per sample
                'targets': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) target/original input per sample
                'target_masks': [],  # list of len(num_folds), each element: (num_samples, seq_len, feat_dim) boolean mask per sample
                'metrics': [],  # list of len(num_folds), each element: (num_samples, num_metrics) metric per sample
                'IDs': []}  # list of len(num_folds), each element: (num_samples,) ID per sample

    for i, tgt_feats in enumerate(target_feats):

        dataset.mask_feats = tgt_feats  # set the transduction target features

        loader = DataLoader(dataset=dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers'],
                            pin_memory=True,
                            collate_fn=lambda x: collate_unsuperv(x, max_len=config['max_seq_len']))

        evaluator = UnsupervisedRunner(model, loader, device, loss_module,
                                       print_interval=config['print_interval'], console=config['console'])

        logger.info("Evaluating {} set, fold: {}, target features: {}".format(dataset_name, i, tgt_feats))
        aggr_metrics, per_batch = evaluate(evaluator)

        metrics_array = convert_metrics_per_batch_to_per_sample(per_batch['metrics'], per_batch['target_masks'])
        metrics_array = np.concatenate(metrics_array, axis=0)
        allfolds['metrics'].append(metrics_array)
        allfolds['predictions'].append(np.concatenate(per_batch['predictions'], axis=0))
        allfolds['targets'].append(np.concatenate(per_batch['targets'], axis=0))
        allfolds['target_masks'].append(np.concatenate(per_batch['target_masks'], axis=0))
        allfolds['IDs'].append(np.concatenate(per_batch['IDs'], axis=0))

        metrics_mean = np.mean(metrics_array, axis=0)
        metrics_std = np.std(metrics_array, axis=0)
        for m, metric_name in enumerate(list(aggr_metrics.items())[1:]):
            logger.info("{}:: Mean: {:.3f}, std: {:.3f}".format(metric_name, metrics_mean[m], metrics_std[m]))

    pred_filepath = os.path.join(config['pred_dir'], dataset_name + '_fold_transduction_predictions.pickle')
    logger.info("Serializing predictions into {} ... ".format(pred_filepath))
    with open(pred_filepath, 'wb') as f:
        pickle.dump(allfolds, f, pickle.HIGHEST_PROTOCOL)


def convert_metrics_per_batch_to_per_sample(metrics, target_masks):
    """
    Args:
        metrics: list of len(num_batches), each element: list of len(num_metrics), each element: (num_active_in_batch,) metric per element
        target_masks: list of len(num_batches), each element: (batch_size, seq_len, feat_dim) boolean mask: 1s active, 0s ignore
    Returns:
        metrics_array = list of len(num_batches), each element: (batch_size, num_metrics) metric per sample
    """
    metrics_array = []
    for b, batch_target_masks in enumerate(target_masks):
        num_active_per_sample = np.sum(batch_target_masks, axis=(1, 2))
        batch_metrics = np.stack(metrics[b], axis=1)  # (num_active_in_batch, num_metrics)
        ind = 0
        metrics_per_sample = np.zeros((len(num_active_per_sample), batch_metrics.shape[1]))  # (batch_size, num_metrics)
        for n, num_active in enumerate(num_active_per_sample):
            new_ind = ind + num_active
            metrics_per_sample[n, :] = np.sum(batch_metrics[ind:new_ind, :], axis=0)
            ind = new_ind
        metrics_array.append(metrics_per_sample)
    return metrics_array


def evaluate(evaluator):
    """Perform a single, one-off evaluation on an evaluator object (initialized with a dataset)"""

    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = evaluator.evaluate(epoch_num=None, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    print()
    print_str = 'Evaluation Summary: '
    for k, v in aggr_metrics.items():
        if v is not None:
            print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)
    logger.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    return aggr_metrics, per_batch


def validate(val_evaluator, tensorboard_writer, config, best_metrics, best_value, epoch):
    """Run an evaluation on the validation set while logging metrics, and handle outcome"""

    logger.info("Evaluating on validation set ...")
    eval_start_time = time.time()
    with torch.no_grad():
        aggr_metrics, per_batch = val_evaluator.evaluate(epoch, keep_all=True)
    eval_runtime = time.time() - eval_start_time
    logger.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

    global val_times
    val_times["total_time"] += eval_runtime
    val_times["count"] += 1
    avg_val_time = val_times["total_time"] / val_times["count"]
    avg_val_batch_time = avg_val_time / len(val_evaluator.dataloader)
    avg_val_sample_time = avg_val_time / len(val_evaluator.dataloader.dataset)
    logger.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
    logger.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
    logger.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

    print()
    print_str = 'Epoch {} Validation Summary: '.format(epoch)
    for k, v in aggr_metrics.items():
        tensorboard_writer.add_scalar('{}/val'.format(k), v, epoch)
        print_str += '{}: {:8f} | '.format(k, v)
    logger.info(print_str)

    if config['key_metric'] in NEG_METRICS:
        condition = (aggr_metrics[config['key_metric']] < best_value)
    else:
        condition = (aggr_metrics[config['key_metric']] > best_value)
    if condition:
        best_value = aggr_metrics[config['key_metric']]
        utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, val_evaluator.model)
        best_metrics = aggr_metrics.copy()

        pred_filepath = os.path.join(config['pred_dir'], 'best_predictions')
        np.savez(pred_filepath, **per_batch)

    return aggr_metrics, best_metrics, best_value



def check_progress(epoch):

    if epoch in [100, 140, 160, 220, 280, 340]:
        return True
    else:
        return False


class BaseRunner(object):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10, console=True):

        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.optimizer = optimizer
        self.loss_module = loss_module
        self.l2_reg = l2_reg
        self.print_interval = print_interval
        self.printer = utils.Printer(console=console)
        self.last_batch_logged = -1  # Track last logged batch to avoid duplicates

        self.epoch_metrics = OrderedDict()

    def train_epoch(self, epoch_num=None):
        raise NotImplementedError('Please override in child class')

    def evaluate(self, epoch_num=None, keep_all=True):
        raise NotImplementedError('Please override in child class')

    def print_callback(self, i_batch, metrics, prefix=''):
        """Print training progress at specified intervals"""
        total_batches = len(self.dataloader)
        
        # Only print first and last batch, and every 50th batch
        should_print = (
            i_batch == 0 or  # First batch
            i_batch == total_batches - 1 or  # Last batch
            i_batch % 50 == 0  # Every 50th batch
        )
        
        if should_print:
            template = "{:5.1f}% | batch: {:9d} of {:9d}"
            content = [100 * (i_batch / total_batches), i_batch, total_batches]
            for met_name, met_value in metrics.items():
                template += "\t|\t{}".format(met_name) + ": {:g}"
                content.append(met_value)

            dyn_string = template.format(*content)
            dyn_string = prefix + dyn_string
            self.printer.print(dyn_string)


class UnsupervisedRunner(BaseRunner):

    def __init__(self, model, dataloader, device, loss_module, optimizer, l2_reg=None, print_interval=10, console=True):
        super().__init__(model, dataloader, device, loss_module, optimizer, l2_reg, print_interval, console)
        self.best_model = None  # Initialize best_model attribute
        
    def _save_best_model(self, epoch, val_loss, is_best):
        """Saves model when validation loss decreases.
        If is_best is True, also saves model_best.pth
        """
        if is_best:
            self.best_model = self.model  # Update best_model when we get a new best
            
    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total unmasked elements in epoch
        epoch_labels = []  # store labels for balance calculation
        
        for i, batch in enumerate(self.dataloader):
            X, targets, target_masks, padding_masks, IDs = batch
            # Store labels for balance calculation
            if hasattr(self.dataloader.dataset.data, 'labels_df'):
                batch_labels = self.dataloader.dataset.data.labels_df.loc[IDs]['soz'].values
                epoch_labels.append(torch.tensor(batch_labels))
            
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore

            predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

            # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
            target_masks = target_masks * padding_masks.unsqueeze(-1)
            loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

            if self.l2_reg is not None:
                l2_term = l2_reg_loss(self.model)
                total_loss = mean_loss + self.l2_reg * l2_term
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else f'Epoch {epoch_num} '
                self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_active_elements += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        # Calculate and log epoch balance
        if epoch_labels:
            epoch_balance = calculate_epoch_balance(epoch_labels)
            logger.info(f"Epoch {epoch_num} SOZ sampling ratio: {epoch_balance:.2%}")
            self.epoch_metrics['soz_ratio'] = epoch_balance
        
        if self.best_model and epoch_num == self.epochs:
            from utils.visualization_utils import create_imputation_visualization
            create_imputation_visualization(
                self.best_model, 
                self.dataloader.dataset.data, 
                self.device, 
                self.config, 
                os.path.join(self.output_dir, 'imputation_visualization.png')
            )
        
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):
        self.model = self.model.eval()
        epoch_loss = 0
        epoch_recon_acc = 0
        total_active_elements = 0

        if keep_all:
            per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        
        for i, batch in enumerate(self.dataloader):
            X, targets, target_masks, padding_masks, IDs = batch
            targets = targets.to(self.device)
            target_masks = target_masks.to(self.device)
            padding_masks = padding_masks.to(self.device)

            with torch.no_grad():
                predictions = self.model(X.to(self.device), padding_masks)
                target_masks = target_masks * padding_masks.unsqueeze(-1)
                
                loss = self.loss_module(predictions, targets, target_masks)
                batch_loss = torch.sum(loss).cpu().item()
                mean_loss = batch_loss / len(loss)

                # Calculate reconstruction accuracy
                masked_pred = torch.masked_select(predictions, target_masks)
                masked_true = torch.masked_select(targets, target_masks)
                batch_recon_acc = torch.mean((torch.abs(masked_pred - masked_true) < 0.25).float())
                # relative_error = torch.abs(masked_pred - masked_true) / (torch.abs(masked_true) + 1e-8)  # add small epsilon to avoid division by zero
                # batch_recon_acc = torch.mean((relative_error < 0.2).float())  # 20% relative error threshold

                if keep_all:
                    per_batch['target_masks'].append(target_masks.cpu().numpy())
                    per_batch['targets'].append(targets.cpu().numpy())
                    per_batch['predictions'].append(predictions.cpu().numpy())
                    # Store both metrics with same shape as loss
                    per_batch['metrics'].append([
                        loss.cpu().numpy(),
                        torch.full_like(loss, batch_recon_acc.item()).cpu().numpy()
                    ])
                    per_batch['IDs'].append(IDs)

                metrics = {
                    "loss": mean_loss,
                    "recon_acc": batch_recon_acc.item()
                }
                
                if i % self.print_interval == 0:
                    ending = "" if epoch_num is None else f'Epoch {epoch_num} '
                    self.print_callback(i, metrics, prefix='Evaluating ' + ending)

                total_active_elements += len(loss)
                epoch_loss += batch_loss
                epoch_recon_acc += batch_recon_acc.item() * len(loss)

        # Properly weight metrics by number of active elements
        epoch_loss = epoch_loss / total_active_elements
        epoch_recon_acc = epoch_recon_acc / total_active_elements
        
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss
        self.epoch_metrics['recon_acc'] = epoch_recon_acc

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics


class SupervisedRunner(BaseRunner):

    def __init__(self, model, dataloader, device, loss_module, optimizer=None, l2_reg=None, print_interval=10, console=True):
        super().__init__(model, dataloader, device, loss_module, print_interval, console)
        self.optimizer = optimizer
        self.l2_reg = l2_reg
        self.epoch_metrics = {}
        self.classification = hasattr(model, 'num_classes')  # Check if model is for classification
        self.analyzer = Analyzer(print_conf_mat=True) if self.classification else Analyzer()

    def train_epoch(self, epoch_num=None):

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_samples = 0  # total samples in epoch
        epoch_labels = []
        
        for i, batch in enumerate(self.dataloader):
            X, targets, padding_masks, IDs = batch
            epoch_labels.append(targets)
            
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)  # 0s: ignore
            # regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
            predictions = self.model(X.to(self.device), padding_masks)

            loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
            batch_loss = torch.sum(loss)
            mean_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization

            if self.l2_reg is not None:
                l2_term = l2_reg_loss(self.model)
                total_loss = mean_loss + self.l2_reg * l2_term
            else:
                total_loss = mean_loss

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            total_loss.backward()

            # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
            self.optimizer.step()

            metrics = {"loss": mean_loss.item()}
            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else f'Epoch {epoch_num} '
                self.print_callback(i, metrics, prefix='Training ' + ending)

            with torch.no_grad():
                total_samples += len(loss)
                epoch_loss += batch_loss.item()  # add total loss of batch

        epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        # Calculate and log epoch balance
        epoch_balance = calculate_epoch_balance(epoch_labels)
        logger.info(f"Epoch {epoch_num} SOZ sampling ratio: {epoch_balance:.2%}")
        self.epoch_metrics['soz_ratio'] = epoch_balance
        
        return self.epoch_metrics

    def evaluate(self, epoch_num=None, keep_all=True):
        self.model = self.model.eval()
        epoch_loss = 0
        total_samples = 0

        per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
        for i, batch in enumerate(self.dataloader):
            X, targets, padding_masks, IDs = batch
            targets = targets.to(self.device)
            padding_masks = padding_masks.to(self.device)
            predictions = self.model(X.to(self.device), padding_masks)
            
            loss = self.loss_module(predictions, targets)
            batch_loss = torch.sum(loss).cpu().item()
            mean_loss = batch_loss / len(loss)

            per_batch['targets'].append(targets.cpu().numpy())
            per_batch['predictions'].append(predictions.detach().cpu().numpy())
            per_batch['metrics'].append([loss.detach().cpu().numpy()])
            per_batch['IDs'].append(IDs)

            if i % self.print_interval == 0:
                ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
                self.print_callback(i, {"loss": mean_loss}, prefix='Evaluating ' + ending)

            total_samples += len(loss)
            epoch_loss += batch_loss

        epoch_loss = epoch_loss / total_samples
        self.epoch_metrics['epoch'] = epoch_num
        self.epoch_metrics['loss'] = epoch_loss

        if self.classification:
            predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
            probs = torch.nn.functional.softmax(predictions, dim=1)
            predictions = torch.argmax(probs, dim=1).cpu().numpy()
            probs = probs.cpu().numpy()
            targets = np.concatenate(per_batch['targets'], axis=0).flatten()
            class_names = np.arange(probs.shape[1])
            metrics_dict = self.analyzer.analyze_classification(predictions, targets, class_names)

            self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']
            self.epoch_metrics['precision'] = metrics_dict['prec_avg']

            if self.model.num_classes == 2:
                false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])
                self.epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

                prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
                self.epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)

        if keep_all:
            return self.epoch_metrics, per_batch
        else:
            return self.epoch_metrics
