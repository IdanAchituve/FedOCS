import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
import torch.utils.data
from tqdm import trange
import copy
from sklearn.metrics import classification_report

from experiments.cub.DataClass import CUBData
from experiments.models import *
from experiments.utils import get_device, set_logger, set_seed, str2bool, detach_to_numpy, \
                              save_experiment, model_load, model_save
from experiments.clients import ClientModelNoise, ClientModelPatches
from experiments.models.classification_head import ClassificationHead

parser = argparse.ArgumentParser(description="FedOCS")

#############################
#       Dataset Args        #
#############################
parser.add_argument("--data-path", type=str, default="./dataset", help="dir path for CUB dataset")
parser.add_argument("--num-clients", type=int, default=4, help="number of simulated nodes")
parser.add_argument("--corruption-type", choices=['gaussian_noise', 'patches'],
                    default='patches')
parser.add_argument("--corruption-severity", type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='5',
                    help="corruption severity")

##################################
#       Optimization args        #
##################################
parser.add_argument("--num-epochs", type=int, default=200)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--test-batch-size", type=int, default=256)
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
parser.add_argument('--scheduler', default=True, type=str2bool, help='use learning rate scheduler')
parser.add_argument('--milestones', type=lambda s: [int(item.strip()) for item in s.split(',')],
                    default='100,150')

################################
#       Model Prop args        #
################################
parser.add_argument("--model-path", type=str, default=None, help="dir to model path")
parser.add_argument('--embed-dim', type=int, default=256, help='embedding dimension')
parser.add_argument('--top-k', type=int, default=1, help='top k features')
parser.add_argument('--bottom-k', type=int, default=0, help='bottom k features')
parser.add_argument('--agg-scheme', type=str, default='max',
                    choices=['max', 'mean', 'min-max', 'cat'], help="aggregation scheme over the clients")
parser.add_argument('--network', type=str, default='MobileNetV2Pytorch',
                    choices=['MobileNetV2Pytorch'], help="network to use")
parser.add_argument('--pretrained', default=True, type=str2bool, help='pre-trained network?')

#############################
#       General args        #
#############################
parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
parser.add_argument('--num-workers', default=4, type=int, help='num workers')
parser.add_argument("--eval-every", type=int, default=1, help="eval every X selected steps")
parser.add_argument("--save-path", type=str, default="./output", help="dir path for output file")
parser.add_argument("--seed", type=int, default=42, help="seed value")

args = parser.parse_args()
        
set_logger()
set_seed(args.seed)

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
num_classes = 200

exp_name = f'FedOCS-CUB_num-clients_{args.num_clients}_' \
           f'seed_{args.seed}_lr_{args.lr}_num-epochs_{args.num_epochs}_embed-dim_{args.embed_dim}_' \
           f'agg-scheme_{args.agg_scheme}_corr-type_{args.corruption_type}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

logging.info(str(args))
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)
logging.info("corruption types: " + args.corruption_type)
if args.corruption_type == 'gaussian_noise':
    logging.info("corruption severities: " + str(args.corruption_severity).strip('[]'))

################################
# get data
###############################
data = CUBData(data_dir=args.data_path)
train_loader, val_loader, test_loader = data.get_loaders(
    max_class=num_classes,
    batch_size=args.batch_size,
    test_batch_size=args.test_batch_size,
    num_workers=args.num_workers
)

###############################
# init clients
###############################
if args.corruption_type == 'gaussian_noise':
    client_models = [ClientModelNoise(id=c, embed_dim=args.embed_dim, model_path=args.model_path,
                                      corruption_type=args.corruption_type, corruption_severity=args.corruption_severity,
                                      network=args.network)
                     for c in range(args.num_clients)]
else:
    grid_side_size = math.sqrt(args.num_clients)
    grid_sizes = [(0, 0), (0, 1), (1, 0), (1, 1)] if grid_side_size == 2 \
        else [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    client_models = [ClientModelPatches(id=c, embed_dim=args.embed_dim, model_path=args.model_path,
                                        grid_side_size=grid_side_size,
                                        row_id=size[0],
                                        col_id=size[1],
                                        network=args.network, **{'pretrained': args.pretrained})
                     for c, size in enumerate(grid_sizes)]
client_models = torch.nn.ModuleList(client_models)
cls_input_dim = args.embed_dim if args.agg_scheme != 'cat' else args.embed_dim * args.num_clients
classifier = ClassificationHead(cls_input_dim, num_classes=num_classes)
if args.model_path:
    model_dir = Path(args.model_path)
    classifier = model_load(classifier, model_dir / "best_model_classifier.pt")

client_models.to(device)
classifier.to(device)

###############################
# optimizer
###############################
param_group = []
param_group.append({'params': client_models.parameters()})
param_group.append({'params': classifier.parameters()})
optimizer = torch.optim.SGD(param_group, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True) \
           if args.optimizer == 'sgd' else torch.optim.Adam(param_group, lr=args.lr, weight_decay=args.wd)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

###############################
# Test/Validation
###############################
@torch.no_grad()
def model_evalution(loader, print_clf_report=False, test=True):
    client_models.eval()
    classifier.eval()

    targets = []
    preds = []
    nll_accum = 0
    num_samples = 0

    step = 0
    for k, batch in enumerate(loader):
        step += 1
        batch_gen = (t.to(device) for t in batch)
        test_data, clf_labels = batch_gen

        features = []
        for client in range(args.num_clients):
            features.append(client_models[client](test_data))

        features = torch.stack(features, dim=-1)
        if args.agg_scheme == 'max':
            features, _ = torch.max(features, dim=-1)
        elif args.agg_scheme == 'min-max':
            features1, _ = torch.max(features, dim=-1)
            features2, _ = torch.min(features, dim=-1)
            features = (features1 + features2) / 2
        elif args.agg_scheme == 'mean':
            features = torch.mean(features, dim=-1)
        elif args.agg_scheme == 'cat':
            features = features.view(clf_labels.shape[0], -1)
        else:
            raise Exception("unsupported aggregation scheme")

        logits = classifier(features)
        loss = criteria(logits, clf_labels)

        nll_accum += loss.item() * test_data.size(0)
        num_samples += test_data.size(0)

        targets.append(clf_labels)
        preds.append(logits)

    nll_accum /= num_samples

    target = detach_to_numpy(torch.cat(targets, dim=0))
    full_pred = detach_to_numpy(torch.cat(preds, dim=0))
    clf_report = classification_report(target, full_pred.argmax(1), output_dict=True, zero_division=0)

    if print_clf_report:
        logging.info('\n' + classification_report(target, full_pred.argmax(1), zero_division=0))

    labels_vs_preds = np.concatenate((target.reshape(-1, 1), full_pred), axis=1)
    return nll_accum, clf_report, labels_vs_preds

###############################
# Train
###############################
def save_models():
    for client in range(args.num_clients):
        client_model = "best_model_" + str(client) + ".pt"
        model_save(client_models[client], out_dir / client_model, args.wandb)
    model_save(classifier, out_dir / "best_model_classifier.pt", args.wandb)

def copy_models():
    return copy.deepcopy(client_models), copy.deepcopy(classifier)


criteria = torch.nn.CrossEntropyLoss()
results = defaultdict(list)

best_loss = best_val_nll = np.inf
best_epoch = 0
best_val_acc = 0
epoch_iter = trange(args.num_epochs)
best_test_labels_vs_preds = best_test_clf_report = None

best_model = copy_models()
best_labels_vs_preds_val = None
best_val_loss = -1

for epoch in epoch_iter:
    client_models.train()
    classifier.train()
    cumm_loss = num_samples = 0

    for k, batch in enumerate(train_loader):

        optimizer.zero_grad()

        # get data
        batch = (t.to(device) for t in batch)
        train_data, clf_labels = batch
        features = []
        for client in range(args.num_clients):
            features.append(client_models[client](train_data))

        features = torch.stack(features, dim=-1)
        if args.agg_scheme == 'max':
            features, _ = torch.max(features, dim=-1)
        elif args.agg_scheme == 'min-max':
            features1, _ = torch.max(features, dim=-1)
            features2, _ = torch.min(features, dim=-1)
            features = (features1 + features2) / 2
        elif args.agg_scheme == 'mean':
            features = torch.mean(features, dim=-1)
        elif args.agg_scheme == 'cat':
            features = features.view(clf_labels.shape[0], -1)
        else:
            raise Exception("unsupported aggregation scheme")

        logits = classifier(features)
        loss = criteria(logits, clf_labels)
        loss.backward()
        optimizer.step()

        epoch_iter.set_description(f"[{epoch} {k}] loss: {loss.item():.3f}")
        cumm_loss += loss.item() * clf_labels.shape[0]
        num_samples += clf_labels.shape[0]

    cumm_loss /= num_samples

    if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.num_epochs:
        val_nll, val_clf_report, _ = model_evalution(val_loader, print_clf_report=False, test=False)
        test_nll, test_clf_report, _ = model_evalution(test_loader, print_clf_report=False, test=True)

        logging.info(f"Train Loss: {cumm_loss:.5f}, "
                     f"Val NLL: {val_nll:.5f}, "
                     f"Val Accuracy: {val_clf_report['accuracy']:.5f}, "
                     f"Test NLL: {test_nll:.5f}, "
                     f"Test Accuracy: {test_clf_report['accuracy']:.5f}\n")

        if best_val_acc < val_clf_report['accuracy']:
            best_epoch = epoch + 1
            best_train_loss = cumm_loss
            best_val_nll = val_nll
            best_val_acc = val_clf_report['accuracy']
            best_test_nll = test_nll
            best_test_acc = test_clf_report['accuracy']
            best_test_clf_report = test_clf_report

            # save model
            save_models()
            best_model = copy_models()
            logging.info(f"Best Epoch: {best_epoch}, "
                         f"Best Train Loss: {cumm_loss:.5f}, "
                         f"Best Val NLL: {val_nll:.5f}, "
                         f"Best Val Accuracy: {val_clf_report['accuracy']:.5f}, "
                         f"Best Test NLL: {test_nll:.5f}, "
                         f"Best Test Accuracy: {test_clf_report['accuracy']:.5f}\n")

    if args.scheduler:
        scheduler.step()
        lrs = scheduler.get_last_lr()

client_models, classifier = best_model
test_nll, test_clf_report, labels_vs_preds_test = model_evalution(test_loader, print_clf_report=True, test=True)

logging.info(f"\nStep: {best_epoch}, Best Val Loss: {best_val_nll:.4f}, Best Val Acc: {best_val_acc:.4f}")
logging.info(f"\nStep: {best_epoch}, Test Loss: {test_nll:.4f}, Test Acc: {test_clf_report['accuracy']:.4f}")

results = dict()
results['final_test_results'] = {
    'best_NLL': best_loss,
    'test_clf_report': best_test_clf_report,
    'test_labels_vs_preds': labels_vs_preds_test.tolist()
}

results_file = "results.json"
with open(str(out_dir / results_file), "w") as file:
    json.dump(results, file, indent=4)