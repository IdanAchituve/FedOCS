import numpy as np
from tqdm import trange
import argparse
import logging
from pathlib import Path
import copy
import torchvision

from experiments.mnist.dataset import MNISTData
from experiments.models import *
from experiments.utils import get_device, set_seed, save_experiment, set_logger, model_save

parser = argparse.ArgumentParser(description="MNIST")

#############################
#       Dataset Args        #
#############################
parser.add_argument("--data-path", type=str, default='./data', help="dataset path")
parser.add_argument("--val-pct", type=int, default=0.1, help="Number of test examples per class")
parser.add_argument('--noise', default=2., type=float, help='Gaussian Noise STD')
parser.add_argument("--num-clients", type=int, default=4, help="number of simulated nodes")

##################################
#       Model args        #
##################################
parser.add_argument('--embed-dim', default=64, type=int, help='Embedding size')
parser.add_argument('--n-hidden-e', default=128, type=int, help='Hiden size encoder')
parser.add_argument('--n-hidden-d', default=128, type=int, help='Hiden size decoder')

##################################
#       Optimization args        #
##################################
parser.add_argument("--num-epochs", type=int, default=200)
parser.add_argument("--optimizer", type=str, default='adam', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--test-batch-size", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")

##################################
#       General        #
##################################
parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
parser.add_argument('--num-workers', default=0, type=int, help='num workers')
parser.add_argument("--eval-every", type=int, default=1, help="eval every X selected steps")
parser.add_argument("--save-path", type=str, default="./output", help="dir path for output file")
parser.add_argument("--seed", type=int, default=0, help="seed value")

args = parser.parse_args()

set_logger()
set_seed(args.seed)

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)

exp_name = f'FedOCS-MNIST_num-clients_{args.num_clients}_' \
           f'seed_{args.seed}_lr_{args.lr}_noise_{args.noise}_embed-dim_{args.embed_dim}_' \
           f'n-hidden-e_{args.n_hidden_e}_n-hidden-d{args.n_hidden_d}_noise_{args.noise}'

if args.exp_name != '':
    exp_name += '_' + args.exp_name

logging.info(str(args))
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)

################################
# Data
################################
data = MNISTData(data_path=args.data_path)
train_loader, val_loader, test_loader = \
    data.get_data_loaders(train_batch_size=args.batch_size, test_batch_size=args.test_batch_size, val_pct=args.val_pct,
                          num_workers=args.num_workers)

###############################
# Model
###############################
client_models = [Encoder2(n_hidden=args.n_hidden_e, n_embedd=args.embed_dim) for c in range(args.num_clients)]
client_models = torch.nn.ModuleList(client_models)
decoder = Decoder2(n_hidden=args.n_hidden_d, n_embedd=args.embed_dim)
client_models.to(device)
decoder.to(device)

###############################
# Optimizer
###############################
param_group = []
param_group.append({'params': client_models.parameters()})
param_group.append({'params': decoder.parameters()})
optimizer = torch.optim.SGD(param_group, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True) \
           if args.optimizer == 'sgd' else torch.optim.Adam(param_group, lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

def save_models():
    for client in range(args.num_clients):
        client_model = "best_model_" + str(client) + ".pt"
        model_save(client_models[client], out_dir / client_model)
    model_save(decoder, out_dir / "decoder.pt")

def copy_models():
    return copy.deepcopy(client_models), copy.deepcopy(decoder)

###############################
# Test/Validation
###############################
@torch.no_grad()
def model_evalution(loader, epoch, test=True):
    client_models.eval()
    decoder.eval()

    nll_accum = 0
    num_samples = 0

    for k, batch in enumerate(loader):
        batch = (t.to(device) for t in batch)
        test_data, clf_labels = batch
        test_data = test_data.view(test_data.size(0), -1)

        zs = []
        xs = []
        for encoder in client_models:
            x = test_data + torch.randn_like(test_data) * args.noise
            z = encoder(x)
            zs.append(z.view(z.shape[0], z.shape[1], 1))
            xs.append(x)
        z = torch.cat(zs, dim=2)
        z, _ = torch.max(z, dim=2)
        recon = decoder(z)
        loss = criterion(recon, test_data)

        nll_accum += loss.item() * test_data.size(0)
        num_samples += test_data.size(0)

    nll_accum /= num_samples
    return nll_accum

###############################
# train loop
###############################
list_train_ll = []
criterion = nn.BCELoss(reduction='mean')
epoch_iter = trange(args.num_epochs)
best_val_nll = best_test_nll = np.inf
best_epoch = 0
best_model = copy_models()

for epoch in epoch_iter:

    client_models.train()
    decoder.train()
    cumm_loss = num_samples = num_batches = 0

    for k, batch in enumerate(train_loader):

        batch = (t.to(device) for t in batch)
        train_data, clf_labels = batch
        train_data = train_data.view(train_data.size(0), -1)

        optimizer.zero_grad()
        zs = []
        for encoder in client_models:
            x = train_data + torch.randn_like(train_data) * args.noise
            z = encoder(x)
            zs.append(z.view(z.shape[0], z.shape[1], 1))
        z = torch.cat(zs, dim=2)
        z, _ = torch.max(z, dim=2)
        recon = decoder(z)
        loss = criterion(recon, train_data)
        loss.backward()
        optimizer.step()

        epoch_iter.set_description(f'[{epoch}] Training loss {loss.item():.5f}')
        cumm_loss += loss.item()
        num_batches += 1
        # break

    cumm_loss /= num_batches

    if (epoch + 1) % args.eval_every == 0 or (epoch + 1) == args.num_epochs:
        val_nll = model_evalution(val_loader, epoch, test=False)
        test_nll = model_evalution(test_loader, epoch, test=True)

        logging.info(f"\nTrain NLL: {cumm_loss:.5f}, "
                     f"Val NLL: {val_nll:.5f}, "
                     f"Test NLL: {test_nll:.5f}")

        if val_nll < best_val_nll:
            best_epoch = epoch + 1
            best_train_loss = cumm_loss
            best_val_nll = val_nll
            best_test_nll = test_nll
            save_models()
            best_model = copy_models()

            # save model
            logging.info(f"\nBest Epoch: {best_epoch}, "
                         f"Best Train NLL: {cumm_loss:.5f}, "
                         f"Best Val NLL: {val_nll:.5f}, "
                         f"Best Test NLL: {test_nll:.5f}\n")


    scheduler.step()

client_models, classifier = best_model
test_nll = model_evalution(test_loader, epoch=9999, test=True)

logging.info(f"\nStep: {best_epoch}, Best Val Loss: {best_val_nll:.4f}")
logging.info(f"\nStep: {best_epoch}, Test Loss: {best_test_nll:.4f}")
