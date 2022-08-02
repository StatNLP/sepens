# coding: utf-8

###############################################################################
# The following code is based on sources from:
#
# https://github.com/pytorch/examples/tree/main/word_language_model
#
###############################################################################

import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import data
import model_ts


parser = argparse.ArgumentParser(description='PyTorch SCIDATOS Time Series RNN/LSTM Model')
parser.add_argument('--data', type=str, default='.',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--nfeatures', type=int, default=43,
                    help='number of input features (time series)')
parser.add_argument('--insize', type=int, default=200,
                    help='size of input for RNN')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--min_epochs', type=int, default=10,
                    help='minimum epoch before early stopping')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=48,
                    help='sequence length')
parser.add_argument('--seqoverlap', type=float, default=0.5,
                    help='sequence overlap')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

###############################################################################
# Load data
###############################################################################

# Load timeseries as torch tensor (cannot be modified)
#timeseries = data.TimeseriesTorch(args.data, args.nfeatures)

# Load timeseries as numpy array (not immutable)
timeseries = data.TimeseriesNumPy(args.data, args.nfeatures)
print("Timeseries loaded")

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(rawdata, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = rawdata[0].size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = rawdata[0].narrow(0, 0, nbatch * bsz)

    # Evenly divide the data across the bsz batches.
    data = data.view(-1, nbatch, args.nfeatures)
    data = data.permute(1,0,2).contiguous()

    ys = rawdata[1].narrow(0, 0, nbatch * bsz)
    ys = ys.view(-1, nbatch, 1)
    ys = ys.permute(1,0,2).contiguous()
    
    return [data.to(device),ys.to(device)]



# Generate overlapping sequences to avoid information loss by unfavorable
# sequence splits

def expand_and_batchify(rawdata, bsz, step=0.5):    
    
    seqlen = args.bptt
    rawlen = int(len(rawdata[0]))
    
    stepsize = int(seqlen*step)
    
    # calculate length of new sequence with overlapa
    timesteps = (math.floor((rawlen-seqlen)/stepsize)+1 ) * seqlen
    timesteps += rawlen - (math.floor((rawlen-seqlen)/stepsize)+1) * stepsize
    
    steps = torch.Tensor(timesteps, args.nfeatures)
    targets = torch.Tensor(timesteps)    # severity

    pos = 0
    for i in range(0, rawlen-seqlen+1, stepsize):
        for j in range(0,seqlen):
            steps[pos] = torch.from_numpy(rawdata[0][i + j])
            targets[pos] = float(rawdata[1][i + j])
            pos += 1
    # copy over remainders
    remainderstart = (math.floor((rawlen-seqlen)/stepsize)+1) * stepsize
    for k in range(remainderstart, rawlen):
        steps[pos] = torch.from_numpy(rawdata[0][k])
        targets[pos] = float(rawdata[1][k])
        pos += 1

    # make sure we have initialized every position
    assert(timesteps == pos)
    
    return batchify([steps,targets], bsz)




eval_batch_size = 10

train_data = expand_and_batchify(timeseries.train, args.batch_size)
val_data = expand_and_batchify(timeseries.valid, eval_batch_size)
test_data = expand_and_batchify(timeseries.test, eval_batch_size)


###############################################################################
# Build the model
###############################################################################

model = model_ts.RNNModelTS(args.model, args.nfeatures, args.insize, 
                            args.nhid, args.nlayers, args.dropout).to(device)

criterion = nn.MSELoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):

    seq_len = min(args.bptt, len(source[0]) - 1 - i)
    data = source[0][i:i+seq_len]

    target = source[1][i:i+seq_len]

    return data, target

def flatten(mydata):
    return mydata.permute(1,0,2).contiguous().view(-1,1)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.

    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source[0].size(0) - 1, args.bptt):

            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            
            total_loss += len(data) * criterion(output, targets).item()
            hidden = repackage_hidden(hidden)

    return total_loss / (len(data_source[0]) - 1)


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    total_xentropy = 0.
    start_time = time.time()
    
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data[0].size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        
        loss = criterion(output, targets)

        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            cur_xentropy = total_xentropy / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | mse {:8.5f}'.format(
                epoch, batch, len(train_data[0]) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, cur_loss))
            total_loss = 0
            start_time = time.time()


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid mse {:8.5f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, val_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if epoch >= args.min_epochs and (not best_val_loss or val_loss < best_val_loss):
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    model.rnn.flatten_parameters()

# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test mse {:8.5f}'.format(
    test_loss, test_loss))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
