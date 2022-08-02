###############################################################################
# The following code is based on sources from:
#
# https://github.com/pytorch/examples/tree/main/word_language_model
#
###############################################################################

import argparse
import torch
import data

# fix "OSError: [Errno 24] Too many open files"
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#

parser = argparse.ArgumentParser(description='PyTorch SCIDATOS Time Series Model')

# Model parameters.
parser.add_argument('--data', type=str, default='.',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--nfeatures', type=int, default=43,
                    help='number of input features (time series)')
parser.add_argument('--timesteps', type=int, default='1000',
                    help='number of timesteps to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

hidden = model.init_hidden(1)

timeseries = data.TimeseriesTorch(args.data, args.nfeatures)

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

eval_batch_size = 1
test_data = batchify(timeseries.test, eval_batch_size)

timesteps = timeseries.test[0].size(0)


with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        for i in range(timesteps):
            input = test_data[0][i].view(1,1,43)
            output, hidden = model(input, hidden)

            step = [str(s) for s in output.data.cpu().flatten().numpy()] 
            outf.write( '-1\t-1\t-1\t' + str(i/2.) + '\t' + '\t'.join(step) + '\n')

            if i % args.log_interval == 0:
                print('| Generated {}/{} timesteps'.format(i, timesteps))
