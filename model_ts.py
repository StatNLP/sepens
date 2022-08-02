###############################################################################
# The following code is based on sources from:
#
# https://github.com/pytorch/examples/tree/main/word_language_model
#
###############################################################################

import torch.nn as nn

class RNNModelTS(nn.Module):
    """Container module with an encoder (inp), a recurrent module, and a decoder (fc)."""

    def __init__(self, rnn_type, nfeat, ninp, nhid, nlayers, dropout=0.5):
        super(RNNModelTS, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.inp = nn.Linear(nfeat, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.fc = nn.Linear(nhid, 1)

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.inp.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):

        emb = self.drop(self.inp(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        # 
        decoded = self.fc(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden 

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
