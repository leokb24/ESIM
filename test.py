import argparse

import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

from model.ESIM import ESIM
from model.utils import SNLI, Quora


def test(model, args, data, mode='test'):
    if mode == 'dev':
        iterator = iter(data.dev_iter)
    else:
        iterator = iter(data.test_iter)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc, loss, size = 0, 0, 0
    losses = []
    for batch in iterator:
        if args.data_type == 'SNLI':
            s1, s2 = 'premise', 'hypothesis'
        else:
            s1, s2 = 'q1', 'q2'

        s1, s1_l = getattr(batch, s1)
        s2, s2_l = getattr(batch, s2)

        kwargs = {'p': s1, 'p_l': s1_l, 'h': s2, 'h_l': s2_l}

        pred = model(**kwargs)

        batch_loss = criterion(pred, batch.label)
        losses.append(batch_loss.item())

        _, pred = pred.max(dim=1)
        acc += (pred == batch.label).sum().float()
        size += len(pred)

    acc /= size
    # acc = acc.cpu().data[0]
    return np.mean(losses), acc.item()


def load_model(args, data):
    model = ESIM(args, data)
    model.load_state_dict(torch.load(args.model_path))

    if args.gpu > -1:
        model.to(args.device)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.001, type=float)
    parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--use-char-emb', default=True, action='store_true')
    parser.add_argument('--word-dim', default=300, type=int)

    parser.add_argument('--model-path', required=True)

    args = parser.parse_args()

    if args.data_type == 'SNLI':
        print('loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('loading Quora data...')
        data = Quora(args)

    setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)

    print('loading model...')
    model = load_model(args, data)

    _, acc = test(model, args, data)

    print(f'test acc: {acc:.3f}')
