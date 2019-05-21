import argparse
import copy
import os
import torch
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from time import localtime, strftime

from model.ESIM import ESIM
from model.utils import SNLI, Quora
from test import test


def train(args, data):
    model = ESIM(args, data)

    if args.gpu > -1:
        model.to(args.device)
    if args.fix_emb:
        # print(args.fix_emb)
        model.embeds.weight.required_grad = False

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.7,
                                                           patience=0)

    criterion = nn.CrossEntropyLoss()

    # writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    # loss, last_epoch = 0, -1
    max_dev_acc, max_test_acc = 0, 0

    iterator = data.train_iter
    patience_counter = 0
    last_dev_score = 0
    for epoch in range(args.epochs):
        iterator.init_epoch()
        n_correct, n_total = 0, 0
        all_losses = []
        print('epoch:', epoch+1)
        for i, batch in enumerate(iterator):
            if args.data_type == 'SNLI':
                s1, s2 = 'premise', 'hypothesis'
            else:
                s1, s2 = 'q1', 'q2'

            s1, s1_l = getattr(batch, s1)
            s2, s2_l = getattr(batch, s2)

            kwargs = {'p': s1, 'p_l': s1_l, 'h': s2, 'h_l': s2_l}

            pred = model(**kwargs)

            optimizer.zero_grad()
            batch_loss = criterion(pred, batch.label)
            all_losses.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()

            _, pred = pred.max(dim=1)
            n_correct += (pred == batch.label).sum().float()
            n_total += len(pred)
            train_acc = n_correct / n_total

            if (i + 1) % args.print_freq == 0:
                dev_loss, dev_acc = test(model, args, data, mode='dev')
                test_loss, test_acc = test(model, args, data)
                train_loss = np.mean(all_losses)
                c = (i + 1) // args.print_freq

                # writer.add_scalar('loss/train', train_loss, c)
                # writer.add_scalar('loss/dev', dev_loss, c)
                # writer.add_scalar('loss/test', test_loss, c)
                # writer.add_scalar('acc/train', train_acc, c)
                # writer.add_scalar('acc/dev', dev_acc, c)
                # writer.add_scalar('acc/test', test_acc, c)

                print(f'train loss: {train_loss:.3f} / dev loss: {dev_loss:.3f} / test loss: {test_loss:.3f}'
                      f' / train acc: {train_acc:.3f} / dev acc: {dev_acc:.3f} / test acc: {test_acc:.3f}')

                if dev_acc > max_dev_acc:
                    max_dev_acc = dev_acc
                    max_test_acc = test_acc
                    best_model = copy.deepcopy(model)
                    # torch.save(best_model.state_dict(), f'saved_models/BIBPM_{args.data_type}_{dev_acc}.pt')
                model.train()

        # dev_loss, dev_acc = test(model, args, data, mode='dev')
        # Early Stopping
        if max_dev_acc < last_dev_score:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping: patience limit reached, stopping...")
                break
        else:
            patience_counter = 0
            last_dev_score = max_dev_acc
        print("patience_counter: ", patience_counter)

        scheduler.step(max_dev_acc)
        model.train()


    # writer.close()
    print(f'max dev acc: {max_dev_acc:.3f} / max test acc: {max_test_acc:.3f}')

    return best_model, max_dev_acc


def main():
    print('add projection-layer, use dropout without bn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--char-dim', default=20, type=int)
    parser.add_argument('--char-hidden-size', default=50, type=int)
    parser.add_argument('--data-type', default='SNLI', help='available: SNLI or Quora')
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=300, type=int)
    parser.add_argument('--learning-rate', default=0.0004, type=float)
    parser.add_argument('--max-sent-len', default=-1, type=int,
                        help='max length of input sentences model can accept, if -1, it accepts any length')
    # parser.add_argument('--num-perspective', default=20, type=int)
    parser.add_argument('--print-freq', default=1000, type=int)
    parser.add_argument('--use-char-emb', default=False, action='store_true')
    parser.add_argument('--word-dim', default=300, type=int)
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(args.use_char_emb)
    if args.data_type == 'SNLI':
        print('loading SNLI data...')
        data = SNLI(args)
    elif args.data_type == 'Quora':
        print('loading Quora data...')
        data = Quora(args)
    else:
        raise NotImplementedError('only SNLI or Quora data is possible')

    # setattr(args, 'char_vocab_size', len(data.char_vocab))
    setattr(args, 'word_vocab_size', len(data.TEXT.vocab))
    setattr(args, 'class_size', len(data.LABEL.vocab))
    setattr(args, 'max_word_len', data.max_word_len)
    setattr(args, 'model_time', strftime('%H:%M:%S', localtime()))

    print('training start!')
    best_model, max_dev_acc = train(args, data)

    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(), f'saved_models/ESIM_{args.data_type}_{max_dev_acc:.3f}.pt')

    print('training finished!')


if __name__ == '__main__':
    main()
