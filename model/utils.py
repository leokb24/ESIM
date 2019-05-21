from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe, Vectors
import spacy
# from nltk import word_tokenize

def tokenizer(text):
    spacy_en = spacy.load('en')
    fileters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    trans_map = str.maketrans(fileters, " " * len(fileters))
    text = text.translate(trans_map)
    text = [tok.text for tok in spacy_en.tokenizer(text) if tok.text != ' ']

    tokenized_text = []
    auxiliary_verbs = ['am', 'is', 'are', 'was', 'were', "'s"]
    for token in text:
        if token == "n't":
            tmp = 'not'
        elif token == "'ll":
            tmp = 'will'
        elif token in auxiliary_verbs:
            tmp = 'be'
        else:
            tmp = token
        tokenized_text.append(tmp)
    return tokenized_text

class SNLI():
    def __init__(self, args):
        self.TEXT = data.Field(batch_first=True, tokenize='spacy', lower=True, include_lengths=True)
        # self.LABEL = data.Field(sequential=False, unk_token=None)
        self.LABEL = data.LabelField()
        self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL,
                                                               root='/media/fch/Data/leo/text-similarity/data')
        vectors = Vectors(name='/media/fch/Data/leo/text-similarity/glove/glove.840B.300d.txt',
                          cache='/media/fch/Data/leo/text-similarity/.vector_cache')
        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=vectors)
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.premise), len(x.hypothesis))
        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.batch_size] * 3,
                                       device=args.device,
                                       sort_key=sort_key)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])


class Quora():
    def __init__(self, args):
        self.RAW = data.RawField(is_target=False)
        self.TEXT = data.Field(batch_first=True, tokenize='spacy', lower=True)
        # self.LABEL = data.Field(sequential=False, unk_token=None)
        self.LABEL = data.LabelField()

        self.train, self.dev, self.test = data.TabularDataset.splits(
            path='/media/fch/Data/leo/text-similarity/data/quora',
            train='train.tsv',
            validation='dev.tsv',
            test='test.tsv',
            format='tsv',
            fields=[('label', self.LABEL),
                    ('q1', self.TEXT),
                    ('q2', self.TEXT),
                    ('id', self.RAW)])
        vectors = Vectors(name='/media/fch/Data/leo/text-similarity/glove/glove.840B.300d.txt',
                          cache='/media/fch/Data/leo/text-similarity/.vector_cache')
        self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=vectors)
        self.LABEL.build_vocab(self.train)

        sort_key = lambda x: data.interleave_keys(len(x.q1), len(x.q2))

        self.train_iter, self.dev_iter, self.test_iter = data.BucketIterator.splits((self.train, self.dev, self.test),
                                                                                    batch_sizes=[args.batch_size] * 3,
                                                                                    device=args.device,
                                                                                    sort_key=sort_key)

        self.max_word_len = max([len(w) for w in self.TEXT.vocab.itos])