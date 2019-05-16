from torch import nn
import torch
import torch.nn.functional as F

class ESIM(nn.Module):
    def __init__(self, args, data):
        super(ESIM, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.embeds_dim = args.word_dim

        self.embeds = nn.Embedding(num_embeddings=args.word_vocab_size, embedding_dim=self.embeds_dim)
        self.embeds.weight.data.copy_(data.TEXT.vocab.vectors)
        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)

        self.context_lstm = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.composition_lstm = nn.LSTM(self.hidden_size * 8, self.hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=args.dropout)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            self.dropout,
            nn.Linear(self.hidden_size * 8, self.hidden_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.hidden_size),
            self.dropout,
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(self.hidden_size),
            self.dropout,
            nn.Linear(self.hidden_size, args.class_size),
            # nn.Softmax(dim=-1)
        )
        self.reset_parameters()

    def reset_parameters(self):

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.005, 0.005)
                nn.init.constant_(layer.bias, val=0)
        # <unk> vectors is randomly initialized
        nn.init.uniform_(self.embeds.weight.data[0], -0.1, 0.1)

        # ----- Context Representation Layer -----
        nn.init.kaiming_normal_(self.context_lstm.weight_ih_l0)
        nn.init.constant_(self.context_lstm.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.context_lstm.weight_hh_l0)
        nn.init.constant_(self.context_lstm.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.context_lstm.weight_ih_l0_reverse)
        nn.init.constant_(self.context_lstm.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.context_lstm.weight_hh_l0_reverse)
        nn.init.constant_(self.context_lstm.bias_hh_l0_reverse, val=0)

        # ----- Aggregation Layer -----
        nn.init.kaiming_normal_(self.composition_lstm.weight_ih_l0)
        nn.init.constant_(self.composition_lstm.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.composition_lstm.weight_hh_l0)
        nn.init.constant_(self.composition_lstm.bias_hh_l0, val=0)

        nn.init.kaiming_normal_(self.composition_lstm.weight_ih_l0_reverse)
        nn.init.constant_(self.composition_lstm.bias_ih_l0_reverse, val=0)
        nn.init.orthogonal_(self.composition_lstm.weight_hh_l0_reverse)
        nn.init.constant_(self.composition_lstm.bias_hh_l0_reverse, val=0)
        
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, **kwargs):
        # batch_size * seq_len
        sent1, sent2 = kwargs['p'], kwargs['h']
        mask1, mask2 = sent1.eq(1), sent2.eq(1)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        # --- Input Encoding ---
        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.context_lstm(x1)
        o2, _ = self.context_lstm(x2)

        # --- Attention ---
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)

        # --- Inference Composition ---
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.composition_lstm(q1_combined)
        q2_compose, _ = self.composition_lstm(q2_combined)

        # --- Aggregate ---
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # --- Prediction ---
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity