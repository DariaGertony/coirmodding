# Инструмент для создания базы знаний из датасета

## Общий пайплайн 

В ```./corpuses_dataset``` лежат распаршенные .txt файлы с корпусами из датасета **codetrans-dl**

В ```./coir/evaluation.py ``` будет вызываться метод ```create_knowledgebase_from_dataset_corpus```, реализованный в ```filter.py```

После вызова данного метода из ```.txt``` файлов созадается ```jsonl``` файл, в котором все фрагменты из датасета представляют собой чанки ```json``` формата с выделенными метаданными: language, imports, classes, functions. 

## extraction.py

- extract_metadata_from_dataset_corpus

    выделяет метаданные из кода на основе AST

- create_knowledgebase_from_dataset_corpus

    на основе выделенных метаданных из ```extract_metadata_from_dataset_corpus``` строит базу знаний (jsonl файл)

## storage.py

Реализует класс ```Storage``` для создания базы знаний и управления ей. 

## Пример работы

Возьмем первый распаршенный датасет из ```./corpuses_dataset```
```
import math
import warnings
import pandas as pd
from d2l import paddle as d2l
warnings.filterwarnings("ignore")
import paddle
from paddle import nn
class PositionWiseFFN(nn.Layer):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
ffn(paddle.ones((2, 3, 4)))[0]
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1D(2)
X = paddle.to_tensor([[1, 2], [2, 3]], dtype=paddle.float32)
print('layer norm:', ln(X), '
batch norm:', bn(X))
class AddNorm(nn.Layer):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
add_norm(paddle.ones((2, 3, 4)), paddle.ones((2, 3, 4))).shape
class EncoderBlock(nn.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)
    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
X = paddle.ones((2, 100, 24))
valid_lens = paddle.to_tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
encoder_blk(X, valid_lens).shape
class TransformerEncoder(d2l.Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_sublayer(str(i), EncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, use_bias))
    def forward(self, X, valid_lens, *args):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
encoder = TransformerEncoder(200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
encoder(paddle.ones((2, 100), dtype=paddle.int64), valid_lens).shape
class DecoderBlock(nn.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]

        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = paddle.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = paddle.arange(1, num_steps + 1).tile((batch_size, 1))
        else:
            dec_valid_lens = None
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = paddle.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state)[0].shape
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_sublayer(str(i), DecoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)
    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]
    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state
    @property
    def attention_weights(self):
        return self._attention_weights
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
enc_attention_weights = paddle.concat(net.encoder.attention_weights, 0).reshape((num_layers, num_heads, -1, num_steps))
enc_attention_weights.shape
d2l.show_heatmaps(enc_attention_weights.cpu(), xlabel='Key positions', ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
dec_attention_weights_2d = [head[0].tolist() for step in dec_attention_weight_seq for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = paddle.to_tensor(pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape((-1, 2, num_layers, num_heads, num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \ dec_attention_weights.transpose((1, 2, 3, 0, 4))
dec_self_attention_weights.shape, dec_inter_attention_weights.shape
```
После работы тулы данный текстовый файл будет представлен json-строкой вида:
```
{"chunk_id": "chunk_1", "language": "python", "imports": ["math", "nn", "paddle", "pandas", "warnings"], "classes": ["PositionWiseFFN", "AddNorm", "EncoderBlock", "TransformerEncoder", "DecoderBlock", "TransformerDecoder"], "functions": [], "content": "fs"}
```
Через тулу прогоняются все текстовые файлы из ./corpuses_dataset
из которых затем строится база знаний в формате json

## Дальнейшие шаги
Тулу можно использовать в ```./coir/evaluation.py```
```
    def run(self, model, output_folder: str, useLLm: bool, llmname: str, prompt: str,to_rerank: bool):
        results = {}
        for task_name, task_data in self.tasks.items():
            output_folder = os.path.join(output_folder, f"{task_name}/{model.model_name}")
            output_file = os.path.join(output_folder, f"{self.type}.json")

            corpus, queries, qrels = task_data

            if task_name == "codetrans-dl":

                # тут можно сделать подготовку базы знаний на датасете на основе написанной тулы
                # желательно добавить проверку на существование базы знаний (чтобы не рассчитывать при каждом запуске)
```
Далее выполнять поиск по отфильтрованной базе знаний и замерять результаты