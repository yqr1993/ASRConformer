import io
import os

import librosa
import soundfile
import yaml

import paddle
from paddle import nn
from paddle.nn import functional as F

import json
import math
import copy
import sys

from collections import OrderedDict
from inspect import signature
from collections import defaultdict

import sentencepiece as spm
from yacs.config import CfgNode
from contextlib import contextmanager
from typeguard import check_argument_types
from python_speech_features.base import logfbank

import numpy as np

# model download path https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/conformer.model.tar.gz'
paddle.half = 'float16'
paddle.float = 'float32'
paddle.double = 'float64'
paddle.short = 'int16'
paddle.int = 'int32'
paddle.long = 'int64'
paddle.uint16 = 'uint16'
paddle.cdouble = 'complex128'


def eq(xs: paddle.Tensor, ys):
    if xs.dtype == paddle.bool:
        xs = xs.astype(paddle.int)
    return xs.equal(paddle.to_tensor(ys, dtype=xs.dtype, place=xs.place))


if not hasattr(paddle.Tensor, 'eq'):
    print("override eq of paddle.Tensor if exists or register, remove this when fixed!")
    paddle.Tensor.eq = eq


def size(xs: paddle.Tensor, *args: int):
    nargs = len(args)
    assert (nargs <= 1)
    s = paddle.shape(xs)
    if nargs == 1:
        return s[args[0]]
    else:
        return s


paddle.Tensor.size = size


def view(xs: paddle.Tensor, *args: int):
    return xs.reshape(args)


if not hasattr(paddle.Tensor, 'view'):
    print("register user view to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.view = view


def is_broadcastable(shp1, shp2):
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def masked_fill(xs: paddle.Tensor, mask: paddle.Tensor, value):
    assert is_broadcastable(xs.shape, mask.shape) is True, (xs.shape, mask.shape)
    bshape = paddle.broadcast_shape(xs.shape, mask.shape)
    mask = mask.broadcast_to(bshape)
    trues = paddle.ones_like(xs) * value
    xs = paddle.where(mask, trues, xs)
    return xs


if not hasattr(paddle.Tensor, 'masked_fill'):
    print("register user masked_fill to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.masked_fill = masked_fill


def repeat(xs: paddle.Tensor, *size):
    return paddle.tile(xs, size)


if not hasattr(paddle.Tensor, 'repeat'):
    print("register user repeat to paddle.Tensor, remove this when fixed!")
    paddle.Tensor.repeat = repeat

"""Transformation module."""


class GlobalCMVN:
    def __init__(self,
                 cmvn_path,
                 norm_means=True,
                 norm_vars=True,
                 std_floor=1.0e-20):
        self.cmvn_path = cmvn_path
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.std_floor = std_floor

        with open(cmvn_path) as f:
            cmvn_stats = json.load(f)
        self.count = cmvn_stats['frame_num']
        self.mean = np.array(cmvn_stats['mean_stat']) / self.count
        self.square_sums = np.array(cmvn_stats['var_stat'])
        self.var = self.square_sums / self.count - self.mean ** 2
        self.std = np.maximum(np.sqrt(self.var), self.std_floor)

    def __call__(self, x, uttid=None):
        # x: [Time, Dim]
        if self.norm_means:
            x = np.subtract(x, self.mean)

        if self.norm_vars:
            x = np.divide(x, self.std)
        return x


class LogMelSpectrogramKaldi:
    def __init__(
            self,
            fs=16000,
            n_mels=80,
            n_fft=512,  # fft point
            n_shift=160,  # unit:sample, 10ms
            win_length=400,  # unit:sample, 25ms
            window="povey",
            fmin=20,
            fmax=None,
            eps=1e-10,
            dither=False):
        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        if n_shift > win_length:
            raise ValueError("Stride size must not be greater than "
                             "window size.")
        self.n_shift = n_shift / fs  # unit: ms
        self.win_length = win_length / fs  # unit: ms

        self.window = window
        self.fmin = fmin
        if fmax is None:
            fmax_ = fmax if fmax else self.fs / 2
        elif fmax > int(self.fs / 2):
            raise ValueError("fmax must not be greater than half of "
                             "sample rate.")
        self.fmax = fmax_

        self.eps = eps
        self.remove_dc_offset = True
        self.preemph = 0.97
        self.dither = dither  # only work in train mode

    def __call__(self, x):
        if x.ndim != 1:
            raise ValueError("Not support x: [Time, Channel]")

        if x.dtype in np.sctypes['float']:
            # PCM32 -> PCM16
            bits = np.iinfo(np.int16).bits
            x = x * 2 ** (bits - 1)

        # logfbank need PCM16 input
        y = logfbank(
            signal=x,
            samplerate=self.fs,
            winlen=self.win_length,  # unit ms
            winstep=self.n_shift,  # unit ms
            nfilt=self.n_mels,
            nfft=self.n_fft,
            lowfreq=self.fmin,
            highfreq=self.fmax,
            dither=False,
            remove_dc_offset=self.remove_dc_offset,
            preemph=self.preemph,
            wintype=self.window)

        return y


class Transformation:
    def __init__(self, conffile):
        self.conf = copy.deepcopy(conffile)
        self.functions = OrderedDict()
        import_class_obj = [LogMelSpectrogramKaldi, GlobalCMVN]

        for idx, process in enumerate(self.conf["process"][:2]):
            assert isinstance(process, dict), type(process)
            opts = dict(process)
            opts.pop("type")
            class_obj = import_class_obj[idx]
            self.functions[idx] = class_obj(**opts)

    def __call__(self, xs, uttid_list=None, **kwargs):
        xs = [xs]

        if isinstance(uttid_list, str):
            uttid_list = [uttid_list for _ in range(len(xs))]

        for idx in range(len(self.conf["process"][:2])):
            func = self.functions[idx]
            try:
                param = signature(func).parameters
            except ValueError:
                # Some function, e.g. built-in function, are failed
                param = {}
            _kwargs = {k: v for k, v in kwargs.items() if k in param}
            try:
                if uttid_list is not None and "uttid" in param:
                    xs = [func(x, u, **_kwargs) for x, u in zip(xs, uttid_list)]
                else:
                    xs = [func(x, **_kwargs) for x in xs]
            except Exception:
                print("Catch a exception from {}th func: {}".format(
                    idx, func))
                raise

        return xs[0]


"""U2 module."""
IGNORE_ID = -1
SOS = "<eos>"
EOS = SOS
UNK = "<unk>"
BLANK = "<blank>"
MASKCTC = "<mask>"
SPACE = "<space>"


def load_dict(dict_path, maskctc=False):
    if dict_path is None:
        return None

    with open(dict_path, "r", encoding='utf-8') as f:
        dictionary = f.readlines()
    char_list = [entry[:-1].split(" ")[0] for entry in dictionary]
    if BLANK not in char_list:
        char_list.insert(0, BLANK)
    if EOS not in char_list:
        char_list.append(EOS)
    # for non-autoregressive maskctc model
    if maskctc and MASKCTC not in char_list:
        char_list.append(MASKCTC)
    return char_list


class TextFeaturizer:
    def __init__(self,
                 unit_type,
                 vocab_filepath,
                 spm_model_prefix=None,
                 maskctc=False):

        assert unit_type in ('char', 'spm', 'word')
        self.unit_type = unit_type
        self.unk = UNK
        self.maskctc = maskctc

        if vocab_filepath:
            self.vocab_dict, self._id2token, self.vocab_list, self.unk_id, self.eos_id, self.blank_id = \
                self._load_vocabulary_from_file(vocab_filepath, maskctc)
            self.vocab_size = len(self.vocab_list)
        else:
            print("TextFeaturizer: not have vocab file.")

        if unit_type == 'spm':
            spm_model = spm_model_prefix + '.model'
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(spm_model)

    def detokenize(self, tokens):
        if self.unit_type == 'char':
            text = self.char_detokenize(tokens)
        elif self.unit_type == 'word':
            text = self.word_detokenize(tokens)
        else:  # spm
            text = self.spm_detokenize(tokens)
        return text

    def defeaturize(self, idxs):
        tokens = []
        for idx in idxs:
            if idx == self.eos_id:
                break
            tokens.append(self._id2token[idx])
        text = self.detokenize(tokens)
        return text

    def char_detokenize(self, tokens):
        tokens = [t.replace(SPACE, " ") for t in tokens]
        return "".join(tokens)

    def word_detokenize(self, tokens):
        return " ".join(tokens)

    def spm_detokenize(self, tokens, input_format='piece'):
        if input_format == "piece":
            def decode(l):
                return "".join(self.sp.DecodePieces(l))
        elif input_format == "id":
            def decode(l):
                return "".join(self.sp.DecodeIds(l))

        return decode(tokens)

    def _load_vocabulary_from_file(self, vocab_filepath: str, maskctc: bool):
        """Load vocabulary from file."""
        vocab_list = load_dict(vocab_filepath, maskctc)
        assert vocab_list is not None

        id2token = dict([(idx, token) for (idx, token) in enumerate(vocab_list)])
        token2id = dict([(token, idx) for (idx, token) in enumerate(vocab_list)])

        blank_id = vocab_list.index(BLANK) if BLANK in vocab_list else -1
        unk_id = vocab_list.index(UNK) if UNK in vocab_list else -1
        eos_id = vocab_list.index(EOS) if EOS in vocab_list else -1

        return token2id, id2token, vocab_list, unk_id, eos_id, blank_id


class PositionalEncoding(nn.Layer):
    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 reverse: bool = False):
        nn.Layer.__init__(self)
        self.d_model = d_model
        self.max_len = max_len
        self.xscale = paddle.to_tensor(math.sqrt(self.d_model))
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = paddle.zeros([self.max_len, self.d_model])  # [T,D]

        position = paddle.arange(
            0, self.max_len, dtype=paddle.float32).unsqueeze(1)  # [T, 1]
        div_term = paddle.exp(
            paddle.arange(0, self.d_model, 2, dtype=paddle.float32) *
            -(math.log(10000.0) / self.d_model))

        self.pe[:, 0::2] = paddle.sin(position * div_term)
        self.pe[:, 1::2] = paddle.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # [1, T, D]

    def forward(self, x: paddle.Tensor, offset: int = 0):
        T = x.shape[1]
        assert offset + x.shape[1] < self.max_len
        pos_emb = self.pe[:, offset:offset + T]
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset: int, size: int) -> paddle.Tensor:
        assert offset + size < self.max_len
        return self.dropout(self.pe[:, offset:offset + size])


class RelPositionalEncoding(PositionalEncoding):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        super().__init__(d_model, dropout_rate, max_len, reverse=True)

    def forward(self, x: paddle.Tensor, offset: int = 0):
        assert offset + x.shape[1] < self.max_len
        x = x * self.xscale
        pos_emb = self.pe[:, offset:offset + x.shape[1]]
        return self.dropout(x), self.dropout(pos_emb)


def make_non_pad_mask(lengths):
    batch_size = int(lengths.shape[0])
    max_len = int(lengths.max())
    seq_range = paddle.arange(0, max_len, dtype=paddle.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand([batch_size, max_len])

    seq_length_expand = lengths.unsqueeze(-1)
    seq_length_expand = seq_length_expand.astype(paddle.int64)

    mask = seq_range_expand >= seq_length_expand
    return mask.logical_not()


def subsequent_mask(size: int) -> paddle.Tensor:
    ret = paddle.ones([size, size], dtype=paddle.bool)
    ret = ret.astype(paddle.float)
    ret = paddle.tril(ret)
    ret = ret.astype(paddle.bool)
    return ret


def subsequent_chunk_mask(size: int, chunk_size: int, num_left_chunks: int = -1, ):
    ret = paddle.zeros([size, size], dtype=paddle.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max(0, (i // chunk_size - num_left_chunks) * chunk_size)
        ending = min(size, (i // chunk_size + 1) * chunk_size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(xs: paddle.Tensor, masks: paddle.Tensor, use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool, decoding_chunk_size: int,
                            static_chunk_size: int, num_decoding_left_chunks: int):
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.shape[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            chunk_size = int(paddle.randint(1, max_len, (1,)))
            num_left_chunks = -1
            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = int(
                        paddle.randint(0, max_left_chunks, (1,)))
        chunk_masks = subsequent_chunk_mask(xs.shape[1], chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        # chunk_masks = masks & chunk_masks  # (B, L, L)
        chunk_masks = masks.logical_and(chunk_masks)  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.shape[1], static_chunk_size,
                                            num_left_chunks)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        # chunk_masks = masks & chunk_masks  # (B, L, L)
        chunk_masks = masks.logical_and(chunk_masks)  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


class MultiHeadedAttention(nn.Layer):
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self,
                    query: paddle.Tensor,
                    key: paddle.Tensor,
                    value: paddle.Tensor
                    ):

        n_batch = query.shape[0]
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose([0, 2, 1, 3])  # (batch, head, time1, d_k)
        k = k.transpose([0, 2, 1, 3])  # (batch, head, time2, d_k)
        v = v.transpose([0, 2, 1, 3])  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self,
                          value: paddle.Tensor,
                          scores: paddle.Tensor,
                          mask) -> paddle.Tensor:
        n_batch = value.shape[0]
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))

            attn = paddle.nn.functional.softmax(scores, axis=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = paddle.nn.functional.softmax(scores, axis=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = paddle.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose([0, 2, 1, 3]).view(n_batch, -1, self.h *
                                           self.d_k)  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self,
                query: paddle.Tensor,
                key: paddle.Tensor,
                value: paddle.Tensor,
                mask) -> paddle.Tensor:

        q, k, v = self.forward_qkv(query, key, value)
        scores = paddle.matmul(q,
                               k.transpose([0, 1, 3, 2])) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, n_head, n_feat, dropout_rate):
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias_attr=False)
        pos_bias_u = self.create_parameter([self.h, self.d_k])
        self.add_parameter('pos_bias_u', pos_bias_u)
        pos_bias_v = self.create_parameter([self.h, self.d_k])
        self.add_parameter('pos_bias_v', pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        zero_pad = paddle.zeros((x.shape[0], x.shape[1], x.shape[2], 1), dtype=x.dtype)
        x_padded = paddle.concat([zero_pad, x], axis=-1)

        x_padded = x_padded.view(x.shape[0], x.shape[1], x.shape[3] + 1, x.shape[2])
        x = x_padded[:, :, 1:].reshape(x.size())  # [B, H, T1, T1]

        if zero_triu:
            ones = paddle.ones((x.shape[2], x.shape[3]))
            x = x * paddle.tril(ones, x.shape[3] - x.shape[2])[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose([0, 2, 1, 3])

        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose([0, 2, 1, 3])

        q_with_bias_u = (q + self.pos_bias_u).transpose([0, 2, 1, 3])
        q_with_bias_v = (q + self.pos_bias_v).transpose([0, 2, 1, 3])

        matrix_ac = paddle.matmul(q_with_bias_u, k.transpose([0, 1, 3, 2]))
        matrix_bd = paddle.matmul(q_with_bias_v, p.transpose([0, 1, 3, 2]))

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)


class PositionwiseFeedForward(nn.Layer):
    """Positionwise feed forward layer."""

    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 activation: nn.Layer = nn.ReLU()):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs: paddle.Tensor) -> paddle.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class BaseSubsampling(nn.Layer):
    def __init__(self, pos_enc_class: nn.Layer = PositionalEncoding):
        super().__init__()
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> paddle.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer = PositionalEncoding):
        super().__init__(pos_enc_class)
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, epsilon=1e-12),
            nn.Dropout(dropout_rate),
            nn.ReLU(), )
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int = 0):
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer = PositionalEncoding):
        super().__init__(pos_enc_class)
        self.conv = nn.Sequential(
            nn.Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2D(odim, odim, 3, 2),
            nn.ReLU(), )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        self.subsampling_rate = 4
        # The right context for every conv layer is computed by:
        self.right_context = 6

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset):
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = paddle.shape(x)
        x = self.out(x.transpose([0, 2, 1, 3]).reshape([b, t, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length)."""

    def __init__(self,
                 idim: int,
                 odim: int,
                 dropout_rate: float,
                 pos_enc_class: nn.Layer = PositionalEncoding):
        super().__init__(pos_enc_class)
        self.conv = nn.Sequential(
            nn.Conv2D(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2D(odim, odim, 5, 3),
            nn.ReLU(), )
        self.linear = nn.Linear(odim * (((idim - 1) // 2 - 2) // 3), odim)
        # The right context for every conv layer is computed by:
        self.subsampling_rate = 6
        self.right_context = 10

    def forward(self, x: paddle.Tensor, x_mask: paddle.Tensor, offset: int = 0):
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = paddle.shape(x)
        x = self.linear(x.transpose([0, 2, 1, 3]).reshape([b, t, c * f]))
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-4:3]


class ConvolutionModule(nn.Layer):
    """ConvolutionModule in Conformer model."""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Layer = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):

        assert check_argument_types()
        super().__init__()
        self.pointwise_conv1 = nn.Conv1D(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None
            if bias else False,  # None for True, using bias as default config
        )

        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        self.depthwise_conv = nn.Conv1D(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias_attr=None
            if bias else False,  # None for True, using bias as default config
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1D(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1D(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=None
            if bias else False,  # None for True, using bias as default config
        )
        self.activation = activation

    def forward(self,
                x: paddle.Tensor,
                mask_pad=None,
                cache=None
                ):

        # exchange the temporal dimension and the feature dimension
        x = x.transpose([0, 2, 1])  # [B, C, T]

        # mask batch padding
        if mask_pad is not None:
            x = x.masked_fill(mask_pad, 0.0)

        if self.lorder > 0:
            if cache is None:
                x = nn.functional.pad(
                    x, [self.lorder, 0], 'constant', 0.0, data_format='NCL')
            else:
                assert cache.shape[0] == x.shape[0]  # B
                assert cache.shape[1] == x.shape[1]  # C
                x = paddle.concat((cache, x), axis=2)

            assert (x.shape[2] > self.lorder)
            new_cache = x[:, :, -self.lorder:]  # [B, C, T]
        else:
            new_cache = paddle.zeros([1], dtype=x.dtype)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, axis=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose([0, 2, 1])  # [B, T, C]
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose([0, 2, 1])  # [B, C, T]
        x = self.pointwise_conv2(x)

        # mask batch padding
        if mask_pad is not None:
            x = x.masked_fill(mask_pad, 0.0)

        x = x.transpose([0, 2, 1])  # [B, T, C]
        return x, new_cache


class TransformerEncoderLayer(nn.Layer):
    """Encoder layer module."""

    def __init__(
            self,
            size: int,
            self_attn: nn.Layer,
            feed_forward: nn.Layer,
            dropout_rate: float,
            normalize_before: bool = True,
            concat_after: bool = False, ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, epsilon=1e-12)
        self.norm2 = nn.LayerNorm(size, epsilon=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = nn.Linear(size + size, size)

    def forward(
            self,
            x: paddle.Tensor,
            mask: paddle.Tensor,
            pos_emb=None,
            mask_pad=None,
            output_cache=None,
            cnn_cache=None,
    ):
        residual = x
        if self.normalize_before:
            x = self.norm1(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.shape[0] == x.shape[0]
            assert output_cache.shape[1] < x.shape[1]
            assert output_cache.shape[2] == self.size
            chunk = x.shape[1] - output_cache.shape[1]
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        if self.concat_after:
            x_concat = paddle.concat(
                (x, self.self_attn(x_q, x, x, mask)), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x_q, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        if output_cache is not None:
            x = paddle.concat([output_cache, x], axis=1)

        fake_cnn_cache = paddle.zeros([1], dtype=x.dtype)
        return x, mask, fake_cnn_cache


class ConformerEncoderLayer(nn.Layer):
    """Encoder layer module."""

    def __init__(
            self,
            size: int,
            self_attn: nn.Layer,
            feed_forward=None,
            feed_forward_macaron=None,
            conv_module=None,
            dropout_rate: float = 0.1,
            normalize_before: bool = True,
            concat_after: bool = False, ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, epsilon=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, epsilon=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, epsilon=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(
                size, epsilon=1e-12)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, epsilon=1e-12)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear = nn.Linear(size + size, size)

    def forward(
            self,
            x: paddle.Tensor,
            mask: paddle.Tensor,
            pos_emb: paddle.Tensor,
            mask_pad=None,
            output_cache=None,
            cnn_cache=None,
    ):
        # whether to use macaron style FFN
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if output_cache is None:
            x_q = x
        else:
            assert output_cache.shape[0] == x.shape[0]
            assert output_cache.shape[1] < x.shape[1]
            assert output_cache.shape[2] == self.size
            chunk = x.shape[1] - output_cache.shape[1]
            x_q = x[:, -chunk:, :]
            residual = residual[:, -chunk:, :]
            mask = mask[:, -chunk:, :]

        x_att = self.self_attn(x_q, x, x, pos_emb, mask)

        if self.concat_after:
            x_concat = paddle.concat((x, x_att), axis=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)

        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = paddle.zeros([1], dtype=x.dtype)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)

            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if output_cache is not None:
            x = paddle.concat([output_cache, x], axis=1)

        return x, mask, new_cnn_cache


class BaseEncoder(nn.Layer):
    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: str = "conv2d",
            pos_enc_layer_type: str = "abs_pos",
            normalize_before: bool = True,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_chunk: bool = False,
            global_cmvn: paddle.nn.Layer = None,
            use_dynamic_left_chunk: bool = False, ):

        assert check_argument_types()
        super().__init__()
        self._output_size = output_size

        pos_enc_class = RelPositionalEncoding

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            idim=input_size,
            odim=output_size,
            dropout_rate=dropout_rate,
            pos_enc_class=pos_enc_class(
                d_model=output_size, dropout_rate=positional_dropout_rate), )

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, epsilon=1e-12)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: paddle.Tensor,
            xs_lens: paddle.Tensor,
            decoding_chunk_size: int = 0,
            num_decoding_left_chunks: int = -1,
    ):
        masks = make_non_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, L)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks.astype(xs.dtype), offset=0)
        masks = masks.astype(paddle.bool)
        mask_pad = masks.logical_not()
        chunk_masks = add_optional_chunk_mask(
            xs, masks, self.use_dynamic_chunk, self.use_dynamic_left_chunk,
            decoding_chunk_size, self.static_chunk_size,
            num_decoding_left_chunks)
        for layer in self.encoders:
            xs, chunk_masks, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_chunk(
            self,
            xs: paddle.Tensor,
            offset: int,
            required_cache_size: int,
            subsampling_cache=None,
            elayers_output_cache=None,
            conformer_cnn_cache=None, ):
        # tmp_masks is just for interface compatibility
        tmp_masks = paddle.ones([1, xs.shape[1]], dtype=paddle.int32)
        tmp_masks = tmp_masks.unsqueeze(1)  # [B=1, C=1, T]

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset=offset)  # xs=(B, T, D), pos_emb=(B=1, T, D)

        if subsampling_cache is not None:
            cache_size = subsampling_cache.shape[1]  # T
            xs = paddle.concat((subsampling_cache, xs), axis=1)
        else:
            cache_size = 0

        # only used when using `RelPositionMultiHeadedAttention`
        pos_emb = self.embed.position_encoding(offset=offset - cache_size, size=xs.shape[1])

        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = xs.shape[1]
        else:
            next_cache_start = xs.shape[1] - required_cache_size
        r_subsampling_cache = xs[:, next_cache_start:, :]

        # Real mask for transformer/conformer layers
        masks = paddle.ones([1, xs.shape[1]], dtype=paddle.bool)
        masks = masks.unsqueeze(1)  # [B=1, L'=1, T]
        r_elayers_output_cache = []
        r_conformer_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            attn_cache = None if elayers_output_cache is None else elayers_output_cache[i]
            cnn_cache = None if conformer_cnn_cache is None else conformer_cnn_cache[i]
            xs, _, new_cnn_cache = layer(xs, masks, pos_emb, output_cache=attn_cache, cnn_cache=cnn_cache)
            r_elayers_output_cache.append(xs[:, next_cache_start:, :])
            r_conformer_cnn_cache.append(new_cnn_cache)
        if self.normalize_before:
            xs = self.after_norm(xs)

        return (xs[:, cache_size:, :], r_subsampling_cache,
                r_elayers_output_cache, r_conformer_cnn_cache)

    def forward_chunk_by_chunk(
            self,
            xs: paddle.Tensor,
            decoding_chunk_size: int,
            num_decoding_left_chunks: int = -1,
    ):
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk

        # feature stride and window for `subsampling` module
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context

        num_frames = xs.shape[1]
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks
        subsampling_cache = None
        elayers_output_cache = None
        conformer_cnn_cache = None
        outputs = []
        offset = 0
        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, subsampling_cache, elayers_output_cache,
             conformer_cnn_cache) = self.forward_chunk(
                chunk_xs, offset, required_cache_size, subsampling_cache,
                elayers_output_cache, conformer_cnn_cache)
            outputs.append(y)
            offset += y.shape[1]
        ys = paddle.concat(outputs, 1)
        # fake mask, just for jit script and compatibility with `forward` api
        masks = paddle.ones([1, ys.shape[1]], dtype=paddle.bool)
        masks = masks.unsqueeze(1)
        return ys, masks


class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""

    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: str = "conv2d",
            pos_enc_layer_type: str = "abs_pos",
            normalize_before: bool = True,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_chunk: bool = False,
            global_cmvn: nn.Layer = None,
            use_dynamic_left_chunk: bool = False, ):
        assert check_argument_types()
        super().__init__(input_size, output_size, attention_heads, linear_units,
                         num_blocks, dropout_rate, positional_dropout_rate,
                         attention_dropout_rate, input_layer,
                         pos_enc_layer_type, normalize_before, concat_after,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk)
        self.encoders = nn.LayerList([
            TransformerEncoderLayer(
                size=output_size,
                self_attn=MultiHeadedAttention(attention_heads, output_size,
                                               attention_dropout_rate),
                feed_forward=PositionwiseFeedForward(output_size, linear_units,
                                                     dropout_rate),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after) for _ in range(num_blocks)
        ])

    def forward_one_step(
            self,
            xs: paddle.Tensor,
            masks: paddle.Tensor,
            cache=None):
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        xs, pos_emb, masks = self.embed(xs, masks.astype(xs.dtype), offset=0)
        masks = masks.astype(paddle.bool)

        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks, _ = e(xs, masks, output_cache=c)
            new_cache.append(xs)
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks, new_cache


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""

    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: str = "conv2d",
            pos_enc_layer_type: str = "rel_pos",
            normalize_before: bool = True,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_chunk: bool = False,
            global_cmvn: nn.Layer = None,
            use_dynamic_left_chunk: bool = False,
            positionwise_conv_kernel_size: int = 1,
            macaron_style: bool = True,
            selfattention_layer_type: str = "rel_selfattn",
            activation_type: str = "swish",
            use_cnn_module: bool = True,
            cnn_module_kernel: int = 15,
            causal: bool = False,
            cnn_module_norm: str = "batch_norm", ):
        assert check_argument_types()
        super().__init__(input_size, output_size, attention_heads, linear_units,
                         num_blocks, dropout_rate, positional_dropout_rate,
                         attention_dropout_rate, input_layer,
                         pos_enc_layer_type, normalize_before, concat_after,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk)

        activation = paddle.nn.Swish()

        # self-attention module definition
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size,
                                       attention_dropout_rate)
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate,
                                   activation)
        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        self.encoders = nn.LayerList([
            ConformerEncoderLayer(
                size=output_size,
                self_attn=encoder_selfattn_layer(*encoder_selfattn_layer_args),
                feed_forward=positionwise_layer(*positionwise_layer_args),
                feed_forward_macaron=positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                conv_module=convolution_layer(*convolution_layer_args)
                if use_cnn_module else None,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after) for _ in range(num_blocks)
        ])


class DecoderLayer(nn.Layer):
    def __init__(
            self,
            size: int,
            self_attn: nn.Layer,
            src_attn: nn.Layer,
            feed_forward: nn.Layer,
            dropout_rate: float,
            normalize_before: bool = True,
            concat_after: bool = False, ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, epsilon=1e-12)
        self.norm2 = nn.LayerNorm(size, epsilon=1e-12)
        self.norm3 = nn.LayerNorm(size, epsilon=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.concat_linear1 = nn.Linear(size + size, size)
        self.concat_linear2 = nn.Linear(size + size, size)

    def forward(
            self,
            tgt: paddle.Tensor,
            tgt_mask: paddle.Tensor,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            cache=None
    ):
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            assert cache.shape == [
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ], f"{cache.shape} == {[tgt.shape[0], tgt.shape[1] - 1, self.size]}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask.cast(paddle.int64)[:, -1:, :].cast(
                paddle.bool)

        if self.concat_after:
            tgt_concat = paddle.concat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), axis=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(
                self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = paddle.concat(
                (x, self.src_attn(x, memory, memory, memory_mask)), axis=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(
                self.src_attn(x, memory, memory, memory_mask))
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = paddle.concat([cache, x], axis=1)

        return x, tgt_mask, memory, memory_mask


class TransformerDecoder(nn.Layer):
    def __init__(
            self,
            vocab_size: int,
            encoder_output_size: int,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            self_attention_dropout_rate: float = 0.0,
            src_attention_dropout_rate: float = 0.0,
            input_layer: str = "embed",
            use_output_layer: bool = True,
            normalize_before: bool = True,
            concat_after: bool = False, ):

        assert check_argument_types()
        nn.Layer.__init__(self)
        self.selfattention_layer_type = 'selfattn'
        attention_dim = encoder_output_size

        if input_layer == "embed":
            self.embed = nn.Sequential(
                nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate), )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(attention_dim, epsilon=1e-12)
        self.use_output_layer = use_output_layer
        self.output_layer = nn.Linear(attention_dim, vocab_size)

        self.decoders = nn.LayerList([
            DecoderLayer(
                size=attention_dim,
                self_attn=MultiHeadedAttention(attention_heads, attention_dim,
                                               self_attention_dropout_rate),
                src_attn=MultiHeadedAttention(attention_heads, attention_dim,
                                              src_attention_dropout_rate),
                feed_forward=PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after, ) for _ in range(num_blocks)
        ])

    def forward(
            self,
            memory: paddle.Tensor,
            memory_mask: paddle.Tensor,
            ys_in_pad: paddle.Tensor,
            ys_in_lens: paddle.Tensor, ):
        tgt = ys_in_pad
        # tgt_mask: (B, 1, L)
        tgt_mask = (make_non_pad_mask(ys_in_lens).unsqueeze(1))
        # m: (1, L, L)
        m = subsequent_mask(tgt_mask.shape[-1]).unsqueeze(0)
        # tgt_mask: (B, L, L)
        # tgt_mask = tgt_mask & m
        tgt_mask = tgt_mask.logical_and(m)

        x, _ = self.embed(tgt)
        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory,
                                                     memory_mask)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)

        olens = tgt_mask.astype(paddle.int).sum(1)
        return x, olens


def pad_sequence(sequences, batch_first=False, padding_value=0.0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:] if max_size.ndim >= 2 else ()
    max_len = max([s.shape[0] for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = paddle.full(out_dims, padding_value, dtype=sequences[0].dtype)
    for i, tensor in enumerate(sequences):
        length = tensor.shape[0]
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            if length != 0:
                out_tensor[i, :length] = tensor
            else:
                out_tensor[i, length] = tensor
        else:
            if length != 0:
                out_tensor[:length, i] = tensor
            else:
                out_tensor[length, i] = tensor

    return out_tensor


def add_sos_eos(ys_pad, sos, eos, ignore_id):
    B = ys_pad.shape[0]
    _sos = paddle.ones([B, 1], dtype=ys_pad.dtype) * sos
    _eos = paddle.ones([B, 1], dtype=ys_pad.dtype) * eos
    ys_in = paddle.concat([_sos, ys_pad], axis=1)
    mask_pad = (ys_in == ignore_id)
    ys_in = ys_in.masked_fill(mask_pad, eos)

    ys_out = paddle.concat([ys_pad, _eos], axis=1)
    ys_out = ys_out.masked_fill(mask_pad, eos)
    mask_eos = (ys_out == ignore_id)
    ys_out = ys_out.masked_fill(mask_eos, eos)
    ys_out = ys_out.masked_fill(mask_pad, ignore_id)
    return ys_in, ys_out


def log_add(args):
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


class CTCDecoder(nn.Layer):
    def __init__(self,
                 odim,
                 enc_n_units,
                 blank_id=0,
                 dropout_rate: float = 0.0):
        super().__init__()

        self.blank_id = blank_id
        self.odim = odim
        self.dropout = nn.Dropout(dropout_rate)
        self.ctc_lo = nn.Linear(enc_n_units, self.odim)

    def log_softmax(self, hs_pad, temperature: float = 1.0):
        return F.log_softmax(self.ctc_lo(hs_pad) / temperature, axis=2)


class U2BaseModel(nn.Layer):
    @classmethod
    def params(cls, config=None) -> CfgNode:
        # network architecture
        default = CfgNode()
        default.cmvn_file = ""
        default.cmvn_file_type = "json"
        default.input_dim = 0
        default.output_dim = 0
        default.encoder = 'transformer'
        default.encoder_conf = CfgNode(
            dict(
                output_size=256,  # dimension of attention
                attention_heads=4,
                linear_units=2048,  # the number of units of position-wise feed forward
                num_blocks=12,  # the number of encoder blocks
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.0,
                input_layer='conv2d',  # encoder input type, you can chose conv2d, conv2d6 and conv2d8
                normalize_before=True,
            ))
        # decoder related
        default.decoder = 'transformer'
        default.decoder_conf = CfgNode(
            dict(
                attention_heads=4,
                linear_units=2048,
                num_blocks=6,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                self_attention_dropout_rate=0.0,
                src_attention_dropout_rate=0.0, ))
        # hybrid CTC/attention
        default.model_conf = CfgNode(
            dict(
                ctc_weight=0.3,
                lsm_weight=0.1,  # label smoothing option
                length_normalized_loss=False, ))

        if config is not None:
            config.merge_from_other_cfg(default)
        return default

    def __init__(self,
                 vocab_size: int,
                 encoder: TransformerEncoder,
                 decoder: TransformerDecoder,
                 ctc: CTCDecoder,
                 ctc_weight: float = 0.5,
                 ignore_id: int = IGNORE_ID,
                 **kwargs):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        nn.Layer.__init__(self)
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc

    def _forward_encoder(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
    ):

        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )
        return encoder_out, encoder_mask

    def _ctc_prefix_beam_search(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            beam_size: int,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            simulate_streaming: bool = False,
            blank_id: int = 0, ):

        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1

        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.shape[1]
        ctc_probs = self.ctc.log_softmax(encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)

        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.numpy().item()
                ps = logp[s].numpy().item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == blank_id:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(
                next_hyps.items(),
                key=lambda x: log_add(list(x[1])),
                reverse=True)
            cur_hyps = next_hyps[:beam_size]

        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out

    def attention_rescoring(
            self,
            speech: paddle.Tensor,
            speech_lengths: paddle.Tensor,
            beam_size: int,
            decoding_chunk_size: int = -1,
            num_decoding_left_chunks: int = -1,
            ctc_weight: float = 0.0,
            simulate_streaming: bool = False, ):

        device = speech.place
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1

        hyps, encoder_out = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)
        assert len(hyps) == beam_size

        hyps_pad = pad_sequence([
            paddle.to_tensor(hyp[0], place=device, dtype=paddle.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        hyps_lens = paddle.to_tensor([len(hyp[0]) for hyp in hyps], place=device, dtype=paddle.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining

        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = paddle.ones(
            (beam_size, 1, encoder_out.shape[1]), dtype=paddle.bool)
        decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad,
            hyps_lens)  # (beam_size, max_hyps_len, vocab_size)
        # ctc score in ln domain
        decoder_out = paddle.nn.functional.log_softmax(decoder_out, axis=-1)
        decoder_out = decoder_out.numpy()

        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            # last decoder output token is `eos`, for laste decoder input token.
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add ctc score (which in ln domain)
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0]

    @paddle.no_grad()
    def decode(self,
               feats: paddle.Tensor,
               feats_lengths: paddle.Tensor,
               text_feature,
               decoding_method: str,
               beam_size: int,
               ctc_weight: float = 0.0,
               decoding_chunk_size: int = -1,
               num_decoding_left_chunks: int = -1,
               simulate_streaming: bool = False):

        batch_size = feats.shape[0]
        if decoding_method in ['ctc_prefix_beam_search', 'attention_rescoring'] and batch_size > 1:
            sys.exit(1)

        assert feats.shape[0] == 1
        hyp = self.attention_rescoring(
            feats,
            feats_lengths,
            beam_size,
            decoding_chunk_size=decoding_chunk_size,
            num_decoding_left_chunks=num_decoding_left_chunks,
            ctc_weight=ctc_weight,
            simulate_streaming=simulate_streaming)
        hyps = [hyp]

        res = [text_feature.defeaturize(hyp) for hyp in hyps]
        res_tokenids = [hyp for hyp in hyps]
        return res, res_tokenids


class U2Model(U2BaseModel):
    def __init__(self, configs: dict):
        vocab_size, encoder, decoder, ctc = U2Model._init_from_config(configs)

        super().__init__(
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            **configs['model_conf'])

    @classmethod
    def _init_from_config(cls, configs: dict):
        # input & output dim
        input_dim = configs['input_dim']
        vocab_size = configs['output_dim']
        assert input_dim != 0, input_dim
        assert vocab_size != 0, vocab_size

        # conformer encoder
        encoder = ConformerEncoder(input_dim, global_cmvn=None, **configs['encoder_conf'])

        # decoder
        decoder = TransformerDecoder(vocab_size, encoder.output_size(), **configs['decoder_conf'])

        # ctc decoder and ctc loss
        model_conf = configs['model_conf']
        dropout_rate = model_conf.get('ctc_dropout_rate', 0.0)

        ctc = CTCDecoder(
            odim=vocab_size,
            enc_n_units=encoder.output_size(),
            blank_id=0,
            dropout_rate=dropout_rate)

        return vocab_size, encoder, decoder, ctc

    @classmethod
    def from_config(cls, configs: dict):
        model = cls(configs)
        return model


@contextmanager
def UpdateConfig(config):
    config.defrost()
    yield
    config.freeze()


class ASRExecutor:
    def __init__(self, device):
        super(ASRExecutor, self).__init__()

        self._inputs = dict()
        self._outputs = dict()

        paddle.set_device(device)
        self._init_from_path()

    def _init_from_path(self):
        res_path = "./conformer.model.tar"  # wenetspeech_zh
        self.res_path = res_path

        self.cfg_path = os.path.join(res_path, "conf/conformer.yaml")
        self.ckpt_path = os.path.join(res_path, "exp/conformer/checkpoints/wenetspeech" + ".pdparams")

        # Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)
        self.config.decoding.decoding_method = "attention_rescoring"

        with UpdateConfig(self.config):
            self.config.collator.vocab_filepath = os.path.join(
                res_path, self.config.collator.vocab_filepath)
            self.config.collator.augmentation_config = os.path.join(
                res_path, self.config.collator.augmentation_config)
            self.config.collator.spm_model_prefix = os.path.join(
                res_path, self.config.collator.spm_model_prefix)
            text_feature = TextFeaturizer(
                unit_type=self.config.collator.unit_type,
                vocab_filepath=self.config.collator.vocab_filepath,
                spm_model_prefix=self.config.collator.spm_model_prefix)
            self.config.model.input_dim = self.config.collator.feat_dim
            self.config.model.output_dim = text_feature.vocab_size

        self.model = U2Model.from_config(self.config.model)
        self.model.eval()

        # load model
        model_dict = paddle.load(self.ckpt_path)
        self.model.set_state_dict(model_dict)

    def preprocess(self, audio_file):
        preprocess_conf_file = self.config.collator.augmentation_config
        # redirect the cmvn path
        with io.open(preprocess_conf_file, encoding="utf-8") as f:
            preprocess_conf = yaml.safe_load(f)

            for idx, process in enumerate(preprocess_conf["process"]):
                if process['type'] == "cmvn_json":
                    preprocess_conf["process"][idx]["cmvn_path"] = os.path.join(
                        self.res_path,
                        preprocess_conf["process"][idx]["cmvn_path"])
                    break

        preprocessing = Transformation(preprocess_conf)
        audio, audio_sample_rate = soundfile.read(audio_file, dtype="int16", always_2d=True)
        audio = audio[:, 0]

        # fbank
        audio = preprocessing(audio)

        audio_len = paddle.to_tensor(audio.shape[0])
        audio = paddle.to_tensor(audio, dtype='float32').unsqueeze(axis=0)

        self._inputs["audio"] = audio
        self._inputs["audio_len"] = audio_len

    @paddle.no_grad()
    def infer(self):
        text_feature = TextFeaturizer(
            unit_type=self.config.collator.unit_type,
            vocab_filepath=self.config.collator.vocab_filepath,
            spm_model_prefix=self.config.collator.spm_model_prefix)
        cfg = self.config.decoding
        audio = self._inputs["audio"]
        audio_len = self._inputs["audio_len"]

        result_transcripts = self.model.decode(
            audio,
            audio_len,
            text_feature=text_feature,
            decoding_method=cfg.decoding_method,
            beam_size=cfg.beam_size,
            ctc_weight=cfg.ctc_weight,
            decoding_chunk_size=cfg.decoding_chunk_size,
            num_decoding_left_chunks=cfg.num_decoding_left_chunks,
            simulate_streaming=cfg.simulate_streaming)
        return result_transcripts[0][0]

    def p_infer(self, audio_file):
        self.preprocess(audio_file)
        return self.infer()
