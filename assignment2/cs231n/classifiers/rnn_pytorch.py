import numpy as np
import torch
from ..rnn_layers_pytorch import *


class CaptioningRNN:
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=torch.float32,
    ):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        - cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        - dtype: numpy datatype to use; use float32 for training and float64 for
          numeric gradient checking.
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # Initialize CNN -> hidden state projection parameters
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = torch.zeros(hidden_dim)

        # Initialize parameters for the RNN
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = torch.zeros(dim_mul * hidden_dim)

        # Initialize output to vocab weights
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size)

        # Cast parameters to correct dtype
        for k, v in self.params.items():
            self.params[k] = v.to(self.dtype)

    def loss(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an RNN (or LSTM) to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Ground-truth captions; an integer array of shape (N, T + 1) where
          each element is in the range 0 <= y[i, t] < V

        Returns a tuple of:
        - loss: Scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last word
        # and will be input to the RNN; captions_out has everything but the first
        # word and this is what we will expect the RNN to generate. These are offset
        # by one relative to each other because the RNN should produce word (t+1)
        # after receiving word t. The first element of captions_in will be the START
        # token, and the first element of captions_out will be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # You'll need this
        mask = captions_out != self._null

        # Weight and bias for the affine transform from image features to initial
        # hidden state
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # Word embedding matrix
        W_embed = self.params["W_embed"]

        # Input-to-hidden, hidden-to-hidden, and biases for the RNN
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # Weight and bias for the hidden-to-vocab transformation.
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss = 0.0
        # (1) features -> initial hidden state
        h0 = affine_forward(features, W_proj, b_proj)

        # (2) word embedding
        x = word_embedding_forward(captions_in, W_embed)

        # (3) recurrent forward
        if self.cell_type == "rnn":
            h = rnn_forward(x, h0, Wx, Wh, b)
        else:
            h = lstm_forward(x, h0, Wx, Wh, b)

        # (4) hidden states -> vocab scores
        scores = temporal_affine_forward(h, W_vocab, b_vocab)

        # (5) temporal softmax loss (with mask)
        loss = temporal_softmax_loss(scores, captions_out, mask)
        return loss

    def sample(self, features, max_length=30):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the input image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Inputs:
        - features: Array of input image features of shape (N, D).
        - max_length: Maximum length T of generated captions.

        Returns:
        - captions: Array of shape (N, max_length) giving sampled captions,
          where each element is an integer in the range [0, V). The first element
          of captions should be the first sampled word, not the <START> token.
        """
        N = features.shape[0]
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)

        # Unpack parameters
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        h = affine_forward(features, W_proj, b_proj)
        c = torch.zeros_like(h) if self.cell_type == "lstm" else None

        prev_word = torch.full((N,), self._start, dtype=torch.long, device=features.device)
        captions = captions.to(features.device)

        for t in range(max_length):
            word_vec = W_embed[prev_word]

            if self.cell_type == "rnn":
                h = rnn_step_forward(word_vec, h, Wx, Wh, b)
            else:
                h, c = lstm_step_forward(word_vec, h, c, Wx, Wh, b)

            scores = affine_forward(h, W_vocab, b_vocab)
            next_word = torch.argmax(scores, dim=1)
            captions[:, t] = next_word
            prev_word = next_word

        return captions
