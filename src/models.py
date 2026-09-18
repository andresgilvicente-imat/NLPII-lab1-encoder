import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionHead(nn.Module):
    """Single Attention Head.

    This class implements a single attention head which is part of the
    multi-head attention mechanism. It computes the attention for a given
    query, key, and value.

    Args:
        d_model (int): The dimension of the input embeddings.
        d_k (int): The dimension of the key vectors.
        d_q (int): The dimension of the query vectors.
        d_v (int): The dimension of the value vectors.
    Attributes:
        wq (nn.Linear): Linear layer to project input to query vectors.
        wk (nn.Linear): Linear layer to project input to key vectors.
        wv (nn.Linear): Linear layer to project input to value vectors.
    """

    def __init__(self, d_model: int, d_k: int, d_q: int, d_v: int):
        super(AttentionHead, self).__init__()

        self.wq = nn.Linear(in_features=d_model, out_features=d_q)
        self.wk = nn.Linear(in_features=d_model, out_features=d_k)
        self.wv = nn.Linear(in_features=d_model, out_features=d_v)

    def scaled_dot_product_attention(self, q, k, v):
        """Calculate the attention weights.

        Args:
            q (Tensor): Query tensor of shape (batch_size, seq_len, d_q).
            k (Tensor): Key tensor of shape (batch_size, seq_len, d_k).
            v (Tensor): Value tensor of shape (batch_size, seq_len, d_v).

        Returns:
            Tensor: Output tensor after applying attention.
            Tensor: Attention weights.
        """

        # The dimension of the key tensor, used to scale the scores.
        dim_k = k.size()[2]

        # Calculate the dot product between query and the transpose of key.
        # The result is then scaled by the square root of dim_k.
        scores = torch.bmm(q, torch.transpose(k, 1, 2)) / math.sqrt(dim_k)  # (q @ k.T) / sqrt(dim_k)

        # Apply the softmax function to obtain the attention weights.
        weights = torch.softmax(scores, dim=2)

        # Compute the output by performing a weighted sum of the value tensor
        # using the attention weights.
        output = torch.bmm(weights, v) 

        return output, weights

    def forward(self, x):
        """Forward pass for the attention head.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_v).
        """
        # Obtain the corresponding query, key, and value vectors of the input tensor.
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        output, _ = self.scaled_dot_product_attention(q, k, v)

        return output

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism.

    This class implements the multi-head attention mechanism, which allows
    the model to focus on different parts of the input sequence at each layer.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads.

    Attributes:
        heads (nn.ModuleList): A list of attention heads.
        output_linear (nn.Linear): Linear layer to project concatenated heads back to d_model.
    """

    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadAttention, self).__init__()

        # we store these to make the division check in the forward
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads

        # d_k = d_q = d_v = fraction (d_model)
        d_k = d_model // num_attention_heads
        d_q, d_v = d_k, d_k

        self.heads = nn.ModuleList(
            [ 
                AttentionHead(d_model, d_k, d_q, d_v ) 
                for _ in range(num_attention_heads) 
            ]
        )

        self.output_linear = nn.Linear(
            in_features=num_attention_heads*d_v, 
            out_features=d_model
        )

    def forward(self, hidden_state):
        """Forward pass for the multi-head attention layer.

        Args:
            hidden_state (Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """

        if self.d_model % self.num_attention_heads != 0:
            raise RuntimeError("d_model must be divisible by num_attention_heads")

        heads_outputs = [head(hidden_state) for head in self.heads]
        concatenated_outputs = torch.cat(heads_outputs, dim=2)
        x = self.output_linear(concatenated_outputs)

        return x
    
class FeedForward(nn.Module):
    """FeedForward module for the Transformer.

    This class implements the feed-forward network used in the Transformer
    model. It consists of two linear layers with a GELU activation in between.

    Args:
        d_model (int): The dimension of the input and output embeddings.
        intermediate_size (int): The dimension of the intermediate layer.

    Attributes:
        linear_1 (nn.Linear): The first linear layer that projects from d_model to intermediate_size.
        linear_2 (nn.Linear): The second linear layer that projects from intermediate_size back to d_model.
        gelu (nn.GELU): GELU activation function applied after the first linear layer.
    """

    def __init__(self, d_model: int, intermediate_size: int):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(
            in_features=d_model,
            out_features=intermediate_size
        )

        self.linear_2 = nn.Linear(
            in_features=intermediate_size,
            out_features=d_model
        )

        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        output_1 = self.linear_1(x)
        output_2 = self.gelu(output_1)
        output_3 = self.linear_2(output_2)

        return output_3

class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer.

    This class implements a single layer of the Transformer encoder, consisting
    of a multi-head attention mechanism followed by a feed-forward neural network.
    Both sub-layers are surrounded by residual connections and layer normalization.

    Args:
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.

    Attributes:
        layer_norm_1 (nn.LayerNorm): Layer normalization applied before the multi-head attention.
        layer_norm_2 (nn.LayerNorm): Layer normalization applied before the feed-forward network.
        attention (MultiHeadAttention): Multi-head attention mechanism.
        feed_forward (FeedForward): Feed-forward neural network.
    """

    def __init__(self, d_model: int, num_attention_heads: int, intermediate_size: int):
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=d_model)
        self.attention = MultiHeadAttention(d_model=d_model, num_attention_heads=num_attention_heads)
        self.feed_forward = FeedForward(d_model=d_model, intermediate_size=intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply layer normalization and then apply multi-head attention
        hidden_state = self.layer_norm_1(x)
        output_attention = self.attention(hidden_state)

        residual_output_1 = x + output_attention
        
        # Apply layer normalization and then apply feed-forward network
        input_feed_forward = self.layer_norm_2(residual_output_1)
        output_feed_forward = self.feed_forward(input_feed_forward)

        residual_output_2 = output_feed_forward + residual_output_1
        
        return residual_output_2

class Embeddings(nn.Module):
    """Embeddings module for the Transformer.

    This module combines token embeddings and positional embeddings and applies
    layer normalization.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.

    Attributes:
        token_embeddings (nn.Embedding): Embedding layer for token embeddings.
        position_embeddings (nn.Embedding): Embedding layer for positional embeddings.
        layer_norm (nn.LayerNorm): Layer normalization applied after combining embeddings.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int):
        super(Embeddings, self).__init__()

        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

        self.position_embeddings = nn.Embedding(
            num_embeddings=max_position_embeddings,
            embedding_dim=d_model
        )

        self.layer_norm = nn.LayerNorm(
            normalized_shape=d_model
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to combine token and positional embeddings.

        Args:
            input_ids (torch.Tensor): Tensor containing input token IDs of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The combined and normalized embeddings of shape (batch_size, seq_len, d_model).
        """
        # Generate position IDs based on the input sequence length
        seq_length = input_ids.size()[1]
        position_ids = torch.tensor(range(seq_length)).to(input_ids.device)

        # Create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine token and position embeddings
        embeddings = self.layer_norm(token_embeddings + position_embeddings)

        return embeddings
    
class TransformerEncoder(nn.Module):
    """Transformer Encoder.

    This class implements the encoder part of the Transformer model, consisting
    of an embeddings layer followed by a stack of Transformer encoder layers.

    Args:
        vocab_size (int): The size of the vocabulary.
        max_position_embeddings (int): The maximum number of positions for positional embeddings.
        d_model (int): The dimension of the input embeddings.
        num_attention_heads (int): The number of attention heads in the multi-head attention mechanism.
        intermediate_size (int): The dimension of the feed-forward network's intermediate layer.
        num_hidden_layers (int): The number of Transformer encoder layers to stack.

    Attributes:
        embeddings (Embeddings): Embeddings layer combining token and positional embeddings.
        layers (nn.ModuleList): List of Transformer encoder layers.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int, d_model: int,
                num_attention_heads: int, intermediate_size: int, num_hidden_layers: int
                 ):
        super(TransformerEncoder, self).__init__()

        self.embeddings = Embeddings(
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
            d_model=d_model
        )

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size
                )
                
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Transformer encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """

        # Embeddings (including positional encoding)
        inputs = self.embeddings(x)

        output = inputs

        # apply each TransformerEncoderLayer
        for layer in self.layers:
            output = layer(output)
        
        return output
    
