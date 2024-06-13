import torch.nn.functional as F
import opensora
from opensora.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region


def VocabParallelEmbeddingForward(self, input_):
    if self.tensor_model_parallel_size > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | \
                     (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input *= ~input_mask
    else:
        masked_input = input_
        # Get the embeddings.
    output_parallel = F.embedding(masked_input, self.weight,
                                  self.padding_idx, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq,
                                  self.sparse)
    # Mask the output embedding.
    if self.tensor_model_parallel_size > 1:
        output_parallel *= ~input_mask[..., None]
    # Reduce across all the model parallel GPUs.
    output = reduce_from_tensor_model_parallel_region(output_parallel)
    return output


opensora.core.tensor_parallel.layers.VocabParallelEmbedding.forward = VocabParallelEmbeddingForward
