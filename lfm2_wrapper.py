import torch as t
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from utils.helpers import get_model_path, find_instruction_end_postion
from utils.llama_tokenize import (
    tokenize_lfm2_chat,
    ADD_FROM_POS_LFM2,
    IM_END,
)


class LFM2AttnWrapper(t.nn.Module):
    """
    Wrapper for LFM2 attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class LFM2BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for LFM2 attention blocks to save activations and unembed them
    Adapted for LFM2's different layer structure
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        # Wrap the self_attn in LFM2 attention layers
        if hasattr(self.block, 'self_attn'):
            self.block.self_attn = LFM2AttnWrapper(self.block.self_attn)

        # LFM2 uses operator_norm and ffn_norm instead of post_attention_layernorm
        self.operator_norm = self.block.operator_norm
        self.ffn_norm = self.block.ffn_norm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.from_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        # LFM2 layers return tensor directly, not tuple
        self.activations = output

        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (
                t.norm(last_token_activations) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append((top_token, dot_product.cpu().item()))

        if self.add_activations is not None:
            from utils.helpers import add_vector_from_position
            output = add_vector_from_position(
                matrix=output,
                vector=self.add_activations,
                position_ids=kwargs["position_ids"],
                from_pos=self.from_position,
            )

        if not self.save_internal_decodings:
            return output

        # Decode activations (adapted for LFM2)
        self.block_output_unembedded = self.unembed_matrix(self.norm(output))

        if hasattr(self.block, 'self_attn'):
            # Self-attention unembedded
            attn_output = self.block.self_attn.activations
            self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

            # Intermediate residual unembedded
            attn_output += args[0]
            self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

            # MLP unembedded (using ffn_norm for LFM2)
            mlp_output = self.block.feed_forward(self.ffn_norm(attn_output))
            self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations):
        self.add_activations = activations

    def reset(self):
        self.add_activations = None
        self.activations = None
        if hasattr(self.block, 'self_attn'):
            self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class LFM2Wrapper:
    """
    Wrapper for LFM2-1.2B model with hybrid conv-attention architecture.
    Only wraps the attention layers for CAA steering.
    """

    def __init__(
        self,
        hf_token: str,
        size: str = "1.2b",
        use_chat: bool = True,  # LFM2 always uses chat template
        override_model_weights_path: Optional[str] = None,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.use_chat = True  # LFM2 always uses chat template
        self.size = size
        self.model_name_path = get_model_path(size, not use_chat)

        # LFM2-1.2B has attention layers at indices [2, 5, 8, 10, 12, 14]
        self.attention_layer_indices = [2, 5, 8, 10, 12, 14]
        self.num_attention_layers = len(self.attention_layer_indices)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_path, token=hf_token
        )
        # Set pad_token to eos_token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_path, token=hf_token
        )

        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))

        # LFM2 models use bfloat16 but float16 works for inference
        # No need to apply half() - keep original precision
        self.model = self.model.to(self.device)

        # Set END_STR for LFM2 (ChatML format uses <|im_end|>)
        self.END_STR = t.tensor(self.tokenizer.encode(ADD_FROM_POS_LFM2)[1:]).to(
            self.device
        )

        # Only wrap the attention layers
        # LFM2 uses embedding_norm instead of norm
        self.wrapped_layers = {}
        for idx in self.attention_layer_indices:
            self.model.model.layers[idx] = LFM2BlockOutputWrapper(
                self.model.model.layers[idx],
                self.model.lm_head,
                self.model.model.embedding_norm,  # LFM2 uses embedding_norm
                self.tokenizer
            )
            self.wrapped_layers[idx] = self.model.model.layers[idx]

    def set_save_internal_decodings(self, value: bool):
        """Set whether to save internal decodings for wrapped attention layers"""
        for idx in self.attention_layer_indices:
            self.model.model.layers[idx].save_internal_decodings = value

    def set_from_positions(self, pos: int):
        """Set the position from which to add activations"""
        for idx in self.attention_layer_indices:
            self.model.model.layers[idx].from_position = pos

    def generate(self, tokens, max_new_tokens=100):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            generated = self.model.generate(
                inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
            )
            return self.tokenizer.batch_decode(generated)[0]

    def generate_text(self, user_input: str, model_output: Optional[str] = None,
                     system_prompt: Optional[str] = None, max_new_tokens: int = 50) -> str:
        """Generate text using LFM2 chat format"""
        tokens = tokenize_lfm2_chat(
            tokenizer=self.tokenizer,
            user_input=user_input,
            model_output=model_output,
            system_prompt=system_prompt
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def get_logits(self, tokens):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            logits = self.model(tokens).logits
            return logits

    def get_logits_from_text(self, user_input: str, model_output: Optional[str] = None,
                            system_prompt: Optional[str] = None) -> t.Tensor:
        """Get logits using LFM2 chat format"""
        tokens = tokenize_lfm2_chat(
            tokenizer=self.tokenizer,
            user_input=user_input,
            model_output=model_output,
            system_prompt=system_prompt
        )
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)

    def get_last_activations(self, layer):
        """
        Get activations from an attention layer.
        layer: logical layer index (0-5 for the 6 attention layers)
        """
        if layer >= self.num_attention_layers:
            raise ValueError(f"Layer {layer} out of range. LFM2-1.2B has {self.num_attention_layers} attention layers (0-{self.num_attention_layers-1})")

        actual_layer_idx = self.attention_layer_indices[layer]
        return self.model.model.layers[actual_layer_idx].activations

    def set_add_activations(self, layer, activations):
        """
        Add steering activations to an attention layer.
        layer: logical layer index (0-5 for the 6 attention layers)
        """
        if layer >= self.num_attention_layers:
            raise ValueError(f"Layer {layer} out of range. LFM2-1.2B has {self.num_attention_layers} attention layers (0-{self.num_attention_layers-1})")

        actual_layer_idx = self.attention_layer_indices[layer]
        self.model.model.layers[actual_layer_idx].add(activations)

    def set_calc_dot_product_with(self, layer, vector):
        """Set vector for dot product calculation for an attention layer"""
        if layer >= self.num_attention_layers:
            raise ValueError(f"Layer {layer} out of range. LFM2-1.2B has {self.num_attention_layers} attention layers (0-{self.num_attention_layers-1})")

        actual_layer_idx = self.attention_layer_indices[layer]
        self.model.model.layers[actual_layer_idx].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        """Get dot products for an attention layer"""
        if layer >= self.num_attention_layers:
            raise ValueError(f"Layer {layer} out of range. LFM2-1.2B has {self.num_attention_layers} attention layers (0-{self.num_attention_layers-1})")

        actual_layer_idx = self.attention_layer_indices[layer]
        return self.model.model.layers[actual_layer_idx].dot_products

    def reset_all(self):
        """Reset all wrapped attention layers"""
        for idx in self.attention_layer_indices:
            self.model.model.layers[idx].reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        """Print decoded activations (compatibility with LlamaWrapper)"""
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def get_activation_data(self, decoded_activations, topk=10):
        """Get activation data (compatibility with LlamaWrapper)"""
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))