import sys
sys.path.append('/workspace')

import os
import torch
from smt_model import SMTConfig, SMTModelForCausalLM
from kern_lm import KernLM
from data_augmentation.data_augmentation import convert_img_to_tensor

checkpoint_path = 'final_4.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"
lm = KernLM.from_checkpoint(checkpoint_path)
lm = lm.to(device)
freeze = True
layers_to_transfer = 4

transfer_kqv_weights = True
transfer_kqv_bias = True
transfer_ffn1_weights = True
transfer_ffn2_weights = True
transfer_norm1_weights = True
transfer_norm2_weights = True
transfer_embeddings = True
transfer_vocab_projection = False

lm_to_smt_tokens = {
    '<t>':'<t>',
    '<end>':'<eos>',
    '<n>':'<b>',
    '**kern':'**ekern_1.0'
}

def map_lm_to_smt(smt):
    remapped_param_names = set()

    def remap_relative_to_mha_layer(source_state_dict, target_state_dict, layer, n_heads=4):
        def stack_heads(keys):
            return torch.cat([source_state_dict[k] for k in keys], dim=0)

        # helper to assign and track
        def assign(key, tensor):
            target_state_dict[key] = tensor
            remapped_param_names.add(key)

        head_dim = source_state_dict[f"blocks.{layer}.sa.heads.0.query.weight"].size(0)
        emb_dim = head_dim * n_heads
        # Query, Key, Value projections
        if transfer_kqv_weights:
          assign(f"decoder.decoder.layers.{layer}.self_attn.q_proj.weight",
                stack_heads([f"blocks.{layer}.sa.heads.{i}.query.weight" for i in range(n_heads)]))
          assign(f"decoder.decoder.layers.{layer}.self_attn.k_proj.weight",
                stack_heads([f"blocks.{layer}.sa.heads.{i}.key.weight" for i in range(n_heads)]))
          assign(f"decoder.decoder.layers.{layer}.self_attn.v_proj.weight",
                stack_heads([f"blocks.{layer}.sa.heads.{i}.value.weight" for i in range(n_heads)]))
          assign(f"decoder.decoder.layers.{layer}.self_attn.out_proj.weight", source_state_dict[f"blocks.{layer}.sa.proj.weight"])


        # Initialize biases to zero
        zero_bias = torch.zeros(emb_dim, device=source_state_dict[f"blocks.{layer}.sa.heads.0.query.weight"].device)
        if transfer_kqv_bias:
          assign(f"decoder.decoder.layers.{layer}.self_attn.q_proj.bias", zero_bias)
          assign(f"decoder.decoder.layers.{layer}.self_attn.k_proj.bias", zero_bias)
          assign(f"decoder.decoder.layers.{layer}.self_attn.v_proj.bias", zero_bias)
          assign(f"decoder.decoder.layers.{layer}.self_attn.out_proj.bias", source_state_dict[f"blocks.{layer}.sa.proj.bias"])


        # Output projection

        # Layer normalization and feed-forward
        if transfer_norm1_weights:
          assign(f"decoder.decoder.layers.{layer}.norm_layers.0.weight", source_state_dict[f"blocks.{layer}.ln1.weight"])
          assign(f"decoder.decoder.layers.{layer}.norm_layers.0.bias", source_state_dict[f"blocks.{layer}.ln1.bias"])
        if transfer_norm2_weights:
          assign(f"decoder.decoder.layers.{layer}.norm_layers.1.weight", source_state_dict[f"blocks.{layer}.ln2.weight"])
          assign(f"decoder.decoder.layers.{layer}.norm_layers.1.bias", source_state_dict[f"blocks.{layer}.ln2.bias"])

        if transfer_ffn1_weights:
          assign(f"decoder.decoder.layers.{layer}.ffn.0.weight", source_state_dict[f"blocks.{layer}.ffwd.net.0.weight"])
          assign(f"decoder.decoder.layers.{layer}.ffn.0.bias", source_state_dict[f"blocks.{layer}.ffwd.net.0.bias"])
        if transfer_ffn2_weights:
          assign(f"decoder.decoder.layers.{layer}.ffn.3.weight", source_state_dict[f"blocks.{layer}.ffwd.net.2.weight"])
          assign(f"decoder.decoder.layers.{layer}.ffn.3.bias", source_state_dict[f"blocks.{layer}.ffwd.net.2.bias"])

        # Optional third norm
        if f"blocks.{layer}.ln3.weight" in source_state_dict:
            assign(f"decoder.decoder.layers.{layer}.norm_layers.2.weight", source_state_dict[f"blocks.{layer}.ln3.weight"])
            assign(f"decoder.decoder.layers.{layer}.norm_layers.2.bias", source_state_dict[f"blocks.{layer}.ln3.bias"])

        return target_state_dict

    def remap_layers(source_state_dict, target_state_dict, n_layers=8):
        for layer in range(n_layers):
            # Check if this layer exists in source before remapping
            if f"blocks.{layer}.sa.proj.weight" in source_state_dict:
                target_state_dict = remap_relative_to_mha_layer(source_state_dict, target_state_dict, layer)
        return target_state_dict

    # Perform remapping
    target_sd = smt.state_dict()
    remapped = remap_layers(lm.state_dict(), target_sd, n_layers=layers_to_transfer)

    k = 0
    for key, i in lm.w2i.items():
        key = lm_to_smt_tokens.get(key, key)
        j = smt.w2i.get(key, None)
        if j is not None:
          if transfer_embeddings:
            smt.decoder.embedding.weight.data[j].copy_(lm.token_embedding_table.weight[i])
            #smt.decoder.embedding.weight.data[j].requires_grad = not freeze
          if transfer_vocab_projection:
            smt.decoder.vocab_projection.weight.data[j].copy_(lm.lm_head.weight[i])
            smt.decoder.vocab_projection.bias.data[j].copy_(lm.lm_head.bias[i])
            #smt.decoder.vocab_projection.weight.data[j].requires_grad = not freeze
            #smt.decoder.vocab_projection.bias.data[j].requires_grad = not freeze

            k += 1

    print(f"Remapped {k} tokens out of {len(lm.w2i)} from LM to SMT. {k/len(lm.w2i)*100}%")
    smt.load_state_dict(remapped)
    print("All applicable keys remapped and loaded into SMT model.")

    # Freeze only the parameters that were actually remapped
    for name, param in smt.named_parameters():
      if name in remapped_param_names:
        param.requires_grad = not freeze
    if freeze: print(f"Froze {len(remapped_param_names)} parameters in mapped SMT layers.")


if __name__ == "__main__":
    # Load the SMT model
    smt = SMTModelForCausalLM.from_pretrained("antoniorv6/smt-grandstaff")
    smt.to(device)

    # Map and selectively freeze
    map_lm_to_smt(smt)

    # Continue with additional setup or fine-tuning as needed
