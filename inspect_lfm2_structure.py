 #!/usr/bin/env python3
"""
Quick script to inspect LFM2 model structure
"""

import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    print("Loading LFM2-1.2B model to inspect structure...")
    model = AutoModelForCausalLM.from_pretrained(
        "LiquidAI/LFM2-1.2B",
        token=HF_TOKEN
    )

    print("\nModel attributes:")
    print(dir(model))

    print("\nModel.model attributes:")
    print(dir(model.model))

    print("\nModel structure:")
    print(model)

    print("\nFirst layer structure:")
    if hasattr(model.model, 'layers'):
        print("First layer:", model.model.layers[0])

    print("\nTesting layer outputs:")
    import torch
    # Create dummy input
    dummy_input = torch.randint(0, 1000, (1, 10)).to(model.device)  # batch=1, seq_len=10

    # Get embeddings
    embeddings = model.model.embed_tokens(dummy_input)
    print(f"Embeddings shape: {embeddings.shape}")

    # Test first conv layer (layer 0)
    with torch.no_grad():
        layer_0_output = model.model.layers[0](embeddings)
        print(f"Layer 0 (conv) output type: {type(layer_0_output)}")
        if isinstance(layer_0_output, tuple):
            print(f"Layer 0 output[0] shape: {layer_0_output[0].shape}")
        else:
            print(f"Layer 0 output shape: {layer_0_output.shape}")

    # Test first attention layer (layer 2)
    with torch.no_grad():
        # Pass through layers 0 and 1 first
        hidden = embeddings
        for i in range(2):
            hidden = model.model.layers[i](hidden)
            if isinstance(hidden, tuple):
                hidden = hidden[0]

        layer_2_output = model.model.layers[2](hidden)
        print(f"\nLayer 2 (attention) output type: {type(layer_2_output)}")
        if isinstance(layer_2_output, tuple):
            print(f"Layer 2 output[0] shape: {layer_2_output[0].shape}")
        else:
            print(f"Layer 2 output shape: {layer_2_output.shape}")
else:
    print("No HF_TOKEN found. Please set it to inspect the model.")