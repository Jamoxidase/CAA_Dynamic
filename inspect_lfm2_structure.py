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
else:
    print("No HF_TOKEN found. Please set it to inspect the model.")