#!/usr/bin/env python3
"""
Test script to verify model support changes don't break existing functionality
and that new Llama 3 8B and LFM2 1.2B models work correctly.
"""

import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
from lfm2_wrapper import LFM2Wrapper
from utils.helpers import get_model_path, model_name_format

load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def test_model_paths():
    """Test that model paths are generated correctly"""
    print("Testing model path generation...")

    # Test existing Llama 2 paths - should remain unchanged
    assert get_model_path("7b", True) == "meta-llama/Llama-2-7b-hf"
    assert get_model_path("7b", False) == "meta-llama/Llama-2-7b-chat-hf"
    assert get_model_path("13b", True) == "meta-llama/Llama-2-13b-hf"
    assert get_model_path("13b", False) == "meta-llama/Llama-2-13b-chat-hf"

    # Test new Llama 3 paths
    assert get_model_path("8b", True) == "meta-llama/Meta-Llama-3-8B"
    assert get_model_path("8b", False) == "mlabonne/Daredevil-8B"

    # Test LFM2 paths
    assert get_model_path("1.2b", True) == "LiquidAI/LFM2-1.2B"  # No base/chat distinction
    assert get_model_path("1.2b", False) == "LiquidAI/LFM2-1.2B"

    print("✓ Model path generation tests passed")

def test_model_name_formatting():
    """Test model name formatting"""
    print("\nTesting model name formatting...")

    # Test existing Llama 2 formats
    assert model_name_format("Llama-2-7b-hf") == "Llama 2 7B"
    assert model_name_format("Llama-2-7b-chat-hf") == "Llama 2 Chat 7B"
    assert model_name_format("Llama-2-13b-hf") == "Llama 2 13B"
    assert model_name_format("Llama-2-13b-chat-hf") == "Llama 2 Chat 13B"

    # Test new Llama 3 formats
    assert model_name_format("Meta-Llama-3-8B") == "Llama 3 8B"
    assert model_name_format("Daredevil-8B") == "Llama 3 Chat 8B"

    # Test LFM2 format
    assert model_name_format("LFM2-1.2B") == "LFM2 1.2B"

    print("✓ Model name formatting tests passed")

def test_tokenization_imports():
    """Test that all tokenization functions are importable"""
    print("\nTesting tokenization imports...")

    from utils.llama_tokenize import (
        tokenize_llama_chat,
        tokenize_llama_base,
        tokenize_llama3_chat,
        tokenize_lfm2_chat,
        ADD_FROM_POS_BASE,
        ADD_FROM_POS_CHAT,
        ADD_FROM_POS_LLAMA3_CHAT,
        ADD_FROM_POS_LFM2,
        E_INST,
        EOT,
        IM_END,
    )

    # Check special tokens are defined
    assert E_INST == "[/INST]"
    assert EOT == "<|eot_id|>"
    assert IM_END == "<|im_end|>"
    assert ADD_FROM_POS_CHAT == E_INST
    assert ADD_FROM_POS_LLAMA3_CHAT == EOT
    assert ADD_FROM_POS_LFM2 == IM_END

    print("✓ Tokenization imports tests passed")

def test_model_initialization(test_new_models_only=False):
    """Test model initialization (requires HF token)"""
    if not HUGGINGFACE_TOKEN:
        print("\n⚠️  Skipping model initialization tests (no HF_TOKEN found)")
        return

    print("\nTesting model initialization...")

    if not test_new_models_only:
        # Test existing 7b model still works
        print("  Testing Llama 2 7B initialization...")
        model_7b = LlamaWrapper(HUGGINGFACE_TOKEN, size="7b", use_chat=False)
        assert model_7b.model_name_path == "meta-llama/Llama-2-7b-hf"
        assert not model_7b.is_llama3
        assert model_7b.size == "7b"
        print("  ✓ Llama 2 7B initialization successful")

    # Test new 8b model
    print("  Testing Llama 3 8B (Daredevil) initialization...")
    model_8b = LlamaWrapper(HUGGINGFACE_TOKEN, size="8b", use_chat=True)
    assert model_8b.model_name_path == "mlabonne/Daredevil-8B"
    assert model_8b.is_llama3
    assert model_8b.size == "8b"
    print("  ✓ Llama 3 8B initialization successful")

    # Test LFM2 model
    print("  Testing LFM2 1.2B initialization...")
    model_lfm2 = LFM2Wrapper(HUGGINGFACE_TOKEN, size="1.2b")
    assert model_lfm2.model_name_path == "LiquidAI/LFM2-1.2B"
    assert model_lfm2.num_attention_layers == 6
    assert model_lfm2.attention_layer_indices == [2, 5, 8, 10, 12, 14]
    print("  ✓ LFM2 1.2B initialization successful")

    print("✓ Model initialization tests passed")

if __name__ == "__main__":
    print("=" * 50)
    print("Testing Llama 3 8B and LFM2 1.2B Integration")
    print("=" * 50)

    test_model_paths()
    test_model_name_formatting()
    test_tokenization_imports()

    # Set to True to only test new models (faster)
    test_model_initialization(test_new_models_only=True)

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)