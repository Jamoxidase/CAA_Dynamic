from typing import List
from transformers import PreTrainedTokenizer

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
BASE_INPUT = "Input:"
BASE_RESPONSE = "\nResponse:"

# Llama 3 special tokens
B_HEADER, E_HEADER = "<|start_header_id|>", "<|end_header_id|>"
EOT = "<|eot_id|>"

# LFM2 special tokens (ChatML format)
IM_START, IM_END = "<|im_start|>", "<|im_end|>"

ADD_FROM_POS_CHAT = E_INST
ADD_FROM_POS_BASE = BASE_RESPONSE
ADD_FROM_POS_LLAMA3_CHAT = EOT
ADD_FROM_POS_LFM2 = IM_END


def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    input_content = ""
    if system_prompt is not None:
        input_content += B_SYS + system_prompt + E_SYS
    input_content += f"{B_INST} {user_input.strip()} {E_INST}"
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    return tokenizer.encode(input_content)


def tokenize_llama_base(
    tokenizer, user_input: str, model_output: str = None
) -> List[int]:
    input_content = ""
    input_content += f"{BASE_INPUT} {user_input.strip()}"
    if model_output is not None:
        input_content += f"{BASE_RESPONSE} {model_output.strip()}"
    return tokenizer.encode(input_content)


def tokenize_llama3_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    input_content = ""

    # Add system prompt if provided
    if system_prompt is not None:
        input_content += f"{B_HEADER}system{E_HEADER}\n\n{system_prompt.strip()}{EOT}"

    # Add user input
    input_content += f"{B_HEADER}user{E_HEADER}\n\n{user_input.strip()}{EOT}"

    # Add assistant header and optional model output
    input_content += f"{B_HEADER}assistant{E_HEADER}\n\n"
    if model_output is not None:
        input_content += f"{model_output.strip()}"

    return tokenizer.encode(input_content)


def tokenize_lfm2_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
    system_prompt: str = None,
) -> List[int]:
    input_content = ""

    # Add system prompt if provided (LFM2 uses ChatML format)
    if system_prompt is not None:
        input_content += f"{IM_START}system\n{system_prompt.strip()}{IM_END}\n"

    # Add user input
    input_content += f"{IM_START}user\n{user_input.strip()}{IM_END}\n"

    # Add assistant header and optional model output
    input_content += f"{IM_START}assistant\n"
    if model_output is not None:
        input_content += f"{model_output.strip()}{IM_END}"

    return tokenizer.encode(input_content)
