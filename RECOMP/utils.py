import torch
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import gc

class DeviceType(Enum):
    CUDA = "cuda"

@dataclass
class ModelConfig:
    """Configuration for model loading and inference"""
    device_type: DeviceType
    dtype: torch.dtype
    device_map: Optional[Union[str, Dict]] = None

def detect_device() -> ModelConfig:
    """
    Detect the best available device and return appropriate configuration for a single GPU.
    Returns: ModelConfig with optimal settings for the current hardware
    """
    if torch.cuda.is_available():
        return ModelConfig(
            device_type=DeviceType.CUDA,
            dtype=torch.bfloat16,
            device_map=None  # Disable automatic mapping to ensure manual placement
        )
    else:
        raise RuntimeError("CUDA device is not available. Please ensure a GPU is accessible.")
    
def load_model_and_tokenizer(
    model_name: str,
    config: Optional[ModelConfig] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model and tokenizer with optimal settings for a single GPU.

    Args:
        model_name: Name or path of the model
        config: Optional ModelConfig, if None will auto-detect

    Returns:
        tuple: (model, tokenizer)
    """
    if config is None:
        config = detect_device()

    print(f"Loading model on {config.device_type.value} with {config.dtype}")

    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",  # Better for chat models
        trust_remote_code=True
    )

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with optimal settings for device
    model_kwargs = {
        "torch_dtype": config.dtype,
        "trust_remote_code": True
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    ).to(config.device_type.value)

    return model, tokenizer, config

def format_chat_prompt(messages: List[Dict[str, str]],
                      tokenizer) -> str:
    """Format chat messages using model's template or fallback format"""
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            tokenize_kwargs = {}
            if "qwen" in getattr(tokenizer, "name_or_path", "").lower():
                tokenize_kwargs = {
                    "enable_thinking" : False
                }
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True, 
                **tokenize_kwargs
            )
            
    except Exception as e:
        raise RuntimeError(f"Error formatting chat prompt: {e}")
    
def prepare_inputs(
    text: str,
    tokenizer: AutoTokenizer,
    device_type: DeviceType, 
    max_length: int = None
) -> Dict[str, torch.Tensor]:
    """Prepare model inputs with proper device placement"""
    
    tokenize_kwargs = {}
    if max_length is not None:
        tokenize_kwargs = {'max_length':max_length}
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,  
        truncation=True, 
        add_special_tokens=False, 
        **tokenize_kwargs
    )

    inputs['input_ids'] = inputs['input_ids']
    inputs['attention_mask'] = inputs['attention_mask']
    
    # Move tensors to appropriate device
    device = device_type.value
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    return inputs

def get_chat_logprobs(
    messages: List[Dict[str, str]], 
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: ModelConfig,
    include_prompt: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 1
) -> Dict:
    """
    Get log probabilities for chat completion with optimal device handling.

    Args:
        messages: List of message dictionaries
        model: Language model
        tokenizer: Associated tokenizer
        config: ModelConfig with device settings
        include_prompt: Whether to include prompt tokens
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        batch_size: Batch size for processing

    Returns:
        dict: Contains tokens, logprobs, and generation info
    """
    # Ensure model is in eval mode
    model.eval()

    # Format prompt and prepare inputs
    formatted_prompt = format_chat_prompt(messages, tokenizer)
    inputs = prepare_inputs(formatted_prompt, tokenizer, config.device_type)

    # Store offset mapping and remove from inputs
    offset_mapping = inputs.pop("offset_mapping")[0]

    with torch.no_grad():
        with torch.amp.autocast('cuda:0'):
            try:
                token_ids = inputs["input_ids"][0]
                
                # Generate completion with appropriate settings
                generation_config = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                    "pad_token_id": tokenizer.pad_token_id,
                    "attention_mask": inputs["attention_mask"],
                    "return_dict_in_generate": True,
                    "output_scores": True
                }

                # Add device-specific settings
                if config.device_type == DeviceType.CUDA:
                    generation_config["use_cache"] = True

                gen_outputs = model.generate(
                    inputs["input_ids"],
                    **generation_config
                )

                # Process generation outputs
                generated_ids = gen_outputs.sequences[0][len(token_ids):]
                generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
                generated_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True
                )

                # Calculate generation logprobs
                gen_logprobs = []
                # gen_top_logprobs = []
                if hasattr(gen_outputs, "scores") and gen_outputs.scores:
                    for token_idx, token_scores in enumerate(gen_outputs.scores):
                        if token_idx < len(generated_ids):
                            current_token_id = generated_ids[token_idx]
                            probs = torch.nn.functional.softmax(token_scores[0], dim=-1)
                            log_probs = torch.log(probs)

                            # Get logprob for next token
                            gen_logprobs.append(
                                log_probs[current_token_id].item()
                            )


                # Create final result
                result = {
                    "tokens": generated_tokens,
                    "token_ids": generated_ids,
                    "logprobs": gen_logprobs,
                    "completion": generated_text,
                    "prompt": formatted_prompt if include_prompt else None
                }
                return result

            except Exception as e:
                raise RuntimeError(f"Error during inference: {e}")

            finally:
                del inputs, gen_outputs, token_ids, generated_ids, generated_tokens, generated_text, gen_logprobs
                torch.cuda.empty_cache()
                gc.collect()

def is_groundtruth_duplicated_in_generation(generation, groundtruth):
    """
    Mengecek apakah ada kata yang muncul lebih dari sekali dalam teks,
    dan apakah kata tersebut juga terdapat di dalam kolom 'answer'.

    Args:
        text (str): Teks yang akan diperiksa (misalnya 'generated_completion').
        answer (str): Jawaban ground truth dari kolom 'answer'.

    Returns:
        tuple: (has_duplicates, duplicates_in_answer, duplicate_words_list)
            - has_duplicates (bool): True jika ada kata yang muncul lebih dari sekali.
            - duplicates_in_answer (bool): True jika kata duplikat juga ada dalam 'answer'.
            - duplicate_words_list (list[str]): Daftar kata yang merupakan duplikat.
    """
    words = re.findall(r'\w+|[^\w\s]', generation, re.UNICODE)  # Pisahkan teks menjadi kata-kata
    word_counts = {}

    # Hitung frekuensi kemunculan tiap kata
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Cari kata-kata yang muncul lebih dari sekali
    duplicate_words = {word for word, count in word_counts.items() if count > 1}

    # Cek apakah kata-kata duplikat ada di dalam 'answer'
    answer_words = set(re.findall(r'\w+|[^\w\s]', groundtruth, re.UNICODE))
    duplicate_words = duplicate_words.intersection(answer_words)  # Filter hanya kata-kata di 'answer'
    duplicates_in_answer = len(duplicate_words) > 0

    return duplicates_in_answer, list(duplicate_words)

def get_w_wo_preceding_space_variants(
    word: str, 
    tokenizer: AutoTokenizer
) -> list:
    """
    Mengembalikan variasi tokenisasi untuk sebuah kata dengan/atau tanpa spasi sebelumnya.
    Kenapa perlu dibedakan antara yg berspasi sebelum dengan sesudah? 
    Karena ada kasus dimana tokenizer memberikan token yang berbeda untuk kata dengan atau tanpa spasi, misal token untuk "nama" adalah 1 dan " nama" adalah 101
    Kata tanpa spasi yang mendahuluinya pun bisa terjadi di banyak kasus, seperti ketika terletak di awal string atau jika didahului:
    1. tanda petik ganda ( " ) atau petik tunggal ( ' )
    2. tanda kurang ( `(` atau `[` )
    3. tanda hubung ( - )
    4. tanda miring ( / )

    Args:
        word (str): Kata yang akan ditokenisasi.
        tokenizer: Tokenizer yang digunakan.

    Returns:
        list: Variasi tokenisasi kata.
    """
    try: 
        supports_prefix_space = "add_prefix_space" in tokenizer.__call__.__code__.co_varnames

        if supports_prefix_space:
            tokenized_variants = [
                tokenizer.encode(word, add_special_tokens=False, add_prefix_space=False),  # Tanpa spasi sebelumnya
                tokenizer.encode(word, add_special_tokens=False, add_prefix_space=True)   # Dengan spasi sebelumnya
            ]
        else:
            tokenized_variants = [
                tokenizer.encode(word, add_special_tokens=False),  # Tanpa spasi sebelumnya
                tokenizer.encode(" " + word, add_special_tokens=False)  # Dengan spasi sebelumnya (manual)
            ]
        return tokenized_variants
    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat tokenisasi '{word}': {e}")
        raise