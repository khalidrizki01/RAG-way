import torch
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

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
            dtype=torch.float16,  # Using float16 for CUDA by default
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
    )

    # Place model explicitly on GPU
    model = model.to("cuda:0")

    return model, tokenizer, config

def format_chat_prompt(messages: List[Dict[str, str]],
                      tokenizer) -> str:
    """Format chat messages using model's template or fallback format"""
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
        else:
            formatted_prompt = ""
            for message in messages:
                role = message['role']
                content = message['content']
                if role == 'sistem':
                    formatted_prompt += f"<|sistem|>\n{content}\n"
                elif role == 'pengguna':
                    formatted_prompt += f"<|pengguna|>\n{content}\n"
                elif role == 'asisten':
                    formatted_prompt += f"<|asisten|>\n{content}\n"
                else:
                    raise ValueError(f"Peran tak diketahui: {role}")
            return formatted_prompt + "<|asisten|>\n"
    except Exception as e:
        raise RuntimeError(f"Error formatting chat prompt: {e}")
    
def prepare_inputs(
    text: str,
    tokenizer: AutoTokenizer,
    device_type: DeviceType
) -> Dict[str, torch.Tensor]:
    """Prepare model inputs with proper device placement"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )
    
    # Remove the first token (always duplicate <|begin_of_text|>)
    inputs['input_ids'] = inputs['input_ids'][:, 1:]
    inputs['attention_mask'] = inputs['attention_mask'][:, 1:]
    inputs['offset_mapping'] = inputs['offset_mapping'][:, 1:]
    
    # Move tensors to appropriate device
    device = device_type.value
    inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    return inputs

def get_chat_logprobs(
    messages: List[Dict[str, str]], # system, user
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

                        # # Get top alternatives
                        # top_values, top_indices = torch.topk(log_probs, 5)
                        # top_logprobs = {
                        #     tokenizer.decode([idx]): prob.item()
                        #     for idx, prob in zip(top_indices, top_values)
                        # }
                        # gen_top_logprobs.append(top_logprobs)

            # Create final result
            result = {
                "tokens": generated_tokens,
                "token_ids": generated_ids,
                "logprobs": gen_logprobs,
                # "top_logprobs": gen_top_logprobs,
                # "text": generated_text,  # tokenizer.decode(token_ids) +
                "completion": generated_text,
                "prompt": formatted_prompt if include_prompt else None
            }
            return result

        except Exception as e:
            raise RuntimeError(f"Error during inference: {e}")

        finally:
            # Clean up CUDA cache if needed
            if config.device_type == DeviceType.CUDA:
                torch.cuda.empty_cache()