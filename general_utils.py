from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import Tensor, cuda
import torch.nn.functional as F
from tqdm import tqdm
import torch
from enum import Enum
from transformers import AutoTokenizer

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
    if cuda.is_available():
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

def truncate_each_passage(
    retriever_tokenizer, 
    text, 
    context_length
):

    # 1. Tokenisasi dengan retriever_tokenizer
    tokenized_example_retriever = retriever_tokenizer(
        text,
        return_tensors='pt',
        max_length=context_length,
        truncation=True,
        add_special_tokens=False
    )

    # token_length = tokenized_example_retriever.input_ids.shape[1]

    # 2. Decode hasil tokenisasi
    decoded_example = retriever_tokenizer.decode(
        tokenized_example_retriever.input_ids[0],
        skip_special_tokens=True
    )

    return decoded_example #, token_length

def extract_topk_texts(
        row, 
        k=None, 
        ranked_units='ranked_chunks_with_labels', 
        returned_units_col="top_chunks", 
        returned_labels_col="top_labels"
):
    if ranked_units not in row:
        raise ValueError(
            f"Kolom '{ranked_units}' tidak ditemukan dalam row. "
            f"Pastikan kolom tersebut ada dalam dataset dan sesuai dengan nama yang dimasukkan."
        )
    if k is None:
        k = len(row[ranked_units])

    ranked_chunks = row[ranked_units]
    
    # Ambil top-k chunk (atau kurang jika tersedia lebih sedikit)
    topk_chunks = ranked_chunks[:k]
    
    # Gabungkan semua 'text' dari top-k dengan pemisah newline
    joined_text = "\n".join(chunk['text'] for chunk in topk_chunks)
    topk_labels = [chunk['is_positive'] for chunk in topk_chunks]

    return {
        returned_units_col: joined_text,
        returned_labels_col: topk_labels
    }

def format_chat_prompt(
        messages: List[Dict[str, str]],
        tokenizer
) -> str:
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
) -> Dict[str, Tensor]:
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
        k: v.to(device) if isinstance(v, Tensor) else v
        for k, v in inputs.items()
    }

    return inputs

def generate_per_row(row, query_col, ctx_col, tokenizer, model, device_type, instruction) -> Tuple[str, str]:
    query = row[query_col]
    if ctx_col is not None:
        context = row[ctx_col]
        prompt = instruction.format_map({'query': query, 'context': context})
    else: 
        prompt = instruction.format_map({'query': query})

    messages = format_chat_prompt([{"role": "user", "content": prompt}], tokenizer)
    inputs = prepare_inputs(messages, tokenizer, device_type)["input_ids"]

    with torch.no_grad():
        with torch.amp.autocast('cuda:0'):
            outputs = model.generate(
                inputs,
                max_new_tokens=52,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True
            )

    # prompt_and_completion = outputs.sequences
    completion = outputs.sequences[:, inputs.shape[1]:]

    # decoded_output = tokenizer.decode(prompt_and_completion[0], skip_special_tokens=False)
    answer = tokenizer.decode(completion[0], skip_special_tokens=True)

    return  answer # decoded_output,

# Fungsi pooling dari model card
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def rank_ctxs_by_query_similarity(query, ctxs, labels, tokenizer, model):
    if not ctxs:
        return []

    input_texts = ["query: " + query] + ["passage: " + c for c in ctxs]
    query_and_ctxs = tokenizer(
        input_texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    query_and_ctxs = {k: v.to(model.device) for k, v in query_and_ctxs.items()}

    with torch.no_grad():
        outputs = model(**query_and_ctxs)

    embeddings = average_pool(outputs.last_hidden_state, query_and_ctxs['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    query_embedding = embeddings[0]
    ctx_embeddings = embeddings[1:]
    scores = (query_embedding @ ctx_embeddings.T) * 100

    scores_list = scores.tolist()

    combined = [{"text": ctxs[i], "score": scores_list[i], "is_positive": labels[i]} for i in range(len(ctxs))]
    combined_sorted = sorted(combined, key=lambda x: x["score"], reverse=True)
    return combined_sorted

def apply_similarity_ranking_to_dataset(
    dataset, 
    text_col: str,
    label_col: Optional[str] =None,
    output_col: str=None, 
    tokenizer = None,
    model = None
):
    ranked_units_all = []

    for example in tqdm(dataset, desc=f"Processing {output_col}"):
        query = example['query']

        if label_col is None:
            units = example[text_col]
            labels = [i == 0 for i in range(len(units))]            
        else:
            units = example[text_col]
            labels = example[label_col]            

        ranked_units = rank_ctxs_by_query_similarity(query, units, labels, tokenizer, model)
        ranked_units_all.append(ranked_units)

    dataset = dataset.add_column(output_col, ranked_units_all)
    return dataset

import re

def clean_parsoid_artifacts(text: str) -> str:
    """
    Bersihkan artefak Parsoid/HTML yang pecah, misal:
    - data-parsoid='...'>s/u>
    - d/b>eoxyribob data-parsoid='...'>n/b>ucleic a/b>cid

    Langkah:
      1) Hilangkan huruf tag tunggal (b|i|u|em|strong|span|a) yang muncul
         PERSIS sebelum 'data-parsoid=' (representasi '<b', '<i', dst. yang kehilangan '<').
      2) Hapus blok data-parsoid='...'>
      3) Hapus fragmen penutup tag pecah: /b>, /i>, /u>, /em>, /strong>, /span>, /a>
      4) Bersihkan spasi berlebih
    """
    # 1) buang huruf/tag tunggal yang berada tepat sebelum data-parsoid=
    text = re.sub(r"\b(?:b|i|u|em|strong|span|a)\s+(?=data-parsoid=)", "", text, flags=re.IGNORECASE)

    # 2) hapus keseluruhan blok data-parsoid='...'>
    text = re.sub(r"\s*data-parsoid='[^']*'>", "", text, flags=re.IGNORECASE)

    # 3) hapus penutup tag yang pecah (tidak ada '<')
    text = re.sub(r"/(?:b|i|u|em|strong|span|a)>", "", text, flags=re.IGNORECASE)

    # 3b) hapus karakter \u200e dan varian "‎\u200e;" jika ada
    text = re.sub(r"[\u200e;]+", "", text)

    # 4) rapikan spasi
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def normalize_units(text: str) -> str:
    """
    Normalisasi angka + satuan agar konsisten:
    - Hilangkan spasi antara angka dan satuan.
    - Ubah superscript ² menjadi angka 2 biasa.
    - Samakan variasi unit area (km², km2 -> km2, dst).
    """
    # Ganti superscript ² jadi 2 biasa
    text = text.replace("²", "2")
    
    # Hilangkan spasi sebelum unit (misal "9.000 km2" -> "9.000km2")
    text = re.sub(r"(\d)\s+(km2|m2|cm2)", r"\1\2", text, flags=re.IGNORECASE)
    
    # Samakan semua bentuk unit area jadi format baku
    unit_map = {
        "km2": "km2",
        "km²": "km2",
        "m2": "m2",
        "m²": "m2",
        "cm2": "cm2",
        "cm²": "cm2",
    }
    for u_from, u_to in unit_map.items():
        text = re.sub(u_from, u_to, text, flags=re.IGNORECASE)
    
    return text