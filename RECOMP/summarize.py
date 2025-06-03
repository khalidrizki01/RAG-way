from utils import detect_device, load_model_and_tokenizer, format_chat_prompt, prepare_inputs, DeviceType
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import torch
import gc

def build_prompts(batch_data, query_col, psgs_col, tokenizer):
    """Buat prompt untuk setiap baris data."""
    batch_texts = []
    for row in batch_data:
        passages = row[psgs_col]
        query = row[query_col]
        messages = [
            {
                "role": "user",
                "content": f'{passages}\n\nRingkaslah teks di atas menjadi kurang dari 2 kalimat (40 kata) agar menjawab pertanyaan secara mendetail. TANPA PENGANTAR. Pertanyaan: "{query}"'
            }
        ]
        formatted_prompt = format_chat_prompt(messages, tokenizer)
        batch_texts.append(formatted_prompt)
    return batch_texts

def generate_summaries(model, tokenizer, batch_texts, max_new_tokens, temperature, max_source_length=None):
    """Generate summary untuk batch prompt."""
    inputs = prepare_inputs(batch_texts, tokenizer, DeviceType.CUDA, max_length=max_source_length)
    token_ids = inputs["input_ids"]
    summaries = []

    with torch.no_grad():
        with torch.amp.autocast('cuda:0'):
            try:
                summary_outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=1.0,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True
                )

                for i in range(len(batch_texts)):
                    generated_ids = summary_outputs.sequences[i][len(token_ids[i]):]
                    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    summaries.append(summary)

            except Exception as e:
                print(f"Error saat generate summary untuk batch: {e}")
                summaries.extend(["Error"] * len(batch_texts))
    return summaries

def cleanup_cuda():
    """Bersihkan cache CUDA dan garbage collector."""
    torch.cuda.empty_cache()
    gc.collect()

def generate_summary_dataset(
    model,
    dataset: Dataset, 
    query_col: str, 
    psgs_col: str, 
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    max_new_tokens: int = 50,
    temperature: float = 0.7, 
    summary_col: str = "summary", 
    max_source_length: int = None
):
    """
    Merangkum kolom 'top_5_combined' menjadi teks pendek 2 kalimat untuk setiap row.

    Args:
        model_name: Nama atau path model Hugging Face.
        dataset: Dataset Hugging Face dengan kolom 'top_5_combined'.
        batch_size: Ukuran batch untuk pemrosesan.
        max_new_tokens: Maksimal token baru yang dihasilkan.
        temperature: Parameter sampling untuk variasi teks.

    Returns:
        Dataset dengan kolom baru 'summary'.
    """
    try:
        # Deteksi perangkat dan muat model serta tokenizer
        if model is None or tokenizer is None:
            raise ValueError("Harap memasukkan model dan tokenizer LLM")

        # Pastikan model dalam mode evaluasi untuk inference
        model.eval()

        # List untuk menyimpan ringkasan
        summaries = []

        # Proses dataset dalam batch
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # Hitung jumlah batch

        for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Summarizing dataset", total=num_batches):
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_data = dataset.select(range(start_idx, end_idx))

            batch_texts = build_prompts(batch_data, query_col, psgs_col, tokenizer)
            batch_summaries = generate_summaries(model, tokenizer, batch_texts, max_new_tokens, temperature, max_source_length=max_source_length)
            summaries.extend(batch_summaries)

            # Bersihkan variabel yang tidak diperlukan
            # del batch_data, batch_texts, inputs, summary_outputs, token_ids
            cleanup_cuda()

        # Tambahkan kolom baru 'summary' ke dataset
        dataset = dataset.add_column(summary_col, summaries)
        cleanup_cuda()
        return dataset

    except Exception as e:
        raise Exception(f"Unexpected error in summarizing: {e}")

    finally:
        # Clean up CUDA cache after processing
        cleanup_cuda()