from utils import detect_device, load_model_and_tokenizer, format_chat_prompt, prepare_inputs, DeviceType
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import torch
import gc

def prepare_truncated_prompt(passages, query, teacher_tokenizer, student_tokenizer, max_source_length, return_truncated_passages=False):
    qwen_instruction = f'\n\nRingkaslah teks di atas menjadi maksimal 2 kalimat (40 kata) agar menjawab pertanyaan secara mendetail. TANPA PENGANTAR. Pertanyaan: "{query}'
    t5_instruction = f"Rangkum Dokumen agar bisa menjawab Pertanyaan.\nPertanyaan: {query}\nDokumen:\nRangkuman: "
    instr_len = len(student_tokenizer(t5_instruction, add_special_tokens=False)['input_ids'])

    available_len = max_source_length - instr_len

    truncated_passages = student_tokenizer.decode(
        student_tokenizer(passages, max_length=available_len, truncation=True, add_special_tokens=False)['input_ids'],
        skip_special_tokens=True
    )

    messages = [{"role": "user", "content": truncated_passages + qwen_instruction}]
    formatted_prompt = format_chat_prompt(messages, teacher_tokenizer)

    if return_truncated_passages:
        return formatted_prompt, truncated_passages
    else:
        return formatted_prompt

def generate_batch_prompts(batch_data, query_col, psgs_col, teacher_tokenizer, student_tokenizer, max_source_length=512, return_truncated_passages=True):
    """Buat prompt yang sudah di-truncate dan diformat untuk setiap baris data."""
    batch_prompts = []
    truncated_passages_list = []

    for row in batch_data:
        passages = row[psgs_col]
        query = row[query_col]
        result = prepare_truncated_prompt(
            passages=passages,
            query=query,
            teacher_tokenizer=teacher_tokenizer,
            student_tokenizer=student_tokenizer,
            max_source_length=max_source_length,
            return_truncated_passages=return_truncated_passages
        )
        if return_truncated_passages:
            prompt, truncated_passages = result
            truncated_passages_list.append(truncated_passages)
        else:
            prompt = result

        batch_prompts.append(prompt)

    if return_truncated_passages:
        return batch_prompts, truncated_passages_list
    else:
        return batch_prompts


def generate_summaries(model, tokenizer, batch_texts, max_new_tokens, temperature):
    """Generate summary untuk batch prompt."""
    inputs = prepare_inputs(batch_texts, tokenizer, DeviceType.CUDA)
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
    dataset: Dataset, 
    query_col: str, 
    psgs_col: str, 
    model: AutoModelForCausalLM,
    teacher_tokenizer: AutoTokenizer,
    student_tokenizer: AutoTokenizer, 
    batch_size: int = 4,
    max_new_tokens: int = 52,
    temperature: float = 0.7, 
    create_truncated_psg_column: bool = True
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
        if model is None or teacher_tokenizer is None or student_tokenizer is None:
            raise ValueError("Harap memasukkan model teacher LLM, tokenizer LLM")

        # Pastikan model dalam mode evaluasi untuk inference
        model.eval()

        # List untuk menyimpan ringkasan
        summaries = []
        truncated_passages_all = [] if create_truncated_psg_column else None

        # Proses dataset dalam batch
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # Hitung jumlah batch

        for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Summarizing dataset", total=num_batches):
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_data = dataset.select(range(start_idx, end_idx))

            if create_truncated_psg_column:
                batch_prompts, batch_truncated = generate_batch_prompts(
                    batch_data, query_col, psgs_col, teacher_tokenizer, student_tokenizer,
                    max_source_length=512,
                    return_truncated_passages=True
                )
                truncated_passages_all.extend(batch_truncated)
            else:
                batch_prompts = generate_batch_prompts(
                    batch_data, query_col, psgs_col, teacher_tokenizer, student_tokenizer,
                    max_source_length=512,
                    return_truncated_passages=False
                )

            batch_summaries = generate_summaries(model, teacher_tokenizer, batch_prompts, max_new_tokens, temperature)
            summaries.extend(batch_summaries)

            # Bersihkan variabel yang tidak diperlukan
            # del batch_data, batch_texts, inputs, summary_outputs, token_ids
            cleanup_cuda()

        # Tambahkan kolom baru 'summary' ke dataset
        dataset = dataset.add_column("summary", summaries)
        if create_truncated_psg_column:
            dataset = dataset.add_column("truncated_passages", truncated_passages_all)
        cleanup_cuda()
        return dataset

    except Exception as e:
        raise Exception(f"Unexpected error in summarizing: {e}")

    finally:
        # Clean up CUDA cache after processing
        cleanup_cuda()