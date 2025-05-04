from utils import detect_device, load_model_and_tokenizer, format_chat_prompt, prepare_inputs, DeviceType
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
import gc

def summarize_top_5_combined(
    model_name: str,
    dataset,
    query_col: str, 
    docs_col: str, 
    model: AutoModelForCausalLM = None, 
    tokenizer: AutoTokenizer = None,
    batch_size: int = 4,
    max_new_tokens: int = 50,
    temperature: float = 0.7
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
            print("Model dan tokenizer belum dimuat. Memuat model sekarang...")
            config = detect_device()
            model, tokenizer, config = load_model_and_tokenizer(model_name, config)

        # Pastikan model dalam mode evaluasi untuk inference
        model.eval()

        # List untuk menyimpan ringkasan
        summaries = []

        # Proses dataset dalam batch
        num_batches = (len(dataset) + batch_size - 1) // batch_size  # Hitung jumlah batch

        for start_idx in tqdm(range(0, len(dataset), batch_size), desc="Summarizing dataset", total=num_batches):
            end_idx = min(start_idx + batch_size, len(dataset))
            batch_data = dataset.select(range(start_idx, end_idx))

            batch_texts = []
            for row in batch_data:
                # Ambil teks dari kolom top_5_combined
                top_5_text = row[docs_col]
                query = row[query_col]

                # Siapkan prompt untuk rangkuman
                messages = [
                    {
                        "role": "user",
                        "content": f'{top_5_text}\n\nRingkaslah teks di atas agar dapat menjawab pertanyaan secara mendetail. Jangan berikan pengantar pada hasil. Pertanyaan: "{query}"'
                    }
                ]

                formatted_prompt = format_chat_prompt(messages, tokenizer)
                batch_texts.append(formatted_prompt)

            # Tokenisasi batch
            inputs = prepare_inputs(batch_texts, tokenizer, DeviceType.CUDA)
            token_ids = inputs["input_ids"]

            with torch.no_grad():
                with torch.amp.autocast('cuda:0'):
                    try:
                        summary_outputs = model.generate(
                            inputs["input_ids"],
                            max_new_tokens=max_new_tokens,  # token hasil generasi dibatasi hingga sebanyak max_new_tokens
                            temperature=temperature,
                            top_p=1.0,
                            do_sample=temperature > 0,
                            pad_token_id=tokenizer.pad_token_id, # Jika teks generasi kurang dari max_new_tokens, maka akan dipadding
                            return_dict_in_generate=True
                        )

                        for i in range(len(batch_texts)):
                            generated_ids = summary_outputs.sequences[i][len(token_ids[i]):]

                            # Decode hanya bagian hasil keluaran (tanpa prompt input)
                            summary = tokenizer.decode(
                                generated_ids,
                                skip_special_tokens=True
                            )
                            summaries.append(summary)

                    except Exception as e:
                        print(f"Error generating summary at index {start_idx}-{end_idx}: {e}")
                        summaries.extend(["Error"] * (end_idx - start_idx))  # Isi dengan "Error" jika gagal

            # Bersihkan variabel yang tidak diperlukan
            del batch_data, batch_texts, inputs, summary_outputs, token_ids
            torch.cuda.empty_cache()
            gc.collect()

        # Tambahkan kolom baru 'summary' ke dataset
        dataset = dataset.add_column("summary", summaries)
        return dataset

    except Exception as e:
        raise Exception(f"Unexpected error in summarizing: {e}")

    finally:
        # Clean up CUDA cache after processing
        torch.cuda.empty_cache()
        gc.collect()

    
from transformers import pipeline
from datasets import Dataset
from tqdm import tqdm

def summarize_with_pipeline(
    model_name: str,
    dataset,
    model: AutoModelForCausalLM = None, 
    tokenizer: AutoTokenizer = None,
    max_length: int = 100,
    batch_size: int = 8
):
    """
    Merangkum kolom 'top_5_combined' menjadi teks pendek 2 kalimat menggunakan pipeline summarization.

    Args:
        model_name: Nama atau path model Hugging Face.
        dataset: Dataset Hugging Face dengan kolom 'top_5_combined'.
        max_length: Maksimum panjang token output yang dirangkum.
        batch_size: Ukuran batch untuk pemrosesan.

    Returns:
        Dataset dengan kolom baru 'summary'.
    """
    try:
        if model is None or tokenizer is None:
            print("Model dan tokenizer belum dimuat. Memuat model sekarang...")
            config = detect_device()
            model, tokenizer, config = load_model_and_tokenizer(model_name, config)

        # Inisialisasi pipeline untuk summarization
        summarizer = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0  # Menggunakan GPU jika tersedia
        )

        summaries = []

        # Loop melalui dataset dalam batch untuk pemrosesan efisien
        for i in tqdm(range(0, len(dataset), batch_size), desc="Summarizing dataset"):
            batch_end = min(i + batch_size, len(dataset))
            batch_texts = [row["top_5_combined"] for row in dataset.select(range(i, batch_end))]

            # Lakukan summarization untuk batch saat ini
            summarized_results = summarizer(batch_texts, max_length=max_length, truncation=True)

            # Ambil teks hasil ringkasan dan tambahkan ke list summaries
            for summary in summarized_results:
                summaries.append(summary["summary_text"])

        # Tambahkan kolom baru ke dataset dengan ringkasan
        dataset = dataset.add_column("summary_pipe", summaries)
        return dataset

    except Exception as e:
        raise Exception(f"Unexpected error in summarization: {e}")
