from utils import detect_device, load_model_and_tokenizer, format_chat_prompt, prepare_inputs
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import torch
import gc
import unicodedata
import re
import string
from collections import Counter
from nltk.corpus import stopwords

try: 
    STOPWORDS_ID = set(stopwords.words('indonesian'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOPWORDS_ID = set(stopwords.words('indonesian'))

def normalize_answer(s):
    """
    Normalisasi jawaban untuk menghilangkan variasi yang tidak penting dalam perhitungan EM dan F1.
    - Menghapus tanda baca
    - Menghapus artikel (a, an, the)
    - Menghapus ekstra whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punctuation(s.lower())))

def remove_stopwords(text):
    """
    Menghapus stopwords dari teks sebelum evaluasi F1.
    """
    words = text.split()
    filtered_words = [word for word in words if word not in STOPWORDS_ID]
    return ' '.join(filtered_words)

def exact_match(prediction, ground_truth):
    """
    Hitung Exact Match (EM), bernilai 1 jika ground truth ada di dalam hasil prediksi.
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    return int(normalized_ground_truth in normalized_prediction)

def f1_score(prediction, ground_truth):
    """
    Hitung Unigram F1 Score antara jawaban prediksi dan jawaban referensi.
    """
    prediction_tokens = remove_stopwords(normalize_answer(prediction)).split()
    ground_truth_tokens = remove_stopwords(normalize_answer(ground_truth)).split()

    common_tokens = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common_tokens.values())

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)  # Jika kosong, hanya bisa sama persis

    if num_same == 0:
        return 0  # Tidak ada kata yang cocok

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1

def evaluate_em_f1(generated_answer, reference_answer):
    """
    Evaluasi EM dan F1 dengan membandingkan jawaban model dengan semua referensi.
    """
    em_score = exact_match(generated_answer, reference_answer)
    f1_score_value = f1_score(generated_answer, reference_answer)

    return em_score, f1_score_value  # Ambil skor tertinggi dari semua referensi


def generate_answers_and_evaluate(
    dataset: Dataset,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 50,
):
    """
    Generate completions with and without summary, and evaluate EM & F1 scores.
    """
    config = detect_device()

    processed_results = []

    for i in tqdm(range(len(dataset)), desc="Generating responses"):
        query = dataset['query'][i]
        summary = dataset['summary'][i]
        answer = dataset['answer'][i] 
        passages = dataset['formatted_passages'][i] 

        # 🔄 **Generate with summary (w_summary)**
        # messages_w_summary = [
        #     {"role": "user", "content": f"Konteks: {summary}\nPertanyaan: {query}"}
        # ]
        messages_w_summary = [
            {
                "role": "user",
                "content": f"Konteks: {dataset['summary'][i]}\nBerdasarkan konteks sebelumnya, jawab pertanyaan berikut. Pertanyaan: {dataset['query'][i]}"
            }
        ]

        inputs_w_summary = prepare_inputs(format_chat_prompt(messages_w_summary, tokenizer), tokenizer, config.device_type)
        result_w_summary = model.generate(
            inputs_w_summary["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True
        )

        # 🔄 **Generate without summary (wo_summary)**
        # messages_wo_summary = [
        #     {"role": "user", "content": query}
        # ]
        messages_wo_summary = [
            {
                "role": "user",
                "content": dataset['query'][i]
            }
        ]

        inputs_wo_summary = prepare_inputs(format_chat_prompt(messages_wo_summary, tokenizer), tokenizer, config.device_type)
        result_wo_summary = model.generate(
            inputs_wo_summary["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True
        )

        # ✅ **Remove prompt input from the completion using token slicing**
        w_summary_token_ids = inputs_w_summary['input_ids'][0]  # Tokenized input
        w_summary_generated_ids = result_w_summary.sequences[0][len(w_summary_token_ids):]  # Generated tokens
        completion_w_summary = tokenizer.decode(w_summary_generated_ids, skip_special_tokens=True)

        wo_summary_token_ids = inputs_wo_summary['input_ids'][0]
        wo_summary_generated_ids = result_wo_summary.sequences[0][len(wo_summary_token_ids):]
        completion_wo_summary = tokenizer.decode(wo_summary_generated_ids, skip_special_tokens=True)

        # 🔍 **Evaluate Exact Match (EM) and F1 Scores**
        em_w_summary, f1_w_summary = evaluate_em_f1(completion_w_summary, answer.strip())
        em_wo_summary, f1_wo_summary = evaluate_em_f1(completion_wo_summary, answer.strip())

        # 🏷 **Determine final_summary based on conditions**
        if (em_wo_summary == 1 and em_wo_summary > em_w_summary) or (f1_wo_summary > f1_w_summary) or (f1_wo_summary == f1_w_summary):
            final_summary = ""
        else:
            final_summary = summary

        # 🔄 **Store results**
        processed_results.append({
            "query": query,
            "passages": passages,
            "summary": summary, 
            "final_summary": final_summary, 
            "answer": answer,
            "generated_results": {
                "w_summary": {
                    "completion": completion_w_summary,
                    "em": em_w_summary,
                    "f1": f1_w_summary
                },
                "wo_summary": {
                    "completion": completion_wo_summary,
                    "em": em_wo_summary,
                    "f1": f1_wo_summary
                }
            }
        })

    return processed_results

def generate_answer_and_do_scoring(
    dataset: Dataset,
    query_col: str, 
    summary_col: str, 
    label_col: str, 
    passages_col: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 50, 
    max_source_length: int = 512
):
    """
    Generate completions with summary only, and evaluate EM & F1 scores.
    Return query, passages, summary, label (answer), generated_answer, em, f1.
    """
    config = detect_device()

    results = []
    tokenize_kwargs = {}
    if max_source_length is not None:
        tokenize_kwargs = {
            'max_length':max_source_length
        }

    for i in tqdm(range(len(dataset)), desc="Generating responses with summary only"):
        query = dataset[query_col][i]
        summary = dataset[summary_col][i]
        label = dataset[label_col][i]
        passages = dataset[passages_col][i]

        # Generate with summary (w_summary) only
        messages_w_summary = [
            {
                "role": "user",
                "content": f"Konteks: {summary}\nBerdasarkan konteks sebelumnya, jawab pertanyaan berikut. Pertanyaan: {query}"
            }
        ]

        inputs_w_summary = prepare_inputs(format_chat_prompt(messages_w_summary, tokenizer), tokenizer, config.device_type, **tokenize_kwargs)
        result_w_summary = model.generate(
            inputs_w_summary["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True
        )

        # Remove prompt tokens from generated tokens
        w_summary_token_ids = inputs_w_summary['input_ids'][0]
        w_summary_generated_ids = result_w_summary.sequences[0][len(w_summary_token_ids):]
        completion_w_summary = tokenizer.decode(w_summary_generated_ids, skip_special_tokens=True)

        # Evaluate EM and F1 for w_summary only
        em_w_summary, f1_w_summary = evaluate_em_f1(completion_w_summary, label.strip())

        # Simpan hasil yang diminta
        results.append({
            "query": query,
            "passages": passages,
            "summary": summary,
            "label": label,
            "generated_answer": completion_w_summary,
            "em": em_w_summary,
            "f1": f1_w_summary,
        })

    return results

