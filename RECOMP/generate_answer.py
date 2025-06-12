from utils import detect_device, format_chat_prompt, prepare_inputs
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import re
import string
from collections import Counter
from nltk.corpus import stopwords
from typing import Optional

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

def evaluate_em_f1(generated_answer: str, reference_answer: str)-> tuple[float, float]:
    """
    Evaluasi EM dan F1 dengan membandingkan jawaban model dengan semua referensi.
    """
    em_score = exact_match(generated_answer, reference_answer)
    f1_score_value = f1_score(generated_answer, reference_answer)

    return em_score, f1_score_value  # Ambil skor tertinggi dari semua referensi


def build_prompt(query: str, summary: Optional[str] = None) -> str:
    if summary:
        return f"Konteks: {summary}\nBerdasarkan konteks sebelumnya, jawab pertanyaan berikut. Pertanyaan: {query}"
    return query

def generate_completion(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device_type: str,
    max_new_tokens: int = 50,
    max_source_length: int = 512
) -> str:
    tokenize_kwargs = {}
    if max_source_length is not None:
        tokenize_kwargs['max_length'] = max_source_length

    inputs = prepare_inputs(format_chat_prompt([{"role": "user", "content": prompt}], tokenizer), tokenizer, device_type, **tokenize_kwargs)
    input_ids = inputs["input_ids"]
    
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True
    )
    generated_ids = output.sequences[0][len(input_ids[0]):]
    completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return completion

def generate_answers_and_compare_between_with_and_without_summary(
    dataset: Dataset,
    passages_column:str,
    query_column: str, 
    label_column: str, 
    summary_column: str,  
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 52,
    max_source_length: int = 512
):
    """
    Generate completions with and without summary, evaluate EM & F1 scores,
    and decide final summary selection strategy.
    """
    config = detect_device()
    results = []

    for i in tqdm(range(len(dataset)), desc="Generating responses (w/ & wo/ summary)"):
        query = dataset[query_column][i]
        summary = dataset[summary_column][i]
        answer = dataset[label_column][i]
        passages = dataset[passages_column][i]

        # Generate with summary
        prompt_w_summary = build_prompt(query, summary)
        completion_w_summary = generate_completion(prompt_w_summary, model, tokenizer, config.device_type, max_new_tokens, max_source_length)
        em_w, f1_w = evaluate_em_f1(completion_w_summary.strip(), answer.strip())

        # Generate without summary
        prompt_wo_summary = build_prompt(query, None)
        completion_wo_summary = generate_completion(prompt_wo_summary, model, tokenizer, config.device_type, max_new_tokens, max_source_length)
        em_wo, f1_wo = evaluate_em_f1(completion_wo_summary.strip(), answer.strip())

        # Determine final_summary
        if (em_wo == 1 and em_wo > em_w) or (f1_wo > f1_w) or (f1_wo == f1_w):
            final_summary = ""
        else:
            final_summary = summary

        results.append({
            "query": query,
            "passages": passages,
            "summary": summary,
            "final_summary": final_summary,
            "answer": answer,
            "generated_results": {
                "w_summary": {
                    "completion": completion_w_summary,
                    "em": em_w,
                    "f1": f1_w
                },
                "wo_summary": {
                    "completion": completion_wo_summary,
                    "em": em_wo,
                    "f1": f1_wo
                }
            }
        })

    return results

def generate_answer_and_do_scoring(
    dataset: Dataset,
    query_col: str, 
    summary_col: str, 
    label_col: str, 
    passages_col: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 52, 
    max_source_length: int = 512
):
    config = detect_device()
    results = []

    for i in tqdm(range(len(dataset)), desc="Generating responses with summary only"):
        query = dataset[query_col][i]
        summary = dataset[summary_col][i]
        label = dataset[label_col][i]
        passages = dataset[passages_col][i]

        prompt = f"Konteks: {summary}\nBerdasarkan konteks sebelumnya, jawab pertanyaan berikut. Pertanyaan: {query}"
        completion = generate_completion(prompt, model, tokenizer, config.device_type, max_new_tokens, max_source_length)
        em, f1 = evaluate_em_f1(completion.strip(), label.strip())

        results.append({
            "query": query,
            "passages": passages,
            "summary": summary,
            "label": label,
            "generated_answer": completion,
            "em": em,
            "f1": f1,
        })

    return results
