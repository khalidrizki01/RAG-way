from utils import get_w_wo_preceding_space_variants
from transformers import AutoTokenizer
from datasets import Dataset
from functools import reduce

def get_nested_value(d, key_path):
    """
    Ambil nilai dari dictionary bersarang berdasarkan string path dengan notasi titik.
    
    Args:
        d (dict): Dictionary yang ingin diakses.
        key_path (str): String dengan format "key.subkey.index" contoh: "answers.text.0".
        
    Returns:
        Nilai yang diambil dari dictionary sesuai dengan path, atau None jika path tidak ditemukan.
    """
    keys = key_path.split('.')  # Pecah string key_path menjadi list berdasarkan titik
    try:
        return reduce(lambda d, key: d[int(key)] if key.isdigit() else d[key], keys, d)
    except (KeyError, IndexError, TypeError):
        return None  # Mengembalikan None jika kunci tidak ditemukan


def find_exact_sequence(answer_tokens_variations, generated_token_ids, logprobs, tokenizer):
    """
    Find exact token sequences in the generated tokens and compute logprobs.

    Args:
        answer_tokens_variations (list): List of tokenized answer variations.
        generated_token_ids (list): List of token IDs from the generated text.
        logprobs (list): List of log probabilities corresponding to the token IDs.
        tokenizer (AutoTokenizer): The tokenizer instance.

    Returns:
        tuple: (found_exact_sequence (bool), answer_logprobs (dict))
    """
    answer_logprobs = {}
    for answer_tokens in answer_tokens_variations:
        for start_idx in range(len(generated_token_ids) - len(answer_tokens) + 1):
            if generated_token_ids[start_idx:start_idx + len(answer_tokens)] == answer_tokens:
                total_logprob = 0.0

                for i, token_id in enumerate(answer_tokens):
                    token_logprob = logprobs[start_idx + i]
                    answer_logprobs[tokenizer.decode([token_id])] = token_logprob
                    total_logprob += token_logprob
                avg_answer_logprobs = total_logprob / len(answer_tokens)
                return True, answer_logprobs, avg_answer_logprobs
            
    return False, answer_logprobs, float('-inf')

def process_word_by_word(answer_words, tokenizer, generated_token_ids, logprobs):
    """
    Process each word in the answer to find logprobs in the generated tokens.

    Args:
        answer_words (list): List of words from the ground truth answer.
        tokenizer (AutoTokenizer): The tokenizer instance.
        generated_token_ids (list): List of token IDs from the generated text.
        logprobs (list): List of log probabilities corresponding to the token IDs.

    Returns:
        tuple: (all_words_found (bool), answer_logprobs (dict))
    """
    answer_tokens_variants = []
    for word in answer_words:
        tokenized_variants = get_w_wo_preceding_space_variants(word, tokenizer)
        if not any(tokenized_variants):
            print(f"[WARNING] Tidak ada token yang dihasilkan untuk kata: {word}")

        answer_tokens_variants.append(tokenized_variants)

    answer_logprobs = {}
    all_words_found = True
    total_logprob = 0.0
    token_count = 0

    for word_variants in answer_tokens_variants:
        word_found = False
        for word_tokens in word_variants:
            for start_idx in range(len(generated_token_ids) - len(word_tokens) + 1):
                if generated_token_ids[start_idx:start_idx + len(word_tokens)] == word_tokens:
                    word_found = True
                    for i, token_id in enumerate(word_tokens):
                        if tokenizer.decode([token_id]) not in answer_logprobs:
                            token_logprob = logprobs[start_idx + i]
                            answer_logprobs[tokenizer.decode([token_id])] = token_logprob
                            total_logprob += token_logprob
                            token_count += 1
                    break
            if word_found:
                break

        if not word_found:
            all_words_found = False
            for token_id in word_variants[0]:
                if tokenizer.decode([token_id]) not in answer_logprobs:
                    answer_logprobs[tokenizer.decode([token_id])] = None

    # Hitung rata-rata log probability jika ditemukan token yang sesuai
    avg_answer_logprobs = total_logprob / token_count if token_count > 0 else float('-inf')

    return all_words_found, answer_logprobs, avg_answer_logprobs

def process_row(row, answer_col, token_col, logprobs_col, tokenizer, suffix=""):
    """
    Process a single row to extract logprobs for ground-truth answers.

    Args:
        row (dict): A row from the dataset.
        answer_col (str): Path to the answer field in the row.
        token_col (str): Path to the token IDs in the row.
        logprobs_col (str): Path to the log probabilities in the row.
        tokenizer (AutoTokenizer): Tokenizer instance.
        prefix (str): Prefix for the returned field names.

    Returns:
        dict: Processed row with prefixed additional fields.
    """
    answer = get_nested_value(row, answer_col)
    generated_token_ids = get_nested_value(row, token_col)
    logprobs = get_nested_value(row, logprobs_col)

    # Periksa apakah ada yang None sebelum melanjutkan
    if answer is None or generated_token_ids is None or logprobs is None:
        print(f"[WARNING] Data tidak ditemukan pada row: answer={answer}, token_ids={generated_token_ids}, logprobs={logprobs}")
        result = {f"answer_logprobs_{suffix}" : {
                    "all_logprobs": {}, 
                    "avg_logprobs": float('-inf'), 
                    "include_answer": False
        }}
        return result

    # Dapatkan variasi tokenisasi dengan dan tanpa spasi
    answer_tokens_variations = get_w_wo_preceding_space_variants(answer, tokenizer)

    # Prioritas 1: Cari subsekuens yang sama persis
    found_exact_sequence, answer_logprobs, avg_logprobs = find_exact_sequence(
        answer_tokens_variations, generated_token_ids, logprobs, tokenizer
    )

    if not found_exact_sequence:
        # Jika tidak ditemukan subsekuens yang sama persis, lakukan splitting dan pencocokan per kata
        answer_words = answer.split()
        all_words_found, answer_logprobs, avg_logprobs = process_word_by_word(
            answer_words, tokenizer, generated_token_ids, logprobs
        )
        include_answer = all_words_found
    else:
        include_answer = True

    result = {f"answer_logprobs_{suffix}" : {
                    "all_logprobs": answer_logprobs, 
                    "avg_logprobs": avg_logprobs, 
                    "include_answer": include_answer
    }}
    return result


# Penggunaan langsung pada dataset
def retrieve_groundtruth_token_logprobs(
        tokenizer: AutoTokenizer, 
        dataset: Dataset, 
        answer_col="answers.text.0", 
        token_col="generation.token_ids", 
        logprobs_col="generation.logprobs", 
        logprobs_col_prefix=""
    ) -> Dataset:
    """
    Calculate log probabilities for the given dataset.

    Args:
        dataset (Dataset): Hugging Face dataset.
        tokenizer (AutoTokenizer): Pre-trained tokenizer.

    Returns:
        Dataset: Processed dataset with added columns.
    """

    return dataset.map(lambda row: process_row(
        row, 
        answer_col, 
        token_col, 
        logprobs_col, 
        tokenizer, 
        suffix=logprobs_col_prefix
        ))

def determine_final_summary(row, w_summary_col="answer_logprobs_w_summary.avg_logprobs", 
                            wo_summary_col="answer_logprobs_NO_summary.avg_logprobs", 
                            summary_col="summary"):
    """
    Menentukan apakah akan melampirkan rangkuman berdasarkan perbandingan avg_logprobs.

    Args:
        row (dict): Satu baris data dari dataset.
        w_summary_col (str): Jalur kolom untuk avg_logprobs dengan rangkuman.
        wo_summary_col (str): Jalur kolom untuk avg_logprobs tanpa rangkuman.
        summary_col (str): Nama kolom yang berisi rangkuman.

    Returns:
        dict: Dengan kolom baru 'final_summary'.
    """
    # Ambil nilai log probabilities dengan dan tanpa summary
    w_summary = get_nested_value(row, w_summary_col)
    wo_summary = get_nested_value(row, wo_summary_col)
    summary = row[summary_col]

    # Tentukan final summary berdasarkan kondisi yang diberikan
    final_summary = summary if w_summary > wo_summary else ""

    return {
        "final_summary": final_summary
    }