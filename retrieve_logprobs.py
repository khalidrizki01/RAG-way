from utils import get_w_wo_preceding_space_variants
from transformers import AutoTokenizer
import pandas as pd

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
                for i, token_id in enumerate(answer_tokens):
                    token_logprob = logprobs[start_idx + i]
                    answer_logprobs[tokenizer.decode([token_id])] = token_logprob
                return True, answer_logprobs
    return False, answer_logprobs

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
        answer_tokens_variants.append(tokenized_variants)

    answer_logprobs = {}
    all_words_found = True

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
                    break
            if word_found:
                break

        if not word_found:
            all_words_found = False
            for token_id in word_variants[0]:
                if tokenizer.decode([token_id]) not in answer_logprobs:
                    answer_logprobs[tokenizer.decode([token_id])] = None

    return all_words_found, answer_logprobs

### - tidak harus berurutan, 
### - menguji 2 variasi: dengan dan tanpa spasi sebelum
### - mengutamakan mengecek subsekuens berurutan dan mengambil logprobsnya dulu. jika ga ada, maka baru ambil per kata (ga harus berurutan)
def retrieve_correct_token_logprobs(
    data: pd.DataFrame,
    tokenizer: AutoTokenizer
) -> pd.DataFrame:
    """
    Menghitung log probabilities untuk tiap kata dalam jawaban ground truth
    berdasarkan jawaban hasil generasi, dengan memberikan prioritas pada subsekuens berurutan.

    Args:
        final_output (pd.DataFrame): DataFrame yang memiliki kolom berikut:
            - 'answer': Jawaban ground truth.
            - 'token_ids': Daftar ID token dari hasil generasi (list[int]).
            - 'logprobs': Log probabilities yang terkait dengan 'token_ids' (list[float]).
        tokenizer (AutoTokenizer): Tokenizer dari Hugging Face yang digunakan untuk tokenisasi.

    Returns:
        pd.DataFrame: DataFrame yang sama dengan kolom tambahan:
                      - 'answer_logprobs': Dictionary dengan token sebagai key dan logprobs sebagai value.
                      - 'include_answer': Boolean, True jika semua kata ditemukan dalam hasil generasi.
    """
    # Inisialisasi kolom baru untuk menyimpan logprobs dan flag keberadaan jawaban
    data["answer_logprobs"] = None
    data["include_answer"] = False  # Default ke False

    # Iterasi menggunakan iterrows
    for idx, row in data.iterrows():
        answer = row['answer']
        generated_token_ids = row['token_ids']
        logprobs = row['logprobs']

        # Tambahkan spasi secara manual
        answer_tokens_variations = get_w_wo_preceding_space_variants(row['answer'], tokenizer)

        # Prioritas 1: Cari subsekuens yang sama persis
        found_exact_sequence, answer_logprobs = find_exact_sequence(
            answer_tokens_variations, generated_token_ids, logprobs, tokenizer
        )

        # Jika tidak ditemukan subsekuens yang sama persis, lakukan splitting dan pencocokan per kata
        if not found_exact_sequence:
            answer_words = answer.split()
            all_words_found, answer_logprobs = process_word_by_word(
                answer_words, tokenizer, generated_token_ids, logprobs
            )
            data.at[idx, 'include_answer'] = all_words_found

        else:
            # Jika ditemukan subsekuens yang sama persis, set 'include_answer' ke True
            data.at[idx, 'include_answer'] = True

        # Simpan hasil ke kolom baru
        data.at[idx, 'answer_logprobs'] = answer_logprobs

    return data