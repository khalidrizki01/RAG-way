### VERSION 4
### - tidak harus berurutan, 
### - menguji 2 variasi: dengan dan tanpa spasi sebelum
### - mengutamakan mengecek subsekuens berurutan dan mengambil logprobsnya dulu. jika ga ada, maka baru ambil per kata (ga harus berurutan)

from transformers import AutoTokenizer
import pandas as pd

def retrieve_correct_token_logprobs(
    final_output: pd.DataFrame,
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
    final_output["answer_logprobs"] = None
    final_output["include_answer"] = False  # Default ke False

    # Periksa apakah tokenizer mendukung `add_prefix_space`
    supports_prefix_space = "add_prefix_space" in tokenizer.__call__.__code__.co_varnames

    # Iterasi menggunakan iterrows
    for idx, row in final_output.iterrows():
        # Encode jawaban ground truth sebagai satu subsekuens
        answer = row['answer']
        answer_tokens_no_space = tokenizer.encode(answer, add_special_tokens=False, add_prefix_space=False)
        answer_tokens_with_space = tokenizer.encode(answer, add_special_tokens=False, add_prefix_space=True)

        # Get generated token IDs dan logprobs
        generated_token_ids = row['token_ids']
        logprobs = row['logprobs']

        # Initialize dictionary untuk menyimpan logprobs
        answer_logprobs = {}

        # Prioritas 1: Cari subsekuens yang sama persis
        found_exact_sequence = False

        for answer_tokens in [answer_tokens_no_space, answer_tokens_with_space]:
            for start_idx in range(len(generated_token_ids) - len(answer_tokens) + 1):
                if generated_token_ids[start_idx:start_idx + len(answer_tokens)] == answer_tokens:
                    found_exact_sequence = True
                    # Ambil logprobs untuk semua token dalam subsekuens
                    for i, token_id in enumerate(answer_tokens):
                        token_logprob = logprobs[start_idx + i]
                        answer_logprobs[tokenizer.decode([token_id])] = token_logprob
                    break
            if found_exact_sequence:
                break

        # Jika tidak ditemukan subsekuens yang sama persis, lakukan splitting dan pencocokan per kata
        if not found_exact_sequence:
            answer_words = answer.split()

            # Tokenisasi setiap kata menjadi token ID dengan dua variasi
            answer_tokens_variants = []
            for word in answer_words:
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
                answer_tokens_variants.append(tokenized_variants)

            # Flag untuk mencatat apakah semua kata ditemukan
            all_words_found = True

            for word_variants in answer_tokens_variants:
                word_found = False
                for word_tokens in word_variants:  # Cek kedua variasi tokenisasi
                    for start_idx in range(len(generated_token_ids) - len(word_tokens) + 1):
                        if generated_token_ids[start_idx:start_idx + len(word_tokens)] == word_tokens:
                            word_found = True
                            # Ambil logprobs untuk token-token kata tersebut
                            for i, token_id in enumerate(word_tokens):
                                if tokenizer.decode([token_id]) not in answer_logprobs:
                                    token_logprob = logprobs[start_idx + i]
                                    answer_logprobs[tokenizer.decode([token_id])] = token_logprob
                            break
                    if word_found:
                        break

                if not word_found:
                    all_words_found = False
                    for token_id in word_variants[0]:  # Ambil tokenisasi normal
                        if tokenizer.decode([token_id]) not in answer_logprobs:
                            answer_logprobs[tokenizer.decode([token_id])] = None

            # Update kolom 'include_answer' berdasarkan hasil pencocokan per kata
            final_output.at[idx, 'include_answer'] = all_words_found

        else:
            # Jika ditemukan subsekuens yang sama persis, set 'include_answer' ke True
            final_output.at[idx, 'include_answer'] = True

        # Simpan hasil ke kolom baru
        final_output.at[idx, 'answer_logprobs'] = answer_logprobs

    return final_output