from nltk.corpus import stopwords
from collections import Counter
import string
import re

try: 
    STOPWORDS_ID = set(stopwords.words('indonesian'))
except LookupError:
    import nltk
    nltk.download('stopwords')
    STOPWORDS_ID = set(stopwords.words('indonesian'))

def normalize_answer(s):
    """
    Normalisasi jawaban untuk menghilangkan variasi yang tidak penting dalam perhitungan substring match dan F1.
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

def substring_match(prediction, ground_truth):
    """
    Hitung substring match, bernilai 1 jika ground truth ada di dalam hasil prediksi.
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

def evaluate_substringmatch_f1(generated_answer: str, reference_answer: str)-> tuple[float, float]:
    """
    Evaluasi substring match dan F1 dengan membandingkan jawaban model dengan semua referensi.
    """
    sm_score = substring_match(generated_answer, reference_answer)
    f1_score_value = f1_score(generated_answer, reference_answer)

    return sm_score, f1_score_value