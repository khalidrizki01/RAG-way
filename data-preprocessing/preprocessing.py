import torch.nn.functional as F
from typing import Optional
from tqdm import tqdm
import torch

# Fungsi pooling dari model card
def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
def rank_ctxs_by_query_similarity(query, ctxs, labels, tokenizer, model):
    if not ctxs:
        return []

    input_texts = ["query: " + query] + ["passage: " + c for c in ctxs]
    query_and_ctxs = tokenizer(input_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

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