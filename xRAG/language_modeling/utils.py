from preprocessing import get_retrieval_embeds
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch

def get_nll_loss(logits, labels, vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    # print("NLL Loss:", loss.item())  # loss.item() untuk menampilkan nilai loss
    return loss

def get_kl_loss(teacher_logits,student_logits,student_labels,teacher_labels,temperature,distill_topk=None):
    
    ## make sure the teacher_logits and student_logits have the same shape
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    _,_,vocab_size = student_logits.shape

    ## only compute loss in the completion part, not propmt
    
    student_mask = (student_labels!=-100).unsqueeze(-1).expand_as(student_logits) ## batch_size,num_tokens,vocab_size
    student_logits_selected = torch.masked_select(student_logits,student_mask).view(-1,vocab_size)

    teacher_mask = (teacher_labels != -100).unsqueeze(-1).expand_as(teacher_logits)
    teacher_logits_selected = torch.masked_select(teacher_logits,teacher_mask).view(-1,vocab_size)

    assert teacher_logits_selected.shape == student_logits_selected.shape, (f"The shape of teacher logits is {teacher_logits_selected.shape}, while that of student is {student_logits_selected.shape}")

    kl_loss = loss_fct(
        F.log_softmax(student_logits_selected / temperature, dim=-1),
        F.softmax(    teacher_logits_selected / temperature, dim=-1),
    ) * temperature ** 2 
    # print("KL  Loss:", kl_loss.item())
    
    return kl_loss

def validate_during_pretrain(model, dataloader, vocab_size):
    model.eval()
    total_loss = []
    
    # Nonaktifkan perhitungan gradien untuk menghemat memori
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", unit="batch"):
            outputs = model(
                input_ids = batch['xrag_input_ids'],
                attention_mask = batch['xrag_attention_mask'],
                retrieval_embeds = batch['retriever_embeddings'],
            )

            del batch['xrag_input_ids']
            del batch['xrag_attention_mask']
            del batch['retriever_embeddings']
            torch.cuda.empty_cache()

            nll_loss = get_nll_loss(
                labels = batch['xrag_labels'],
                logits = outputs.logits,
                vocab_size = vocab_size,
            )
            total_loss.append(nll_loss.item())
            del batch['xrag_labels']
    
    model.train()  # Kembali ke mode training
    ppl = torch.exp(torch.tensor(sum(total_loss)/len(total_loss)))
    return ppl