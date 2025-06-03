from preprocessing import get_retrieval_embeds
from transformers import PreTrainedTokenizer, StoppingCriteria, StoppingCriteriaList
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from typing import List
import torch

class MultiTokenEOSCriteria(StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker

def stop_sequences_criteria(
    tokenizer: PreTrainedTokenizer,
    initial_decoder_input_length: int,
    batch_size: int,
    stop_sequences: List[str] = ['\n', '.', ','],
    ) -> StoppingCriteriaList:
    return StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

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