import random, copy, torch, tqdm
from datasets import Dataset, load_from_disk, load_dataset
import yaml
from types import SimpleNamespace
from utils import stop_sequences_criteria

XRAG_TOKEN = "<xRAG>" 

ParaphraseInstructions = [
    'Informasi: {xrag_token} berarti sama dengan',
    "Konteks: {xrag_token} Bisakah kamu mengungkapkan kalimat di atas dengan kata-katamu sendiri?",
    "Penjelasan: {xrag_token} Harap berikan penafsiran ulang dari teks penjelasan sebelumnya.",
    "Pernyataan ini pada dasarnya setara:\n(1) {xrag_token}\n(2)",
    "Konteks: {xrag_token} adalah parafrase dari apa?",
    "Penjelasan: {xrag_token} Bisakah kamu memberikan versi yang berbeda dari kalimat penjelasan di atas?",
    "Dengan kata lain, informasi: {xrag_token} hanyalah cara lain untuk mengatakan:",
    "Kamu menyampaikan poin yang sama baik kamu mengatakan konteks: {xrag_token} atau",
    "Penjelasan: {xrag_token} Setelah mengurai ide-ide dalam informasi di atas, kita mendapatkan:",
    "Informasi: {xrag_token} Harap tawarkan pernyataan ulang dari kalimat penjelasan yang baru saja saya baca.",
    "Konteks: {xrag_token}, yang juga berarti:",
    "Hilangkan misterinya, dan kamu akan menemukan informasi: {xrag_token} hanyalah versi lain dari:",
    "Esensi dari konteks: {xrag_token} terulang lagi dalam pernyataan berikut:",
]

def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)

def load_and_format_dataset(dataset_path, task_type,psg_col,query_col=None, answer_col=None, max_rows=None):
    """
    Mengubah format dari dataset raw ke format yang sesuai untuk xRAG
    """
    # Memuat dataset yang disimpan dalam format DatasetDict (train, dev, test)
    dataset = load_dataset(dataset_path)
    # Iterasi untuk setiap split dalam DatasetDict
    for split_name, split_data in dataset.items():
        # Jika max_rows diterapkan, hanya pilih jumlah baris yang ditentukan
        if max_rows is not None:
            split_data = split_data.select(range(max_rows))

        if task_type == 'finetune':
            # Gunakan fungsi untuk task_type 'finetune'
            formatted_data = process_finetune_dataset(split_data, query_col, answer_col, psg_col)
        elif task_type == 'pretrain':
            # Gunakan fungsi untuk task_type 'pretrain'
            formatted_data = process_pretrain_dataset(split_data, psg_col)

        # Setelah selesai memproses seluruh split, buat dataset baru untuk masing-masing split
        formatted_split_dataset = Dataset.from_dict(formatted_data)

        # Simpan hasil format untuk setiap split (train, dev, test)
        dataset[split_name] = formatted_split_dataset

    return dataset


def process_finetune_dataset(split_data, query_col, answer_col, psg_col):
    """
    Fungsi untuk memproses dataset saat task_type adalah 'finetune'
    """
    formatted_data = {
        "background": [],
        "messages": []
    }
    
    # Iterasi untuk setiap contoh dalam split_data
    for example in split_data:
        # Ambil query dan answers untuk messages
        query = example[query_col]
        answers = example[answer_col]

        # Membuat pasangan message
        messages = [
            {"role": "user", "content": f"{query}\n"},
            {"role": "assistant", "content": answers}
        ]

        background = example[psg_col] 

        # Menambahkan data ke formatted_data
        formatted_data['background'].append(background)
        formatted_data['messages'].append(messages)
    
    return formatted_data


def process_pretrain_dataset(split_data, psg_col):
    """
    Fungsi untuk memproses dataset saat task_type adalah 'pretrain'
    """
    formatted_data = {
        "text": []  # Menyimpan hanya kolom 'text' untuk task_type 'pretrain'
    }

    # Iterasi untuk setiap contoh dalam split_data
    for example in split_data:
        # Ambil background dan extend hasilnya ke dalam list of strings
        text = example[psg_col]  # .split('\n\n')
        formatted_data['text'].append(text)  # Menggunakan extend, bukan append
    
    return formatted_data

def _concat_messages_llama(messages, llm_tokenizer):

    message_text = "<|begin_of_text|>"
    for message in messages:
        if message["role"] == "user":
            message_text += f"<|start_header_id|>user<|end_header_id|>{message['content'].strip()}<|eot_id|>"
        elif message["role"] == "assistant":
            message_text += f"<|start_header_id|>assistant<|end_header_id|>{message['content'].strip()}<|eot_id|>"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

def _concat_messages_qwen(messages, llm_tokenizer, add_generation_prompt=False):
    message_text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    return message_text

def truncate_using_retriever_tokenizer(retriever_tokenizer, text, context_length) :
    # 1. Tokenisasi dengan retriever_tokenizer
    tokenized_example_retriever = retriever_tokenizer(text, return_tensors='pt', max_length=context_length, truncation=True, add_special_tokens=False)
    
    # 2. Decode hasil tokenisasi untuk memastikan input yang setara
    decoded_example = retriever_tokenizer.decode(tokenized_example_retriever.input_ids[0], skip_special_tokens=True)
    return decoded_example

def _encode_chat_format(
        messages,
        llm_tokenizer,
        max_seq_length,
        chat_format='qwen',
    ):

    # _concat_messages = eval(f"_concat_messages_{chat_format}")
    if chat_format == 'llama':
        _concat_messages = _concat_messages_llama
    elif chat_format =='qwen':
        _concat_messages = _concat_messages_qwen
    else:
        raise ValueError(f"Invalid chat_format: {chat_format}. Must be 'qwen' or 'llama'.")
    
    example_text = _concat_messages(messages, llm_tokenizer=llm_tokenizer).strip()
    tokenized_example = llm_tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, padding=True, truncation=True, add_special_tokens=False)
    # KALAU BACKGROUND KEPANJANGAN (melebihi max_sequence_length), MAKA CONTENT DR ASSISTANT TIDAK AKAN TERTOKENISASI KARENA SUDAH KEPOTONG OLEH TRUNCATION
    # alhasil akan kena filter oleh train_dataset.filter(lambda example: (example['labels'] != -100).any())
    # dan itu gapapa, karena hanya menyusun sekitar 123 baris train dan 17 baris dev. masih ada 4419 baris untuk latihan dan 1126 baris untuk dev
    
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    
    # Temukan indeks pertama dari id 151668
    stop_index = (input_ids == 151668).nonzero(as_tuple=True)[1].item() if (input_ids == 151668).any() else -1
    
    # Jika ada id 151668 dalam input_ids, atur label hingga indeks tersebut menjadi -100
    if stop_index != -1:
        labels[:, :stop_index+1] = -100

        # Jika ada id 271 setelah 151668, beri label -100
        next_index = stop_index + 1
        if next_index < input_ids.shape[1] and input_ids[0, next_index] == 271:
            labels[:, next_index] = -100

    # Membagi input_ids menjadi dua bagian: yang tidak dimask dan yang dimask
    prompt_ids = input_ids[:, :next_index+1]  # Bagian sebelum 151668
    label_ids = input_ids[:, next_index+1:]  # Bagian setelah 151668 yang sudah dimask (-100)
    # Decoding untuk mendapatkan teks dari prompt dan label
    prompt_text = llm_tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
    label_text = llm_tokenizer.decode(label_ids[0], skip_special_tokens=False)
    
    # assert tokenizer.eos_token_id in input_ids, input_ids
    return {
        "input_ids":input_ids.flatten(),
        "labels":labels.flatten(),
        "prompt_ids": prompt_ids,
        "label_ids": label_ids, 
        "prompt_text": prompt_text,  
        "label_text": label_text,  
    }

def encode_with_chat_format_pretrain(
        example,
        llm_tokenizer,
        max_seq_length,
        retrieval_embed_length,
        chat_format='qwen',
        ):
    """
    encode messages into input_ids and labels for paraphrase pretrain
    
    Return:
        input_ids,labels and retriever_input_text
    """

    document = "passage: " + example['text'].strip()
    # truncated_document = truncate_using_retriever_tokenizer(retriever_tokenizer, document, retrieval_context_length)
    xrag_token = " ".join([XRAG_TOKEN] * retrieval_embed_length)  
    instruction = random.choice(ParaphraseInstructions).format_map(dict(xrag_token=xrag_token))

    messages = [
        {"role":"user","content":instruction},
        {"role":"assistant","content":document},  # truncated_document
    ]

    encoded = _encode_chat_format(
        messages=messages,
        llm_tokenizer=llm_tokenizer, 
        max_seq_length=max_seq_length,
        chat_format=chat_format)

    return {
        "xrag_input_ids":encoded['input_ids'],
        "xrag_labels":encoded['labels'],
        "retriever_input_text":document,  # truncated_document
        "prompt_ids": encoded['prompt_ids'],
        "label_ids": encoded['label_ids'] ,
        "prompt_text": encoded['prompt_text'], 
        "label_text": encoded['label_text']
    }

import random
import copy

def encode_with_chat_format_finetune(
        example, 
        llm_tokenizer,
        max_seq_length,
        use_rag_tuning=True,
        use_retriever_embed=False,
        chat_format='qwen'
    ):
    """
    Fungsi untuk melakukan encoding dengan format chat, mendukung multiple background.
    """
    messages, backgrounds = example['messages'], example['background']

    ret = {}
    
    # # Menentukan template instruksi secara acak jika tidak diberikan
    # instruction_templates = [
    #     "Berdasarkan latar belakang berikut, {backgrounds} jawab pertanyaan ini. {query}",
    #     "Lihat informasi berikut, {backgrounds} dan berikan jawaban Anda. {query}",
    #     "Dengan latar belakang ini, {backgrounds} tolong jawab pertanyaan di bawah. {query}",
    #     "Gunakan informasi berikut, {backgrounds} untuk menjawab pertanyaan. {query}"
    # ]
    
    # # Memilih template instruksi secara acak
    # instruct = random.choice(instruction_templates)

    instruct = "Rujuklah latar belakang: {backgrounds} Pertanyaan: {query}"

    if use_rag_tuning and use_retriever_embed:
        formatted_backgrounds = ["passage: " + background for background in backgrounds]
        num_bg = len(formatted_backgrounds)  # Total jumlah potongan background
        ret['retriever_input_text'] = formatted_backgrounds

    if use_rag_tuning:
        # Penyematan token xrag untuk setiap background yang sudah dipotong
        _messages = copy.deepcopy(messages)
        xrag_tokens = " ".join([XRAG_TOKEN] * num_bg)
        joined_background = " ".join(formatted_backgrounds)
        
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                # Menggunakan instruksi yang dipilih acak dan memasukkan latar belakang ke dalam format
                _messages[idx]['content'] = instruct.format_map({'backgrounds': xrag_tokens, 'query': messages[idx]['content']})
                break
        # ret['xrag_messages']=_messages
        
        encoded_for_xrag = _encode_chat_format(_messages, llm_tokenizer, max_seq_length, chat_format=chat_format)
        ret['xrag_input_ids'] = encoded_for_xrag['input_ids']
        ret['xrag_labels'] = encoded_for_xrag['labels']

        # Jika menggunakan vanilla RAG
        _messages = copy.deepcopy(messages)
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                # Menggunakan instruksi yang dipilih acak dan latar belakang yang digabungkan
                _messages[idx]['content'] = instruct.format_map({'backgrounds': joined_background, 'query': messages[idx]['content']})
                break
        # ret['formatted_messages']= _messages

        encoded_for_teacher_model = _encode_chat_format(_messages, llm_tokenizer, max_seq_length, chat_format=chat_format)
        ret['input_ids'] = encoded_for_teacher_model['input_ids']
        ret['labels'] = encoded_for_teacher_model['labels']
    
    return ret
        
def get_retrieval_embeds(retriever,input_ids,attention_mask=None):
    with torch.no_grad():
        embeds = retriever.get_doc_embedding(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
    embeds = embeds.view(-1,embeds.shape[-1])
    return embeds 

# Fungsi untuk menghitung retriever_embeddings
def add_retriever_embeddings(example, retriever, retriever_tokenizer, retrieval_context_length, text_col):
    # Tokenize the retrieval text 
    tokenized_retrieval_text = retriever_tokenizer(
        example[text_col], 
        max_length=retrieval_context_length, truncation=True,
        padding=True, return_tensors="pt"
    )
    
    tokenized_retrieval_text['input_ids']=tokenized_retrieval_text['input_ids'].to('cuda:0')
    tokenized_retrieval_text['attention_mask']=tokenized_retrieval_text['attention_mask'].to('cuda:0')

    # Dapatkan embeddings dari retriever
    retrieval_embeds = get_retrieval_embeds(
        retriever=retriever,
        input_ids=tokenized_retrieval_text['input_ids'],
        attention_mask=tokenized_retrieval_text['attention_mask']
    )
    
    # Tambahkan retriever_embeddings ke contoh
    del tokenized_retrieval_text['input_ids']
    del tokenized_retrieval_text['attention_mask']
    torch.cuda.empty_cache()

    example['retriever_embeddings'] = retrieval_embeds.cpu()

    return example

def collator(samples, 
             llm_tokenizer, 
             xrag_input_ids_col='xrag_input_ids', 
             xrag_labels_col='xrag_labels', 
             text_input_ids_col='input_ids', 
             text_labels_col='labels', 
             retriever_embeds_col='retriever_embeddings'
    ):
    """
    Collate tokenized input_ids and labels with left and right side padding supported
    
    Args:
        samples (dict)          : A dict contains input_ids, labels and maybe retrieval_text
        llm_tokenizer           : Tokenizer for LLM
        xrag_input_ids_col      : nama untuk kolom tokenized messages yang diselipkan placeholder xrag
        xrag_labels_col         : nama untuk kolom xrag_input_ids_col tetapi message user di-mask dengan -100
        text_input_ids_col:     : nama untuk kolom tokenized messages dimana konteks dibiarkan seperti original (tidak diganti xrag)
        text_labels_col         : nama untuk kolom text_input_ids_col tetapi message user di-mask dengan -100
        retriever_embeds_col    : nama untuk kolom yang berisi vektor embedding dari konteks original
    
    Returns:
        A batch dictionary containing input_ids, attention_mask, labels, and retriever_embeddings
    """
    
    def padding(input_ids, labels=None, padding_side='right'):
        """
        Perform padding on input_ids and labels.
        """
        def _padding(ids, padding_value, padding_side='right'):
            if padding_side == 'right':
                return torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=padding_value)
            elif padding_side == 'left':
                flipped_ids = [torch.flip(x, dims=[0]) for x in ids]
                return torch.flip(
                    torch.nn.utils.rnn.pad_sequence(flipped_ids, batch_first=True, padding_value=padding_value),
                    dims=[1],
                )

        pad_token_id = llm_tokenizer.pad_token_id

        input_ids = _padding(input_ids,padding_value=pad_token_id,padding_side=padding_side)
        attention_mask = (input_ids != pad_token_id).long()
        if labels is not None:
            labels = _padding(labels,padding_value=-100,padding_side=padding_side)
        return input_ids,attention_mask,labels
    # temp_batch_input_ids = [x[xrag_input_ids_col] for x in samples]
    # print("before padding xrag_input_ids:")
    # print(temp_batch_input_ids)
    # print()
    # temp_batch_labels = [x[xrag_labels_col] for x in samples]
    # print('before padding xrag_labels:')
    # print(temp_batch_labels)
    # print()
    xrag_input_ids, xrag_attention_mask, xrag_labels = padding(
        input_ids=[x[xrag_input_ids_col] for x in samples],
        labels=[x[xrag_labels_col] for x in samples] if xrag_labels_col in samples[0].keys() else None,
        padding_side=llm_tokenizer.padding_side
    )
    # print("after padding xrag_input_ids:")
    # print(xrag_input_ids)
    # print()
    # print("after padding xrag_labels:")
    # print(xrag_labels)
    # print()
    # print('created attention_mask:')
    # print(xrag_attention_mask)
    # print()
    ret = {
        "xrag_input_ids": xrag_input_ids.to('cuda:0'),  # token hasil embedding llm_tokenizer 
        "xrag_attention_mask": xrag_attention_mask.to('cuda:0'),  # memberi masking agar pad tidak dipelajari selama training
        "xrag_labels": xrag_labels.to('cuda:0'),  # token hasil embedding llm_tokenizer, dimana instruksi sistem sudah disensor (dengan -100)
    }
    
    ret['retriever_embeddings'] = torch.stack([torch.tensor(x[retriever_embeds_col]).to('cuda:0') for x in samples])
    
    if text_input_ids_col in samples[0].keys():
        input_ids = [x[text_input_ids_col] for x in samples]
        labels =    [x[text_labels_col] for x in samples]
     
        input_ids,attention_mask,labels = padding(input_ids,labels,padding_side=llm_tokenizer.padding_side)
        
        ret['input_ids'] = input_ids.to('cuda:0')
        ret['attention_mask'] = attention_mask.to('cuda:0')
        ret['labels'] = labels.to('cuda:0')

    return ret

def load_xrag_dataset(test_path, test_split, background_col, retriever_name):
    if test_path.startswith('khalidrizki'):
        test_data = load_dataset(test_path)[test_split]   
    else: 
        test_data = load_from_disk(test_path)[test_split]

    if retriever_name is not None and retriever_name.lower().startswith('intfloat/multilingual-e5'):
        def add_passage_prefix(example):
            example[background_col] = ["passage: " + x for x in example[background_col]]
            return example
        
        test_data = test_data.map(add_passage_prefix)
    return test_data

def str_format(row, background_col, query_col):
    prompt_template = "Rujuklah latar belakang: {background} Pertanyaan: {query}"
    retrieval_embed_length = len(row[background_col])
    background = " ".join([XRAG_TOKEN]*retrieval_embed_length)
    prompt = prompt_template.format_map(dict(background=background, query=row[query_col].strip()))
    return prompt

def prepare_prompt(row, tokenizer, query_col, background_col):
    prompt = str_format(
        row, 
        background_col, 
        query_col
    )

    messages = [{'role':'user', 'content':prompt}]
    prompt = _concat_messages_qwen(messages, tokenizer, add_generation_prompt=True)
    return {'prompt':prompt}

@torch.no_grad()
def compute_retrieval_embeds_per_example(example, background_col, retriever, retriever_tokenizer):
    passages = example[background_col]
    
    tokenized = retriever_tokenizer(
        passages,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    input_ids = tokenized["input_ids"].to("cuda")
    attention_mask = tokenized["attention_mask"].to("cuda")

    # get embeddings: tensor [num_passages, dim]
    embeds = get_retrieval_embeds(
        retriever=retriever,
        input_ids=input_ids,
        attention_mask=attention_mask,
    ).cpu()

    # convert each vector to list[float] for serialization
    embed_list = [embed.tolist() for embed in embeds]

    return {"retrieval_embeds": embed_list}

from utils import stop_sequences_criteria
from tqdm import tqdm
import torch

def llm_for_open_generation_dataset(
    llm,
    llm_tokenizer,
    formatted_dataset,
    batch_size=2,
    enable_progress_bar=True,
):
    full_completion = []
    
    total_test_number = len(formatted_dataset)
    device = llm.device

    progress_bar = tqdm(range(0, total_test_number, batch_size), ncols=60, disable=not enable_progress_bar)

    for start_idx in range(0, total_test_number, batch_size):
        batch = formatted_dataset.select(range(start_idx, min(start_idx + batch_size, total_test_number)))

        prompts = batch["prompt"]
        tokenized_prompt = llm_tokenizer(prompts, padding='longest', return_tensors='pt', truncation=True, max_length=512)

        input_ids = tokenized_prompt.input_ids.to(device)
        attention_mask = tokenized_prompt.attention_mask.to(device)   

        stopping_criteria = stop_sequences_criteria(llm_tokenizer, input_ids.shape[1], input_ids.shape[0])
        retrieval_kwargs = {}

        if "retrieval_embeds" in batch.column_names:
            embeds_batch = batch["retrieval_embeds"]
            # Flatten and convert to tensor
            embeds = [torch.tensor(vec, dtype=torch.float32) for sublist in embeds_batch for vec in sublist]
            embeds = torch.stack(embeds).to(device)
            retrieval_kwargs["retrieval_embeds"] = embeds
            stopping_criteria = stop_sequences_criteria(llm_tokenizer, 0, input_ids.shape[0])

        with torch.no_grad():
            generated_output = llm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # stopping_criteria=stopping_criteria,
                do_sample=False,
                max_new_tokens=52,
                pad_token_id=llm_tokenizer.pad_token_id,
                **retrieval_kwargs,
            )

        output = llm_tokenizer.batch_decode(generated_output, skip_special_tokens=True)
        full_completion.extend([x.strip() for x in output])

        progress_bar.update(batch_size)

    return full_completion