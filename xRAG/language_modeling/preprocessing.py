import random, copy, torch
from datasets import Dataset, load_from_disk

XRAG_TOKEN = "<xRAG>" 

ParaphraseInstructions = [
    'Background: {xrag_token} means the same as',
    "Background: {xrag_token} Can you put the above sentences in your own terms?",
    "Background: {xrag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {xrag_token}\n(2)",
    "Background: {xrag_token} is a paraphrase of what?",
    "Background: {xrag_token} Could you give me a different version of the background sentences above?",
    "In other words, background: {xrag_token} is just another way of saying:",
    "You're getting across the same point whether you say background: {xrag_token} or",
    "Background: {xrag_token} After uppacking the ideas in the background information above, we got:",
    "Background: {xrag_token} Please offer a restatement of the background sentences I've just read.",
    "Background: {xrag_token}, which also means:",
    "Strip away the mystery, and you'll find background: {xrag_token} is simply another rendition of:",
    "The essence of background: {xrag_token} is captured again in the following statement:",
]

def load_and_format_dataset(dataset_path, query_col, answer_col, psg_col, task_type, max_rows=None, include_psg_len=False):
    """
    Mengubah format dari dataset raw ke format yang sesuai untuk xRAG
    """
    # Memuat dataset yang disimpan dalam format DatasetDict (train, dev, test)
    dataset = load_from_disk(dataset_path)

    # Iterasi untuk setiap split dalam DatasetDict
    for split_name, split_data in dataset.items():
        # Jika max_rows diterapkan, hanya pilih jumlah baris yang ditentukan
        if max_rows is not None:
            split_data = split_data.select(range(max_rows))

        if task_type == 'finetune':
            # Gunakan fungsi untuk task_type 'finetune'
            formatted_data = process_finetune_dataset(split_data, query_col, answer_col, psg_col, include_psg_len)
        elif task_type == 'pretrain':
            # Gunakan fungsi untuk task_type 'pretrain'
            formatted_data = process_pretrain_dataset(split_data, psg_col)

        # Setelah selesai memproses seluruh split, buat dataset baru untuk masing-masing split
        formatted_split_dataset = Dataset.from_dict(formatted_data)

        # Simpan hasil format untuk setiap split (train, dev, test)
        dataset[split_name] = formatted_split_dataset

    return dataset


def process_finetune_dataset(split_data, query_col, answer_col, psg_col, include_psg_len=False):
    """
    Fungsi untuk memproses dataset saat task_type adalah 'finetune'
    """
    formatted_data = {
        "background": [],
        "messages": []
    }

    if include_psg_len:
        formatted_data["background_length"] = []  # Menambahkan kolom untuk panjang background
    
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
        
        # Ambil background dan hitung panjangnya jika include_psg_len True
        background = example[psg_col]  # .split('\n\n')

        # Menambahkan data ke formatted_data
        formatted_data['background'].append(background)
        formatted_data['messages'].append(messages)

        if include_psg_len:
            # Jika include_psg_len True, hitung panjang background dan tambahkan ke formatted_data
            bg_length = len(example[psg_col])
            formatted_data['background_length'].append(bg_length)
    
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
        formatted_data['text'].extend(text)  # Menggunakan extend, bukan append
    
    return formatted_data

def split_multiple_backgrounds(backgrounds, retriever_tokenizer, total_max_len, single_max_len, single_min_len=20):
    """
    Membagi beberapa background menjadi potongan-potongan kecil.
    
    Args:
        backgrounds (list): List dari background (list string).
        tokenizer: Tokenizer yang digunakan.
        total_max_len: Panjang maksimum token untuk masing-masing background.
        single_max_len: Panjang maksimum untuk setiap potongan.
        single_min_len: Panjang minimum untuk sebuah potongan.
    
    Return:
        list: List potongan-potongan background, masing-masing dipotong sesuai dengan panjang token yang ditentukan.
    """
    all_sharded_backgrounds = []
    
    for background in backgrounds:
        # print(background)
        ids = retriever_tokenizer(background, add_special_tokens=False, max_length=total_max_len, truncation=True).input_ids
        # print("panjang total:", len(ids))
        background_chunks = [ids[idx:idx+single_max_len] for idx in range(0, len(ids), single_max_len)]
        # for chunk in background_chunks:
        #     print(len(chunk))
        
        # Menghapus chunk terakhir jika terlalu kecil
        if len(background_chunks[-1]) <= single_min_len and len(background_chunks) > 1:
            background_chunks = background_chunks[:-1]
        
        all_sharded_backgrounds.extend([retriever_tokenizer.decode(x) for x in background_chunks])
    
    return all_sharded_backgrounds

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

def _concat_messages_qwen(messages, llm_tokenizer):
    message_text = llm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
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
        chat_format='llama',
    ):
    """
    encode messages to input_ids and make non-assistant part

    Args:
        messages (list): list of dict with 'role' and 'content' field
        tokenizer: llm tokenizer
        max_seq_lengh: maximun context length  
    
    Return:
        input_ids and labels
    """
    # _concat_messages = eval(f"_concat_messages_{chat_format}")
    if chat_format == 'llama':
        # print("Menambahkan token spesial dengan format LLAMA")
        # example_text = _concat_messages_llama(messages).strip()
        _concat_messages = _concat_messages_llama
    elif chat_format =='qwen':
        # print("Menambahkan token spesial dengan format QWEN")
        # example_text = _concat_messages_qwen(messages, llm_tokenizer).strip()
        _concat_messages = _concat_messages_qwen
    else:
        raise ValueError(f"Invalid chat_format: {chat_format}. Must be 'qwen' or 'llama'.")
    
    example_text = _concat_messages(messages, llm_tokenizer=llm_tokenizer).strip()
    tokenized_example = llm_tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True, add_special_tokens=False)
    # KALAU BACKGROUND KEPANJANGAN (melebihi 1024 token), MAKA CONTENT DR ASSISTANT TIDAK AKAN TERTOKENISASI KARENA SUDAH KEPOTONG OLEH TRUNCATION
    # alhasil akan kena filter oleh train_dataset.filter(lambda example: (example['labels'] != -100).any())
    # dan itu gapapa, karena hanya menyusun sekitar 123 baris train dan 17 baris dev. masih ada 4419 baris untuk latihan dan 1126 baris untuk dev
    
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    labels_all_masked=False
    # Check if tokenized_example contains the id 78191 ("assistant")
    if 78191 not in input_ids.flatten().tolist():
        # print("ID 78191 not found in input_ids.") 
        labels_all_masked = True
    # else:
        # print("ID 78191 found in input_ids.")
    
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = llm_tokenizer(
                    _concat_messages(messages[:message_idx]), llm_tokenizer=llm_tokenizer, return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            
            if chat_format in ['llama', 'qwen']:
                # Karena elemen messages bergantian antara role:user dengan role:assistant, 
                # ambil messages_so_far sebagai messages indeks role:user saja
                messages_so_far = _concat_messages(messages[:message_idx+1], llm_tokenizer=llm_tokenizer)         

            # Tokenisasi message role:user
            message_end_idx = llm_tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length,
                truncation=True, 
                add_special_tokens=False
            ).input_ids.shape[1]  # Panjang keseluruhan message role:user diidentifikasi

            # Labels untuk message role:user di-mask dengan -100
            labels[:, message_start_idx:message_end_idx] = -100

            # # Kode untuk menge-print jika keseluruhan labels di-mask
            # if labels_all_masked: 
            #     print("LABELS ENCODE_CHAT_FORMAT:", labels)
            #     print('selesai mencetak labels')

            if message_end_idx >= max_seq_length:
                break
    
    # assert tokenizer.eos_token_id in input_ids, input_ids
    return {
        "input_ids":input_ids.flatten(),
        "labels":labels.flatten(),
    }

def encode_with_chat_format_pretrain(
        example,
        llm_tokenizer,
        retriever_tokenizer, 
        max_seq_length,
        retrieval_context_length, 
        retrieval_embed_length,
        chat_format='llama',
        ):
    """
    encode messages into input_ids and labels for paraphrase pretrain
    
    Return:
        input_ids,labels and retriever_input_text
    """

    document = "passage: " + example['text'].strip()
    truncated_document = truncate_using_retriever_tokenizer(retriever_tokenizer, document, retrieval_context_length)
    xrag_token = " ".join([XRAG_TOKEN] * retrieval_embed_length)  
    instruction = random.choice(ParaphraseInstructions).format_map(dict(xrag_token=xrag_token))

    messages = [
        {"role":"user","content":instruction},
        {"role":"assistant","content":truncated_document},
    ]

    encoded = _encode_chat_format(
        messages=messages,
        llm_tokenizer=llm_tokenizer, 
        max_seq_length=max_seq_length,
        chat_format=chat_format)

    return {
        "xrag_input_ids":encoded['input_ids'],
        "xrag_labels":encoded['labels'],
        "retriever_input_text":truncated_document,
    }

def encode_with_chat_format_finetune(
        example, 
        llm_tokenizer,
        max_seq_length,
        retrieval_context_length,
        lang, 
        use_rag_tuning=True,
        use_retriever_embed=False,
        retriever_tokenizer=None,
        chat_format='llama'
    ):
    """
    Fungsi untuk melakukan encoding dengan format chat, mendukung multiple background.
    """
    messages, backgrounds = example['messages'], example['background']
    # print("encode_with_finetune: LIST OF BACKGROUNDS#################################")
    # print(backgrounds)

    ret = {}

    if lang == "english":
        instruct = "Refer to the background document: "
        question="Question: "
    elif lang == "indonesian":
        instruct = "Rujuklah dokumen konteks: "
        question="Pertanyaan: "

    if use_rag_tuning and use_retriever_embed:
        # # Split setiap background menjadi potongan-potongan lebih kecil jika panjangnya melebihi batas
        # sharded_backgrounds = split_multiple_backgrounds(backgrounds, retriever_tokenizer, total_max_len=max_seq_length, single_max_len=retrieval_context_length)
        # sharded_backgrounds = ["passage: "+background for background in sharded_backgrounds]
        # num_split = len(sharded_backgrounds)  # Total jumlah potongan background
        # ret['retriever_input_text'] = sharded_backgrounds
        truncated_backgrounds = ["passage: " + truncate_using_retriever_tokenizer(retriever_tokenizer, background, retrieval_context_length) for background in backgrounds]
        num_bg = len(truncated_backgrounds)  # Total jumlah potongan background
        ret['retriever_input_text'] = truncated_backgrounds

    if use_rag_tuning:
        # Penyematan token xrag untuk setiap background yang sudah dipotong
        _messages = copy.deepcopy(messages)
        xrag_tokens = " ".join([XRAG_TOKEN] * num_bg)
        joined_background = " ".join(truncated_backgrounds)
        
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"{instruct}{xrag_tokens}\n\n{question}{messages[idx]['content']}"
                break
        # ret['xrag_messages']=_messages
        
        encoded = _encode_chat_format(_messages, llm_tokenizer, max_seq_length, chat_format=chat_format)
        ret['xrag_input_ids'] = encoded['input_ids']
        ret['xrag_labels'] = encoded['labels']

        # Jika menggunakan vanilla RAG
        _messages = copy.deepcopy(messages)
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"{instruct}{joined_background}\n\n{question}{messages[idx]['content']}"
                break
        # ret['formatted_messages']= _messages
        
        encoded = _encode_chat_format(_messages, llm_tokenizer, max_seq_length, chat_format=chat_format)
        ret['input_ids'] = encoded['input_ids']
        ret['labels'] = encoded['labels']
    
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
            #  llm_tokenizer_pad_token, 
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
        llm_tokenizer_pad_token : pad token for tokenizer
        xrag_input_ids_col      : nama untuk kolom tokenized messages yang diselipkan placeholder xrag
        xrag_labels_col         : nama untuk kolom xrag_input_ids_col tetapi message user di-mask dengan -100
        text_input_ids_col:     : nama untuk kolom tokenized messages dimana konteks dibiarkan seperti original (tidak diganti xrag)
        text_labels_col         : nama untuk kolom text_input_ids_col tetapi message user di-mask dengan -100
    
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
        # print(input_ids)
        attention_mask = (input_ids != pad_token_id).long()
        if labels is not None:
            labels = _padding(labels,padding_value=-100,padding_side=padding_side)
        return input_ids,attention_mask,labels
    
    xrag_input_ids, xrag_attention_mask, xrag_labels = padding(
        input_ids=[x[xrag_input_ids_col] for x in samples],
        labels=[x[xrag_labels_col] for x in samples] if xrag_labels_col in samples[0].keys() else None,
        padding_side=llm_tokenizer.padding_side
    )

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