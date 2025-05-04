from utils import detect_device, load_model_and_tokenizer, get_chat_logprobs, format_chat_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
import torch
import gc
import json
from tqdm import tqdm

def generate_answer_with_logprobs(
    model_name: str,
    dataset: Dataset,
    model: AutoModelForCausalLM = None, 
    tokenizer: AutoTokenizer = None,
    use_context: bool = False,
    loop_range: int = None,
    max_new_tokens: int = 50,
    batch_size: int = 8
):
    """Generate answers using LLM model"""
    try:
        config = detect_device()
        if model is None or tokenizer is None:
            print("Model dan tokenizer belum dimuat. Memuat model sekarang...")
            model, tokenizer, config = load_model_and_tokenizer(model_name, config)

        print(f"\nLoading model on {config.device_type.value}")

        # Check required columns in dataset
        required_columns = ['query', 'summary', 'answers']
        for column in required_columns:
            # if column not in dataset['train'].column_names:
            if column not in dataset.column_names:
                raise ValueError(f"Dataset is missing required column: {column}")

        # Set default loop_range if not provided
        if loop_range is None:
            loop_range = len(dataset)
            # loop_range = len(dataset['train'])

        # Initialize results list
        results = []            

        count = 0
        for i in tqdm(range(loop_range), desc="Processing dataset"):
            if use_context:
                messages = [
                    {
                        "role": "sistem",
                        "content": f"Kamu adalah asisten AI yang bertugas menjawab pertanyaan pengguna. Gunakan teks berikut sebagai konteks pendukung dalam menjawab pertanyaan: {dataset['summary'][i]}"  # ['train']
                    },
                    {
                        "role": "pengguna",
                        "content": dataset['query'][i]  # ['train']
                    }
                ]
            else:
                messages = [
                    {
                        "role": "sistem",
                        "content": f"Kamu adalah asisten AI yang bertugas menjawab pertanyaan pengguna."
                    },
                    {
                        "role": "pengguna",
                        "content": dataset['query'][i]
                    }
                ]
            if count ==0:
                print(format_chat_prompt(messages, tokenizer))

            # Get completion with logprobs
            result = get_chat_logprobs(
                messages,
                model,
                tokenizer,
                config,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                batch_size=batch_size  # Adjust based on available memory
            )

            # Convert tensor fields to list or plain values
            if isinstance(result['token_ids'], torch.Tensor):
                result['token_ids'] = result['token_ids'].tolist()
            if isinstance(result['logprobs'], torch.Tensor):
                result['logprobs'] = [logprob.item() if logprob is not None else None for logprob in result['logprobs']]
            if 'top_logprobs' in result:  # Convert top_logprobs if tensor exists
                for logprobs in result['top_logprobs']:
                    for key, value in logprobs.items():
                        if isinstance(value, torch.Tensor):
                            logprobs[key] = value.item()

            # Store result along with corresponding question and passage
            results.append({
                # "id": dataset['id'][i],  #['train']
                # "question": dataset['question_text'][i],  #['train'] 
                # "passage": dataset['passage_text'][i],  # ['train']
                "generated_completion": result["completion"],  # 
                # "answer": dataset['answers'][i]['text'][0],  #['train']
                "tokens": result['tokens'],
                "token_ids": result['token_ids'],
                "logprobs": result["logprobs"]
                # "top_logprobs": result["top_logprobs"]
            })

            # Hapus variabel untuk membebaskan memori
            del messages, result
            torch.cuda.empty_cache()
            gc.collect()
            count+=1

        if use_context:
            column_name = 'generation'
        else:
            column_name = 'generation_wo_summary'
            
        dataset = dataset.add_column(column_name, results)
        return dataset

        # Save results to file
        # output_file = f"{model_name.replace('/', '_')}-{loop_range}_results.json"
        # with open(output_file, "w") as f:
        #     json.dump(results, f, indent=4)
        # print(f"Processing completed. Results saved to {output_file}")

    except Exception as e:
        raise Exception(f"Unexpected error in processing: {e}")