from utils import detect_device, load_model_and_tokenizer, get_chat_logprobs
from datasets import load_dataset
import torch
import json
from tqdm import tqdm

def generate_answer_with_logprobs(
    model_name: str,
    dataset_name: str = "khalidalt/tydiqa-goldp",
    loop_range: int = None,
    max_new_tokens: int = 50,
    batch_size: int = 8
):
    """Generate answers using LLM model"""
    try:
        # Detect device and load model
        config = detect_device()

        print(f"\nLoading model on {config.device_type.value}")
        model, tokenizer, config = load_model_and_tokenizer(model_name, config)

        # Load dataset
        dataset = load_dataset(dataset_name, 'indonesian', trust_remote_code=True)

        # Check required columns in dataset
        required_columns = ['id', 'question_text', 'passage_text', 'answers']
        for column in required_columns:
            if column not in dataset['train'].column_names:
                raise ValueError(f"Dataset is missing required column: {column}")

        # Set default loop_range if not provided
        if loop_range is None:
            loop_range = len(dataset['train'])

        # Initialize results list
        results = []

        for i in tqdm(range(loop_range), desc="Processing dataset"):
            messages = [
                {
                    "role": "sistem",
                    "content": f"Kamu adalah asisten AI. Berdasarkan teks berikut, jawab pertanyaan pengguna: {dataset['train']['passage_text'][i]}"
                },
                {
                    "role": "pengguna",
                    "content": f"{dataset['train']['question_text'][i]}"
                }
            ]

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
                "id": dataset['train']['id'][i],
                "question": dataset['train']['question_text'][i],
                "passage": dataset['train']['passage_text'][i],
                "generated_completion": result["completion"],
                "answer": dataset['train']['answers'][i]['text'][0],
                "tokens": result['tokens'],
                "token_ids": result['token_ids'],
                "logprobs": result["logprobs"]
                # "top_logprobs": result["top_logprobs"]
            })

        # Save results to file
        output_file = f"{model_name.replace('/', '_')}-{loop_range}_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Processing completed. Results saved to {output_file}")

    except Exception as e:
        raise Exception(f"Unexpected error in processing: {e}")