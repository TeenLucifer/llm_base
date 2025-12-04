import json
import tqdm
from transformers import AutoTokenizer

from load_config import CONFIG

tokenizer = AutoTokenizer.from_pretrained(CONFIG["tokenizer_path"])

# 从 deepctrl-sft-data 的 sft_data_zh.jsonl 中取一部分长度大于 1024 toekn 的序列作为训练数据集
if __name__ == "__main__":
    with open(CONFIG["raw_data_path"], 'r', encoding='utf-8') as f:
        with open(CONFIG["warmup_data_path"], 'a', encoding='utf-8') as fw:
            warmup_data = []
            for line in tqdm.tqdm(f):
                raw_line = line
                line = json.loads(line)
                instruction_text = line['instruction']
                input_text = line['input']
                output_text = line['output']
                history = line['history']
                query = instruction_text + input_text
                answer = output_text + tokenizer.eos_token
                messages = []
                if history:
                    for i in history:
                        messages.append({'role': 'user', 'content': i[0]})
                        messages.append({'role': 'assistant', 'content': i[1]})

                messages.append({'role': 'user', 'content': query})   
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                prompt_input_ids = tokenizer.encode(prompt)
                answer_input_ids = tokenizer.encode(answer)
                input_ids = prompt_input_ids + answer_input_ids
                if len(input_ids) >= 1024:
                    warmup_data.append(raw_line)

                if len(warmup_data) == 100:
                    fw.writelines(warmup_data)
                    warmup_data = []
