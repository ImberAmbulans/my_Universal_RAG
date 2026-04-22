"""用户提问和回答生成"""
import pandas as pd

# %%
query_df = pd.read_parquet(r'D:\WorkDirectory\PythonProject\RAG\my_dataset\wikidata\data\rag-mini-wikipedia\data\test.parquet\part.0.parquet')

print(query_df.head(5))
# USE_DATA_COUNT = query_df.shape[0]
USE_DATA_COUNT = 5
query_df = query_df.head(USE_DATA_COUNT)

# %%
questions_list = []
answers_list = []
for _,row in query_df.iterrows():
    question = row['question']
    answer = row['answer']
    questions_list.append(question)
    answers_list.append(answer)

questions_embedding = embed_model.encode(questions_list, batch_size=64, convert_to_tensor=False)
questions_embedding = questions_embedding.astype('float32')
# embeddings = model.encode(answers_list, batch_size=64, convert_to_tensor=False)


# %%
k = 5
distances, indices = index.search(questions_embedding, k)


# %%
print(f"chunks 长度: {len(all_chunks)}")
print(f"索引中向量总数: {index.ntotal}")

# %%
# 假设 chunks 是你分块后的原始文本列表，顺序与 embeddings 相同
# retrieved_chunks = [all_chunks[i] for i in indices[0]]

# %%
# 拼接检索到的内容作为上下文
# context = "\n\n".join(retrieved_chunks)
prompts_list = []
contexts_list = []
for i in range(len(questions_list)):
    retrieved_chunks = [all_chunks[j] for j in indices[i]]
    context = "\n\n".join(retrieved_chunks)
    contexts_list.append(context)
    prompt = f"""基于以下参考信息回答问题。
    参考信息：
    {context}
    只回答问题，不要生成新的问题或者回答问题以外的回答。
    回答结束时，输出eos标志
    问题：{questions_list[i]}
    回答："""
    prompts_list.append(prompt)
prompts_list


# %%


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
torch.cuda.empty_cache()
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 启用4-bit量化[reference:1]
    bnb_4bit_quant_type="nf4",      # 使用nf4量化类型，质量更高[reference:2]
    bnb_4bit_compute_dtype=torch.float16, # 计算时使用float16，平衡速度与精度[reference:3]
    bnb_4bit_use_double_quant=True, # 启用双重量化，进一步压缩显存[reference:4]
)
# 加载模型和分词器 (以4-bit量化为例)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
generate_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb_config,
    device_map="auto"
)
responses_list = []
for i in range(len(prompts_list)):
    inputs = tokenizer(prompts_list[i],return_tensors="pt").to(generate_model.device)
    response_ids = generate_model.generate(inputs.input_ids,max_new_tokens=32,do_sample=False,stop_strings=["\n"],tokenizer=tokenizer,eos_token_id=tokenizer.eos_token_id)
    input_len = inputs.input_ids.shape[1]
    response_text = tokenizer.decode(response_ids[0][input_len:],skip_special_tokens=True)
    
    print(f"Prompt: {prompts_list[i]}{response_text}")
    responses_list.append(response_text)
    print("-" * 50)



# %%
# 我现在有了参考内容、问题、回答、标准回答的矩阵
# 进行评估
import re
import json
def extract_json(text):
    match = re.search(r'\{.*?"factual_score".*?"completeness_score".*?"redundancy_score".*?\}', text, re.DOTALL)
    print("匹配结果",match)
    if match:
        return json.loads(match.group())
    return None
eval_scores = []
# for i in range(len(prompts_list)):
for i in range(1):
    eval_prompt = f"""你是一个RAG评估专家。请根据标准回答，评估生成回答的质量。

    问题：{questions_list[i]}
    标准回答：{answers_list[i]}
    检索到的上下文：{contexts_list[i]}
    生成回答：{responses_list[i]}

    请从以下维度打分（1-5分）：
    1. 事实一致性：生成回答的信息是否与标准回答一致？（5=完全一致，无矛盾）
    2. 完整性：生成回答是否覆盖了标准回答的所有关键点？（5=完全覆盖）
    3. 冗余性：生成回答是否包含多余或无关信息？（1=严重冗余，5=简洁相关）

    最终输出JSON格式：{{"factual_score": x, "completeness_score": y, "redundancy_score": z}}\n\n
    """
    eval_input = tokenizer(eval_prompt,return_tensors="pt").to(generate_model.device)
    input_len = eval_input.input_ids.shape[0]
    eval_response_ids = generate_model.generate(eval_input.input_ids,max_new_tokens=128,do_sample=False,eos_token_id = tokenizer.eos_token_id)
    eval_response_text = tokenizer.decode(eval_response_ids[0][input_len:],skip_special_tokens=True)
    print("评估结果",eval_response_text)
    try:
        eval_json = extract_json(eval_response_text)
    except Exception as e:
        # print(eval_response_text,"\n",e)
        eval_json = None
    eval_scores.append(eval_json)
eval_scores