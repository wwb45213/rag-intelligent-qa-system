import requests
import os
import numpy as np

# 国产大模型API配置（兼容OpenAI格式，可替换为通义千问、智谱、DeepSeek等）
API_KEY = os.getenv("AI_API_KEY", "0f05e784c6ab4da597d7cb9a560bb0e1.lyW46nMiYaLH96Tu")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
# 嵌入模型接口（用于文本向量化）
EMBEDDING_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
EMBEDDING_MODEL = "embedding-3"
def get_embedding(text):
    """调用智谱接口把文本转成向量"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    try:
        response = requests.post(EMBEDDING_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["data"][0]["embedding"]
    except Exception as e:
        print(f"向量化出错：{str(e)}")
        return None
def load_knowledge_base(file_path):
    """读取本地txt文档，按行切分成知识块，并全部转成向量保存"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    chunks = []
    embeddings = []
    for line in lines:
        vec = get_embedding(line)
        if vec:
            chunks.append(line)
            embeddings.append(vec)
    return chunks, np.array(embeddings)

def retrieve_relevant_chunks(query, chunks, embeddings, top_k=2):
    """计算问题与知识库的语义相似度，返回最相关的top_k个片段"""
    query_vec = get_embedding(query)
    if not query_vec:
        return []
    
    # 计算余弦相似度
    query_vec = np.array(query_vec)
    similarities = np.dot(embeddings, query_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec))
    
    # 取相似度最高的top_k条
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def chat_with_ai(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "glm-4-flash",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"调用出错：{str(e)}"

if __name__ == "__main__":
    print("=== RAG知识库问答系统 ===")
    print("正在加载知识库并生成向量...")
    
    # 加载本地知识库并向量化
    chunks, embeddings = load_knowledge_base("src/knowledge.txt")
    print(f"知识库加载完成，共 {len(chunks)} 条知识")
    print("输入 exit 退出程序\n")

    # 对话历史记录
    chat_history = []
    # 最多保留最近5轮对话
    max_history = 5

    while True:
        query = input("请输入你的问题：")
        if query.lower() == "exit":
            break

        print("AI正在思考...")
        # 第一步：从知识库检索相关片段
        relevant_chunks = retrieve_relevant_chunks(query, chunks, embeddings)

        # 第二步：拼接对话历史 + 检索资料 + 当前问题
        context = "\n".join(relevant_chunks)
        history_str = "\n".join(chat_history)

        rag_prompt = f"""你是一个智能问答助手，请结合对话历史和参考资料回答用户问题。
如果参考资料中没有相关内容，请说明无法回答。

对话历史：
{history_str}

参考资料：
{context}

用户当前问题：{query}"""

        # 第三步：调用大模型生成答案
        answer = chat_with_ai(rag_prompt)
        print(f"AI回复:{answer}\n")

        # 把本轮对话加入历史记录
        chat_history.append(f"用户：{query}")
        chat_history.append(f"AI:{answer}")

        # 超过最大轮数就删掉最早的对话
        if len(chat_history) > max_history * 2:
            chat_history = chat_history[2:]
