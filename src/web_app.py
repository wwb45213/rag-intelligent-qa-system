import gradio as gr
import requests
import numpy as np
import faiss
import os

# ========== 配置项（和 main.py 完全一致）==========
API_KEY = os.getenv("AI_API_KEY", "0f05e784c6ab4da597d7cb9a560bb0e1.lyW46nMiYaLH96Tu")
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
EMBEDDING_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
EMBEDDING_MODEL = "embedding-3"
CHAT_MODEL = "glm-4-flash"
KNOWLEDGE_FILE = "./src/knowledge.txt"
VECTOR_INDEX_PATH = "./vector_store/faiss.index"

# ========== 工具函数 ==========
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
        print(f"向量生成失败: {e}")
        return None

def split_text(text, chunk_size=500, chunk_overlap=50):
    """按语义切分文本"""
    chunks = []
    sentences = text.split("\n")
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence[max(0, len(sentence)-chunk_overlap):] + "\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def build_vector_db():
    """加载知识库并构建向量库"""
    # 读取文档
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 切分文本
    chunks = split_text(text)
    
    # 生成所有片段的向量
    vectors = []
    valid_chunks = []
    for chunk in chunks:
        vec = get_embedding(chunk)
        if vec:
            vectors.append(vec)
            valid_chunks.append(chunk)
    
    # 构建 FAISS 索引
    vector_array = np.array(vectors).astype("float32")
    dimension = vector_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vector_array)
    
    # 保存索引
    os.makedirs(os.path.dirname(VECTOR_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, VECTOR_INDEX_PATH)
    
    # 保存文本片段
    with open("./vector_store/chunks.txt", "w", encoding="utf-8") as f:
        f.write("\n=====CHUNK=====\n".join(valid_chunks))
    
    return index, valid_chunks, f"知识库构建完成，共 {len(valid_chunks)} 个文本片段"

def load_vector_db():
    """加载本地向量库"""
    index = faiss.read_index(VECTOR_INDEX_PATH)
    with open("./vector_store/chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("\n=====CHUNK=====\n")
    return index, chunks

def search_similar(index, chunks, query, top_k=3):
    """检索最相似的文本片段"""
    query_vec = get_embedding(query)
    if not query_vec:
        return []
    
    query_array = np.array([query_vec]).astype("float32")
    distances, indices = index.search(query_array, top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])
    return results

def chat_with_llm(prompt):
    """调用大模型生成回答"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    try:
        response = requests.post(BASE_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"调用大模型失败: {e}"

# ========== RAG 问答核心 ==========
def rag_answer(question):
    """完整 RAG 问答流程"""
    # 检索相关片段
    related_docs = search_similar(index, chunks, question)
    
    # 拼接提示词
    context = "\n\n".join(related_docs)
    prompt = f"""你是基于文档内容的智能问答助手，请严格根据下面的上下文内容回答用户的问题。
如果上下文里没有相关答案，请回答“抱歉，知识库中没有找到相关内容”。

上下文：
{context}

用户问题：{question}

回答："""
    
    # 生成回答
    answer = chat_with_llm(prompt)
    
    # 格式化原文片段
    source_text = ""
    for i, doc in enumerate(related_docs):
        source_text += f"片段 {i+1}：\n{doc}\n\n"
    
    return answer, source_text.strip()

# ========== 初始化向量库 ==========
if os.path.exists(VECTOR_INDEX_PATH):
    index, chunks = load_vector_db()
    init_msg = "已加载本地向量库"
else:
    index, chunks, init_msg = build_vector_db()

# ========== Gradio 网页界面 ==========
with gr.Blocks(title="RAG 智能问答系统") as demo:
    gr.Markdown("# 🤖 RAG 智能问答系统")
    gr.Markdown("基于检索增强生成技术，结合本地知识库实现精准问答")
    gr.Markdown(f"✅ {init_msg}")
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(label="请输入你的问题", placeholder="在这里输入问题...", lines=2)
            submit_btn = gr.Button("提交问题", variant="primary")
            
            answer_output = gr.Textbox(label="AI 回答", lines=8)
        
        with gr.Column(scale=1):
            source_output = gr.Textbox(label="检索到的原文片段", lines=12)
    
    submit_btn.click(
        fn=rag_answer,
        inputs=question_input,
        outputs=[answer_output, source_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
