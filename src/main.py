import requests
import os

# 国产大模型API配置（兼容OpenAI格式，可替换为通义千问、智谱、DeepSeek等）
API_KEY = os.getenv("AI_API_KEY", "你的API密钥")
BASE_URL = "https://api.deepseek.com/v1/chat/completions"

def chat_with_ai(prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": "deepseek-chat",
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
    print("=== AI对话Demo ===")
    print("输入 exit 退出程序")
    while True:
        user_input = input("\n请输入你的问题：")
        if user_input.lower() == "exit":
            print("程序结束")
            break
        print("AI正在思考...")
        reply = chat_with_ai(user_input)
        print(f"AI回复：{reply}")
