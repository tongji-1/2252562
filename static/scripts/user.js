let qaDatabase = []; // 保留以备后用，如果需要基于数据库进一步优化答案

// 从外部JSON文件加载Q&A数据
fetch('/static/scripts/qaDatabase.json')  
    .then(response => response.json())
    .then(data => {
        qaDatabase = data;
    })
    .catch(error => {
        console.error("加载Q&A数据库时出错:", error);
    });

// 调用 OpenAI API 获取最佳答案
async function findBestAnswer(userInput) {
    try {
        const response = await fetch("https://api.openai.com/v1/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": 'Bearer ' + 'sk-xxx' // 请替换为您的 OpenAI API 密钥
            },
            body: JSON.stringify({
                model: "gpt-4o-mini", // 可以选择 gpt-4 或 gpt-3.5-turbo
                messages: [
                    { role: "system", content: "你是一个帮助用户回答问题的智能助手。" },
                    { role: "user", content: userInput }
                ],
                max_tokens: 200, // 根据需要调整输出长度
                temperature: 0.7 // 根据需要调整生成内容的随机性
            })
        });

        const data = await response.json();

        if (data.choices && data.choices.length > 0) {
            return data.choices[0].message.content.trim();
        } else {
            throw new Error("API 返回的结果无效");
        }
    } catch (error) {
        console.error("调用 OpenAI API 时出错:", error);
        return "抱歉，我无法处理您的请求，请稍后再试或联系技术支持。";
    }
}

// 事件监听器
document.getElementById('send-button').onclick = async () => {
    const userInput = document.getElementById('user-input').value;
    const chatDisplay = document.getElementById('chat-display');

    if (userInput.trim() !== '') {
        const userMessage = document.createElement('div');
        userMessage.className = 'user-message';
        userMessage.textContent = userInput;
        chatDisplay.appendChild(userMessage);

        document.getElementById('user-input').value = '';

        const aiMessage = document.createElement('div');
        aiMessage.className = 'ai-message';
        aiMessage.textContent = '正在处理您的请求...';
        chatDisplay.appendChild(aiMessage);

        // 调用 OpenAI API 获取答案
        const bestAnswer = await findBestAnswer(userInput);
        aiMessage.textContent = bestAnswer;

        // 滚动到最新消息
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }
};

// 按回车发送消息
document.getElementById('user-input').addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        document.getElementById('send-button').click();
    }
});
