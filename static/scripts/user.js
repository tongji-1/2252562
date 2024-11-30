let qaDatabase = [];

// 从外部JSON文件加载Q&A数据
fetch('/static/scripts/qaDatabase.json')  
    .then(response => response.json())
    .then(data => {
        qaDatabase = data;
    })
    .catch(error => {
        console.error("加载Q&A数据库时出错:", error);
    });

function findBestAnswer(userInput) {
    const lowerCaseInput = userInput.toLowerCase();
    let bestMatch = null;
    let highestScore = 0;

    for (let qa of qaDatabase) {
        const lowerCaseQuestion = qa.question.toLowerCase();
        const score = calculateSimilarityScore(lowerCaseInput, lowerCaseQuestion);

        if (score > highestScore) {
            highestScore = score;
            bestMatch = qa.answer;
        }
    }

    return highestScore > 0.3 ? bestMatch : "抱歉，我无法理解您的问题，请联系技术支持。";
}

// 停用词去除函数
function removeStopWords(words) {
    const stopWords = ["的", "了", "在", "是"];
    return words.filter(word => !stopWords.includes(word));
}

// 余弦相似度计算函数
function calculateSimilarityScore(input, question) {
    let inputWords = removeStopWords(input.split(" "));
    let questionWords = removeStopWords(question.split(" "));

    const wordSet = new Set([...inputWords, ...questionWords]);
    const inputVector = Array.from(wordSet).map(word => inputWords.filter(w => w === word).length);
    const questionVector = Array.from(wordSet).map(word => questionWords.filter(w => w === word).length);

    const dotProduct = inputVector.reduce((sum, count, index) => sum + count * questionVector[index], 0);
    const inputMagnitude = Math.sqrt(inputVector.reduce((sum, count) => sum + count * count, 0));
    const questionMagnitude = Math.sqrt(questionVector.reduce((sum, count) => sum + count * count, 0));

    return dotProduct / (inputMagnitude * questionMagnitude);
}

// 事件监听器
document.getElementById('send-button').onclick = () => {
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

        setTimeout(() => {
            const bestAnswer = findBestAnswer(userInput);
            aiMessage.textContent = bestAnswer;

            // 滚动到最新消息
            chatDisplay.scrollTop = chatDisplay.scrollHeight;
        }, 1000);
    }
};

// 按回车发送消息
document.getElementById('user-input').addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        document.getElementById('send-button').click();
    }
});
