
function updateFileStatus() {
    const fileInput = document.getElementById('image-upload');
    const fileStatus = document.getElementById('file-status');
    
    if (fileInput.files.length > 0) {
        fileStatus.textContent = "✔ 文件已选择";
        fileStatus.classList.add("success"); // 添加成功样式
    } else {
        fileStatus.textContent = "未选择任何文件";
        fileStatus.classList.remove("success"); // 移除成功样式
    }
}

