function toggleFooter() {
    var footerMain = document.getElementById("footer-main");
    var toggleButton = document.getElementById("toggle-footer");

    footerMain.classList.toggle("collapsed");

    if (footerMain.classList.contains("collapsed")) {
        toggleButton.textContent = "展开更多";
    } else {
        toggleButton.textContent = "收起";
    }
}
