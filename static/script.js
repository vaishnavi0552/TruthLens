
const newsInput = document.getElementById("newsInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultBox = document.getElementById("result");
const charCount = document.getElementById("charCount");
const buttonText = document.getElementById("buttonText");
const clearBtn = document.getElementById("clearBtn");

// Character counter
newsInput.addEventListener("input", () => {
    charCount.textContent =
        `${newsInput.value.length.toLocaleString()} / 20,000 characters`;
});

// Clear article
clearBtn.addEventListener("click", () => {
    newsInput.value = "";
    charCount.textContent = "0 / 20,000 characters";

    resultBox.className = "result-box";
    resultBox.innerHTML = `
        <div class="result-placeholder">
            <div class="result-icon">✦</div>
            <h3>Your analysis will appear here</h3>
            <p>Enter a news article and click Analyze.</p>
        </div>
    `;

    newsInput.focus();
});

// Show error safely
function showError(message) {
    resultBox.className = "result-box error";

    const heading = document.createElement("h3");
    heading.className = "result-title";
    heading.textContent = "Analysis Error";

    const paragraph = document.createElement("p");
    paragraph.className = "result-description";
    paragraph.textContent = message;

    resultBox.replaceChildren(heading, paragraph);
}

// Analyze news
async function checkNews() {

    const news = newsInput.value.trim();

    if (!news) {
        showError("Please paste a news article before analyzing.");
        return;
    }

    if (news.length > 20000) {
        showError("Your article exceeds the character limit.");
        return;
    }

    analyzeBtn.disabled = true;
    buttonText.textContent = "Analyzing...";

    resultBox.className = "result-box";

    resultBox.innerHTML = `
        <div class="result-placeholder">
            <div class="result-icon">⟳</div>
            <h3>Analyzing your article</h3>
            <p>Our Linear SVM model is processing your news.</p>
        </div>
    `;

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            body: JSON.stringify({ news: news })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(
                data.error || `Server error: ${response.status}`
            );
        }

        const prediction = data.prediction;

        if (
            prediction !== "Fake News" &&
            prediction !== "Real News"
        ) {
            throw new Error("The server returned an invalid prediction.");
        }

        const isFake = prediction === "Fake News";

        resultBox.className =
            isFake ? "result-box fake" : "result-box success";

        const icon = document.createElement("div");
        icon.className = "result-icon";
        icon.textContent = isFake ? "⚠" : "✓";

        const title = document.createElement("h3");
        title.className = "result-title";
        title.textContent = prediction;

        const description = document.createElement("p");
        description.className = "result-description";
        description.textContent = data.message;

        const modelInfo = document.createElement("p");
        modelInfo.className = "result-description";
        modelInfo.style.marginTop = "12px";
        modelInfo.textContent = "Detection Model: " +
            (data.model || "Linear SVM");

        const disclaimer = document.createElement("p");
        disclaimer.className = "result-description";
        disclaimer.style.marginTop = "12px";
        disclaimer.textContent = data.disclaimer || "";

        resultBox.replaceChildren(
            icon,
            title,
            description,
            modelInfo,
            disclaimer
        );

    } catch (error) {

        console.error("TruthLens prediction error:", error);

        showError(
            error.message.includes("Failed to fetch")
                ? "Cannot connect to Flask. Check whether your server is running."
                : error.message
        );

    } finally {

        analyzeBtn.disabled = false;
        buttonText.textContent = "✦ Analyze News Article";
    }
}

// Keep button and keyboard interaction working
analyzeBtn.addEventListener("click", checkNews);

newsInput.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter") {
        checkNews();
    }
});

let bulkResults = [];

document.getElementById("uploadBtn").addEventListener("click", async () => {

    const fileInput = document.getElementById("csvFile");
    const resultBox = document.getElementById("bulkResult");
    const uploadBtn = document.getElementById("uploadBtn");
    const downloadBtn = document.getElementById("downloadBtn");

    if (!fileInput.files.length) {
        resultBox.textContent = "Please select a CSV file.";
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    uploadBtn.disabled = true;
    uploadBtn.textContent = "Analyzing Dataset...";
    downloadBtn.hidden = true;
    resultBox.textContent = "Processing your news articles...";

    try {

        const response = await fetch("/predict_csv", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "CSV analysis failed.");
        }

        bulkResults = data.results;

        resultBox.replaceChildren();

        const summary = document.createElement("h3");
        summary.textContent =
            `Analysis completed: ${data.total} articles`;
        resultBox.appendChild(summary);

        const stats = document.createElement("p");
        stats.textContent =
            `Fake News: ${data.fake_count} | Real News: ${data.real_count} | Skipped: ${data.skipped}`;
        resultBox.appendChild(stats);

        const table = document.createElement("table");
        table.style.width = "100%";
        table.style.marginTop = "20px";

        const header = document.createElement("tr");

        ["Article", "Prediction"].forEach(name => {
            const th = document.createElement("th");
            th.textContent = name;
            header.appendChild(th);
        });

        table.appendChild(header);

        data.results.forEach(item => {

            const row = document.createElement("tr");

            const title = document.createElement("td");
            title.textContent = item.title;

            const prediction = document.createElement("td");
            prediction.textContent = item.prediction;

            row.appendChild(title);
            row.appendChild(prediction);

            table.appendChild(row);
        });

        resultBox.appendChild(table);
        downloadBtn.hidden = false;

    } catch (error) {

        resultBox.textContent = error.message;

    } finally {

        uploadBtn.disabled = false;
        uploadBtn.textContent = "Analyze CSV Dataset";

    }

});

document.getElementById("downloadBtn").addEventListener("click", () => {

    if (!bulkResults.length) return;

    const csv = [
        ["Article", "Prediction"],
        ...bulkResults.map(item => [item.title, item.prediction])
    ].map(row =>
        row.map(value =>
            '"' + String(value).replace(/"/g, '""') + '"'
        ).join(",")
    ).join("\r\n");

    const blob = new Blob([csv], {
        type: "text/csv;charset=utf-8;"
    });

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");

    link.href = url;
    link.download = "truthlens_predictions.csv";
    link.click();

    URL.revokeObjectURL(url);

});
