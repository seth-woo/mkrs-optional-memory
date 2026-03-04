let uploadedImages = [];
let currentMemoryCount = 0;

// DOM references
const imageInput = document.getElementById("imageInput");
const previewBox = document.getElementById("preview-box");
const imageStatus = document.getElementById("imageStatus");
const promptInput = document.getElementById("promptInput");
const promptStatus = document.getElementById("promptStatus");
const outputBox = document.getElementById("output-box");
const submitBtn = document.getElementById("submitBtn");
const resetBtn = document.getElementById("resetBtn");

// NEW
const saveMemoryToggle = document.getElementById("saveMemoryToggle");
const memoryCountEl = document.getElementById("memoryCount");

// -----------------------------
// Helpers
// -----------------------------
async function refreshMemoryCount() {
    try {
        const res = await fetch("/memory/count");
        const data = await res.json();
        currentMemoryCount = data.count ?? 0;
        memoryCountEl.textContent = `Memory: ${data.count}`;
    } catch {
        currentMemoryCount = 0;
        memoryCountEl.textContent = "Memory: ?";
    }
}

async function clearMemory() {
    if (!confirm("Clear all stored memory?")) return;

    try {
        await fetch("/memory/clear", { method: "POST" });
        await refreshMemoryCount();
        alert("Memory cleared.");
    } catch {
        alert("Failed to clear memory.");
    }
}

// expose to global (for index.html button)
window.clearMemory = clearMemory;

// -----------------------------
// Image upload handler
// -----------------------------
imageInput.addEventListener("change", () => {
    uploadedImages = Array.from(imageInput.files);
    previewBox.innerHTML = "";

    uploadedImages.forEach(file => {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        previewBox.appendChild(img);
    });

    if (uploadedImages.length > 0) {
        imageStatus.textContent = "Image(s) uploaded successfully.";
    }
});

// -----------------------------
// Submit prompt
// -----------------------------
submitBtn.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    const saveToMemory = saveMemoryToggle.checked;

    if (!prompt || uploadedImages.length === 0) {
        alert("Please upload an image and enter a prompt.");
        return;
    }

    promptStatus.textContent = "Prompt submitted successfully.";
    outputBox.textContent = "Running inference...";

    const formData = new FormData();

    // Mode 1: backend uses first image only
    formData.append("images", uploadedImages[0]);
    formData.append("question", prompt);
    formData.append("save_to_memory", saveToMemory);

    // Refresh memory count before routing to avoid stale client state.
    await refreshMemoryCount();

    // Routing policy:
    // - memory == 0 -> single_qa (state-less baseline)
    // - memory >= 1 -> multi_qa (RAG mode), regardless of save toggle
    const endpoint = currentMemoryCount >= 1 ? "/qa/multi" : "/qa/single";
    promptStatus.textContent = `Prompt submitted successfully. (${endpoint})`;

    try {
        const res = await fetch(endpoint, {
            method: "POST",
            body: formData
        });

        const data = await res.json();
        outputBox.textContent = data.answer ?? JSON.stringify(data, null, 2);

        if (saveToMemory) {
            await refreshMemoryCount();
        }
    } catch (err) {
        outputBox.textContent = "Error: " + err;
    }
});

// -----------------------------
// Reset
// -----------------------------
resetBtn.addEventListener("click", () => {
    imageInput.value = "";
    promptInput.value = "";
    outputBox.textContent = "VLM Output";
    imageStatus.textContent = "";
    promptStatus.textContent = "";
    uploadedImages = [];

    previewBox.innerHTML = '<span id="preview-placeholder">Image preview</span>';
});

// -----------------------------
// Initial load
// -----------------------------
refreshMemoryCount();
