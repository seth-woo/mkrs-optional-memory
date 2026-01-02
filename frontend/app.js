let uploadedImages = [];

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
        memoryCountEl.textContent = `Memory: ${data.count}`;
    } catch {
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

    try {
        const res = await fetch("/qa/single", {
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