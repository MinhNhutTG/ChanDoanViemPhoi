function previewImage(event) {
    const reader = new FileReader();
    const preview = document.getElementById('image-preview');

    reader.onload = function () {
        preview.src = reader.result;
        preview.style.display = 'block';
    }
    reader.readAsDataURL(event.target.files[0]);
}

async function predict() {
    const fileInput = document.getElementById('image');

    if (fileInput.files.length === 0) {
        alert("Vui lòng chọn một file ảnh X-ray!");
        return;
    }

    document.getElementById("result-text").innerText = "⏳ Đang phân tích...";
    document.getElementById("llm-content").innerText = "AI đang tạo báo cáo y khoa...";

    const formData = new FormData();
    formData.append("image", fileInput.files[0]);

    try {
        const res = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        const result = await res.json();
        document.getElementById("result-text").innerText = result["ket_qua"];
        document.getElementById("llm-content").innerText = result["bao_cao"];

        // const data = {
        //     data_label: result.label,
        //     data_confidence: (result.confidence * 100).toFixed(2) + "%"
        // };

        // // ✅ HIỂN THỊ KẾT QUẢ CHẨN ĐOÁN
        // document.getElementById("result-text").innerText =
        //     `${data.data_label} (${data.data_confidence})`;

       
    } catch (err) {
        console.error(err);
        document.getElementById("result-text").innerText = "❌ Lỗi server";
    }
}




async function genareteLLM(data) {
    const API_KEY = "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"; // Thay key thật của bạn vào đây

    const prompt = `<s>[INST] You are a medical AI assistant.
    Create a short clinical report based on:
    Diagnosis: ${data.data_label}
    Confidence: ${data.data_confidence}

    Return the report with 3 sections: Summary, Analysis, Recommendation. [/INST]`;

    try {
        const response = await fetch("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${API_KEY}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                inputs: prompt,
                parameters: { max_new_tokens: 500, return_full_text: false }
            })
        });

        const result = await response.json();
        
        // Inference API trả về một mảng: [{ generated_text: "..." }]
        const aiText = result[0].generated_text;

        // Vì Mistral (Inference API) thường trả về text thuần thay vì JSON chuẩn, 
        // ta hiển thị trực tiếp vào div
        document.getElementById("llm-content").innerHTML = `<pre style="white-space: pre-wrap;">${aiText}</pre>`;

    } catch (err) {
        console.error("Lỗi gọi LLM:", err);
        document.getElementById("llm-content").innerText = "❌ Không thể tạo báo cáo lúc này.";
    }
}