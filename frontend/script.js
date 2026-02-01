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




