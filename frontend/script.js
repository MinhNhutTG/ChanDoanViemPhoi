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

        if (!res.ok) {
        // Lấy nội dung lỗi dưới dạng văn bản để đọc
        const errorDetail = await res.text();
        console.error("Nội dung lỗi từ server:", errorDetail);
        throw new Error(`Server báo lỗi ${res.status}`);
    }

        const result = await res.json();
        document.getElementById("result-text").innerText = result["ket_qua"];
        document.getElementById("llm-content").innerText = result["bao_cao"];
        document.getElementById("heatmap-image").src = result["image_url"];
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




function switchTab(tabId) {
    // Ẩn tất cả tab
    document.querySelectorAll('.tab-panel').forEach(tab => {
        tab.classList.remove('active');
    });

    // Bỏ active tất cả button
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Hiện tab được chọn
    document.getElementById(tabId).classList.add('active');

    // Active button tương ứng
    event.target.classList.add('active');
}