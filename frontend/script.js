// Đợi tài liệu tải xong để đảm bảo các ID đã tồn tại
document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadPrompt = document.getElementById('upload-prompt');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultContent = document.getElementById('result-content');
    const noResult = document.getElementById('no-result');
    const gradcamImage = document.getElementById('gradcam-image');
    const predictionText = document.getElementById('prediction-text');
    const resultBadge = document.getElementById('result-badge');

    // 1. KÍCH HOẠT CHỌN FILE: Khi click vào vùng dropzone, mở bảng chọn ảnh
    dropzone.addEventListener('click', () => {
        fileInput.click();
    });

    // 2. XỬ LÝ PREVIEW: Khi đã chọn ảnh xong
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                // Hiển thị ảnh vừa chọn
                imagePreview.src = event.target.result;
                imagePreview.classList.remove('hidden');
                // Ẩn hướng dẫn upload
                uploadPrompt.classList.add('hidden');
                
                // Reset vùng kết quả về trạng thái ban đầu
                resultContent.classList.add('hidden');
                noResult.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    // 3. HÀM DỰ ĐOÁN (Chạy khi nhấn nút "Bắt đầu chẩn đoán")
    window.predict = async function() {
        if (!fileInput.files[0]) {
            alert("Vui lòng chọn một file ảnh X-ray trước!");
            return;
        }

        // Trạng thái chờ (Loading)
        const originalBtnHTML = analyzeBtn.innerHTML;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner animate-spin mr-2"></i> Đang phân tích...';
        analyzeBtn.disabled = true;

        // Hiện khung kết quả trống
        noResult.classList.add('hidden');
        resultContent.classList.remove('hidden');
        predictionText.innerText = "Đang xử lý...";

        const formData = new FormData();
        formData.append("image", fileInput.files[0]);

        try {
            // Gửi tới API của bạn
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Lỗi Server");

            const data = await response.json();

            
            const label = data["ket_qua"];

            const percent = data["do_tin_cay"] ? (data["do_tin_cay"] * 100).toFixed(2) + "%" : "";
            
            predictionText.innerText = `${label} (${percent})`;

            // Đổi màu khung dựa trên kết quả
            if (label.toUpperCase().includes("PNEUMONIA") || label.toUpperCase().includes("VIÊM PHỔI")) {
                resultBadge.className = "p-4 rounded-lg border-l-4 bg-red-50 border-red-500 shadow-sm";
                predictionText.className = "text-2xl font-black text-red-700";
            } else {
                resultBadge.className = "p-4 rounded-lg border-l-4 bg-green-50 border-green-500 shadow-sm";
                predictionText.className = "text-2xl font-black text-green-700";
            }

            // Hiển thị ảnh GradCAM
           if (data["image_url"]) {
               gradcamImage.src = data["image_url"];
           }

        } catch (error) {
            console.error(error);
            predictionText.innerText = "❌ LỖI KẾT NỐI";
            predictionText.className = "text-2xl font-black text-orange-600";
        } finally {
            // Khôi phục nút bấm
            analyzeBtn.innerHTML = originalBtnHTML;
            analyzeBtn.disabled = false;
        }
    };
});