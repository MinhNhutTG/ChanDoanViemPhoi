from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_image
from llm import generate_medical_json_report_1
from grad_CAM import run_gradcam
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route("/api/predict", methods=["POST"])
def predict_api():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    
    # 2. Đọc vào bộ nhớ để tái sử dụng nhiều lần
    img_bytes = file.read()
    
    # Tạo các bản sao file ảo từ img_bytes
    file_for_predict = BytesIO(img_bytes)
    file_for_gradcam = BytesIO(img_bytes)

    result = predict_image(request.files["image"])
    pred_class, confidence, heatmap_base64 = run_gradcam(request.files["image"],
                              checkpoint_path="models/best_resnet_pneumonia.pth",
                              save_path="models/output/grad_cam_api.png",
                              device="cpu")

    print(heatmap_base64)
    report = generate_medical_json_report_1(forecast=result["Ket_qua"], reliability=result["Do_tin_cay"])
    
    print(report)

    return jsonify({
        "ket_qua": result["Ket_qua"],
        "do_tin_cay": result["Do_tin_cay"],
        "bao_cao": report,
        "image_url": f"data:image/png;base64,{heatmap_base64}" # Gửi kèm prefix
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
