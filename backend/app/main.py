from flask import Flask, request, jsonify
from flask_cors import CORS
from services.predict_service import predict_image
from services.gradcam_service import run_gradcam
from schemas.response_schema import PredictionData, ApiResponse
# Các import khác giữ nguyên...
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

    result = predict_image(file_for_predict)
    pred_class, confidence, heatmap_base64 = run_gradcam(file_for_gradcam,
                              checkpoint_path="models/best_resnet_pneumonia.pth",
                              save_path="models/output/grad_cam_api.png",
                              device="cpu")



    prediction_data = PredictionData(
                            prediction=result["prediction_service"],
                            confidence=result["confidence_service"],
                            heatmap_url=f"data:image/png;base64,{heatmap_base64}"
                        )
    
    response = ApiResponse(
        success=True,
        prediction=prediction_data
    )


    return jsonify(response.to_dict())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
