from flask import Flask, request, jsonify
from flask_cors import CORS
from services.predict_service import predict_image
from services.gradcam_service import run_gradcam
from schemas.response_schema import PredictionData, ApiResponse
from utils.dicom_util import dicom_to_preview
# Các import khác giữ nguyên...
from io import BytesIO
import base64


app = Flask(__name__)
CORS(app)


@app.route("/api/preview", methods=["POST"])
def preview():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
   
    file = request.files["file"]

    try:

        jpg_buffer = dicom_to_preview(file)

        img_base64 = base64.b64encode(jpg_buffer.read()).decode("utf-8")
       
        return jsonify({
            "image": img_base64

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


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


    file_for_predict = BytesIO(img_bytes)
    result = predict_image(file_for_predict)

    file_for_gradcam.seek(0)
    pred_class, heatmap_base64 = run_gradcam(file_for_gradcam,
                              checkpoint_path="models/best_resnet50_rsna.pth",
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
    app.run(host="0.0.0.0", port=5000)
