from flask import Blueprint, request, jsonify
from services.predict_service import predict_image
from services.gradcam_service import run_gradcam
from schemas.response_schema import PredictionData, ApiResponse
from utils.generate_report import generate_medical_json_report
from io import BytesIO
import os
import gdown
import torch



predict_bp = Blueprint("predict", __name__, url_prefix="/api")

MODEL_PATH = "models/best_resnet50.pth"




@predict_bp.route("/predict", methods=["POST"])
def predict_api():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Đọc 1 lần vào RAM, tái sử dụng nhiều lần
    img_bytes = file.read()
    file_for_predict = BytesIO(img_bytes)
    file_for_gradcam = BytesIO(img_bytes)

    # Dự đoán
    result = predict_image(file_for_predict)

    # GradCAM
    pred_class, heatmap_base64 , affected_region = run_gradcam(
        file_for_gradcam,
        checkpoint_path=MODEL_PATH,
        device="cpu"
    )
    report = generate_medical_json_report(result["prediction_service"], result["confidence_service"] , affected_region)
    
    # Đóng gói phản hồi
    prediction_data = PredictionData(
        prediction=result["prediction_service"],
        confidence=result["confidence_service"],
        probabilities=result["probabilities"], 
        affected_region = affected_region,
        heatmap_url=f"data:image/png;base64,{heatmap_base64}",
        report = report
    )

    response = ApiResponse(success=True, prediction=prediction_data)
    return jsonify(response.to_dict())
