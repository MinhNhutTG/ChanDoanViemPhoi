from flask import Blueprint, request, jsonify
from utils.dicom_util import dicom_to_preview
import base64

# Tên blueprint + prefix URL chung cho nhóm route này
preview_bp = Blueprint("preview", __name__, url_prefix="/api")


@preview_bp.route("/preview", methods=["POST"])
def preview():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    try:
        jpg_buffer = dicom_to_preview(file)
        img_base64 = base64.b64encode(jpg_buffer.read()).decode("utf-8")
        return jsonify({"image": img_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500