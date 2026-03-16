import os
import base64
from io import BytesIO
from flask_cors import CORS
from flask import render_template
from flask import Flask, request, jsonify
from routes.preview_routes import preview_bp
from routes.predict_routes import predict_bp
from utils.dicom_util import dicom_to_preview
from services.gradcam_service import run_gradcam
from services.predict_service import predict_image
from schemas.response_schema import PredictionData, ApiResponse


app = Flask(
    __name__,
    template_folder="templates",   # Flask tìm HTML ở đây
    static_folder="static"         # Flask tìm JS/CSS ở đây
)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")


app.register_blueprint(preview_bp)
app.register_blueprint(predict_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # mặc định 5001 nếu không có env
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"✅ FLASK STARTED — http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)