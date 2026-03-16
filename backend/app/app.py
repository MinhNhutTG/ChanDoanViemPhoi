from flask import Flask, request, jsonify
from flask import render_template
from flask_cors import CORS
from services.predict_service import predict_image
from services.gradcam_service import run_gradcam
from schemas.response_schema import PredictionData, ApiResponse
from utils.dicom_util import dicom_to_preview

from routes.preview_routes import preview_bp
from routes.predict_routes import predict_bp
# Các import khác giữ nguyên...
from io import BytesIO
import base64


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
    print("FLASK STARTED SUCCESSFULLY")
    app.run(host="0.0.0.0", port=5001)
  
