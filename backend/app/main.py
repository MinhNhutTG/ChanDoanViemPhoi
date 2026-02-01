from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_image
from llm import generate_medical_json_report_1

app = Flask(__name__)
CORS(app)

@app.route("/api/predict", methods=["POST"])
def predict_api():
    data = {

    }
    data["ket_qua"] = None
    data["bao_cao"] = None
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    result = predict_image(request.files["image"])
    data["ket_qua"] = result["Ket qua"]
    report = generate_medical_json_report_1(forecast=result["Ket qua"], reliability=result["Do tin cay"])
    data["bao_cao"] =  report
    print(report)

    return data

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
