# llm.py
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()



client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)

def generate_medical_json_report_1(forecast,reliability) -> dict:
    forecast = forecast
    reliability = reliability

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
           {
                "role": "system",
                "content": (
                    "You are a licensed physician. Interpret AI predictions cautiously. "
                    "Write a formal medical report in professional clinical language. "
                    "The report MUST be written in Vietnamese using standard medical terminology. "
                    "Do NOT output JSON. Output plain text only."
                )

            },
            {
               "role": "user",
                "content": (
                    "AI image analysis predicts the label '{forecast}' with a confidence score of {reliability}.\n"
                    "Based solely on this information, generate a formal medical report interpreting the result,\n"
                    "including diagnostic impression, limitations, and clinical recommendations."
                ).format(forecast=forecast,reliability=reliability)
            }
        ],
        
    )
    # Lấy content và parse JSON
    content = completion.choices[0].message.content
    return content



def generate_medical_json_report() -> dict:
    """
    Gọi Groq LLM để tạo báo cáo chẩn đoán phổi dưới dạng JSON.
    Trả về: dict (đã parse JSON)
    """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a professional Medical AI. Write a clinical diagnostic report in formal medical style. Output TEXT ONLY. No JSON, no bullet points, no explanation."
            },
            {
                "role": "user",
                "content": (
                    "Generate a lung diagnostic report based on available data:\n"
                    "- Patient_Info\n"
                    "- Lab_Results (SpO2, bpm)\n"
                    "- AI_Analysis (Accuracy, Confidence)\n"
                    "- Diagnosis\n"
                    "- AI_Metadata"
                )
            }
        ],
        response_format={"type": "json_object"}
    )

    # Lấy content và parse JSON
    content = completion.choices[0].message.content
    return json.loads(content)
