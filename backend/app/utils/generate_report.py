import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()



client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)

def generate_medical_json_report(forecast,reliability,affected_region) -> dict:
    forecast = forecast
    reliability = reliability
    affected_region = affected_region
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
                    "Mô hình AI phân tích hình ảnh dự đoán nhãn '{forecast}' với độ tin cậy {reliability}.\n"
                    "Vùng giải phẫu mà mô hình tập trung trong quá trình phân tích: '{affected_region}'.\n\n"
                    "Dựa solely vào thông tin này, hãy tạo một báo cáo y tế chính thức theo cấu trúc sau:\n\n"
                    "1. KẾT QUẢ CHẨN ĐOÁN\n"
                    "Nêu rõ có phát hiện viêm phổi hay không. Đánh giá độ tin cậy {reliability} "
                    "ở mức thấp / trung bình .\n\n"
                    "2. VÙNG MÔ HÌNH TẬP TRUNG\n"
                    "Giải thích '{affected_region}' là vùng giải phẫu gì trên phim X-quang ngực. "
                    "3. KHUYẾN NGHỊ\n"
                    "Đưa ra 2-3 khuyến nghị ngắn gọn cho bác sĩ lâm sàng dựa trên kết quả trên. "
                    "Lưu ý rõ rằng đây là công cụ hỗ trợ chẩn đoán bằng AI, không thay thế đánh giá của bác sĩ.\n\n"
                    "Viết bằng tiếng Việt, ngôn ngữ chuyên môn nhưng dễ hiểu. "
                    "Không bịa thêm thông tin ngoài dữ liệu đã cung cấp."
                ).format(forecast=forecast, reliability=reliability, affected_region=affected_region)
            }
        ],
        
    )
    # Lấy content và parse JSON
    content = completion.choices[0].message.content
    return content



