# NLP Legal Contract Analysis

Pipeline phân tích văn bản hợp đồng pháp lý tiếng Anh.

## Cấu trúc dự án

- **BTL1/**: Tiền xử lý văn bản. Xử lý tách câu (Clause Splitting), nhận diện cụm danh từ (Noun Chunking) và phân tích cú pháp (Dependency Analysis) bằng spaCy.
- **BTL2/**: Trích xuất thông tin & Phân tích ngữ nghĩa. Huấn luyện và dự đoán nhận diện thực thể (NER), gán nhãn vai trò ngữ nghĩa (SRL) và phân loại câu (Intent Classification).
- **BTL3/**: Hệ thống Question Answering sử dụng RAG (Retrieval-Augmented Generation) kết hợp LangChain, ChromaDB và Google Gemini.
- **data/**: Dữ liệu huấn luyện và file dữ liệu thô.
- **report_assets/**: Nơi lưu biểu đồ, hình ảnh trực quan dùng để viết báo cáo.

## Cài đặt chung

Cài đặt thư viện cơ bản cho BTL1 & BTL2:
```bash
pip install spacy scikit-learn pandas matplotlib seaborn
python -m spacy download en_core_web_sm
```

## Hướng dẫn sử dụng

### 1. Chạy BTL1 (Tiền xử lý)
Đọc file `BTL1/input/raw_contracts.txt` và phân tích cú pháp:
```bash
cd BTL1
python src/main.py
```

### 2. Chạy BTL2 (Trích xuất & Huấn luyện)
Đọc kết quả từ BTL1 và tiến hành chạy mô hình NER, SRL, Intent:
```bash
cd BTL2
# Train và Inference (chạy mặc định)
python src/main.py 

# Chạy nhanh Inference (bỏ qua train mô hình DistilBERT)
python src/main.py --no-transformer
```

### 3. Chạy BTL3 (Chatbot RAG)
Phần ứng dụng Chatbot yêu cầu cài thêm thư viện chuyên dụng và cấu hình API Key.

**Cài đặt thư viện:**
```bash
cd BTL3
pip install -r requirements.txt
```

**Cấu hình API Key:**
Cập nhật file `.env` trong thư mục `BTL3` với API Key của Google Gemini:
```env
GOOGLE_API_KEY=your_api_key_here
```

**Khởi tạo Database (Chỉ chạy 1 lần):**
```bash
python src/data_ingestion.py
```

**Khởi chạy Giao diện Chatbot:**
```bash
streamlit run src/app.py
```