import json
import os
from collections import Counter

def check_ner_data(data_path):
    if not os.path.exists(data_path):
        print(f"Error: File {data_path} not found.")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"--- THỐNG KÊ DỮ LIỆU ---")
    print(f"Tổng số câu: {len(data)}")
    
    label_counts = Counter()
    issues = []
    text_label_map = {} # Để kiểm tra sự mâu thuẫn

    for i, item in enumerate(data):
        text = item.get("text", "")
        entities = item.get("entities", [])
        
        if not entities:
            # Không phải lỗi, nhưng cần biết số lượng câu không nhãn
            continue

        last_end = -1
        # Sắp xếp thực thể theo vị trí bắt đầu
        sorted_entities = sorted(entities, key=lambda x: x[0])

        for start, end, label in sorted_entities:
            label_counts[label] += 1
            ent_text = text[start:end]

            # 1. Kiểm tra tọa độ hợp lệ
            if start >= end:
                issues.append(f"Câu {i}: Tọa độ sai ({start}, {end}) cho nhãn {label}")
            if end > len(text):
                issues.append(f"Câu {i}: Tọa độ vượt quá chiều dài câu (end={end}, len={len(text)})")

            # 2. Kiểm tra chồng lấn (Overlapping)
            if start < last_end:
                issues.append(f"Câu {i}: Thực thể '{ent_text}' ({label}) bị chồng lấn với thực thể trước đó.")
            last_end = end

            # 3. Kiểm tra sự nhất quán (Inconsistency)
            if ent_text in text_label_map:
                if text_label_map[ent_text] != label:
                    # Đây là cảnh báo quan trọng
                    if len(ent_text) > 3: # Bỏ qua các từ quá ngắn như 'A', 'B', '1'
                        issues.append(f"CẢNH BÁO NHẤT QUÁN: '{ent_text}' được gán là {label}, nhưng ở câu khác lại là {text_label_map[ent_text]}")
            else:
                text_label_map[ent_text] = label

    print("\n--- SỐ LƯỢNG NHÃN ---")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    print("\n--- CÁC VẤN ĐỀ PHÁT HIỆN ---")
    if not issues:
        print("Chúc mừng! Dữ liệu của bạn rất sạch sẽ.")
    else:
        # In ra tối đa 20 lỗi đầu tiên
        for issue in issues[:20]:
            print(f"[!] {issue}")
        if len(issues) > 20:
            print(f"... và {len(issues) - 20} lỗi khác.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "ner_training_data.json")
    check_ner_data(data_path)
