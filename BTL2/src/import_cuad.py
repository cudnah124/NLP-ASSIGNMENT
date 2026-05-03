import pandas as pd
import json
import os
import ast
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # CUAD values are often string representations of lists: "['text']"
    if text.startswith('[') and text.endswith(']'):
        try:
            items = ast.literal_eval(text)
            if isinstance(items, list):
                text = " ".join(items)
        except:
            pass
    
    # Remove extra quotes and whitespace
    text = text.replace('""', '"').strip()
    # Remove simple artifacts
    text = re.sub(r'\s+', ' ', text)
    return text

def import_cuad():
    cuad_path = r'c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\252\NLP\BTL\data\archive\CUAD_v1\master_clauses.csv'
    output_path = r'c:\Users\nhanha213\OneDrive - hcmut.edu.vn\Desktop\STUDY\252\NLP\BTL\BTL2\data\intent_training_data.json'
    
    if not os.path.exists(cuad_path):
        print(f"Error: CUAD file not found at {cuad_path}")
        return

    print(f"Reading CUAD data from {cuad_path}...")
    df = pd.read_csv(cuad_path)
    
    mapping = {
        'Termination Condition': [
            'Termination For Convenience-Answer', 
            'Notice Period To Terminate Renewal- Answer', 
            'Change Of Control-Answer'
        ],
        'Right': [
            'Renewal Term-Answer', 
            'Rofr/Rofo/Rofn-Answer', 
            'License Grant-Answer', 
            'Audit Rights-Answer'
        ],
        'Prohibition': [
            'Non-Compete-Answer', 
            'Exclusivity-Answer', 
            'No-Solicit Of Customers-Answer', 
            'No-Solicit Of Employees-Answer', 
            'Anti-Assignment-Answer'
        ],
        'Obligation': [
            'Insurance-Answer', 
            'Minimum Commitment-Answer', 
            'Ip Ownership Assignment-Answer',
            'Governing Law-Answer'
        ]
    }
    
    new_samples = []
    
    for label, columns in mapping.items():
        count = 0
        for col in columns:
            if col in df.columns:
                # Filter out empty or "[]" or "No" values
                valid_rows = df[df[col].notna() & (df[col] != '[]') & (df[col] != 'No')]
                for val in valid_rows[col]:
                    text = clean_text(val)
                    if len(text) > 20: # Skip too short fragments
                        new_samples.append({'text': text, 'label': label})
                        count += 1
        print(f"  Extracted {count} samples for {label}")

    # Load existing data
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # Merge and deduplicate by text
    seen_texts = {s['text'] for s in existing_data}
    added_count = 0
    for s in new_samples:
        if s['text'] not in seen_texts:
            existing_data.append(s)
            seen_texts.add(s['text'])
            added_count += 1
            
    print(f"Successfully added {added_count} unique samples from CUAD.")
    print(f"Total training samples: {len(existing_data)}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    import_cuad()
