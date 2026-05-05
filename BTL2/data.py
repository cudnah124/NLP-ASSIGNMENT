import json
from collections import Counter

def ann(text, phrase, label):
    idx = text.find(phrase)
    if idx == -1:
        raise ValueError(f"'{phrase}' not found in:\n  {text}")
    return [idx, idx + len(phrase), label]

def make(text, *pairs):
    entities = [ann(text, p, l) for p, l in pairs]
    return {"text": text, "entities": entities}

records = [
    make("Party A shall pay Party B the sum of 10,000,000 VND on or before 01/01/2024.",
         ("Party A","PARTY"), ("Party B","PARTY"), ("10,000,000 VND","MONEY"), ("01/01/2024","DATE")),
    make("Party B shall pay the full rental amount before the 5th of each month.",
         ("Party B","PARTY"), ("the 5th of each month","DATE")),
    make("The Employer shall transfer 5,000,000 VND to the Employee by 15/06/2024.",
         ("The Employer","PARTY"), ("5,000,000 VND","MONEY"), ("the Employee","PARTY"), ("15/06/2024","DATE")),
    make("Party A agrees to pay Party B a monthly fee of 2,000,000 VND.",
         ("Party A","PARTY"), ("Party B","PARTY"), ("2,000,000 VND","MONEY")),
    make("The Lessor shall receive the security deposit of 20,000,000 VND from the Lessee within 7 days of signing.",
         ("The Lessor","PARTY"), ("20,000,000 VND","MONEY"), ("the Lessee","PARTY"), ("7 days of signing","DATE")),
    make("Party A shall deliver the goods to Party B no later than 30/09/2024.",
         ("Party A","PARTY"), ("Party B","PARTY"), ("30/09/2024","DATE")),
    make("The Employee shall receive a monthly salary of 15,000,000 VND from the Employer.",
         ("The Employee","PARTY"), ("15,000,000 VND","MONEY"), ("the Employer","PARTY")),
    make("Party B shall reimburse Party A for all expenses not exceeding 3,000,000 VND.",
         ("Party B","PARTY"), ("Party A","PARTY"), ("3,000,000 VND","MONEY")),
    make("The Contractor shall complete the project by 31/12/2024 and submit the final report to the Client.",
         ("The Contractor","PARTY"), ("31/12/2024","DATE"), ("the Client","PARTY")),
    make("Party A shall pay an advance of 50,000,000 VND on the effective date.",
         ("Party A","PARTY"), ("50,000,000 VND","MONEY"), ("the effective date","DATE")),
    
    # RATE + PENALTY
    make("If payment is delayed, a penalty of 1% per day shall apply.",
         ("a penalty of 1% per day","PENALTY")),
    make("Late payments shall incur an interest rate of 0.05% per day on the outstanding balance.",
         ("0.05% per day","RATE")),
    make("Party B shall be subject to a penalty of 5,000,000 VND for each day of delay.",
         ("Party B","PARTY"), ("a penalty of 5,000,000 VND","PENALTY")),
    make("The penalty rate for early termination is 10% of the total contract value.",
         ("10%","RATE")),
    make("A late fee of 2% per month will be charged on overdue amounts.",
         ("2% per month","RATE")),
    make("Party A shall pay liquidated damages equal to 15% of the contract price.",
         ("Party A","PARTY"), ("liquidated damages equal to 15% of the contract price","PENALTY")),
    make("If Party B terminates the contract early, a penalty of 3,000,000 VND shall be imposed.",
         ("Party B","PARTY"), ("a penalty of 3,000,000 VND","PENALTY")),
    make("The interest rate applicable shall be 8% per annum as specified in the agreement.",
         ("8% per annum","RATE")),
    make("Any breach by the Contractor shall result in liquidated damages of 10,000,000 VND.",
         ("the Contractor","PARTY"), ("liquidated damages of 10,000,000 VND","PENALTY")),
    make("Overdue invoices will attract a surcharge of 1.5% per month.",
         ("a surcharge of 1.5% per month","PENALTY")),
    make("A penalty of 0.1% per day shall be imposed on overdue payments.",
         ("A penalty of 0.1% per day","PENALTY"), ("0.1% per day","RATE")),
    make("The default interest rate shall not exceed 20% per annum.",
         ("20% per annum","RATE")),
    make("Party A shall pay a penalty fee of 2,000,000 VND for each week of delay.",
         ("Party A","PARTY"), ("a penalty fee of 2,000,000 VND","PENALTY")),
    
    # LAW
    make("This agreement is governed by the Civil Code 2015 of Vietnam.",
         ("Civil Code 2015","LAW")),
    make("The parties agree to comply with the Labor Code 2019 in all employment matters.",
         ("Labor Code 2019","LAW")),
    make("Disputes shall be resolved in accordance with Article 45 of Decree No. 15/2015/ND-CP.",
         ("Article 45 of Decree No. 15/2015/ND-CP","LAW")),
    make("This contract shall be interpreted pursuant to the Commercial Law 2005.",
         ("Commercial Law 2005","LAW")),
    make("Party A warrants compliance with Circular 09/2015/TT-NHNN issued by the State Bank.",
         ("Party A","PARTY"), ("Circular 09/2015/TT-NHNN","LAW")),
    make("The Employer agrees to adhere to the provisions of the Law on Enterprises 2020.",
         ("The Employer","PARTY"), ("Law on Enterprises 2020","LAW")),
    make("All tax obligations shall be fulfilled pursuant to the Tax Administration Law 2019.",
         ("Tax Administration Law 2019","LAW")),
    make("The confidentiality clause is governed by Article 17 of the Intellectual Property Law.",
         ("Article 17 of the Intellectual Property Law","LAW")),
    make("Party B must comply with Decree No. 98/2020/ND-CP on penalties for commercial violations.",
         ("Party B","PARTY"), ("Decree No. 98/2020/ND-CP","LAW")),
    make("The arbitration procedure shall follow the Law on Commercial Arbitration 2010.",
         ("Law on Commercial Arbitration 2010","LAW")),
    make("Liability shall be limited as provided under Article 302 of the Commercial Law 2005.",
         ("Article 302 of the Commercial Law 2005","LAW")),
    make("Party A must register the contract under Decree No. 163/2006/ND-CP.",
         ("Party A","PARTY"), ("Decree No. 163/2006/ND-CP","LAW")),
    
    # DATE
    make("This contract is effective from 01/03/2024 and expires on 28/02/2025.",
         ("01/03/2024","DATE"), ("28/02/2025","DATE")),
    make("Payment must be received within 30 days of invoice.",
         ("within 30 days of invoice","DATE")),
    make("The probationary period shall last 60 days from the date of joining.",
         ("60 days from the date of joining","DATE")),
    make("Party A shall submit the report no later than the 10th business day of each quarter.",
         ("Party A","PARTY"), ("the 10th business day of each quarter","DATE")),
    make("The lease term commences on 01/01/2024 and terminates on 31/12/2026.",
         ("01/01/2024","DATE"), ("31/12/2026","DATE")),
    make("The notice period for termination is 30 days prior to the intended termination date.",
         ("30 days prior to the intended termination date","DATE")),
    make("Party B must renew the contract before the expiry date of 30/06/2025.",
         ("Party B","PARTY"), ("30/06/2025","DATE")),
    make("Deliverables are due on the last working day of each calendar month.",
         ("the last working day of each calendar month","DATE")),
    make("Party A shall provide written notice at least 15 days before the termination date.",
         ("Party A","PARTY"), ("15 days before the termination date","DATE")),
    make("The warranty period is 12 months from the date of acceptance.",
         ("12 months from the date of acceptance","DATE")),
    make("The final payment is due on 15 June 2024.",
         ("15 June 2024", "DATE")),
    make("This agreement shall commence on 01 August 2025.",
         ("01 August 2025", "DATE")),
    make("Party A must deliver the goods by 20th December 2023.",
         ("Party A", "PARTY"), ("20th December 2023", "DATE")),
    
    # MONEY
    make("The total contract value is 500,000,000 VND.",
         ("500,000,000 VND","MONEY")),
    make("Party A shall provide a security deposit of USD 2,000 upon signing.",
         ("Party A","PARTY"), ("USD 2,000","MONEY")),
    make("The monthly retainer fee is 8,000,000 VND payable in advance.",
         ("8,000,000 VND","MONEY")),
    make("Party B shall pay a refundable deposit of 10,000,000 VND.",
         ("Party B","PARTY"), ("10,000,000 VND","MONEY")),
    make("The service fee of 1,200,000 VND per month shall be paid by Party A to Party B.",
         ("1,200,000 VND","MONEY"), ("Party A","PARTY"), ("Party B","PARTY")),
    make("The Lessor shall refund 20,000,000 VND to the Lessee within 15 days of contract termination.",
         ("The Lessor","PARTY"), ("20,000,000 VND","MONEY"), ("the Lessee","PARTY"), ("15 days","DATE")),
    make("The Borrower shall repay the Lender the principal of 100,000,000 VND.",
         ("The Borrower","PARTY"), ("the Lender","PARTY"), ("100,000,000 VND","MONEY")),
    make("Party A shall provide a performance bond of 50,000,000 VND to Party B.",
         ("Party A","PARTY"), ("50,000,000 VND","MONEY"), ("Party B","PARTY")),
    
    # PARTY variations
    make("The Seller agrees to transfer ownership to the Buyer upon receipt of full payment.",
         ("The Seller","PARTY"), ("the Buyer","PARTY")),
    make("The Service Provider shall assign a dedicated team to the Client for the duration of the project.",
         ("The Service Provider","PARTY"), ("the Client","PARTY")),
    make("Party A, hereinafter referred to as the Licensor, grants Party B the right to use the software.",
         ("Party A","PARTY"), ("the Licensor","PARTY"), ("Party B","PARTY")),
    make("The Franchisor shall provide training and support to the Franchisee within 30 days.",
         ("The Franchisor","PARTY"), ("the Franchisee","PARTY"), ("30 days","DATE")),
    
    # Multi-entity
    make("Party A shall pay Party B 25,000,000 VND by 15/03/2024, failing which a penalty rate of 2% per month shall apply.",
         ("Party A","PARTY"), ("Party B","PARTY"), ("25,000,000 VND","MONEY"),
         ("15/03/2024","DATE"), ("a penalty rate of 2% per month","PENALTY"), ("2% per month","RATE")),
    make("The Employer shall pay the Employee a monthly salary of 12,000,000 VND in accordance with the Labor Code 2019.",
         ("The Employer","PARTY"), ("the Employee","PARTY"), ("12,000,000 VND","MONEY"), ("Labor Code 2019","LAW")),
    make("Any breach of this agreement by Party B shall incur a penalty of 50,000,000 VND as stipulated in Article 12 of the Civil Code 2015.",
         ("Party B","PARTY"), ("a penalty of 50,000,000 VND","PENALTY"), ("Article 12 of the Civil Code 2015","LAW")),
    make("The interest rate is set at 9% per annum pursuant to Circular 39/2016/TT-NHNN.",
         ("9% per annum","RATE"), ("Circular 39/2016/TT-NHNN","LAW")),
    make("Party B shall indemnify Party A for any losses not exceeding 200,000,000 VND within 30 days of the claim.",
         ("Party B","PARTY"), ("Party A","PARTY"), ("200,000,000 VND","MONEY"), ("30 days of the claim","DATE")),
    make("The Contractor must complete all work by 31/03/2025 or face a penalty of 1% per week of the contract value.",
         ("The Contractor","PARTY"), ("31/03/2025","DATE"),
         ("a penalty of 1% per week of the contract value","PENALTY"), ("1% per week","RATE")),
    make("The Lessee shall pay the Lessor 6,000,000 VND per month for the lease of the premises.",
         ("The Lessee","PARTY"), ("the Lessor","PARTY"), ("6,000,000 VND","MONEY")),
    make("Party A must notify Party B in writing at least 60 days before the expiry date of 01/06/2025.",
         ("Party A","PARTY"), ("Party B","PARTY"), ("60 days","DATE"), ("01/06/2025","DATE")),
    make("Under the Enterprise Income Tax Law 2008, Party A shall withhold 10% of the payment.",
         ("Enterprise Income Tax Law 2008","LAW"), ("Party A","PARTY"), ("10%","RATE")),
    make("The Buyer shall pay the Seller USD 10,000 within 14 days of delivery.",
         ("The Buyer","PARTY"), ("the Seller","PARTY"), ("USD 10,000","MONEY"), ("14 days of delivery","DATE")),
    make("All disputes shall be settled by arbitration in accordance with the Law on Commercial Arbitration 2010 within 60 days.",
         ("Law on Commercial Arbitration 2010","LAW"), ("60 days","DATE")),
    make("Party B shall pay a late charge equal to 0.03% per day on any overdue amounts.",
         ("Party B","PARTY"), ("a late charge equal to 0.03% per day","PENALTY"), ("0.03% per day","RATE")),
    make("The Service Provider shall complete the implementation phase by 30/11/2024 for the Client.",
         ("The Service Provider","PARTY"), ("30/11/2024","DATE"), ("the Client","PARTY")),
    make("The Employer shall pay the Employee an annual bonus of 2 months salary no later than 31/01 of each year.",
         ("The Employer","PARTY"), ("the Employee","PARTY"), ("31/01 of each year","DATE")),
    make("Party A shall pay Party B a maintenance fee of 3,500,000 VND per quarter starting from 01/04/2024.",
         ("Party A","PARTY"), ("Party B","PARTY"), ("3,500,000 VND","MONEY"), ("01/04/2024","DATE")),
    make("The Seller warrants that all goods comply with the Law on Product Quality 2007.",
         ("The Seller","PARTY"), ("Law on Product Quality 2007","LAW")),
    make("Party B shall pay interest at a rate of 12% per annum on overdue amounts.",
         ("Party B","PARTY"), ("12% per annum","RATE")),
    # SỬA LỖI Ở ĐÂY: "the Client" -> "The Client" vì nó đứng đầu câu
    make("The Client shall pay the Service Provider 15,000,000 VND upon completion of each milestone.",
         ("The Client","PARTY"), ("the Service Provider","PARTY"), ("15,000,000 VND","MONEY")),
]

# Thống kê kết quả
label_counts = Counter()
for r in records:
    for e in r["entities"]:
        label_counts[e[2]] += 1

print(f"Total records : {len(records)}")
print("Label distribution:")
for lbl, cnt in sorted(label_counts.items()):
    print(f"  {lbl:<10}: {cnt}")

# Lưu file
output_path = "ner_training_data.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {output_path}")