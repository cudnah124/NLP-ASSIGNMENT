import json

dataset = []

def add(text, pairs):
    """pairs = [(substring, label, nth=0), ...]"""
    entities = []
    for p in pairs:
        sub, lbl = p[0], p[1]
        nth = p[2] if len(p) > 2 else 0
        idx = -1; count = 0
        for i in range(len(text)):
            if text[i:i+len(sub)] == sub:
                if count == nth: idx = i; break
                count += 1
        if idx == -1:
            raise ValueError(f"NOT FOUND (nth={nth}): '{sub}'")
        entities.append([idx, idx+len(sub), lbl])

    entities.sort(key=lambda x: x[0])

    # strict non-overlap / non-nested check
    for i in range(len(entities)):
        s1,e1,l1 = entities[i]
        for j in range(i+1, len(entities)):
            s2,e2,l2 = entities[j]
            if s2 < e1:          # overlap or nesting
                raise ValueError(
                    f"OVERLAP: [{s1},{e1},'{l1}'] vs [{s2},{e2},'{l2}']\n"
                    f"  '{text[s1:e1]}'\n  '{text[s2:e2]}'\nIn: {text[:80]}..."
                )
    dataset.append({"text": text, "entities": entities})

# ═══════════════════════════════════════════════════════════════
# KEY DESIGN RULE – no nested spans:
#  • PENALTY  = the trigger / mechanism phrase, ends BEFORE the rate/amount
#  • RATE     = the percentage value phrase, separate token group
#  • MONEY    = the currency amount, separate token group
#  • DATE     = date string, separate token group
#  • LAW      = law reference, separate token group
#  • PARTY    = party name, separate token group
# ═══════════════════════════════════════════════════════════════

# ── 1 ─────────────────────────────────────────────────────────
add(
    "Pursuant to the Civil Code 2015 and the Commercial Law 2005, "
    "ABC Trading JSC (Party A) shall pay VietBank Corp. (Party B) "
    "a principal of 5,000,000,000 VND at 9.5% per annum from 01/03/2025; "
    "failure to pay by 01/09/2025 shall attract a late-payment penalty "
    "accruing at 0.05% of the outstanding balance per day until settlement.",
    [
        ("Civil Code 2015","LAW"),
        ("Commercial Law 2005","LAW"),
        ("ABC Trading JSC","PARTY"),
        ("Party A","PARTY"),
        ("VietBank Corp.","PARTY"),
        ("Party B","PARTY"),
        ("5,000,000,000 VND","MONEY"),
        ("9.5% per annum","RATE"),
        ("01/03/2025","DATE"),
        ("01/09/2025","DATE"),
        ("a late-payment penalty","PENALTY"),
        ("0.05%","RATE"),
    ]
)

# ── 2 ─────────────────────────────────────────────────────────
add(
    "Under the Labor Code 2019 and Decree No. 145/2020/ND-CP, "
    "Horizon Tech Ltd. (Employer) shall pay Ms. Nguyen Thi Lan (Employee) "
    "a gross monthly salary of 45,000,000 VND with an annual increment of 8% "
    "effective from January 1, 2026; wrongful termination before June 30, 2027 "
    "shall trigger a wrongful-termination penalty equal to six months gross salary.",
    [
        ("Labor Code 2019","LAW"),
        ("Decree No. 145/2020/ND-CP","LAW"),
        ("Horizon Tech Ltd.","PARTY"),
        ("Employer","PARTY"),
        ("Ms. Nguyen Thi Lan","PARTY"),
        ("Employee","PARTY"),
        ("45,000,000 VND","MONEY"),
        ("8%","RATE"),
        ("January 1, 2026","DATE"),
        ("June 30, 2027","DATE"),
        ("a wrongful-termination penalty equal to six months gross salary","PENALTY"),
    ]
)

# ── 3 ─────────────────────────────────────────────────────────
add(
    "Under the Law on Real Estate Business 2023 and the Land Law 2024, "
    "Sunrise Realty JSC (Seller) shall transfer the apartment to "
    "Mr. Tran Van Minh (Buyer) for 3,800,000,000 VND; "
    "a deposit of 380,000,000 VND is due by 15/02/2025, "
    "balance by 30/06/2025, and handover by 31/12/2025; "
    "delay in handover shall incur a daily delay penalty "
    "charged at 0.05% of the total purchase price per calendar day.",
    [
        ("Law on Real Estate Business 2023","LAW"),
        ("Land Law 2024","LAW"),
        ("Sunrise Realty JSC","PARTY"),
        ("Seller","PARTY"),
        ("Mr. Tran Van Minh","PARTY"),
        ("Buyer","PARTY"),
        ("3,800,000,000 VND","MONEY"),
        ("380,000,000 VND","MONEY"),
        ("15/02/2025","DATE"),
        ("30/06/2025","DATE"),
        ("31/12/2025","DATE"),
        ("a daily delay penalty","PENALTY"),
        ("0.05%","RATE"),
    ]
)

# ── 4 ─────────────────────────────────────────────────────────
add(
    "Pursuant to Decree No. 65/2022/ND-CP and Decree No. 08/2023/ND-CP, "
    "Green Bond Issuer JSC shall issue bonds totaling 100,000,000,000 VND "
    "at a coupon of 10.5% per annum, issuance date 01/04/2025, "
    "maturity date 01/04/2030; failure to redeem on maturity "
    "shall trigger a bond-default penalty "
    "equivalent to 150% of the outstanding face value.",
    [
        ("Decree No. 65/2022/ND-CP","LAW"),
        ("Decree No. 08/2023/ND-CP","LAW"),
        ("Green Bond Issuer JSC","PARTY"),
        ("100,000,000,000 VND","MONEY"),
        ("10.5% per annum","RATE"),
        ("01/04/2025","DATE"),
        ("01/04/2030","DATE"),
        ("a bond-default penalty","PENALTY"),
        ("150%","RATE"),
    ]
)

# ── 5 ─────────────────────────────────────────────────────────
add(
    "Under the Law on Credit Institutions 2024 and Circular No. 39/2016/TT-NHNN, "
    "Mega Finance Bank (Lender) shall extend a syndicated loan of $20,000,000 "
    "to Pacific Infrastructure Co. (Borrower) at SOFR plus 4.5% per annum "
    "from 15/05/2025 to 14/05/2030; upon default "
    "a compounding default penalty shall apply "
    "at 2% of the outstanding principal per month until fully cured.",
    [
        ("Law on Credit Institutions 2024","LAW"),
        ("Circular No. 39/2016/TT-NHNN","LAW"),
        ("Mega Finance Bank","PARTY"),
        ("Lender","PARTY"),
        ("$20,000,000","MONEY"),
        ("Pacific Infrastructure Co.","PARTY"),
        ("Borrower","PARTY"),
        ("4.5% per annum","RATE"),
        ("15/05/2025","DATE"),
        ("14/05/2030","DATE"),
        ("a compounding default penalty","PENALTY"),
        ("2%","RATE"),
    ]
)

# ── 6 ─────────────────────────────────────────────────────────
add(
    "MekongSoft JSC (Service Provider) agrees to deliver Phase I of the ERP system "
    "to Vietnam Retail Group (Client) by 30/09/2025 "
    "for a contract fee of 8,500,000,000 VND; "
    "30% is payable on signing, 40% on UAT acceptance by 31/07/2025, "
    "and 30% on go-live; delay beyond 30/09/2025 "
    "shall incur a service-delay penalty "
    "capped at 10% of the contract fee, "
    "accruing at 0.1% per day, pursuant to the Commercial Law 2005.",
    [
        ("MekongSoft JSC","PARTY"),
        ("Service Provider","PARTY"),
        ("Vietnam Retail Group","PARTY"),
        ("Client","PARTY"),
        ("30/09/2025","DATE",0),
        ("8,500,000,000 VND","MONEY"),
        ("30%","RATE",0),
        ("40%","RATE"),
        ("31/07/2025","DATE"),
        ("30%","RATE",1),
        ("30/09/2025","DATE",1),
        ("a service-delay penalty","PENALTY"),
        ("10%","RATE"),
        ("0.1% per day","RATE"),
        ("Commercial Law 2005","LAW"),
    ]
)

# ── 7 ─────────────────────────────────────────────────────────
add(
    "Saigon Food Chain LLC (Franchisee) shall pay "
    "Global Taste International Ltd. (Franchisor) "
    "an initial franchise fee of $150,000 by 01/06/2025, "
    "a monthly royalty of 6% of gross revenue, "
    "and a marketing levy of 2% of gross revenue; "
    "underpayment for three consecutive months "
    "shall trigger a franchise-breach penalty "
    "equal to twenty-four months of average monthly royalties, "
    "pursuant to the Commercial Law 2005.",
    [
        ("Saigon Food Chain LLC","PARTY"),
        ("Franchisee","PARTY"),
        ("Global Taste International Ltd.","PARTY"),
        ("Franchisor","PARTY"),
        ("$150,000","MONEY"),
        ("01/06/2025","DATE"),
        ("6%","RATE"),
        ("2%","RATE"),
        ("a franchise-breach penalty","PENALTY"),
        ("Commercial Law 2005","LAW"),
    ]
)

# ── 8 ─────────────────────────────────────────────────────────
add(
    "Under the Law on Investment 2020 and the Law on Enterprises 2020, "
    "FPT Industrial Corp. (Party A) and Hanoi Smart City JSC (Party B) "
    "shall each contribute 250,000,000,000 VND to the joint venture by 01/07/2025; "
    "Party A shall receive a preferred return at 12% per annum "
    "before any distributions to Party B; "
    "a capital-call penalty shall apply to any late contributor "
    "at 0.1% of the unpaid amount per day.",
    [
        ("Law on Investment 2020","LAW"),
        ("Law on Enterprises 2020","LAW"),
        ("FPT Industrial Corp.","PARTY"),
        ("Party A","PARTY",0),
        ("Hanoi Smart City JSC","PARTY"),
        ("Party B","PARTY",0),
        ("250,000,000,000 VND","MONEY"),
        ("01/07/2025","DATE"),
        ("Party A","PARTY",1),
        ("12% per annum","RATE"),
        ("Party B","PARTY",1),
        ("a capital-call penalty","PENALTY"),
        ("0.1%","RATE"),
    ]
)

# ── 9 ─────────────────────────────────────────────────────────
add(
    "In accordance with the Law on Insurance Business 2022 and Decree No. 46/2023/ND-CP, "
    "Bao Viet Insurance Co. (Insurer) shall provide coverage "
    "to Nam Long Industrial Park JSC (Insured) "
    "from 01/01/2025 to 31/12/2025 "
    "on a total insured value of 500,000,000,000 VND "
    "at a premium rate of 0.15% per annum; "
    "non-payment within thirty days shall trigger a lapse-and-reinstatement penalty "
    "charged at 5% of the annual premium.",
    [
    ("Law on Insurance Business 2022","LAW"),
    ("Decree No. 46/2023/ND-CP","LAW"),
    ("Bao Viet Insurance Co.","PARTY"),
    ("Nam Long Industrial Park JSC","PARTY"),
    ("01/01/2025","DATE"),
    ("31/12/2025","DATE"),
    ("500,000,000,000 VND","MONEY"),
    ("0.15% per annum","RATE"),
    ("a lapse-and-reinstatement penalty charged at 5% of the annual premium","PENALTY")
    ]
)

# ── 10 ────────────────────────────────────────────────────────
add(
    "Pursuant to the Law on Anti-Money Laundering 2022 "
    "and Decree No. 13/2023/ND-CP on personal data protection, "
    "TechPay Vietnam JSC (Payment Processor) must complete KYC "
    "by 31/03/2025 and deploy transaction monitoring by 30/06/2025 "
    "with a compliance budget of 2,000,000,000 VND; "
    "missing either deadline shall constitute a regulatory-breach penalty "
    "of 500,000,000 VND per missed milestone.",
    [
        ("Law on Anti-Money Laundering 2022","LAW"),
        ("Decree No. 13/2023/ND-CP on personal data protection","LAW"),
        ("TechPay Vietnam JSC","PARTY"),
        ("Payment Processor","PARTY"),
        ("31/03/2025","DATE"),
        ("30/06/2025","DATE"),
        ("2,000,000,000 VND","MONEY"),
        ("a regulatory-breach penalty","PENALTY"),
        ("500,000,000 VND","MONEY"),
    ]
)

# ── 11 ────────────────────────────────────────────────────────
add(
    "Vietnam Export Logistics Co. (Carrier) shall transport goods "
    "for Mekong Agro JSC (Shipper) departing by 15/04/2025 "
    "and arriving by 15/06/2025 "
    "for a total freight of $320,000 "
    "at a demurrage rate of $5,000 per day; "
    "if cargo is not loaded by 15/04/2025 due to the Carrier's fault "
    "a vessel-detention penalty shall accrue "
    "at $10,000 per day from that date, "
    "under the Vietnam Maritime Code 2015.",
    [
        ("Vietnam Export Logistics Co.","PARTY"),
        ("Carrier","PARTY",0),
        ("Mekong Agro JSC","PARTY"),
        ("Shipper","PARTY"),
        ("15/04/2025","DATE",0),
        ("15/06/2025","DATE"),
        ("$320,000","MONEY"),
        ("$5,000 per day","RATE"),
        ("Carrier","PARTY",1),
        ("15/04/2025","DATE",1),
        ("a vessel-detention penalty","PENALTY"),
        ("$10,000 per day","RATE"),
        ("Vietnam Maritime Code 2015","LAW"),
    ]
)

# ── 12 ────────────────────────────────────────────────────────
add(
    "Under the Law on Intellectual Property 2005 and Law No. 07/2022/QH15, "
    "Creative Studio Vietnam (Licensor) grants DataViz Corp. (Licensee) "
    "an exclusive license from 01/07/2025 to 30/06/2030 "
    "for an upfront fee of 1,500,000,000 VND "
    "and an annual maintenance rate of 5% of the original fee; "
    "unauthorized sub-licensing shall trigger an IP-breach penalty "
    "equal to five times the applicable annual license fee.",
    [
        ("Law on Intellectual Property 2005","LAW"),
        ("Law No. 07/2022/QH15","LAW"),
        ("Creative Studio Vietnam","PARTY"),
        ("Licensor","PARTY"),
        ("DataViz Corp.","PARTY"),
        ("Licensee","PARTY"),
        ("01/07/2025","DATE"),
        ("30/06/2030","DATE"),
        ("1,500,000,000 VND","MONEY"),
        ("5%","RATE"),
        ("an IP-breach penalty","PENALTY"),
    ]
)

# ── 13 ────────────────────────────────────────────────────────
add(
    "Under the Construction Law 2020 and QCVN 06:2022/BXD, "
    "Delta Construction Group (EPC Contractor) "
    "shall complete a warehouse for Logistics Hub Vietnam JSC (Owner) "
    "by 30/09/2026 for a lump-sum price of 450,000,000,000 VND; "
    "an advance of 20% is disbursed on 01/12/2025 "
    "and the remaining 80% in milestones; "
    "schedule overrun shall incur a delay-LD penalty "
    "accruing at 0.1% of the contract price per day, "
    "subject to a 10% aggregate cap.",
    [
        ("Construction Law 2020","LAW"),
        ("QCVN 06:2022/BXD","LAW"),
        ("Delta Construction Group","PARTY"),
        ("EPC Contractor","PARTY"),
        ("Logistics Hub Vietnam JSC","PARTY"),
        ("Owner","PARTY"),
        ("30/09/2026","DATE"),
        ("450,000,000,000 VND","MONEY"),
        ("20%","RATE"),
        ("01/12/2025","DATE"),
        ("80%","RATE"),
        ("a delay-LD penalty","PENALTY"),
        ("0.1%","RATE"),
        ("10%","RATE"),
    ]
)

# ── 14 ────────────────────────────────────────────────────────
add(
    "Under Decree No. 65/2022/ND-CP and the Law on Securities 2019, "
    "BIDV Securities JSC (Underwriter) and Phu Quoc Tourism Corp. (Issuer) "
    "agree to issue 1,000,000,000,000 VND in convertible bonds "
    "at par of 100,000 VND per bond, coupon 9% per annum, "
    "issuance date 01/06/2025, conversion window 01/06/2026 to 31/05/2027; "
    "failure to redeem at maturity shall trigger a bondholder-protection penalty "
    "at 110% of par value.",
    [
        ("Decree No. 65/2022/ND-CP","LAW"),
        ("Law on Securities 2019","LAW"),
        ("BIDV Securities JSC","PARTY"),
        ("Underwriter","PARTY"),
        ("Phu Quoc Tourism Corp.","PARTY"),
        ("Issuer","PARTY"),
        ("1,000,000,000,000 VND","MONEY"),
        ("100,000 VND","MONEY"),
        ("9% per annum","RATE"),
        ("01/06/2025","DATE"),
        ("01/06/2026","DATE"),
        ("31/05/2027","DATE"),
        ("a bondholder-protection penalty","PENALTY"),
        ("110%","RATE"),
    ]
)

# ── 15 ────────────────────────────────────────────────────────
add(
    "Under the Law on Environmental Protection 2020 and Decree No. 08/2022/ND-CP, "
    "EcoPlastic Vietnam JSC (Operator) must achieve a recycling rate of 50% "
    "by 31/12/2025 and 70% by 31/12/2027, "
    "with a compliance bond of 10,000,000,000 VND deposited by 01/04/2025; "
    "failure to meet either target shall trigger "
    "a full bond-forfeiture penalty "
    "monitored by GreenWatch NGO (Monitor).",
    [
        ("Law on Environmental Protection 2020","LAW"),
        ("Decree No. 08/2022/ND-CP","LAW"),
        ("EcoPlastic Vietnam JSC","PARTY"),
        ("Operator","PARTY"),
        ("50%","RATE"),
        ("31/12/2025","DATE"),
        ("70%","RATE"),
        ("31/12/2027","DATE"),
        ("10,000,000,000 VND","MONEY"),
        ("01/04/2025","DATE"),
        ("a full bond-forfeiture penalty","PENALTY"),
        ("GreenWatch NGO","PARTY"),
        ("Monitor","PARTY"),
    ]
)

# ── 16 ────────────────────────────────────────────────────────
add(
    "Sun Life Assurance Vietnam (Insurer) shall pay a death benefit of "
    "2,000,000,000 VND to Ms. Le Thi Hoa (Beneficiary) "
    "within thirty days of a valid claim "
    "under the Law on Insurance Business 2022; "
    "delayed payment shall attract a late-settlement penalty "
    "accruing at 12% per annum from day thirty-one "
    "compounding at 0.05% per day until full disbursement.",
    [
        ("Sun Life Assurance Vietnam","PARTY"),
        ("Insurer","PARTY"),
        ("2,000,000,000 VND","MONEY"),
        ("Ms. Le Thi Hoa","PARTY"),
        ("Beneficiary","PARTY"),
        ("Law on Insurance Business 2022","LAW"),
        ("a late-settlement penalty","PENALTY"),
        ("12% per annum","RATE"),
        ("0.05% per day","RATE"),
    ]
)

# ── 17 ────────────────────────────────────────────────────────
add(
    "Under the Law on Tax Administration 2019 and Circular No. 80/2021/TT-BTC, "
    "Alpha Manufacturing Ltd. (Taxpayer) must file CIT at 15% on profits "
    "for the year ending 31/12/2025 by the deadline of 31/03/2026; "
    "Beta Audit JSC (Tax Agent) shall assist; "
    "late filing shall trigger a statutory late-filing penalty "
    "accruing at 0.03% per day "
    "up to a maximum fine of 3,000,000,000 VND "
    "under the Civil Code 2015.",
    [
        ("Law on Tax Administration 2019","LAW"),
        ("Circular No. 80/2021/TT-BTC","LAW"),
        ("Alpha Manufacturing Ltd.","PARTY"),
        ("Taxpayer","PARTY"),
        ("15%","RATE"),
        ("31/12/2025","DATE"),
        ("31/03/2026","DATE"),
        ("Beta Audit JSC","PARTY"),
        ("Tax Agent","PARTY"),
        ("a statutory late-filing penalty","PENALTY"),
        ("0.03% per day","RATE"),
        ("3,000,000,000 VND","MONEY"),
        ("Civil Code 2015","LAW"),
    ]
)

# ── 18 ────────────────────────────────────────────────────────
add(
    "Vietnam Power Grid Corp. (Off-taker) and SolarMax Energy JSC (Generator) "
    "enter a twenty-year Power Purchase Agreement from 01/01/2026 to 31/12/2045 "
    "under the Electricity Law 2022 at a feed-in tariff of $0.0838 per kWh; "
    "the Generator's minimum annual output obligation is 200,000,000 kWh "
    "worth approximately 1,500,000,000,000 VND; "
    "an energy-shortfall penalty shall apply "
    "at 120% of the contracted tariff per kWh of shortfall.",
    [
        ("Vietnam Power Grid Corp.","PARTY"),
        ("Off-taker","PARTY"),
        ("SolarMax Energy JSC","PARTY"),
        ("Generator","PARTY",0),
        ("01/01/2026","DATE"),
        ("31/12/2045","DATE"),
        ("Electricity Law 2022","LAW"),
        ("$0.0838 per kWh","RATE"),
        ("1,500,000,000,000 VND","MONEY"),
        ("Generator","PARTY",1),
        ("an energy-shortfall penalty","PENALTY"),
        ("120%","RATE"),
    ]
)

# ── 19 ────────────────────────────────────────────────────────
add(
    "Under the Law on Competition 2018 and the Civil Code 2015, "
    "PharmaCo Vietnam JSC (Licensor) and MedDistrib Co. (Licensee) "
    "agree that the Licensee shall not distribute competing products "
    "for a term from 01/06/2025 to 31/05/2028 "
    "in exchange for exclusivity payments of 800,000,000 VND per year "
    "at an escalation rate of 5% annually; "
    "breach of exclusivity shall trigger a non-compete violation penalty "
    "of $500,000 per confirmed incident.",
    [
        ("Law on Competition 2018","LAW"),
        ("Civil Code 2015","LAW"),
        ("PharmaCo Vietnam JSC","PARTY"),
        ("Licensor","PARTY"),
        ("MedDistrib Co.","PARTY"),
        ("Licensee","PARTY",0),
        ("Licensee","PARTY",1),
        ("01/06/2025","DATE"),
        ("31/05/2028","DATE"),
        ("800,000,000 VND","MONEY"),
        ("5%","RATE"),
        ("a non-compete violation penalty","PENALTY"),
        ("$500,000","MONEY"),
    ]
)

# ── 20 ────────────────────────────────────────────────────────
add(
    "Pursuant to the Law on Cybersecurity 2018 and Decree No. 13/2023/ND-CP, "
    "CloudSec Vietnam JSC (Data Processor) must achieve ISO 27001 certification "
    "by 31/08/2025 and complete annual penetration testing by 31/08/2026 "
    "with a security investment budget of 5,000,000,000 VND; "
    "failure to certify by the deadline shall incur "
    "a data-security non-compliance penalty "
    "at 0.1% of the annual service contract value per week of delay "
    "for FinanceHub Corp. (Data Controller).",
    [
        ("Law on Cybersecurity 2018","LAW"),
        ("Decree No. 13/2023/ND-CP","LAW"),
        ("CloudSec Vietnam JSC","PARTY"),
        ("Data Processor","PARTY"),
        ("31/08/2025","DATE"),
        ("31/08/2026","DATE"),
        ("5,000,000,000 VND","MONEY"),
        ("a data-security non-compliance penalty","PENALTY"),
        ("0.1%","RATE"),
        ("FinanceHub Corp.","PARTY"),
        ("Data Controller","PARTY"),
    ]
)

# ── 21 ────────────────────────────────────────────────────────
add(
    "Under Decree No. 99/2022/ND-CP on security interests and the Civil Code 2015, "
    "Vingroup JSC (Pledgor) pledges assets of 400,000,000,000 VND "
    "to HSBC Vietnam Ltd. (Secured Creditor) "
    "securing a facility of $15,000,000 at 7.2% per annum "
    "from 01/10/2025 to 30/09/2028; "
    "any unauthorized disposal of pledged assets shall give rise to "
    "a pledge-breach penalty "
    "equal to 200% of the disposed asset value.",
    [
        ("Decree No. 99/2022/ND-CP on security interests","LAW"),
        ("Civil Code 2015","LAW"),
        ("Vingroup JSC","PARTY"),
        ("Pledgor","PARTY"),
        ("400,000,000,000 VND","MONEY"),
        ("HSBC Vietnam Ltd.","PARTY"),
        ("Secured Creditor","PARTY"),
        ("$15,000,000","MONEY"),
        ("7.2% per annum","RATE"),
        ("01/10/2025","DATE"),
        ("30/09/2028","DATE"),
        ("a pledge-breach penalty","PENALTY"),
        ("200%","RATE"),
    ]
)

# ── 22 ────────────────────────────────────────────────────────
add(
    "Pursuant to the Law on Foreign Trade Management 2017 "
    "and Decree No. 69/2018/ND-CP, "
    "Saigon Import Hub JSC (Importer) shall purchase 10,000 MT of steel "
    "from Korea Steel Corp. (Exporter) for $8,500,000 "
    "with a letter of credit due by 15/03/2025 and shipment by 30/04/2025; "
    "the applicable import duty rate is 5% ad valorem; "
    "failure to open the LC by 15/03/2025 "
    "shall trigger an LC-default penalty "
    "of 2% of the contract value per week of delay.",
    [
        ("Law on Foreign Trade Management 2017","LAW"),
        ("Decree No. 69/2018/ND-CP","LAW"),
        ("Saigon Import Hub JSC","PARTY"),
        ("Importer","PARTY"),
        ("Korea Steel Corp.","PARTY"),
        ("Exporter","PARTY"),
        ("$8,500,000","MONEY"),
        ("15/03/2025","DATE",0),
        ("30/04/2025","DATE"),
        ("5%","RATE"),
        ("15/03/2025","DATE",1),
        ("an LC-default penalty","PENALTY"),
        ("2%","RATE"),
    ]
)

# ── 24 ────────────────────────────────────────────────────────
add(
    "Pursuant to the Law on Accounting 2015 and the Law on Tax Administration 2019, "
    "Deloitte Vietnam Ltd. (Auditor) shall deliver audited financial statements "
    "to TechGroup Corp. (Auditee) by 31/03/2026 "
    "for an audit fee of 2,500,000,000 VND; "
    "TechGroup Corp. shall pay 50% on engagement and 50% on delivery; "
    "late delivery by the Auditor beyond 31/03/2026 "
    "shall incur an audit-delay penalty "
    "of 0.1% of the audit fee per day up to a maximum of 15% of the total fee.",
    [
        ("Law on Accounting 2015","LAW"),
        ("Law on Tax Administration 2019","LAW"),
        ("Deloitte Vietnam Ltd.","PARTY"),
        ("Auditor","PARTY",0),
        ("TechGroup Corp.","PARTY",0),
        ("Auditee","PARTY"),
        ("31/03/2026","DATE",0),
        ("2,500,000,000 VND","MONEY"),
        ("TechGroup Corp.","PARTY",1),
        ("50%","RATE",0),
        ("50%","RATE",1),
        ("Auditor","PARTY",1),
        ("31/03/2026","DATE",1),
        ("an audit-delay penalty","PENALTY"),
        ("0.1%","RATE"),
        ("15%","RATE"),
    ]
)

# ── 25 ────────────────────────────────────────────────────────
add(
    "Under the Law on Investment in the Form of Public-Private Partnership 2020 "
    "and Decree No. 35/2021/ND-CP, "
    "the Ministry of Transport (Grantor) awards a thirty-year concession "
    "to Highway Star JSC (Concessionaire) from 01/01/2026 to 31/12/2055 "
    "for total investment of 15,000,000,000,000 VND; "
    "a government viability gap funding of 4,500,000,000,000 VND "
    "shall be disbursed at 5% per year of total capex; "
    "non-achievement of traffic volume targets shall trigger "
    "a performance-shortfall penalty "
    "equal to 1% of annual concession revenue per percentage point of shortfall.",
    [
        ("Law on Investment in the Form of Public-Private Partnership 2020","LAW"),
        ("Decree No. 35/2021/ND-CP","LAW"),
        ("Ministry of Transport","PARTY"),
        ("Grantor","PARTY"),
        ("Highway Star JSC","PARTY"),
        ("Concessionaire","PARTY"),
        ("01/01/2026","DATE"),
        ("31/12/2055","DATE"),
        ("15,000,000,000,000 VND","MONEY"),
        ("4,500,000,000,000 VND","MONEY"),
        ("5% per year","RATE"),
        ("a performance-shortfall penalty","PENALTY"),
        ("1%","RATE"),
    ]
)

# ── 26 ────────────────────────────────────────────────────────
add(
    "Pursuant to the Law on Securities 2019 "
    "and Circular No. 96/2020/TT-BTC on corporate disclosure, "
    "Hanoi Listed Co. JSC (Issuer) must publish quarterly financials "
    "within forty-five days of each quarter-end: "
    "31/03/2025, 30/06/2025, 30/09/2025, and 31/12/2025; "
    "the State Securities Commission (Regulator) may impose "
    "a disclosure-failure penalty "
    "of 0.05% of charter capital per day of late disclosure "
    "on a charter capital of 500,000,000,000 VND.",
    [
        ("Law on Securities 2019","LAW"),
        ("Circular No. 96/2020/TT-BTC on corporate disclosure","LAW"),
        ("Hanoi Listed Co. JSC","PARTY"),
        ("Issuer","PARTY"),
        ("31/03/2025","DATE"),
        ("30/06/2025","DATE"),
        ("30/09/2025","DATE"),
        ("31/12/2025","DATE"),
        ("State Securities Commission","PARTY"),
        ("Regulator","PARTY"),
        ("a disclosure-failure penalty","PENALTY"),
        ("0.05%","RATE"),
        ("500,000,000,000 VND","MONEY"),
    ]
)

# ── 27 ────────────────────────────────────────────────────────
add(
    "Under the Law on Enterprises 2020 and the Civil Code 2015, "
    "Vietnam Dairy JSC (Company) shall conduct a rights issue "
    "at an issuance price of 25,000 VND per share "
    "offering 100,000,000 new shares with total proceeds of 2,500,000,000,000 VND "
    "to existing shareholders by 30/06/2025; "
    "shares not subscribed by 30/06/2025 "
    "shall be offered to a standby underwriter "
    "at a standby discount of 3% below issue price; "
    "failure of the underwriter to purchase unsubscribed shares "
    "shall trigger an underwriting-default penalty "
    "of 5% of the underwritten value.",
    [
        ("Law on Enterprises 2020","LAW"),
        ("Civil Code 2015","LAW"),
        ("Vietnam Dairy JSC","PARTY"),
        ("Company","PARTY"),
        ("25,000 VND","MONEY"),
        ("2,500,000,000,000 VND","MONEY"),
        ("30/06/2025","DATE",0),
        ("30/06/2025","DATE",1),
        ("3%","RATE"),
        ("an underwriting-default penalty","PENALTY"),
        ("5%","RATE"),
    ]
)

# ── 28 ────────────────────────────────────────────────────────
add(
    "Under the Law on Consumer Rights Protection 2023 "
    "and Decree No. 55/2024/ND-CP, "
    "FreshMart Vietnam JSC (Retailer) and "
    "Organic Farm Co. (Supplier) agree that the Supplier "
    "shall deliver 500 MT of certified organic produce per month "
    "valued at 10,000,000,000 VND from 01/01/2025 to 31/12/2025 "
    "with a food safety compliance rate of 99.5%; "
    "any recall event causing harm to consumers "
    "shall attract a product-liability penalty "
    "of 10% of the monthly delivery value per recall incident.",
    [
        ("Law on Consumer Rights Protection 2023","LAW"),
        ("Decree No. 55/2024/ND-CP","LAW"),
        ("FreshMart Vietnam JSC","PARTY"),
        ("Retailer","PARTY"),
        ("Organic Farm Co.","PARTY"),
        ("Supplier","PARTY",0),
        ("10,000,000,000 VND","MONEY"),
        ("01/01/2025","DATE"),
        ("31/12/2025","DATE"),
        ("Supplier","PARTY",1),
        ("99.5%","RATE"),
        ("a product-liability penalty","PENALTY"),
        ("10%","RATE"),
    ]
)


# ── 30 ────────────────────────────────────────────────────────
add(
    "Under the Labor Code 2019, Decree No. 12/2022/ND-CP on administrative penalties, "
    "and the Law on Social Insurance 2014, "
    "TechHire JSC (Employer) must register and contribute social insurance "
    "for all employees at the statutory rate of 21.5% of salary "
    "from the first working day; contributions for Q1-2025 are due by 30/04/2025; "
    "underpayment shall trigger a social-insurance arrears penalty "
    "accruing at 0.03% per day on the shortfall amount "
    "pursuant to authority of the Vietnam Social Security Agency (VSSA).",
    [
        ("Labor Code 2019","LAW"),
        ("Decree No. 12/2022/ND-CP on administrative penalties","LAW"),
        ("Law on Social Insurance 2014","LAW"),
        ("TechHire JSC","PARTY"),
        ("Employer","PARTY"),
        ("21.5%","RATE"),
        ("30/04/2025","DATE"),
        ("a social-insurance arrears penalty","PENALTY"),
        ("0.03% per day","RATE"),
        ("Vietnam Social Security Agency","PARTY"),
        ("VSSA","PARTY"),
    ]
)

# ── 31 ────────────────────────────────────────────────────────
add(
    "In accordance with the Law on Anti-Money Laundering 2022, "
    "the Law on Credit Institutions 2024, "
    "and Circular No. 06/2023/TT-NHNN on credit risk classification, "
    "Techcombank (Lead Arranger) and "
    "Vietnam Infrastructure Fund JSC (Co-arranger) "
    "shall jointly extend a project finance loan of 8,000,000,000,000 VND "
    "to Moc Bai SEZ Co. (Borrower) "
    "at a blended rate of 9.8% per annum from 15/06/2025 to 14/06/2040; "
    "any drawing default shall trigger an event-of-default penalty "
    "accruing at 1.5% per annum above the contract rate until cured.",
    [
        ("Law on Anti-Money Laundering 2022","LAW"),
        ("Law on Credit Institutions 2024","LAW"),
        ("Circular No. 06/2023/TT-NHNN on credit risk classification","LAW"),
        ("Techcombank","PARTY"),
        ("Lead Arranger","PARTY"),
        ("Vietnam Infrastructure Fund JSC","PARTY"),
        ("Co-arranger","PARTY"),
        ("8,000,000,000,000 VND","MONEY"),
        ("Moc Bai SEZ Co.","PARTY"),
        ("Borrower","PARTY"),
        ("9.8% per annum","RATE"),
        ("15/06/2025","DATE"),
        ("14/06/2040","DATE"),
        ("an event-of-default penalty","PENALTY"),
        ("1.5% per annum","RATE"),
    ]
)

# ── 32 ────────────────────────────────────────────────────────
add(
    "Pursuant to the Law on Real Estate Business 2023, the Land Law 2024, "
    "and Circular No. 11/2024/TT-BXD, "
    "Vinhomes JSC (Developer) shall sell 500 apartments to individual buyers "
    "at an average price of 5,000,000,000 VND per unit "
    "with a total contract value of 2,500,000,000,000,000 VND; "
    "buyers must pay a booking deposit of 5% of unit price by 01/03/2025, "
    "first installment of 20% by 30/06/2025, "
    "and final balance of 75% by 31/12/2025; "
    "delayed handover by Vinhomes JSC beyond 31/12/2025 "
    "shall incur a developer-delay penalty "
    "at 0.02% of the unit price per day "
    "until actual delivery, "
    "as monitored by the Ho Chi Minh City Department of Construction (Regulator).",
    [
        ("Law on Real Estate Business 2023","LAW"),
        ("Land Law 2024","LAW"),
        ("Circular No. 11/2024/TT-BXD","LAW"),
        ("Vinhomes JSC","PARTY",0),
        ("Developer","PARTY"),
        ("5,000,000,000 VND","MONEY"),
        ("2,500,000,000,000,000 VND","MONEY"),
        ("5%","RATE"),
        ("01/03/2025","DATE"),
        ("20%","RATE"),
        ("30/06/2025","DATE"),
        ("75%","RATE"),
        ("31/12/2025","DATE",0),
        ("Vinhomes JSC","PARTY",1),
        ("31/12/2025","DATE",1),
        ("a developer-delay penalty","PENALTY"),
        ("0.02%","RATE"),
        ("Ho Chi Minh City Department of Construction","PARTY"),
        ("Regulator","PARTY"),
    ]
)

# ═══════════════════ VALIDATION ═══════════════════
print(f"Total clauses: {len(dataset)}")

errors = []

for i, item in enumerate(dataset):
    t = item["text"]
    ents = item["entities"]

    # 1. check empty / invalid span
    for e in ents:
        s, epos, label = e

        if s < 0 or epos > len(t) or s >= epos:
            errors.append(
                f"Clause {i+1}: INVALID SPAN [{s},{epos},{label}]"
            )
            continue

        if not t[s:epos].strip():
            errors.append(
                f"Clause {i+1}: EMPTY SPAN [{s},{epos},{label}]"
            )

    # 2. sort entities by start position (IMPORTANT)
    ents_sorted = sorted(ents, key=lambda x: x[0])

    # 3. check overlap
    for a in range(len(ents_sorted)):
        s1, e1, l1 = ents_sorted[a]

        for b in range(a + 1, len(ents_sorted)):
            s2, e2, l2 = ents_sorted[b]

            # nếu entity b bắt đầu sau hoặc bằng end của a → OK
            if s2 >= e1:
                break

            # overlap thật sự
            errors.append(
                f"Clause {i+1}: OVERLAP "
                f"[{s1},{e1},{l1}] vs [{s2},{e2},{l2}]"
            )

if errors:
    for err in errors: print("❌", err)
else:
    print("✅ All spans verified — no overlaps, no nesting")

from collections import Counter
cnt = Counter(e[2] for item in dataset for e in item["entities"])
avg = sum(len(item["entities"]) for item in dataset) / len(dataset)
print(f"Entity counts: {dict(cnt)}")
print(f"Avg entities per clause: {avg:.1f}")

with open("dense_32_clean.json","w",encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
print("Written dense_32_clean.json")