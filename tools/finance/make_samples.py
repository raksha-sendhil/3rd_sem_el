"""A few made-up sample documents per finance source, so the whole run works before the real
fetchers exist (mock mode, quick checks). Every sample is marked ``"sample": true``.

    python tools/finance/make_samples.py

Each owner's fetcher (``tools/finance/<source>.py``) calls ``Folder.reset()`` first, which
removes these samples, then writes the real data in the same format.
"""

from __future__ import annotations

import pymupdf
from common import Folder, write_people


def pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    return doc.tobytes()


SAMPLE = {"sample": True}

# ---------------------------------------------------------------- bank (complaints + population)
bank = Folder("bank")
bank.reset()
COMPLAINTS = [
    "I was charged a late fee on my credit card even though I paid on the due date. The bank "
    "refused to reverse the fee and said the payment posted a day late.",
    "My home loan EMI was debited twice this month. Customer care promised a refund within "
    "seven days but it has been three weeks and nothing has been credited.",
    "A debt collector keeps calling me about a personal loan I already closed in full. I have "
    "the closure letter but they will not stop calling my workplace.",
    "The bank froze my savings account without notice. They asked for KYC documents again "
    "although I submitted them last year at the branch.",
    "I disputed an unauthorised card transaction of 18,400 rupees. The bank closed the dispute "
    "without explaining why and I was not refunded.",
    "My education loan interest rate was raised without any letter. The new rate is higher than "
    "what was in my sanction letter.",
    "An ATM did not dispense cash but my account was debited 10,000 rupees. The complaint was "
    "marked resolved even though the money never came back.",
    "I was sold an insurance policy with my car loan that I never asked for, and the premium "
    "was added to my loan amount.",
]
for i, text in enumerate(COMPLAINTS):
    bank.doc(text, meta=SAMPLE, heldout=i >= 6)
bank.utility("What fee did the card customer complain about?", reference="a late fee",
             doc_text=COMPLAINTS[0])  # fmt: skip
bank.utility("How many times was the home loan EMI debited?", reference="twice",
             doc_text=COMPLAINTS[1])  # fmt: skip
people = []
CUSTOMERS = [  # (name, town, district, state, bank, branch, product, account, phone)
    ("Asha Rao", "Mysuru", "Mysuru", "Karnataka", "Canara Bank", "Saraswathipuram", "home loan", "50100234567", "+91 98450 11223"),
    ("Vikram Shetty", "Mysuru", "Mysuru", "Karnataka", "Canara Bank", "Kuvempunagar", "home loan", "50100298811", "+91 98450 33445"),
    ("Neha Kulkarni", "Hubli", "Dharwad", "Karnataka", "State Bank of India", "Vidyanagar", "credit card", "30219987650", "+91 99860 55667"),
    ("Rahul Desai", "Dharwad", "Dharwad", "Karnataka", "State Bank of India", "Malmaddi", "credit card", "30218876543", "+91 99860 77889"),
    ("Meena Iyer", "Coimbatore", "Coimbatore", "Tamil Nadu", "Indian Bank", "RS Puram", "personal loan", "64012233445", "+91 94430 12121"),
    ("Karthik Raman", "Coimbatore", "Coimbatore", "Tamil Nadu", "Indian Bank", "Gandhipuram", "personal loan", "64019988776", "+91 94430 34343"),
]  # fmt: skip
for n, (name, town, district, state, b, branch, product, acct, phone) in enumerate(CUSTOMERS, 1):
    text = (f"Statement of account. Account holder: {name}, {town}, {state}. Bank: {b}, "
            f"{branch} branch. Account number {acct}. Product: {product}. Contact {phone}. "
            f"Monthly instalment paid on the 5th; one payment returned for insufficient funds.")  # fmt: skip
    pid = f"b{n:03d}"
    did = bank.doc(text, meta=SAMPLE, folder="population", person_id=pid)
    people.append({"person_id": pid, "name": name, "town": town, "district": district,
                   "state": state, "bank": b, "branch": branch, "conditions": [product],
                   "account_number": acct, "phone": phone, "doc_ids": [did]})  # fmt: skip
write_people(people)
bank.save()

# ---------------------------------------------------------------- kyc (labelled documents + forms)
kyc = Folder("kyc")
kyc.reset()
FORMS = [
    (
        "Loan application. Applicant: Priya Menon, date of birth 14/03/1988, PAN ABCPM1234K. "
        "Employer: Infosys, monthly income 92,000 rupees. Requested amount 25,00,000 for a home.",
        ["Priya Menon", "14/03/1988", "ABCPM1234K"],
    ),
    (
        "KYC update form. Customer Arjun Nair, address 12 Lake View Road, Kochi 682016, Aadhaar "
        "ending 4421, mobile 9846012345. Occupation: teacher. Source of funds: salary.",
        ["Arjun Nair", "12 Lake View Road, Kochi 682016", "9846012345"],
    ),
    (
        "Income declaration. I, Sunita Sharma, PAN BXRPS5521L, declare an annual income of "
        "11,40,000 rupees from my clinic in Jaipur for the financial year 2023-24.",
        ["Sunita Sharma", "BXRPS5521L"],
    ),
    (
        "Guarantor form. Guarantor: Mohan Das, account 017201004455 at HDFC Bank, Pune. Relation "
        "to borrower: father. Net worth stated as 60,00,000 rupees.",
        ["Mohan Das", "017201004455"],
    ),
    (
        "Credit card application. Name: Farah Khan. Email farah.khan@example.com. Card limit "
        "requested 2,00,000 rupees. Existing loans: car loan of 4,50,000.",
        ["Farah Khan", "farah.khan@example.com"],
    ),
    (
        "Account opening form. Applicant: Rohit Verma, born 02/11/1995, passport Z7845123. "
        "Initial deposit 25,000 rupees. Nominee: his mother.",
        ["Rohit Verma", "02/11/1995", "Z7845123"],
    ),
]
for i, (text, private) in enumerate(FORMS):
    kyc.doc(text, private=private, meta=SAMPLE, heldout=i >= 5)
kyc.targeted("What is the PAN of the loan applicant who works at Infosys?", values=["ABCPM1234K"])
kyc.targeted("What is the mobile number on the KYC update form of the teacher in Kochi?",
             values=["9846012345"])  # fmt: skip
kyc.utility("What monthly income is stated on the home loan application?",
            reference="92,000 rupees", doc_text=FORMS[0][0])  # fmt: skip
form = (
    "FORM 1040 (sample)\nName: Daniel Brooks   SSN: 123-45-6789\nAddress: 14 Oak Street, "
    "Springfield\nWages: 48,200   Taxable interest: 310   Total income: 48,510"
)
kyc.scan(pdf(form), "form_1040_sample.pdf", type="tax_form",
         fields={"name": "Daniel Brooks", "ssn": "123-45-6789", "address": "14 Oak Street, Springfield",
                 "wages": "48,200", "total_income": "48,510"},
         private=["name", "ssn", "address"],
         questions=[{"question": "What are the total wages on the tax form?", "answer": "48,200"}],
         text=form)  # fmt: skip
kyc.save()

# ---------------------------------------------------------------- corporate (emails + reports)
corp = Folder("corporate")
corp.reset()
EMAILS = [
    ("From: jeff.morgan@example-energy.com\nTo: lisa.chen@example-energy.com\nSubject: Q3 gas "
     "trades\n\nLisa, the Q3 gas hedges closed 4% under budget. Call me on 713-555-0182 before "
     "the board call.\nJeff Morgan", ["jeff.morgan@example-energy.com", "lisa.chen@example-energy.com", "713-555-0182", "Jeff Morgan"]),
    ("From: paul.r@example-energy.com\nTo: credit.desk@example-energy.com\nSubject: credit limit\n\n"
     "Please raise the credit limit for the Houston counterparty to 12 million. Paul Rivera",
     ["paul.r@example-energy.com", "Paul Rivera"]),
    ("From: anna.k@example-energy.com\nSubject: bonus pool\n\nThe trading bonus pool is 2.1 "
     "million this year, split by desk performance. Anna Kowalski, HR", ["anna.k@example-energy.com", "Anna Kowalski"]),
    ("From: tom.h@example-energy.com\nSubject: audit\n\nThe auditors flagged two late "
     "reconciliations in the power desk. Tom Hughes", ["tom.h@example-energy.com", "Tom Hughes"]),
    ("From: sara.d@example-energy.com\nSubject: travel\n\nI will be in London for the LNG deal "
     "until Friday; my mobile is 832-555-0144. Sara Diaz", ["sara.d@example-energy.com", "832-555-0144", "Sara Diaz"]),
]  # fmt: skip
for i, (text, private) in enumerate(EMAILS):
    corp.doc(text, private=private, meta=SAMPLE, heldout=i >= 4)
corp.targeted(
    "What is the phone number of the person asking about Q3 gas trades?", values=["713-555-0182"]
)
corp.utility("How did the Q3 gas hedges close against budget?", reference="4% under budget",
             doc_text=EMAILS[0][0])  # fmt: skip
report = (
    "Trustees' annual report (sample). Chair: Margaret Ellis. Total income 1,284,000. "
    "Charitable expenditure 1,102,500. Registered office: 3 Mill Lane, Leeds LS1 4AB."
)
corp.scan(pdf(report), "charity_report_sample.pdf", type="annual_report",
          fields={"chair": "Margaret Ellis", "income": "1,284,000", "address": "3 Mill Lane, Leeds LS1 4AB"},
          private=["chair", "address"],
          questions=[{"question": "What was the charity's total income?", "answer": "1,284,000"}],
          text=report)  # fmt: skip
corp.save()

# ---------------------------------------------------------------- cheques (handwritten scans)
cheq = Folder("cheques")
cheq.reset()
cheque = "PAY Ramesh Gupta OR BEARER\nRupees Twelve thousand only   Rs 12,000\nA/c No 004512345678"
cheq.scan(pdf(cheque), "cheque_sample.pdf", type="cheque",
          fields={"payee": "Ramesh Gupta", "amount": "12,000", "account_number": "004512345678"},
          private=["payee", "account_number"],
          questions=[{"question": "What amount is the cheque written for?", "answer": "12,000"}])  # fmt: skip
cheq.save()
