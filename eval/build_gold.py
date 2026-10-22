"""Builds data/gold/*.json (spec section 8.3 format) from hand annotations below.

Annotation policy (matches README): ages at any value; durations are not DATE;
weekday/anchored dates are DATE; employer names not annotated; named hospitals
are LOCATION; job titles are OCCUPATION. MEDICAL_CONDITION / QUASI_COMBO are left
to the LLM pass and not annotated here, since this set scores the rule pass only.
Gold spans are written as (text, category[, occurrence]) and offsets computed.
"""

import hashlib
import json
from pathlib import Path

DOCS = {
    "c01": (
        "Hello doctor, my name is Priya Sharma and I am 34 years old. I work as a software engineer in Bangalore. For the last 3 months I have had irregular periods and weight gain. My gynecologist Dr. Meena Rao at Manipal Hospital did an ultrasound on 12th March 2023 which showed multiple cysts. My height is 5 feet 2 inches and I weigh 72 kg. Please advise. You can reach me at priya.sharma88@yahoo.co.in.",
        [
            ("Priya Sharma", "NAME"),
            ("34 years old", "AGE"),
            ("software engineer", "OCCUPATION"),
            ("Bangalore", "LOCATION"),
            ("Meena Rao", "NAME"),
            ("Manipal Hospital", "LOCATION"),
            ("12th March 2023", "DATE"),
            ("5 feet 2 inches", "PHYSICAL"),
            ("72 kg", "PHYSICAL"),
            ("priya.sharma88@yahoo.co.in", "EMAIL"),
        ],
    ),
    "c02": (
        "hi, my daughter is 7yrs old and has had a fever of 102 since monday. she also complains of ear pain. we took her to the pediatrician, dr. john mathews, who gave her amoxicillin. she weighs 22 kg. should i be worried? thanks, sarah",
        [
            ("my daughter", "RELATION"),
            ("7yrs old", "AGE"),
            ("monday", "DATE"),
            ("john mathews", "NAME"),
            ("22 kg", "PHYSICAL"),
            ("sarah", "NAME"),
        ],
    ),
    "c03": (
        "Dear Sir, I am a 58 year old male, retired army officer, living in Pune. I have diabetes for 15 years and BP. Recently my creatinine was 2.1 on 05/06/2022. My wife Sunita also has thyroid. My mobile is 9823456710. Please suggest.",
        [
            ("58 year old", "AGE"),
            ("retired army officer", "OCCUPATION"),
            ("Pune", "LOCATION"),
            ("05/06/2022", "DATE"),
            ("My wife", "RELATION"),
            ("Sunita", "NAME"),
            ("9823456710", "PHONE"),
        ],
    ),
    "c04": (
        "Hi, I'm Mike, 41, a long-haul truck driver from Ohio. I've had lower back pain for about six weeks, worse after long drives. I'm 6'1\" and about 260 lbs. My doctor at Cleveland Clinic said it may be a herniated disc. Should I get an MRI?",
        [
            ("Mike", "NAME"),
            ("41", "AGE"),
            ("long-haul truck driver", "OCCUPATION"),
            ("Ohio", "LOCATION"),
            ("6'1\"", "PHYSICAL"),
            ("about 260 lbs", "PHYSICAL"),
            ("Cleveland Clinic", "LOCATION"),
        ],
    ),
    "c05": (
        "My mother, aged 76, had a fall at home last night and now cannot bear weight on her right leg. She has osteoporosis. We live in Leeds and the GP practice is on 14 Kirkstall Road, LS4 2AA. Her NHS number is 943 476 5919. What should we do?",
        [
            ("My mother", "RELATION"),
            ("aged 76", "AGE"),
            ("Leeds", "LOCATION"),
            ("14 Kirkstall Road, LS4 2AA", "ADDRESS"),
            ("943 476 5919", "ID_NUMBER"),
        ],
    ),
    "c06": (
        "Hello, I am a 26 yr old female nurse working night shifts at St. Mary's Hospital. I have been having palpitations and dizziness for two weeks. My ECG on 3/14 was normal. My BMI is 19. My sister had similar issues at age 25. Thank you, Emily Carter",
        [
            ("26 yr old", "AGE"),
            ("nurse", "OCCUPATION"),
            ("St. Mary's Hospital", "LOCATION"),
            ("3/14", "DATE"),
            ("BMI is 19", "PHYSICAL"),
            ("My sister", "RELATION"),
            ("age 25", "AGE"),
            ("Emily Carter", "NAME"),
        ],
    ),
    "c07": (
        "My son has been experiencing left-sided abdominal pain and bilateral pelvic pain for approximately 10 weeks. Tests showed enlarged lymph nodes and mild leukocytosis. He is 32 years old, 6 ft 3 inches tall, and weighs approximately 220 pounds.",
        [
            ("My son", "RELATION"),
            ("32 years old", "AGE"),
            ("6 ft 3 inches", "PHYSICAL"),
            ("approximately 220 pounds", "PHYSICAL"),
        ],
    ),
    "c08": (
        "Hi doc. I'm a 19 year old college student and a competitive swimmer. I have had a sore shoulder for a month. My coach thinks it's tendonitis. I'm 180cm and 78kg. I live in Austin, Texas. My insurance member ID is XKP448921.",
        [
            ("19 year old", "AGE"),
            ("college student", "OCCUPATION"),
            ("competitive swimmer", "OCCUPATION"),
            ("180cm", "PHYSICAL"),
            ("78kg", "PHYSICAL"),
            ("Austin, Texas", "LOCATION"),
            ("XKP448921", "ID_NUMBER"),
        ],
    ),
    "c09": (
        "Respected doctor, my father is 67 and had a heart attack in January 2021. He underwent angioplasty at Apollo Hospital, Chennai. Now he gets breathless climbing stairs. His EF is 40%. He was a school principal before retiring. Our contact: +91 44 2829 3333.",
        [
            ("my father", "RELATION"),
            ("67", "AGE"),
            ("January 2021", "DATE"),
            ("Apollo Hospital", "LOCATION"),
            ("Chennai", "LOCATION"),
            ("school principal", "OCCUPATION"),
            ("+91 44 2829 3333", "PHONE"),
        ],
    ),
    "c10": (
        "I am 45, female, and work as a cashier at Walmart. Since June 3rd I have had a rash on both arms. I had a similar rash in 2019. I'm 5 foot 6 and 190 pounds. My husband Tom thinks it is from the new detergent.",
        [
            ("45", "AGE"),
            ("cashier", "OCCUPATION"),
            ("June 3rd", "DATE"),
            ("2019", "DATE"),
            ("5 foot 6", "PHYSICAL"),
            ("190 pounds", "PHYSICAL"),
            ("My husband", "RELATION"),
            ("Tom", "NAME"),
        ],
    ),
    "c11": (
        "Hello, my 3 month old baby boy has been vomiting after every feed for 4 days. He was born on 2 August 2023 at 36 weeks. His birth weight was 2.4 kg. We are in Dubai. Please help urgently, my number is 050 123 4567.",
        [
            ("3 month old", "AGE"),
            ("baby boy", "RELATION"),
            ("2 August 2023", "DATE"),
            ("2.4 kg", "PHYSICAL"),
            ("Dubai", "LOCATION"),
            ("050 123 4567", "PHONE"),
        ],
    ),
    "c12": (
        "Hi, my name is Ahmed Khan, 29 years old, working as a civil engineer in Karachi. I was diagnosed with hepatitis B in 2018. My father passed away from liver cancer at 55. My email is ahmed.k@gmail.com. CNIC 42101-1234567-1.",
        [
            ("Ahmed Khan", "NAME"),
            ("29 years old", "AGE"),
            ("civil engineer", "OCCUPATION"),
            ("Karachi", "LOCATION"),
            ("2018", "DATE"),
            ("My father", "RELATION"),
            ("55", "AGE"),
            ("ahmed.k@gmail.com", "EMAIL"),
            ("42101-1234567-1", "ID_NUMBER"),
        ],
    ),
    "e01": (
        "From: Vince J Kaminski\nTo: Stinson Gibner\nDate: 10/23/2000 02:15 PM\n\nStinson,\n\nPlease set up a meeting with Jeff Shankman for Thursday. My cell is (713) 410-5396. I will be in London from Nov 6 through Nov 10.\n\nVince",
        [
            ("Vince J Kaminski", "NAME"),
            ("Stinson Gibner", "NAME"),
            ("10/23/2000 02:15 PM", "DATE"),
            ("Stinson", "NAME", 1),
            ("Jeff Shankman", "NAME"),
            ("Thursday", "DATE"),
            ("(713) 410-5396", "PHONE"),
            ("London", "LOCATION"),
            ("Nov 6", "DATE"),
            ("Nov 10", "DATE"),
            ("Vince", "NAME", 1),
        ],
    ),
    "e02": (
        "Sally,\n\nAttached is the revised schedule for deal #456172. Please have Kim Ward review the volumes before Friday, March 9, 2001. The counterparty contact is Bob Hall at 503-464-8100, bob.hall@pgn.com.\n\nThanks,\nKate Symes\nEnron North America\n121 SW Salmon Street, Portland, OR 97204",
        [
            ("Sally", "NAME"),
            ("456172", "ID_NUMBER"),
            ("Kim Ward", "NAME"),
            ("Friday, March 9, 2001", "DATE"),
            ("Bob Hall", "NAME"),
            ("503-464-8100", "PHONE"),
            ("bob.hall@pgn.com", "EMAIL"),
            ("Kate Symes", "NAME"),
            ("121 SW Salmon Street, Portland, OR 97204", "ADDRESS"),
        ],
    ),
    "e03": (
        "Louise,\n\nJohn Lavorato asked me to forward this. The trading desk will move to the 31st floor of the Enron Building, 1400 Smith Street, on May 1. Questions go to Tammie Schoppe at x3-5761 or 713-853-5761.\n\nRegards,\nGreg Whalley\nPresident, Enron Wholesale Services",
        [
            ("Louise", "NAME"),
            ("John Lavorato", "NAME"),
            ("1400 Smith Street", "ADDRESS"),
            ("May 1", "DATE"),
            ("Tammie Schoppe", "NAME"),
            ("x3-5761", "PHONE"),
            ("713-853-5761", "PHONE"),
            ("Greg Whalley", "NAME"),
            ("President", "OCCUPATION"),
        ],
    ),
    "e04": (
        "Hi Mom,\n\nI'll be flying home to Omaha on December 22nd. Can Dad pick me up at the airport around 6pm? My flight is UA 1123. Tell Grandma I said happy 80th birthday!\n\nLove,\nCara\n\nCara Semperger\nEnron, 503.464.3814",
        [
            ("Mom", "RELATION"),
            ("Omaha", "LOCATION"),
            ("December 22nd", "DATE"),
            ("Dad", "RELATION"),
            ("Grandma", "RELATION"),
            ("80th birthday", "AGE"),
            ("Cara", "NAME"),
            ("Cara Semperger", "NAME"),
            ("503.464.3814", "PHONE"),
        ],
    ),
    "e05": (
        "Mark,\n\nPer our conversation, the ISDA with Morgan Stanley is signed. Contract no. 96012345 has been booked. Sara Shackleton (713-853-5620) in Legal has the originals. My new home address is 2207 Bellefontaine St, Houston, TX 77025 in case you need to courier anything.\n\nTana",
        [
            ("Mark", "NAME"),
            ("96012345", "ID_NUMBER"),
            ("Sara Shackleton", "NAME"),
            ("713-853-5620", "PHONE"),
            ("2207 Bellefontaine St, Houston, TX 77025", "ADDRESS"),
            ("Tana", "NAME"),
        ],
    ),
    "e06": (
        "Team,\n\nReminder that the offsite is on Wednesday, Oct 4 at the Woodlands Resort. Please RSVP to Shirley Crenshaw by Monday. Employee ID numbers are required for registration - mine is 90012457. Directions: take I-45 north to exit 76.\n\nThanks,\nVince Kaminski\nManaging Director, Research",
        [
            ("Wednesday, Oct 4", "DATE"),
            ("Woodlands Resort", "LOCATION"),
            ("Shirley Crenshaw", "NAME"),
            ("Monday", "DATE"),
            ("90012457", "ID_NUMBER"),
            ("Vince Kaminski", "NAME"),
            ("Managing Director", "OCCUPATION"),
        ],
    ),
    "e07": (
        "Jeff,\n\nMy wife Linda and I would love to have you over on Saturday. We are at 5 Briar Hollow Lane, Houston 77027. Our home number is 713-621-0384. Our son Jake just turned 16 and is looking for a summer internship, maybe you could talk to him.\n\nBest,\nRick Buy",
        [
            ("Jeff", "NAME"),
            ("My wife", "RELATION"),
            ("Linda", "NAME"),
            ("Saturday", "DATE"),
            ("5 Briar Hollow Lane, Houston 77027", "ADDRESS"),
            ("713-621-0384", "PHONE"),
            ("Our son", "RELATION"),
            ("Jake", "NAME"),
            ("16", "AGE"),
            ("Rick Buy", "NAME"),
        ],
    ),
    "e08": (
        "Please be advised that effective 01/15/2001, Janet Dietrich (SSN on file) will assume responsibility for the West Power desk. Her assistant, Maria Valdes, can be reached at maria.valdes@enron.com or ext. 33487. Janet's direct line is 713.853.9662.\n\nHR Department",
        [
            ("01/15/2001", "DATE"),
            ("Janet Dietrich", "NAME"),
            ("Maria Valdes", "NAME"),
            ("maria.valdes@enron.com", "EMAIL"),
            ("ext. 33487", "PHONE"),
            ("Janet", "NAME", 1),
            ("713.853.9662", "PHONE"),
        ],
    ),
}


# Held-out set: written before the round-2 rule fixes, never used to tune them.
HELDOUT = {
    "h01": (
        "Hello doctor, I am Rahul Verma from Lucknow, a 37 year old bank manager. I have had chest tightness on and off since 2nd January. My father died of a heart attack at the age of 52. I am 5 ft 9 in and weigh 88 kg. I visited Fortis Hospital last week where Dr. Anil Gupta did an ECG. Please call me on 9415012345.",
        [
            ("Rahul Verma", "NAME"),
            ("Lucknow", "LOCATION"),
            ("37 year old", "AGE"),
            ("bank manager", "OCCUPATION"),
            ("2nd January", "DATE"),
            ("My father", "RELATION"),
            ("age of 52", "AGE"),
            ("5 ft 9 in", "PHYSICAL"),
            ("88 kg", "PHYSICAL"),
            ("Fortis Hospital", "LOCATION"),
            ("Anil Gupta", "NAME"),
            ("9415012345", "PHONE"),
        ],
    ),
    "h02": (
        "hi doc, my grandson is 14 months old and still not walking. he was born premature at 32 weeks at Royal Children's Hospital in Melbourne on 9 July 2022. his mum, my daughter Jess, works as a primary school teacher and is worried. he weighs 9.8 kg.",
        [
            ("my grandson", "RELATION"),
            ("14 months old", "AGE"),
            ("Royal Children's Hospital", "LOCATION"),
            ("Melbourne", "LOCATION"),
            ("9 July 2022", "DATE"),
            ("his mum", "RELATION"),
            ("my daughter", "RELATION"),
            ("Jess", "NAME"),
            ("primary school teacher", "OCCUPATION"),
            ("9.8 kg", "PHYSICAL"),
        ],
    ),
    "h03": (
        "Sir I am 62 yrs, retired from railways as a station master. Diabetic since 2005. Last HbA1c on 14/08/2023 was 9.2. My son Vikram lives in Nagpur and handles my treatment. Height 168 cm, weight 80 kg. Patient ID: AIIMS-778213.",
        [
            ("62 yrs", "AGE"),
            ("station master", "OCCUPATION"),
            ("2005", "DATE"),
            ("14/08/2023", "DATE"),
            ("My son", "RELATION"),
            ("Vikram", "NAME"),
            ("Nagpur", "LOCATION"),
            ("168 cm", "PHYSICAL"),
            ("80 kg", "PHYSICAL"),
            ("AIIMS-778213", "ID_NUMBER"),
        ],
    ),
    "h04": (
        "Hello, I'm a 30-year-old pharmacist in Toronto. Three days ago I developed a severe headache with neck stiffness. My partner, Daniel, drove me to Sunnybrook Health Sciences Centre. I'm 5'7\" and 140 lbs. Please reply to kelly.nguyen@outlook.com. Thanks, Kelly",
        [
            ("30-year-old", "AGE"),
            ("pharmacist", "OCCUPATION"),
            ("Toronto", "LOCATION"),
            ("My partner", "RELATION"),
            ("Daniel", "NAME"),
            ("Sunnybrook Health Sciences Centre", "LOCATION"),
            ("5'7\"", "PHYSICAL"),
            ("140 lbs", "PHYSICAL"),
            ("kelly.nguyen@outlook.com", "EMAIL"),
            ("Kelly", "NAME"),
        ],
    ),
    "h05": (
        "Dear doctor, my wife (age 44) has had heavy bleeding for 2 months. She is a police officer and does night duty. Her gynaecologist at Lilavati Hospital, Mumbai, suggested hysterectomy. We are from Thane. Our landline is 022 2541 7788. Her Aadhaar is 4567 8912 3456.",
        [
            ("my wife", "RELATION"),
            ("age 44", "AGE"),
            ("police officer", "OCCUPATION"),
            ("Lilavati Hospital", "LOCATION"),
            ("Mumbai", "LOCATION"),
            ("Thane", "LOCATION"),
            ("022 2541 7788", "PHONE"),
            ("4567 8912 3456", "ID_NUMBER"),
        ],
    ),
    "h06": (
        "Hi, I'm a 52 y/o male, 6 feet tall and 250 pounds, working as a chef in a busy restaurant in Chicago. I had a gout attack on Christmas Eve. My brother Steve has the same problem. My doctor is Dr. Laura Kim at Northwestern Memorial Hospital.",
        [
            ("52 y/o", "AGE"),
            ("6 feet", "PHYSICAL"),
            ("250 pounds", "PHYSICAL"),
            ("chef", "OCCUPATION"),
            ("Chicago", "LOCATION"),
            ("Christmas Eve", "DATE"),
            ("My brother", "RELATION"),
            ("Steve", "NAME"),
            ("Laura Kim", "NAME"),
            ("Northwestern Memorial Hospital", "LOCATION"),
        ],
    ),
    "h07": (
        "From: Sherri Sera\nSent: Tuesday, April 17, 2001 9:04 AM\nTo: Rosalee Fleming\n\nRosalee,\n\nKen would like to meet with Andy Fastow on Thursday at 3:00. Please call me at 713-853-5984 or on my cell 281-830-2347.\n\nSherri Sera\nAssistant to Jeff Skilling",
        [
            ("Sherri Sera", "NAME"),
            ("Tuesday, April 17, 2001 9:04 AM", "DATE"),
            ("Rosalee Fleming", "NAME"),
            ("Rosalee", "NAME", 1),
            ("Ken", "NAME"),
            ("Andy Fastow", "NAME"),
            ("Thursday", "DATE"),
            ("713-853-5984", "PHONE"),
            ("281-830-2347", "PHONE"),
            ("Sherri Sera", "NAME", 1),
            ("Assistant", "OCCUPATION"),
            ("Jeff Skilling", "NAME"),
        ],
    ),
    "h08": (
        "Debra,\n\nThe confirmation for trade ID 1093457 still shows the wrong delivery point. Can you fix it before Monday? Also, my new address for the 401k paperwork is 3810 Locke Lane, Houston, TX 77027.\n\nThanks,\nChris Germany\nEnron Capital & Trade Resources\n(713) 853-4743",
        [
            ("Debra", "NAME"),
            ("1093457", "ID_NUMBER"),
            ("Monday", "DATE"),
            ("3810 Locke Lane, Houston, TX 77027", "ADDRESS"),
            ("Chris Germany", "NAME"),
            ("(713) 853-4743", "PHONE"),
        ],
    ),
    "h09": (
        "Hi Dad,\n\nHappy 60th! Sorry I can't make it to Austin this weekend. Mom said you're having the party at Uncle Joe's place on Lake Travis. I'll call Sunday.\n\nLove,\nSusan\n\nSusan Scott\nEnron Transwestern Pipeline\nsusan.scott@enron.com",
        [
            ("Dad", "RELATION"),
            ("60th", "AGE"),
            ("Austin", "LOCATION"),
            ("Mom", "RELATION"),
            ("Uncle", "RELATION"),
            ("Joe", "NAME"),
            ("Lake Travis", "LOCATION"),
            ("Sunday", "DATE"),
            ("Susan", "NAME"),
            ("Susan Scott", "NAME"),
            ("susan.scott@enron.com", "EMAIL"),
        ],
    ),
    "h10": (
        "All,\n\nPlease welcome Tracy Ngo, who joins the credit group as Senior Analyst effective June 4, 2001. Tracy previously worked at Portland General Electric. She can be reached at extension 3-7154 or tracy.ngo@enron.com. Her office is in ECN 2841.\n\nTim Belden\nVice President, West Power Trading",
        [
            ("Tracy Ngo", "NAME"),
            ("Senior Analyst", "OCCUPATION"),
            ("June 4, 2001", "DATE"),
            ("Tracy", "NAME", 1),
            ("extension 3-7154", "PHONE"),
            ("tracy.ngo@enron.com", "EMAIL"),
            ("ECN 2841", "ADDRESS"),
            ("Tim Belden", "NAME"),
            ("Vice President", "OCCUPATION"),
        ],
    ),
}


def locate(text, needle, occ):
    i = -1
    for _ in range(occ + 1):
        i = text.find(needle, i + 1)
        if i < 0:
            raise ValueError(f"{needle!r} occurrence {occ} not found")
    return i


def main(out_dir: Path, docs: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (text, spans) in docs.items():
        gold = []
        for s in spans:
            needle, cat, occ = (*s, 0)[:3]
            start = locate(text, needle, occ)
            gold.append(
                {"text": needle, "category": cat, "start": start, "end": start + len(needle)}
            )
        doc = {
            "doc_id": hashlib.sha256(text.encode()).hexdigest()[:16],
            "name": name,
            "text": text,
            "spans": gold,
            "annotator": "claude-draft",
            "reviewed_by": None,
        }
        (out_dir / f"{name}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    print(
        f"wrote {len(docs)} docs, {sum(len(s) for _, s in docs.values())} gold spans -> {out_dir}"
    )


if __name__ == "__main__":
    data = Path(__file__).resolve().parent.parent / "data"
    main(data / "gold_draft", DOCS)
    main(data / "gold_heldout", HELDOUT)
