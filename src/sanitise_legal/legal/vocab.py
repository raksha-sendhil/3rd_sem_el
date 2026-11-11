"""The fictional world the legal corpus is generated from.  OWNERS: Person D (generator) and
Person B (ontology tables).

One vocabulary feeds both the data generator (``generator.py``) and the three ontology tables
(``ontology_tables``), so every city, job and legal issue that appears in a document has a
generalisation ladder, and the candidate-pool counts in ``clients.json`` use the same values
the session monitor matches. Everything here is invented; no real person or matter is
described.

The legal issues carry the useful content of a matter: for each one a few fact patterns, and
for each pattern the trigger (what happened), the advice the firm gave, the documents it asked
for, the outcome and the next step. The advice and document phrases are what the QA set asks
about, so they are written without numbers, dates or proper nouns that the rule pass would
remove.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------- places


@dataclass(frozen=True)
class City:
    name: str
    district: str
    state: str
    localities: tuple[str, ...]
    pin: str  # first four digits of the PIN codes used
    kind: str = "city"


CITIES: tuple[City, ...] = (
    City("Pune", "Pune", "Maharashtra", ("Koregaon Park", "Baner", "Kothrud", "Aundh"), "4110"),
    City("Mumbai", "Mumbai Suburban", "Maharashtra", ("Andheri West", "Powai", "Chembur"), "4000"),
    City("Nagpur", "Nagpur", "Maharashtra", ("Dharampeth", "Sitabuldi"), "4400"),
    City("Nashik", "Nashik", "Maharashtra", ("Gangapur Road", "College Road"), "4220"),
    City("Thane", "Thane", "Maharashtra", ("Naupada", "Vartak Nagar"), "4006"),
    City("Kolhapur", "Kolhapur", "Maharashtra", ("Rajarampuri", "Shahupuri"), "4160"),
    City("Bengaluru", "Bengaluru Urban", "Karnataka", ("Indiranagar", "Jayanagar"), "5600"),
    City("Mysuru", "Mysuru", "Karnataka", ("Gokulam", "Kuvempunagar"), "5700"),
    City("Mangaluru", "Dakshina Kannada", "Karnataka", ("Kadri", "Bejai"), "5750"),
    City("Hubballi", "Dharwad", "Karnataka", ("Vidyanagar", "Keshwapur"), "5800"),
    City("Chennai", "Chennai", "Tamil Nadu", ("Adyar", "Anna Nagar"), "6000"),
    City("Coimbatore", "Coimbatore", "Tamil Nadu", ("Peelamedu", "Saibaba Colony"), "6410"),
    City("Madurai", "Madurai", "Tamil Nadu", ("KK Nagar", "Tallakulam"), "6250"),
    City("Kochi", "Ernakulam", "Kerala", ("Kakkanad", "Edappally"), "6820"),
    City("Thiruvananthapuram", "Thiruvananthapuram", "Kerala", ("Pattom", "Kowdiar"), "6950"),
    City("Kozhikode", "Kozhikode", "Kerala", ("Nadakkavu", "Mavoor Road"), "6730"),
    City("Hyderabad", "Hyderabad", "Telangana", ("Banjara Hills", "Madhapur"), "5000"),
    City("Warangal", "Hanamkonda", "Telangana", ("Hanamkonda", "Kazipet"), "5060"),
    City("Ahmedabad", "Ahmedabad", "Gujarat", ("Navrangpura", "Satellite"), "3800"),
    City("Surat", "Surat", "Gujarat", ("Adajan", "Vesu"), "3950"),
    City("Vadodara", "Vadodara", "Gujarat", ("Alkapuri", "Fatehgunj"), "3900"),
    City("New Delhi", "New Delhi", "Delhi", ("Lajpat Nagar", "Dwarka"), "1100"),
    City("Kolkata", "Kolkata", "West Bengal", ("Salt Lake", "Ballygunge"), "7000"),
    City("Lucknow", "Lucknow", "Uttar Pradesh", ("Gomti Nagar", "Aliganj"), "2260"),
    City("Jaipur", "Jaipur", "Rajasthan", ("Malviya Nagar", "Vaishali Nagar"), "3020"),
    City("Panaji", "North Goa", "Goa", ("Altinho", "Miramar"), "4030"),
)
CITY_BY_NAME = {c.name: c for c in CITIES}

REGION = {
    "Maharashtra": "western India",
    "Gujarat": "western India",
    "Goa": "western India",
    "Karnataka": "southern India",
    "Tamil Nadu": "southern India",
    "Kerala": "southern India",
    "Telangana": "southern India",
    "Delhi": "northern India",
    "Uttar Pradesh": "northern India",
    "Rajasthan": "northern India",
    "West Bengal": "eastern India",
}

HIGH_COURT = {
    "Maharashtra": "Bombay High Court",
    "Goa": "Bombay High Court",
    "Karnataka": "Karnataka High Court",
    "Tamil Nadu": "Madras High Court",
    "Kerala": "Kerala High Court",
    "Telangana": "Telangana High Court",
    "Gujarat": "Gujarat High Court",
    "Delhi": "Delhi High Court",
    "West Bengal": "Calcutta High Court",
    "Uttar Pradesh": "Allahabad High Court",
    "Rajasthan": "Rajasthan High Court",
}

# ---------------------------------------------------------------- occupations

# job -> occupation group (the clients.json value; the ontology phrase adds an article)
OCCUPATIONS: dict[str, str] = {
    "cardiologist": "doctor",
    "dentist": "doctor",
    "orthopaedic surgeon": "doctor",
    "paediatrician": "doctor",
    "general physician": "doctor",
    "schoolteacher": "teacher",
    "college lecturer": "teacher",
    "professor": "teacher",
    "software engineer": "engineer",
    "civil engineer": "engineer",
    "mechanical engineer": "engineer",
    "chartered accountant": "finance professional",
    "bank manager": "finance professional",
    "insurance agent": "finance professional",
    "shopkeeper": "small business owner",
    "textile trader": "small business owner",
    "restaurant owner": "small business owner",
    "hardware dealer": "small business owner",
    "government clerk": "government employee",
    "railway employee": "government employee",
    "truck driver": "driver",
    "taxi driver": "driver",
    "farmer": "farmer",
    "architect": "design professional",
    "fashion designer": "design professional",
    "pharmacist": "healthcare worker",
    "nurse": "healthcare worker",
    "journalist": "media professional",
    "electrician": "skilled tradesperson",
    "retired bank officer": "retired person",
}
GROUP_PHRASE = {
    "doctor": "a doctor",
    "teacher": "an educator",
    "engineer": "an engineer",
    "finance professional": "a finance professional",
    "small business owner": "a small business owner",
    "government employee": "a public servant",
    "driver": "a transport worker",
    "farmer": "an agricultural worker",
    "design professional": "a design professional",
    "healthcare worker": "a healthcare worker",
    "media professional": "a media professional",
    "skilled tradesperson": "a skilled tradesperson",
    "retired person": "a retired person",
}

# ---------------------------------------------------------------- names
# None of these may appear in a prompt's few-shot examples (tests/prompts checks).

MALE_FIRST = (
    "Vikram Arjun Suresh Rahul Nikhil Anand Prakash Sanjay Rajesh Manoj Karthik Vivek Harish "
    "Deepak Ashok Mahesh Naveen Siddharth Gaurav Ramesh Ajay Varun Ravi Kiran Abhishek Pranav "
    "Sameer Tushar Yogesh Girish Mohan Venkat Farhan Joseph Thomas Harpreet Debashish Anirudh"
).split()
FEMALE_FIRST = (
    "Ananya Meera Kavya Pooja Shruti Neha Lakshmi Deepa Swati Rekha Priyanka Aarti Nandini "
    "Divya Shalini Asha Sunanda Vandana Radhika Smita Gayatri Rupali Bhavana Sneha Anjali "
    "Madhuri Pallavi Revathi Fatima Mary Simran Ishita Charulata Leena Sujata Tanvi Uma"
).split()
SURNAMES = (
    "Deshpande Kulkarni Patil Joshi Gokhale Bhosale Pawar Shinde Jadhav Rane Iyer Iyengar "
    "Menon Pillai Nambiar Krishnan Subramanian Reddy Rao Naidu Hegde Shetty Gowda Desai Shah "
    "Patel Mehra Malhotra Kapoor Saxena Agarwal Banerjee Chatterjee Mukherjee Sen Bose Das "
    "Khan Qureshi DSouza Fernandes Gill Sandhu Chauhan Rathore Tiwari Mishra Pandey Srivastava "
    "Bhatt Trivedi Dutta Ghosh Varghese Kurian Chandran Rajan Natarajan Swamy Kamath Pai "
    "Bhandari Ahuja Sethi Goyal Bansal Mittal Ranade Apte Karve Sathe Dalvi Gaikwad More"
).split()
# Advocates of the firm (they sign the advice emails) and judges named in case notes.
LAWYERS = (
    "Aditi Ranganathan",
    "Kunal Bhagat",
    "Sharmila Venkatesan",
    "Tejas Oberoi",
    "Mrinalini Saraf",
    "Zubin Contractor",
    "Hema Lakshmanan",
    "Rohit Wadhwa",
)
JUDGES = ("Meenakshi Raghavan", "Arvind Bhargava", "Sudhir Dhingra", "Lalitha Sundaram")

ORG_WORDS = (
    "Sai Lotus Sahyadri Everest Pioneer Sunrise Meridian Indus Konark Vindhya Coastal Harbour "
    "Skyline Trident Orchid Banyan Crescent Zenith Saffron Aravali Deccan Sterling Emerald"
).split()
CODENAMES = (
    "Falcon Heron Kestrel Osprey Peacock Harrier Condor Pelican Ibis Kingfisher Flamingo Eagle"
).split()

# ---------------------------------------------------------------- legal issues


@dataclass(frozen=True)
class Pattern:
    key: str
    trigger: str  # a clause: "the cheque was returned unpaid for insufficient funds"
    advice: str  # completes "Our advice is to ..."
    documents: str  # completes "Please collect ..."
    outcome: str
    next_step: str
    question: str  # the QA question this pattern answers
    expected: tuple[str, ...]  # key phrases from the advice or documents


@dataclass(frozen=True)
class Issue:
    name: str  # the LEGAL_ISSUE span text and clients.json value
    category: str  # ontology: one rung up
    team: str  # practice team: commercial, property, family, criminal
    court_kind: str  # district, magistrate, consumer, family, sessions, labour, commercial, high
    case_styles: tuple[str, ...]  # "{n}" and "{y}" are filled in
    opposing: str  # "person", "company" or "either"
    org_suffixes: tuple[str, ...]  # for company opponents
    amount_range: tuple[int, int]  # rupees
    narrative: str  # third person; {opp}, {amount}, {trigger}
    narrative_first: str  # the client's own words; {opp}, {amount}, {trigger}
    patterns: tuple[Pattern, ...]
    rare: bool = False
    codename: bool = False
    weight: int = 0  # number of clients in clients.json
    aliases: tuple[str, ...] = field(default_factory=tuple)


ISSUES: tuple[Issue, ...] = (
    Issue(
        name="cheque bounce",
        category="a cheque-bounce dispute",
        team="commercial",
        court_kind="magistrate",
        case_styles=("CC/{n}/{y}", "S.C.C. No. {n}/{y}"),
        opposing="either",
        org_suffixes=("Traders", "Enterprises", "Textiles Pvt Ltd"),
        amount_range=(150_000, 4_500_000),
        narrative="A cheque for {amount} issued by {opp} bounced: {trigger}.",
        narrative_first="{opp} gave me a cheque for {amount} to repay a loan, but {trigger}.",
        weight=40,
        aliases=("cheque-bounce", "cheque dishonour", "Section 138"),
        patterns=(
            Pattern(
                "insufficient_funds",
                "the cheque was returned unpaid for insufficient funds",
                "send a statutory demand notice to the drawer and file the complaint if the "
                "drawer does not pay",
                "the original dishonoured cheque, the bank return memo and proof that the "
                "demand notice was delivered",
                "the drawer paid the full cheque amount after the first hearing",
                "withdraw the complaint once the payment clears",
                "What did we advise in past cheque-bounce matters where the cheque was returned "
                "for insufficient funds?",
                ("statutory demand notice",),
            ),
            Pattern(
                "account_closed",
                "the cheque came back with the remark account closed",
                "treat the closure of the account as evidence of dishonest intent and oppose "
                "any compounding below the full cheque amount",
                "the cheque, the return memo showing the account was closed and the loan agreement",
                "the court issued summons and the drawer offered a settlement",
                "insist on payment of the full amount with costs",
                "How did we handle cheque-bounce matters where the drawer's account was closed?",
                ("evidence of dishonest intent",),
            ),
            Pattern(
                "signature_mismatch",
                "the cheque was dishonoured because the signature did not match",
                "obtain the bank's certificate on the signature mismatch and treat it as a "
                "dishonour under the cheque-bounce law",
                "the bank's certificate on the signature mismatch and the drawer's earlier cheques",
                "the magistrate accepted that a signature mismatch is a dishonour",
                "lead evidence through the bank official",
                "What did we advise when a bounced cheque was dishonoured for a signature "
                "mismatch?",
                ("certificate on the signature mismatch",),
            ),
            Pattern(
                "stop_payment",
                "the drawer stopped the cheque with a stop-payment instruction",
                "argue that a stop-payment instruction does not protect the drawer when the "
                "debt is admitted",
                "the written acknowledgement of the debt and the stop-payment memo",
                "the drawer's application to discharge the complaint was rejected",
                "press for an early date for cross-examination",
                "What was our advice in cheque-bounce cases where the drawer issued a "
                "stop-payment instruction?",
                ("stop-payment instruction does not protect the drawer",),
            ),
        ),
    ),
    Issue(
        name="tenancy eviction",
        category="a tenancy dispute",
        team="property",
        court_kind="district",
        case_styles=("R.C.S. No. {n}/{y}", "R.C.A. No. {n}/{y}"),
        opposing="person",
        org_suffixes=(),
        amount_range=(60_000, 900_000),
        narrative="The tenancy dispute with {opp} concerns rent arrears of {amount}; {trigger}.",
        narrative_first="I have a tenancy problem with {opp} involving {amount} in rent, "
        "and {trigger}.",
        weight=14,
        aliases=("eviction", "tenant", "tenancy"),
        patterns=(
            Pattern(
                "rent_arrears",
                "the tenant has stopped paying rent for many months",
                "issue a notice terminating the tenancy and claim mesne profits for the period "
                "of occupation",
                "the rent agreement, the rent receipts and the notice terminating the tenancy",
                "the court passed an eviction decree with arrears",
                "file for execution of the decree",
                "What did we advise landlords in tenancy eviction disputes over unpaid rent?",
                ("mesne profits",),
            ),
            Pattern(
                "bona_fide_need",
                "the landlord needs the premises for the family's own use",
                "plead bona fide personal need with proof of the family's housing requirement",
                "proof of the family's housing requirement and the property tax records",
                "the eviction was granted on the ground of personal need",
                "give the tenant time to vacate under the decree",
                "How have tenancy eviction disputes based on the landlord's own need gone before "
                "district courts?",
                ("bona fide personal need",),
            ),
            Pattern(
                "subletting",
                "the tenant has sublet the flat without permission",
                "collect evidence of unauthorised subletting such as utility bills in the "
                "occupant's name",
                "utility bills in the occupant's name and statements of the neighbours",
                "the tenant admitted the subletting and agreed to vacate",
                "record the compromise before the court",
                "What evidence did we ask for in tenancy eviction matters involving subletting?",
                ("unauthorised subletting",),
            ),
            Pattern(
                "tenant_defence",
                "the landlord is trying to evict the tenant after a rent increase was refused",
                "deposit the admitted rent in court to avoid an order striking off the defence",
                "all rent receipts and the bank record of the rent deposited in court",
                "the tenant's defence was kept on record and the suit continues",
                "keep depositing the rent every month",
                "What did we advise tenants who were facing eviction after refusing a rent "
                "increase?",
                ("deposit the admitted rent in court",),
            ),
        ),
    ),
    Issue(
        name="consumer complaint",
        category="a consumer dispute",
        team="property",
        court_kind="consumer",
        case_styles=("CC/{n}/{y}", "C.C. No. {n}/{y}"),
        opposing="company",
        org_suffixes=("Builders Pvt Ltd", "Developers", "Realty", "Constructions"),
        amount_range=(1_800_000, 95_000_000),
        narrative="The consumer complaint against {opp} concerns a flat bought for {amount}; "
        "{trigger}.",
        narrative_first="I paid {opp} {amount} for a flat, and {trigger}.",
        weight=14,
        aliases=("consumer", "builder", "builders"),
        patterns=(
            Pattern(
                "delayed_possession",
                "possession of the flat is long overdue",
                "claim interest on the amount paid for every month of delay in possession",
                "the allotment letter, the builder-buyer agreement and every payment receipt",
                "the commission directed the builder to pay interest for the delay",
                "file for execution if the builder does not pay",
                "What did we advise in consumer complaints against builders for delayed "
                "possession?",
                ("every month of delay in possession",),
            ),
            Pattern(
                "defects",
                "the flat was handed over with serious construction defects",
                "commission a structural audit report on the defects before the first hearing",
                "photographs of the defects and a structural audit report",
                "the builder was ordered to repair the defects or pay their cost",
                "inspect the repairs before accepting them",
                "What documents did we ask clients to collect in consumer complaints against "
                "builders over construction defects?",
                ("structural audit report",),
            ),
            Pattern(
                "refund",
                "the builder never obtained the occupancy certificate",
                "seek a full refund with interest because the builder failed to obtain the "
                "occupancy certificate",
                "the builder-buyer agreement, payment receipts and the municipal reply on the "
                "occupancy certificate",
                "a full refund with interest was ordered",
                "track the refund instalments",
                "What did we advise when a builder had not obtained the occupancy certificate?",
                ("full refund with interest",),
            ),
            Pattern(
                "documents",
                "the builder keeps promising possession by email but never delivers",
                "keep every written promise of possession because each one extends the period "
                "of limitation",
                "the allotment letter, the builder-buyer agreement, payment receipts and all "
                "emails promising possession",
                "the complaint was admitted and notice was issued to the builder",
                "file the evidence affidavit",
                "What documents did we ask clients to collect in consumer complaints against "
                "builders who kept promising possession?",
                ("all emails promising possession",),
            ),
        ),
    ),
    Issue(
        name="land dispute",
        category="a property dispute",
        team="property",
        court_kind="district",
        case_styles=("O.S. No. {n}/{y}", "R.C.S. No. {n}/{y}"),
        opposing="person",
        org_suffixes=(),
        amount_range=(800_000, 60_000_000),
        narrative="The land dispute with {opp} concerns agricultural land valued at {amount}; "
        "{trigger}.",
        narrative_first="I have a land dispute with {opp} over land worth {amount}, and {trigger}.",
        weight=12,
        aliases=("land", "plot", "property"),
        patterns=(
            Pattern(
                "encroachment",
                "the neighbour has encroached on the boundary of the plot",
                "apply for a court commissioner to measure the boundary",
                "the sale deed, the survey map and the land records",
                "the court commissioner confirmed the encroachment",
                "seek removal of the encroachment",
                "What did we advise in land disputes about boundary encroachment?",
                ("court commissioner",),
            ),
            Pattern(
                "title",
                "relatives dispute who owns the ancestral land",
                "trace the chain of title through the mutation entries",
                "the mutation entries, the partition deed and the old sale deeds",
                "the title of the client was upheld",
                "apply for correction of the land records",
                "How did we handle land disputes with relatives over title to ancestral land?",
                ("mutation entries",),
            ),
            Pattern(
                "adverse_possession",
                "the occupant claims ownership by adverse possession",
                "rebut the adverse possession claim with tax receipts showing continuous ownership",
                "tax receipts showing continuous ownership and electricity bills",
                "the adverse possession claim was dismissed",
                "seek possession of the land",
                "What did we advise when an occupant claimed land by adverse possession?",
                ("tax receipts showing continuous ownership",),
            ),
            Pattern(
                "injunction",
                "construction has started on the disputed plot",
                "seek a temporary injunction to stop construction on the plot",
                "photographs of the construction and the land records",
                "a temporary injunction was granted",
                "file a contempt petition if construction continues",
                "What relief did we seek in land disputes where construction started on the "
                "disputed plot?",
                ("temporary injunction",),
            ),
        ),
    ),
    Issue(
        name="divorce by mutual consent",
        category="a matrimonial matter",
        team="family",
        court_kind="family",
        case_styles=("Petition No. {n}/{y}",),
        opposing="person",
        org_suffixes=(),
        amount_range=(500_000, 9_000_000),
        narrative="The divorce by mutual consent with {opp} includes a settlement of {amount}; "
        "{trigger}.",
        narrative_first="My spouse {opp} and I want a divorce by mutual consent with a "
        "settlement of {amount}, and {trigger}.",
        weight=8,
        aliases=("divorce", "mutual consent"),
        patterns=(
            Pattern(
                "cooling_off",
                "both spouses have lived apart for a long time",
                "apply to waive the statutory cooling-off period",
                "proof of separate residence and the consent terms",
                "the cooling-off period was waived",
                "attend the second motion hearing",
                "What did we advise in divorces by mutual consent where the spouses had lived "
                "apart for a long time?",
                ("waive the statutory cooling-off period",),
            ),
            Pattern(
                "alimony",
                "the spouses agree on a single payment instead of maintenance",
                "record the one-time alimony settlement in the consent terms",
                "income tax returns and the draft consent terms",
                "the settlement was recorded and the divorce granted",
                "transfer the settlement amount before the decree",
                "How did we structure alimony in divorces by mutual consent?",
                ("one-time alimony settlement",),
            ),
            Pattern(
                "custody",
                "the couple has a young child",
                "agree a joint parenting plan before the second motion",
                "the school records of the child and the draft parenting plan",
                "the joint parenting plan was made part of the decree",
                "review the parenting plan after a year",
                "What did we advise about the child in divorces by mutual consent?",
                ("joint parenting plan",),
            ),
            Pattern(
                "stridhan",
                "the wife wants her jewellery and gifts returned",
                "list the stridhan items to be returned in the settlement deed",
                "the list of jewellery and gifts with any receipts",
                "the stridhan items were returned before the decree",
                "sign the receipt for the returned items",
                "What did we advise about returning jewellery and gifts in a mutual consent "
                "divorce?",
                ("stridhan items",),
            ),
        ),
    ),
    Issue(
        name="anticipatory bail",
        category="a criminal matter",
        team="criminal",
        court_kind="sessions",
        case_styles=("Anticipatory Bail Application No. {n}/{y}", "FIR No. {n}/{y}"),
        opposing="person",
        org_suffixes=(),
        amount_range=(200_000, 7_500_000),
        narrative="The anticipatory bail matter arises from a complaint by {opp} about "
        "{amount}; {trigger}.",
        narrative_first="{opp} has filed a police complaint against me over {amount}, and "
        "{trigger}.",
        weight=8,
        aliases=("bail", "FIR", "arrest"),
        patterns=(
            Pattern(
                "business_fir",
                "the complaint turns a business disagreement into a criminal case",
                "argue that the dispute is civil in nature and offer to cooperate with the "
                "investigation",
                "the partnership papers and the correspondence with the complainant",
                "anticipatory bail was granted with conditions",
                "attend the police station when called",
                "What did we argue in anticipatory bail matters arising from business disputes?",
                ("civil in nature",),
            ),
            Pattern(
                "matrimonial_fir",
                "the complaint was filed during a matrimonial dispute",
                "seek interim protection and offer to attend mediation",
                "the marriage certificate and messages showing attempts to settle",
                "interim protection was granted and the parties were sent to mediation",
                "attend every mediation session",
                "What did we advise in anticipatory bail matters arising from matrimonial "
                "complaints?",
                ("interim protection",),
            ),
            Pattern(
                "cheating",
                "the complainant alleges cheating over a failed deal",
                "produce the contract and payment records to show there was no intent to cheat "
                "at the outset",
                "the signed contract and the payment records",
                "bail was granted because the dispute was about performance of a contract",
                "seek quashing of the complaint",
                "How did we defend clients accused of cheating in anticipatory bail applications?",
                ("no intent to cheat at the outset",),
            ),
            Pattern(
                "negligence",
                "the complaint alleges negligence during treatment",
                "rely on the expert committee opinion that there was no gross negligence",
                "the treatment records and the expert committee opinion",
                "anticipatory bail was granted on the strength of the expert opinion",
                "apply for quashing of the case",
                "What did we rely on in anticipatory bail matters alleging negligence during "
                "treatment?",
                ("no gross negligence",),
            ),
        ),
    ),
    Issue(
        name="breach of contract",
        category="a commercial contract dispute",
        team="commercial",
        court_kind="commercial",
        case_styles=("CS(COMM) {n}/{y}", "Com. S. No. {n}/{y}"),
        opposing="company",
        org_suffixes=("Logistics Ltd", "Infotech Pvt Ltd", "Industries", "Motors Ltd"),
        amount_range=(2_500_000, 180_000_000),
        narrative="The breach of contract claim against {opp} is for {amount}; {trigger}.",
        narrative_first="Our company has a breach of contract claim of {amount} against {opp}, "
        "and {trigger}.",
        codename=True,
        weight=8,
        aliases=("contract", "commercial"),
        patterns=(
            Pattern(
                "unpaid_invoices",
                "the counterparty has not paid the invoices for delivered goods",
                "invoke the arbitration clause and seek interim relief to secure the amount",
                "the contract, the invoices and the delivery challans",
                "the tribunal ordered a deposit of the disputed amount",
                "proceed with the arbitration",
                "What did we advise in breach of contract matters over unpaid invoices?",
                ("invoke the arbitration clause",),
            ),
            Pattern(
                "defective_supply",
                "the goods supplied were defective",
                "claim liquidated damages as fixed in the contract",
                "the inspection reports and the rejection notices",
                "the claim for liquidated damages was allowed",
                "recover the damages from the security deposit",
                "What did we claim in breach of contract cases involving defective supplies?",
                ("liquidated damages",),
            ),
            Pattern(
                "termination",
                "the counterparty terminated the contract without notice",
                "send a notice of breach giving the cure period required by the contract",
                "the contract, the termination letter and the board minutes",
                "the termination was held to be wrongful",
                "claim loss of profit for the remaining term",
                "What did we advise when a counterparty terminated a contract without notice?",
                ("notice of breach",),
            ),
            Pattern(
                "mediation",
                "the claim must be filed as a commercial suit",
                "complete pre-institution mediation before filing the commercial suit",
                "the mediation request and the statement of accounts",
                "mediation failed and the suit was filed",
                "seek summary judgment",
                "What step did we take before filing commercial suits for breach of contract?",
                ("pre-institution mediation",),
            ),
        ),
    ),
    Issue(
        name="wrongful termination",
        category="an employment dispute",
        team="commercial",
        court_kind="labour",
        case_styles=("Complaint No. {n}/{y}",),
        opposing="company",
        org_suffixes=("Industries", "Technologies Pvt Ltd", "Pharma Ltd"),
        amount_range=(300_000, 6_000_000),
        narrative="The wrongful termination claim against {opp} is for {amount}; {trigger}.",
        narrative_first="My employer {opp} owes me {amount} after firing me, and {trigger}.",
        weight=6,
        aliases=("termination", "dismissal", "employment"),
        patterns=(
            Pattern(
                "reinstatement",
                "the employee was dismissed without any enquiry",
                "claim reinstatement with full back wages",
                "the appointment letter, the termination letter and salary slips",
                "reinstatement was ordered with back wages",
                "report for duty within the time fixed",
                "What did we advise employees dismissed without an enquiry?",
                ("reinstatement with full back wages",),
            ),
            Pattern(
                "enquiry",
                "the dismissal followed a one-sided domestic enquiry",
                "demand the domestic enquiry report and challenge it for breach of natural justice",
                "the charge sheet, the enquiry report and the reply to the charge sheet",
                "the enquiry was set aside for breach of natural justice",
                "attend the fresh enquiry",
                "How did we challenge domestic enquiries in wrongful termination cases?",
                ("breach of natural justice",),
            ),
            Pattern(
                "settlement",
                "the employer offered to settle",
                "negotiate a settlement of gratuity and notice pay",
                "the salary slips and the gratuity calculation",
                "the employer paid gratuity and notice pay under a settlement",
                "sign the settlement and withdraw the complaint",
                "What did we negotiate when employers offered to settle wrongful termination "
                "claims?",
                ("gratuity and notice pay",),
            ),
            Pattern(
                "wages",
                "the employer withheld the last salary",
                "file a claim for unpaid wages before the labour authority",
                "the attendance record and the salary slips",
                "the unpaid wages were recovered",
                "close the file",
                "What did we advise when an employer withheld salary after a dismissal?",
                ("claim for unpaid wages",),
            ),
        ),
    ),
    Issue(
        name="maritime salvage claim",
        category="a specialised admiralty dispute",
        team="commercial",
        court_kind="high",
        case_styles=("A.S. No. {n}/{y}",),
        opposing="company",
        org_suffixes=("Shipping Pvt Ltd", "Marine Ltd"),
        amount_range=(40_000_000, 600_000_000),
        narrative="The maritime salvage claim against {opp} is for {amount}; {trigger}.",
        narrative_first="Our salvage crew rescued a vessel owned by {opp} and we claim {amount}, "
        "and {trigger}.",
        rare=True,
        codename=True,
        weight=3,
        aliases=("salvage", "vessel", "admiralty"),
        patterns=(
            Pattern(
                "arrest",
                "the vessel is about to leave port",
                "arrest the vessel to secure the salvage claim",
                "the salvage agreement and the logbook entries",
                "the vessel was arrested until security was furnished",
                "negotiate the salvage award",
                "What did we advise in salvage claims where the vessel was about to sail?",
                ("arrest the vessel",),
            ),
            Pattern(
                "award",
                "the owners dispute the value of the salvage",
                "claim a salvage award based on the value of the property saved",
                "the valuation of the vessel and cargo",
                "a salvage award was fixed by the court",
                "recover the award from the security",
                "How did we value salvage awards in admiralty matters?",
                ("value of the property saved",),
            ),
        ),
    ),
    Issue(
        name="geographical-indication dispute",
        category="a specialised intellectual-property dispute",
        team="commercial",
        court_kind="high",
        case_styles=("CS(COMM) {n}/{y}",),
        opposing="company",
        org_suffixes=("Foods Pvt Ltd", "Exports"),
        amount_range=(5_000_000, 80_000_000),
        narrative="The geographical-indication dispute with {opp} involves sales of {amount}; "
        "{trigger}.",
        narrative_first="Our producers' association has a geographical-indication dispute with "
        "{opp} over sales of {amount}, and {trigger}.",
        rare=True,
        weight=3,
        aliases=("geographical indication", "GI"),
        patterns=(
            Pattern(
                "label",
                "the rival product uses the protected regional name on its label",
                "rely on the registered GI certificate and seek an injunction against the "
                "infringing label",
                "the registered GI certificate and samples of the infringing label",
                "an injunction against the infringing label was granted",
                "seek damages",
                "What did we advise producers whose geographical indication was used on a rival "
                "label?",
                ("registered GI certificate",),
            ),
            Pattern(
                "trademark",
                "a conflicting trademark was registered",
                "file a rectification petition against the conflicting trademark",
                "the trademark registration and the GI registration",
                "the conflicting trademark was removed from the register",
                "monitor new trademark applications",
                "What did we do about trademarks that conflicted with a geographical indication?",
                ("rectification petition",),
            ),
        ),
    ),
    Issue(
        name="aircraft lease repossession",
        category="a specialised aviation finance dispute",
        team="commercial",
        court_kind="high",
        case_styles=("CS(COMM) {n}/{y}",),
        opposing="company",
        org_suffixes=("Aviation Ltd", "Airways Ltd"),
        amount_range=(300_000_000, 2_000_000_000),
        narrative="The aircraft lease repossession against {opp} concerns arrears of {amount}; "
        "{trigger}.",
        narrative_first="Our leasing company wants to repossess aircraft leased to {opp} over "
        "arrears of {amount}, and {trigger}.",
        rare=True,
        codename=True,
        weight=2,
        aliases=("aircraft", "lease", "aviation"),
        patterns=(
            Pattern(
                "deregistration",
                "the airline has stopped paying lease rentals",
                "seek deregistration of the aircraft under the international lease convention",
                "the lease agreement and the irrevocable deregistration authorisation",
                "the aviation regulator deregistered the aircraft",
                "arrange the export of the aircraft",
                "What did we advise lessors whose airline lessee stopped paying rentals?",
                ("deregistration of the aircraft",),
            ),
            Pattern(
                "insolvency_moratorium",
                "the airline sought protection from its creditors",
                "argue that leased aircraft fall outside the insolvency moratorium",
                "the lease agreement and the moratorium order",
                "the aircraft were released from the moratorium",
                "repossess the aircraft",
                "How did we respond when an airline lessee entered insolvency?",
                ("outside the insolvency moratorium",),
            ),
        ),
    ),
    Issue(
        name="cross-border insolvency",
        category="a specialised insolvency matter",
        team="commercial",
        court_kind="tribunal",
        case_styles=("C.P.(IB) No. {n}/{y}",),
        opposing="company",
        org_suffixes=("Holdings Ltd", "Industries"),
        amount_range=(100_000_000, 900_000_000),
        narrative="The cross-border insolvency matter involving {opp} concerns claims of "
        "{amount}; {trigger}.",
        narrative_first="Our company is owed {amount} by {opp}, which is insolvent here and "
        "abroad, and {trigger}.",
        rare=True,
        weight=2,
        aliases=("insolvency", "cross-border"),
        patterns=(
            Pattern(
                "claim",
                "the debtor is in insolvency proceedings in two countries",
                "file the claim with the resolution professional before the deadline",
                "the loan documents and the statement of the amount due",
                "the claim was admitted in full",
                "vote on the resolution plan",
                "What did we advise creditors in cross-border insolvency matters?",
                ("resolution professional",),
            ),
            Pattern(
                "recognition",
                "the foreign administrator wants control of the assets in this country",
                "seek recognition of the foreign insolvency proceeding",
                "the foreign court's order and the list of local assets",
                "the foreign proceeding was recognised",
                "coordinate with the foreign administrator",
                "How did we deal with foreign administrators in cross-border insolvency?",
                ("recognition of the foreign insolvency proceeding",),
            ),
        ),
    ),
)
ISSUE_BY_NAME = {i.name: i for i in ISSUES}

TEAM_LABEL = {
    "commercial": "Commercial",
    "property": "Property",
    "family": "Family",
    "criminal": "Criminal",
}


def court_name(kind: str, city: City) -> str:
    """The court's name as written in the documents (every form is matched by the rules)."""
    high = HIGH_COURT[city.state]
    return {
        "district": f"{city.name} District Court",
        "magistrate": f"Judicial Magistrate First Class Court at {city.name}",
        "consumer": f"{city.name} District Consumer Disputes Redressal Commission",
        "family": f"Family Court at {city.name}",
        "sessions": f"{city.name} Sessions Court",
        "labour": f"{city.name} Labour Court",
        "commercial": f"{city.name} Commercial Court",
        "high": high,
        "tribunal": "NCLT Mumbai",
    }[kind]


# ---------------------------------------------------------------- ontology tables (Person B)


def ontology_tables() -> dict[str, dict]:
    """``data/legal/ontology/{places,occupations,legal_issues}.json`` in the docs/phase3_plan.md 6.3
    format, plus a ``region`` on every place (city -> district -> state -> region)."""
    places: dict[str, dict] = {}
    for c in CITIES:
        places[c.name.lower()] = {
            "district": c.district,
            "state": c.state,
            "kind": c.kind,
            "region": REGION[c.state],
        }
        for locality in c.localities:
            places.setdefault(
                locality.lower(),
                {
                    "district": c.district,
                    "state": c.state,
                    "kind": "neighbourhood",
                    "region": REGION[c.state],
                },
            )
    occupations = {job: GROUP_PHRASE[group] for job, group in OCCUPATIONS.items()}
    legal_issues = {i.name: {"category": i.category, "rare": i.rare} for i in ISSUES}
    return {"places": places, "occupations": occupations, "legal_issues": legal_issues}
