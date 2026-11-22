# Private Matter Assistant: user guide

For lawyers at the firm. The assistant answers questions about the firm's past matters
without telling you who the clients were.

## What you can ask

Ask about the law and the firm's experience, the way you would ask a senior colleague:

- "What did we advise in past cheque-bounce matters where the cheque was returned for
  insufficient funds?"
- "How have tenancy eviction disputes before district courts in Maharashtra gone for us?"
- "What documents did we ask clients to collect in consumer complaints against builders?"

Answers draw on four kinds of past documents: intake notes, advice emails, case notes and
clients' enquiry messages. Each answer is built from the four most relevant documents.

## What you will see instead of names

The assistant reads cleaned versions of those documents:

| In the original | In the answer |
|---|---|
| The client's name | `[CLIENT_1]` |
| The opposing party | `[PARTY_1]` |
| A witness | `[WITNESS_1]` |
| Phone, email, PAN, Aadhaar, case number, address, bank details, matter code name | `[PHONE_1]`, `[EMAIL_1]`, `[ID_1]`, `[CASE_1]`, `[ADDRESS_1]`, `[ACCOUNT_1]`, `[MATTER_1]` |
| Bombay High Court | a High Court |
| Pune | a city in Maharashtra |
| ₹4,25,00,000 | between ₹1 crore and ₹10 crore |
| 12 March 2024 | March 2024 (or 2024) |
| cardiologist, 52-year-old | a doctor, in their 50s |
| A company such as "Mehta Builders Pvt Ltd" | a real-estate developer |

The legal issue, the advice, the documents asked for and the outcome are kept as written,
because that is what you are looking for. Rare issues (for example a maritime salvage claim)
are described more generally, since there are so few of them that the issue alone could
point to the client.

Placeholder numbers only mean something within one document: `[CLIENT_1]` in two answers can
be two different people.

## When the assistant refuses

The assistant keeps track of what a conversation has revealed. If an answer would leave fewer
than five clients matching everything said so far, it replies:

> This question cannot be answered: it would narrow the records down too far.

That is not a malfunction. A series of harmless-looking questions ("any land disputes in
Pune?", "which one involves a doctor?", "what year was it filed?") can identify a client as
surely as asking for the name. Start a new conversation with a broader question, or ask the
partner responsible for the matter.

## Ethical walls

The assistant does not reveal clients' identities, so it is safe to use across practice
teams. It is not a substitute for the conflict check: to find out whether the firm has acted
for or against a named party, use the conflicts register.

## Limits

- It only knows what is in the documents it retrieves. If it says "The records do not say",
  rephrase the question with the legal issue and the court type.
- It is tested on fictional data only. Before it is used on real client files, the firm
  needs access control, audit logs and a legal review (DPDP Act 2023, Section 132 BSA).
- It gives no formal privacy guarantee such as differential privacy.
