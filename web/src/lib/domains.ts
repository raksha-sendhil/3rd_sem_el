// The use cases the dashboard can switch between. The id is what the API receives (?domain=...),
// matching config/domains.json ("medical", "legal"). A domain that is not
// ready yet shows in the toggle, greyed out, and cannot be picked.

export type DomainId = "medical" | "legal" | "finance";

export type Domain = {
  id: DomainId;
  label: string;
  ready: boolean;
  why?: string; // shown when not ready
  accent: string;
  blurb: string; // one line about the documents it protects
  placeholder: string; // the document box before anything is pasted or chosen
  accept: string; // file types the upload button offers
};

const FILES = ".txt,.md,text/plain,.pdf,application/pdf,image/*,.heic,.tif,.tiff";

export const DOMAINS: Domain[] = [
  {
    id: "medical",
    label: "Healthcare",
    ready: true,
    accent: "#0FB5A0",
    blurb: "Patient records: discharge summaries, lab reports, prescriptions, health records",
    placeholder:
      "Paste a clinical note, or choose a file: a typed or scanned PDF, a photo of a report or prescription (printed or handwritten), or a health record (.hl7, .fhir, .xml)",
    accept: `${FILES},.hl7,.fhir,.xml,.json`,
  },
  {
    id: "legal",
    label: "Legal",
    ready: true,
    accent: "#6D4AFF",
    blurb: "Client files: intake notes, case notes, advice emails, contracts",
    placeholder: "Paste a case note or advice email, or choose a contract or scanned filing",
    accept: FILES,
  },
  {
    id: "finance",
    label: "Finance",
    ready: false,
    why: "Coming soon: the finance pipeline has not been built yet",
    accent: "#F5A300",
    blurb: "Customer records: statements, invoices, KYC forms, loan files",
    placeholder: "Paste a customer note, or choose a statement, invoice or KYC scan",
    accept: FILES,
  },
];

export const DEFAULT_DOMAIN: DomainId = "medical";

export const domainById = (id: DomainId): Domain => DOMAINS.find((d) => d.id === id) ?? DOMAINS[0];
