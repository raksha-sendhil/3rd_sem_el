"""Human-readable output for the CLI - the layout of the spec section 6 demo.

ORIGINAL DOCUMENT
   My son has been experiencing ...      (rule spans highlighted)

RULE PASS        (Presidio + spaCy + regex)       4 spans    0.21 s
   AGE           "32 years old"                   chars 182-194
LLM PASS         (llama3:8b, constrained JSON)    3 spans    4.84 s
   QUASI_COMBO   left-sided abdominal + bilateral pelvic pain
MERGED OUTPUT    7 spans   validation OK   retries 0   total 5.09 s
"""

from __future__ import annotations

from rich.console import Console, RenderableType
from rich.markup import escape
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from sanitise.agents.detector.pipeline import DetectorTrace
from sanitise.errors import PipelineError
from sanitise.loader import Document
from sanitise.schemas import DetectorResult, PIICategory, SensitiveSpan, SpanMetrics

INDENT = 3
LABEL_W = 17
CATEGORY_W = 18

CATEGORY_STYLE: dict[PIICategory, str] = {
    PIICategory.NAME: "bold red",
    PIICategory.AGE: "magenta",
    PIICategory.DATE: "cyan",
    PIICategory.PHONE: "bold yellow",
    PIICategory.EMAIL: "bold yellow",
    PIICategory.ID_NUMBER: "bold yellow",
    PIICategory.ADDRESS: "red",
    PIICategory.LOCATION: "red",
    PIICategory.OCCUPATION: "blue",
    PIICategory.RELATION: "green",
    PIICategory.PHYSICAL: "bright_magenta",
    PIICategory.MEDICAL_CONDITION: "bright_cyan",
    PIICategory.SYMPTOM: "bright_blue",
    PIICategory.QUASI_COMBO: "bold bright_red",
    PIICategory.OTHER: "white",
}


def _style(cat: PIICategory) -> str:
    return CATEGORY_STYLE.get(cat, "white")


def _secs(ms: int) -> str:
    return f"{ms / 1000:.2f} s"


def highlighted(text: str, spans: list[SensitiveSpan]) -> Text:
    """The document with every offset-bearing span underlined in its category colour."""
    out = Text(text)
    for span in spans:
        if span.start is not None and span.end is not None:
            out.stylize(f"{_style(span.category)} underline", span.start, span.end)
    return out


def _header(label: str, detail: str, count: int, ms: int | None) -> Text:
    line = Text()
    line.append(f"{label:<{LABEL_W}}", style="bold")
    # Long model names overflow the column; always keep a gap before the count.
    line.append(f"{detail:<36}" if len(detail) < 36 else f"{detail}  ", style="dim")
    line.append(f"{count} span{'s' if count != 1 else ''}")
    if ms is not None:
        line.append(f"   {_secs(ms)}", style="dim")
    return line


def _span_line(span: SensitiveSpan, *, show_rationale: bool) -> list[RenderableType]:
    """Category column + body; long text and rationales wrap under the body, not at column 0."""
    body = Text()
    if span.start is not None:
        quoted = f'"{span.text}"'
        body.append(f"{quoted:<34}")
        body.append(f" chars {span.start}-{span.end}", style="dim")
    else:
        body.append(span.text)
    if span.confidence < 1.0:
        body.append(f"  p={span.confidence:.2f}", style="dim")
    grid = Table.grid()
    grid.add_column(width=CATEGORY_W, no_wrap=True)
    grid.add_column()
    grid.add_row(Text(span.category.value, style=_style(span.category)), body)
    lines: list[RenderableType] = [Padding(grid, (0, 0, 0, INDENT))]
    if show_rationale and span.rationale and span.method == "llm":
        why = Text(f"why: {span.rationale}", style="dim italic")
        lines.append(Padding(why, (0, 0, 0, INDENT + CATEGORY_W)))
    return lines


def render_verbose(
    console: Console, doc: Document, result: DetectorResult, trace: DetectorTrace
) -> None:
    console.print()
    console.print(Text("ORIGINAL DOCUMENT", style="bold"))
    console.print(Padding(highlighted(doc.text, trace.rule_spans), (0, 0, 0, INDENT)))
    console.print(
        Text(f"doc_id {doc.doc_id}   {len(doc.text)} chars   {doc.source or ''}", style="dim")
    )
    console.print()

    console.print(
        _header("RULE PASS", "(Presidio + spaCy + regex)", len(trace.rule_spans), trace.rule_ms)
    )
    for span in trace.rule_spans:
        for line in _span_line(span, show_rationale=False):
            console.print(line)
    console.print()

    if trace.llm_ran:
        detail = f"({result.meta.model}, constrained JSON)"
        console.print(_header("LLM PASS", detail, len(trace.llm_spans), trace.llm_ms))
        for span in trace.llm_spans:
            for line in _span_line(span, show_rationale=True):
                console.print(line)
    else:
        console.print(
            Text(f"{'LLM PASS':<{LABEL_W}}", style="bold")
            + Text("skipped (--rules-only)", style="dim")
        )
    console.print()

    merged = Text()
    merged.append(f"{'MERGED OUTPUT':<{LABEL_W}}", style="bold")
    n = len(result.output.spans)
    merged.append(f"{n} span{'s' if n != 1 else ''}   ")
    merged.append("validation OK", style="bold green")
    merged.append(f"   retries {result.meta.retries}   total {_secs(result.meta.total_ms)}")
    console.print(merged)
    console.print()


def render_summary_line(console: Console, doc: Document, result: DetectorResult) -> None:
    m = result.meta
    console.print(
        f"[bold]{doc.doc_id}[/]  {escape(doc.source or '')}  "
        f"rule {m.rule_span_count}  llm {m.llm_span_count}  merged {len(result.output.spans)}  "
        f"retries {m.retries}  [dim]{_secs(m.total_ms)}[/]"
    )


def render_error(console: Console, exc: PipelineError, source: str | None = None) -> None:
    prefix = f"{escape(source)}: " if source else ""
    console.print(f"[bold red]error[/] {prefix}{escape(str(exc))}")
    console.print_json(exc.to_envelope().model_dump_json(exclude_none=True))


def render_metrics(console: Console, metrics: SpanMetrics, *, documents: int) -> None:
    console.print()
    console.print(
        f"[bold]Precision[/] {metrics.precision:.2f}   [bold]Recall[/] {metrics.recall:.2f}   "
        f"[bold]F1[/] {metrics.f1:.2f}      "
        f"[dim]({documents} documents, {metrics.support} gold spans)[/]"
    )
    if not metrics.per_category:
        return
    table = Table(show_edge=False, box=None, pad_edge=False, header_style="bold dim")
    table.add_column("category")
    keys: list[str] = []
    for row in metrics.per_category.values():
        keys.extend(k for k in row if k not in keys)
    for key in keys:
        table.add_column(key, justify="right")
    for cat, row in sorted(metrics.per_category.items()):
        cells = []
        for k in keys:
            v = row.get(k)
            cells.append(
                ""
                if v is None
                else (f"{v:.0f}" if k in {"support", "tp", "fp", "fn"} else f"{v:.2f}")
            )
        table.add_row(cat, *cells)
    console.print()
    console.print(table)
    console.print()
