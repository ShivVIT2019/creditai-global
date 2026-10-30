"""
Gradio UI tab for batch underwriting.

Wraps the underwriting pipeline (validate -> features -> score -> fraud -> policy -> log)
in a simple interface: paste JSON, click button, get decisions table + downloadable audit log.
"""
import json
import gradio as gr
from underwriting.batch_loop import run_batch
from underwriting.audit_logger import write_log


SAMPLE_INPUT = [
    {
        "application_id": "APP-001-CLEAN",
        "country": "USA",
        "loan_amount": 15000,
        "annual_income": 85000,
        "dti_ratio": 15,
        "credit_history_years": 8,
        "open_credit_accounts": 5,
        "revolving_balance": 2000,
        "revolving_utilization": 12,
        "total_credit_accounts": 10,
        "credit_inquiries_6mo": 1,
        "bankruptcy_history": False
    },
    {
        "application_id": "APP-002-RISKY",
        "country": "USA",
        "loan_amount": 50000,
        "annual_income": 25000,
        "dti_ratio": 48,
        "credit_history_years": 1,
        "open_credit_accounts": 12,
        "revolving_balance": 15000,
        "revolving_utilization": 95,
        "total_credit_accounts": 20,
        "credit_inquiries_6mo": 8,
        "bankruptcy_history": False
    },
    {
        "application_id": "APP-003-BANKRUPTCY",
        "country": "USA",
        "loan_amount": 20000,
        "annual_income": 60000,
        "dti_ratio": 25,
        "credit_history_years": 5,
        "open_credit_accounts": 8,
        "revolving_balance": 5000,
        "revolving_utilization": 35,
        "total_credit_accounts": 12,
        "credit_inquiries_6mo": 2,
        "bankruptcy_history": True
    }
]


def process_batch(json_text: str):
    """Called when user clicks the Run Batch button."""
    try:
        records = json.loads(json_text)
    except json.JSONDecodeError as e:
        return (
            [["ERROR", "Invalid JSON", str(e), "", ""]],
            None,
            f"❌ JSON parse error: {e}",
        )

    if not isinstance(records, list):
        records = [records]  # single-record support

    decisions, audit_entries = run_batch(records)

    # Build table rows for display
    rows = []
    for d in decisions:
        rows.append([
            d["application_id"],
            d["decision"],
            d.get("reason", ""),
            f"{d['risk_score']:.3f}" if d.get("risk_score") is not None else "N/A",
            ", ".join(d.get("fraud_flags", [])) or "none",
        ])

    # Write audit log to a temp file for download
    log_path = write_log(audit_entries, "/tmp/audit_log.jsonl")

    # Summary stats
    stats = {"APPROVE": 0, "DECLINE": 0, "REVIEW": 0, "ERROR": 0}
    for d in decisions:
        stats[d["decision"]] = stats.get(d["decision"], 0) + 1
    summary = (
        f"Processed {len(decisions)} records: "
        f"✅ {stats['APPROVE']} approved, "
        f"❌ {stats['DECLINE']} declined, "
        f"🔍 {stats['REVIEW']} for review, "
        f"⚠️ {stats['ERROR']} errors"
    )

    return rows, log_path, summary


def build_batch_tab():
    """Returns the Gradio Blocks subcomponent for the batch tab."""
    with gr.Column() as tab:
        gr.Markdown(
            "## 📦 Batch Underwriting Loop\n"
            "Paste a JSON array of trade line records (or a single record). "
            "The pipeline validates each record, computes risk features, applies "
            "fraud rules, and makes a policy decision — producing an auditable log."
        )

        with gr.Row():
            with gr.Column(scale=2):
                json_input = gr.Textbox(
                    label="📝 Input JSON (array of records)",
                    value=json.dumps(SAMPLE_INPUT, indent=2),
                    lines=20,
                    max_lines=30,
                )
                run_btn = gr.Button("🚀 Run Batch Underwriting", variant="primary")

            with gr.Column(scale=3):
                summary_box = gr.Textbox(label="📊 Summary", lines=2, interactive=False)
                decisions_table = gr.Dataframe(
                    headers=["Application ID", "Decision", "Reason", "Risk Score", "Fraud Flags"],
                    label="📋 Decisions",
                    wrap=True,
                )
                audit_download = gr.File(label="📥 Download Audit Log (.jsonl)")

        gr.Markdown(
            "### 🧠 What the pipeline does\n"
            "1. **Validate** schema (Pydantic) — reject malformed inputs\n"
            "2. **Engineer features** — DTI buckets, utilization tiers, thin-file flag, loan-to-income ratio\n"
            "3. **Score risk** — transparent rule-weighted probability (replaceable with ML model)\n"
            "4. **Check fraud** — deterministic rules (velocity, abnormal amount, thin-file+high-ask, etc.)\n"
            "5. **Apply policy** — bankruptcy auto-decline, high risk → decline, fraud → review, medium risk → review, else approve\n"
            "6. **Audit log** — every decision captured as JSON Lines for regulator audit\n\n"
            "*This loop fills the gaps a regulated lending workflow actually requires: "
            "deterministic rules, human-in-the-loop for fraud, complete audit trail, and explainable decisions.*"
        )

        run_btn.click(
            fn=process_batch,
            inputs=[json_input],
            outputs=[decisions_table, audit_download, summary_box],
        )

    return tab
