"""Pull eligibility facts out of grant text with NuExtract 2.0 (numind/NuExtract-2.0-2B).

NuExtract fills a JSON template from the input text. Fields it can't find come back as
null or [], which is what lets the rules treat "not stated" differently from "excluded".

Two calls per grant: the 2B model was more accurate answering "is this an AI program?" as
its own question than as one more field in the main template (see Week 6 attempts).
"""

import hashlib
import json

MODEL_NAME = "numind/NuExtract-2.0-2B"

TEMPLATE = {
    "applicant_countries": ["string"],
    "impact_countries": ["string"],
    "applicant_types": [["nonprofit", "for-profit company", "individual"]],
    "women_led_required": ["yes", "no"],
    "min_women_ownership_percent": "number",
    "max_company_age_years": "integer",
    "max_annual_revenue_usd": "number",
    "award": "verbatim-string",
    "deadline": "verbatim-string",
}

AI_TEMPLATE = {"program_focus_is_ai_or_deep_tech": ["yes", "no"]}

# Synthetic grants (not from the dataset) used as in-context examples.
_BRIGHT_FUTURES = (
    "Program: Bright Futures Fund\nType: Grant\nAmount / Benefit: $25,000\nDeadline: 2026-03-01\n"
    "Core Requirements: Registered charity based in Canada; projects must serve communities in Kenya; "
    "women-led organizations only; organization under 3 years old"
)
_OPEN_HORIZONS = (
    "Program: Open Horizons AI Prize\nType: Prize\nAmount / Benefit: Up to €50k\nDeadline: Rolling\n"
    "Core Requirements: Open worldwide; for-profit startups building AI products; "
    "annual revenue below $2M; at least 51% women-owned"
)
_ROBOTICS_NIGHT = (
    "Program: Women in Robotics Demo Night\nType: Pitch event\nAmount / Benefit: $5,000 prize\n"
    "Deadline: 2026-12-01\nCore Requirements: Women-led robotics venture; pitch to investors"
)
_MAIN_STREET = (
    "Program: Main Street Relief Grant\nType: Grant\nAmount / Benefit: $2,500\nDeadline: Rolling\n"
    "Core Requirements: Small business owner; retail, food or household services"
)

EXAMPLES = [
    {
        "input": _BRIGHT_FUTURES,
        "output": json.dumps(
            {
                "applicant_countries": ["Canada"],
                "impact_countries": ["Kenya"],
                "applicant_types": ["nonprofit"],
                "women_led_required": "yes",
                "min_women_ownership_percent": None,
                "max_company_age_years": 3,
                "max_annual_revenue_usd": None,
                "award": "$25,000",
                "deadline": "2026-03-01",
            }
        ),
    },
    {
        "input": _OPEN_HORIZONS,
        "output": json.dumps(
            {
                "applicant_countries": ["worldwide"],
                "impact_countries": [],
                "applicant_types": ["for-profit company"],
                "women_led_required": "yes",
                "min_women_ownership_percent": 51,
                "max_company_age_years": None,
                "max_annual_revenue_usd": 2000000,
                "award": "Up to €50k",
                "deadline": "Rolling",
            }
        ),
    },
]

AI_EXAMPLES = [
    {"input": _BRIGHT_FUTURES, "output": json.dumps({"program_focus_is_ai_or_deep_tech": "no"})},
    {"input": _OPEN_HORIZONS, "output": json.dumps({"program_focus_is_ai_or_deep_tech": "yes"})},
    {"input": _ROBOTICS_NIGHT, "output": json.dumps({"program_focus_is_ai_or_deep_tech": "yes"})},
    {"input": _MAIN_STREET, "output": json.dumps({"program_focus_is_ai_or_deep_tech": "no"})},
]

# Changes whenever the model, templates or examples change, so cached extractions get redone.
PROMPT_VERSION = hashlib.sha256(
    json.dumps([MODEL_NAME, TEMPLATE, EXAMPLES, AI_TEMPLATE, AI_EXAMPLES]).encode()
).hexdigest()[:12]


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_output(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_error": "model output was not valid JSON", "_raw": raw}


class NuExtractor:
    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        import torch
        from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.device = device or pick_device()
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        # Text-only use, so the tokenizer replaces the image processor from the model card.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, dtype=dtype)
        self.model.to(self.device).eval()

    def _run(self, text: str, template: dict, examples: list[dict]) -> dict:
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            template=json.dumps(template, indent=2),
            examples=examples,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            # Greedy decoding (temperature 0), as the model card recommends for extraction.
            out = self.model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=512)
        raw = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return parse_output(raw.strip())

    def extract(self, text: str) -> dict:
        main = self._run(text, TEMPLATE, EXAMPLES)
        ai = self._run(text, AI_TEMPLATE, AI_EXAMPLES)
        if "_error" in main or "_error" in ai:
            return main if "_error" in main else ai
        return {**main, **ai}
