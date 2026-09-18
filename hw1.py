#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    Suggested imports:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_deepseek import ChatDeepSeek

    Use the vision-capable DeepSeek Flash model named
    ``deepseek-v4-flash-vision-exp``. The API key is loaded from .env.
    """
    ### YOUR CODE HERE
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_tokens=None,
        timeout=90,
        max_retries=2,
    )

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You extract monetary fields from one supermarket receipt image.
Read the receipt itself carefully, including English and Chinese labels. Return
exactly one compact JSON object and nothing else: no Markdown, code fence, or
explanation. Use monetary strings and this schema:
{{"paid":"394.70","subtotal":"394.72","discounts":["10.00","5.48"]}}

Definitions:
- paid: the final amount payable after the ROUNDING adjustment.
- subtotal: the SUBTOTAL/小计 amount before the ROUNDING adjustment, after
  discounts have already been applied.
- discounts: every genuine discount line as a positive absolute amount, even
  when the receipt prints it with a minus sign.

Every monetary string must have exactly two decimal places. Read each discount
amount actually printed on the receipt; never calculate an amount from a printed
discount percentage. If there are no discounts, discounts must be [].

Discounts include Buy N Save promotions, percentage OFF reductions, coupons,
member discounts, App Upgrade discounts, and discounts for damaged, broken,
or deformed packaging. Exclude ROUNDING, change/找续, cash tendered, card
balance, loyalty points, and duplicate bank-card transaction records. Do not
mistake payment/tender or repeated card slips for purchases or discounts.
Preserve cents exactly as printed and include each genuine discount once.""",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "Extract paid, subtotal, and discounts from this single receipt.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                ],
            ),
        ]
    )

    audit_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are the independent second-pass auditor for one supermarket
receipt. Re-read the original image yourself, then check the first-pass draft.
Correct every error or omission you find. Return exactly one compact JSON object
and nothing else: no Markdown, code fence, or explanation. Use monetary strings
and this schema:
{{"paid":"394.70","subtotal":"394.72","discounts":["10.00","5.48"]}}

The audited fields mean:
- paid is the final payable amount after ROUNDING.
- subtotal is SUBTOTAL/小计 before ROUNDING and after discounts.
- discounts lists all genuine discount lines as positive absolute amounts.

Every monetary string must have exactly two decimal places. Read each discount
amount actually printed on the receipt; never calculate an amount from a printed
discount percentage. If there are no discounts, discounts must be [].

Count Buy N Save, percentage OFF, coupon, member, App Upgrade, and damaged,
broken, or deformed-packaging discounts. Never count ROUNDING, change/找续,
cash tendered, card balance, loyalty points, or duplicate bank-card transaction
records. Check labels, signs, decimal places, duplicated lines, and arithmetic;
prefer the original receipt over the draft whenever they disagree.""",
            ),
            (
                "human",
                [
                    {
                        "type": "text",
                        "text": "First-pass draft:\n{draft}\nAudit it against the receipt image.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image_url}"},
                    },
                ],
            ),
        ]
    )

    extract_chain = extraction_prompt | model | StrOutputParser()
    audit_chain = audit_prompt | model | StrOutputParser()
    return {"extract": extract_chain, "audit": audit_chain}


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    ### YOUR CODE HERE
    def parse_amount(value: Any, field_name: str) -> Decimal:
        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be a monetary value")

        if isinstance(value, Decimal):
            amount = value
        elif isinstance(value, (str, int)):
            cleaned = str(value).strip().replace("−", "-")
            cleaned = re.sub(r"HK\$", "", cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.replace("$", "").replace(",", "").strip()
            try:
                amount = Decimal(cleaned)
            except InvalidOperation as exc:
                raise ValueError(
                    f"{field_name} is not a valid monetary value: {value!r}"
                ) from exc
        else:
            raise ValueError(f"{field_name} has unsupported type {type(value).__name__}")

        if not amount.is_finite():
            raise ValueError(f"{field_name} must be finite")
        try:
            return amount.quantize(Decimal("0.01"))
        except InvalidOperation as exc:
            raise ValueError(f"{field_name} cannot be rounded to cents") from exc

    def parse_receipt_output(value: Any) -> tuple[Decimal, Decimal]:
        if isinstance(value, BaseException):
            raise ValueError(
                f"model call failed with {type(value).__name__}: {value}"
            )

        text = response_text(value)
        match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
        if match is None:
            raise ValueError("response does not contain a JSON object")

        try:
            data = json.loads(
                match.group(0),
                parse_float=Decimal,
                parse_int=Decimal,
            )
        except (json.JSONDecodeError, InvalidOperation) as exc:
            raise ValueError(f"invalid JSON object: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("JSON value must be an object")
        if "paid" not in data or "subtotal" not in data:
            raise ValueError("JSON object must contain paid and subtotal")

        discounts = data.get("discounts", [])
        if not isinstance(discounts, list):
            raise ValueError("discounts must be a list")

        paid = parse_amount(data["paid"], "paid")
        subtotal = parse_amount(data["subtotal"], "subtotal")
        discount_total = sum(
            (abs(parse_amount(item, f"discounts[{index}]"))
             for index, item in enumerate(discounts)),
            Decimal("0.00"),
        )

        if abs(paid - subtotal) > Decimal("0.50"):
            raise ValueError(
                f"paid {paid:.2f} and subtotal {subtotal:.2f} differ by more than 0.50"
            )

        without_discount = (subtotal + discount_total).quantize(Decimal("0.01"))
        return paid, without_discount

    image_inputs = [
        {"image_url": image_data_url(image)}
        for image in images
    ]
    batch_config = {"max_concurrency": 4}

    extract_outputs = chain["extract"].batch(
        image_inputs,
        config=batch_config,
        return_exceptions=True,
    )

    audit_inputs = []
    for image_input, extract_output in zip(image_inputs, extract_outputs):
        if isinstance(extract_output, BaseException):
            draft = (
                "The first extraction pass failed. Independently extract all "
                "required fields from the receipt image."
            )
        else:
            draft = response_text(extract_output)
        audit_inputs.append(
            {
                "image_url": image_input["image_url"],
                "draft": draft,
            }
        )

    audit_outputs = chain["audit"].batch(
        audit_inputs,
        config=batch_config,
        return_exceptions=True,
    )

    total_paid = Decimal("0.00")
    total_without_discount = Decimal("0.00")

    for index, (image, image_input) in enumerate(zip(images, image_inputs)):
        errors = []
        selected = None

        candidates = (
            ("audit", audit_outputs[index]),
            ("extract", extract_outputs[index]),
        )
        for label, candidate in candidates:
            try:
                selected = parse_receipt_output(candidate)
                break
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")

        if selected is None:
            try:
                retry_output = chain["extract"].invoke(image_input)
                selected = parse_receipt_output(retry_output)
            except Exception as exc:
                errors.append(f"retry: {type(exc).__name__}: {exc}")

        if selected is None:
            print(
                f"Warning: {image.name}: receipt extraction failed; "
                + "; ".join(errors)
            )
            selected = (Decimal("0.00"), Decimal("0.00"))

        paid, without_discount = selected
        total_paid += paid
        total_without_discount += without_discount

    total_paid = total_paid.quantize(Decimal("0.01"))
    total_without_discount = total_without_discount.quantize(Decimal("0.01"))
    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_without_discount:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
