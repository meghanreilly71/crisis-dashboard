import argparse
import json
import os
import re
import sys
import time
import pandas as pd
from pathlib import Path
from typing import Any, Optional

import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
SAMPLED_DIR = ROOT / "data" / "sampled"
ANNOTATED_DIR = ROOT / "data" / "annotated"

# Flagged sample files produced by flag_topics.py (Prompt 0b) and check_strata.py (Prompt 1).
# Climate: keyword-threshold classifier applied; migration: topped up after strata check.
SAMPLE_FILES = {
    "climate": "climate_sample_flagged.csv",
    "migration": "migration_sample_topped_up.csv",
}

CARRY_COLS = [
    "outlet_clean",
    "date",
    "year",
    "corpus",
    "corpus_overlap",
    "label",
    "meta",
    "word_count",
    "on_topic_flag",
]

MODEL = "gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS_A = 1500  # Prompt A: JSON only, compact
MAX_TOKENS_B = 4000  # Prompt B: chain-of-thought reasoning + JSON
MAX_RETRIES = 3

# ── frame groupings ───────────────────────────────────────────────────────────
PASS1_FRAMES = [
    "conflict",
    "human_interest",
    "economic",
    "deservingness",
    "responsibility",
]
PASS2_FRAMES = ["humanitarian", "security", "policy"]
PASS3_FRAMES = ["scientific", "crisis", "solutions", "victim", "skepticism"]
PASS4_FRAMES = ["securitization", "othering", "agency"]
ALL_FRAMES = PASS1_FRAMES + PASS2_FRAMES + PASS3_FRAMES + PASS4_FRAMES

# Extra fields beyond present + indicators, keyed by frame slug
FRAME_EXTRA_FIELDS: dict[str, list[str]] = {
    "deservingness": ["direction"],
    "responsibility": ["responsible_actor"],
    "agency": ["agency_type"],
    "othering": ["othering_type", "othering_evidence"],
}

# ── frame definitions (hardcoded codebook) ────────────────────────────────────

PASS1_DEFS = """\
CONFLICT
Definition: The article presents the issue in terms of conflict, contest, disagreement,
or opposition between identifiable parties (political actors, institutions, social groups).
JSON key: "conflict"
Indicators (answer yes or no for each):
  - "Does the article portray two or more parties in explicit opposition or conflict?"
  - "Does the article use language emphasising disagreement, debate, or confrontation?"
  - "Does the article present the issue in terms of winners and losers?"
  - "Is political or social opposition between groups a central organising element?"
Presence rule: present if any indicator = yes
Special fields: none

HUMAN INTEREST
Definition: The article uses individual human stories, emotional appeals, or personal
anecdotes to illustrate an abstract or collective issue.
JSON key: "human_interest"
Indicators:
  - "Does the article feature individual people (not institutions) as central characters?"
  - "Does the article include emotional language or direct appeals to empathy?"
  - "Does the article describe personal circumstances, suffering, or hardship in detail?"
  - "Does the article humanise those affected through narrative rather than statistics?"
Presence rule: present if any indicator = yes
Special fields: none

ECONOMIC
Definition: The article frames the issue primarily in terms of economic costs, financial
burdens, benefits, or market impacts.
JSON key: "economic"
Indicators:
  - "Does the article mention specific costs, prices, financial figures, or economic projections?"
  - "Does the article discuss economic burdens or benefits to government, taxpayers, or businesses?"
  - "Does the article frame the issue in terms of economic growth, competitiveness, or fiscal risk?"
  - "Does the article mention labour market effects, housing costs, or welfare expenditure?"
Presence rule: present if any indicator = yes
Special fields: none

DESERVINGNESS
Definition: The article evaluates whether individuals or groups merit public sympathy,
support, or resources based on moral criteria such as effort, victimhood, or rule compliance.
JSON key: "deservingness"
Indicators:
  - "Does the article make an explicit or implicit judgement about whether affected people deserve help or support?"
  - "Does the article distinguish between those who have contributed (e.g. workers, taxpayers) and those who have not?"
  - "Does the article invoke moral criteria (effort, victimhood, compliance with rules) to evaluate whether a GROUP merits support? Admiring or sympathetic description of one individual's effort, achievement, or hardship does NOT by itself count — the moral criteria must be applied to a category of people, not to a person as an individual."
  - "Does the article question or affirm the legitimacy of claims to support or asylum?"
  - "Does the article describe or report policy, legal, or administrative measures that differentiate between categories of migrant by merit, prospects, or legitimacy of claim (e.g. safe-country designations, kansrijk/kansarm classifications, conditional access to benefits, housing, or procedures) — regardless of whether the article endorses that differentiation?"
Presence rule: present if any indicator = yes
Special fields:
  direction — "deserving" (article frames subjects as deserving of sympathy/support),
               "undeserving" (article frames subjects as not deserving),
               "contested" (article presents both views); set to null if not present.

RESPONSIBILITY
Definition: The article assigns responsibility for causing or solving a problem to a
specific actor or institution — explicitly or implicitly.
JSON key: "responsibility"
Indicators:
  - "Does the article assign blame for a problem to a specific actor or institution?"
  - "Does the article call on a specific actor to take action or solve the problem?"
  - "Does the article evaluate whether an actor has fulfilled their responsibilities?"
  - "Is there an explicit accountability claim against a named or institutionally identifiable party (a government, company, agency, or organisation)? Diffuse causes such as 'human activity', 'society', or 'the international community' do not count."
Presence rule: present if any indicator = yes
Special fields:
  responsible_actor — name the primary actor identified as responsible. The value
  must resolve to an identifiable actor: a named body, institution, company, or
  level of government (e.g. "national government", "European Commission",
  "Utrecht municipality", "Shell", "COA"). Diffuse or abstract causes
  ("human activity", "society", "the global community", "societal beliefs")
  are NOT permitted — set responsible_actor to null in those cases.
  Set to null if the frame is not present.

"""

PASS2_DEFS = """\
HUMANITARIAN
Definition: The article frames migration primarily in terms of human suffering, vulnerability,
and a moral obligation to protect people fleeing persecution, violence, or severe hardship.
JSON key: "humanitarian"
Indicators:
  - "Does the article describe refugees or migrants primarily as victims of violence, persecution, or dire conditions?"
  - "Does the article invoke international protection obligations, human rights, or refugee law?"
  - "Does the article use empathetic language emphasising the suffering or vulnerability of migrants?"
  - "Does the article include personal stories that emphasise trauma, loss, or danger faced by migrants?"
Presence rule: present if any indicator = yes
Special fields: none

SECURITY
Definition: The article frames migration as a threat to national security, public order,
social cohesion, or cultural identity.
JSON key: "security"
Indicators:
  - "Does the article associate migration with crime, terrorism, or public safety risks?"
  - "Does the article describe migration as a threat to national culture, identity, or social cohesion?"
  - "Does the article use militarised or emergency language (e.g. invasion, surge, wave, flood)?"
  - "Does the article frame border control or immigration enforcement as a protective necessity?"
Presence rule: present if any indicator = yes
Special fields: none

POLICY
Definition: The article frames migration primarily through the lens of laws, regulations,
administrative procedures, government policy decisions, or legal debates.
JSON key: "policy"
Indicators:
  - "Does the article focus on government policy decisions, legislative debates, or legal proceedings?"
  - "Does the article discuss legal categories such as asylum, refugee status, or residence permits?"
  - "Does the article evaluate policy effectiveness or compliance with legal obligations?"
  - "Is the central focus on institutional or bureaucratic processes rather than on affected individuals?"
Presence rule: present if any indicator = yes
Special fields: none

"""

PASS3_DEFS = """\
SCIENTIFIC
Definition: The article grounds its coverage in scientific evidence, empirical data,
expert authority, or research findings about climate change.
JSON key: "scientific"
Indicators:
  - "Does the article cite peer-reviewed research, scientific institutions, or systematic measurement? Estimates or figures produced by advocacy groups, campaign organisations, or commercial actors do NOT count."
  - "Does the article quote scientists, researchers, or technical experts?"
  - "Is the scientific consensus presented as the primary basis for the article's claims?"
Presence rule: present if any indicator = yes
Special fields: none

CRISIS
Definition: The article presents climate change as an urgent, existential crisis
requiring immediate and large-scale action to prevent catastrophe.
JSON key: "crisis"
Indicators:
  - "Does the article use urgent or alarmist language about climate impacts (e.g. catastrophe, emergency, tipping point, irreversible)?"
  - "Does the article emphasise time pressure, deadlines, or the possibility of missing critical windows for action?"
  - "Does the article describe extreme weather events or their consequences as evidence of a climate emergency?"
  - "Does the article call for immediate, large-scale collective action to prevent disaster?"
Presence rule: present if any indicator = yes
Special fields: none

SOLUTIONS
Definition: The article focuses on responses to climate change — mitigation strategies,
technological solutions, policy measures, or adaptation efforts.
JSON key: "solutions"
Indicators:
  - "Does the article describe concrete measures being taken or proposed to reduce emissions or adapt to climate change?"
  - "Does the article highlight technological innovations, renewable energy, or green infrastructure as central elements?"
  - "Does the article present individual, corporate, or governmental actions as effective or promising responses to climate change?"
  - "Is the article primarily forward-looking, emphasising what can or should be done rather than impacts?"
Presence rule: present if any indicator = yes
Special fields: none

VICTIM
Definition: The article portrays specific communities, regions, or individuals as
victims suffering from the concrete impacts of climate change.
JSON key: "victim"
Indicators:
  - "Does the article describe identifiable people or communities suffering concrete harm from climate-related events?"
  - "Does the article emphasise vulnerability, loss, or displacement caused by climate change?"
  - "Does the article give direct voice to people experiencing the negative effects of climate change?"
  - "Are the affected people presented as relatively powerless in the face of climate impacts?"
Presence rule: present if any indicator = yes
Special fields: none

SKEPTICISM
Definition: The article expresses, amplifies, or gives significant platform to doubt
about climate science, the severity of climate change, or the necessity of climate policy.
JSON key: "skepticism"
Indicators:
  - "Does the article quote sources who dispute the scientific consensus on anthropogenic climate change?"
  - "Does the article frame climate policy as economically damaging, disproportionate, or unnecessary?"
  - "Does the article present climate concerns as exaggerated, politically motivated, or scientifically uncertain?"
  - "Does the article give significant space to contrarian or climate-sceptic perspectives?"
Presence rule: present if any indicator = yes
Special fields: none

"""

PASS4_DEFS = """\
SECURITIZATION
Definition: The article treats a non-military issue (migration, climate) as an existential
threat that justifies emergency measures exceeding normal democratic or legal procedures.
JSON key: "securitization"
Indicators:
  - "Does the article use existential or survival-level language to describe a threat not conventionally military?"
  - "Does the article call for or justify extraordinary measures (e.g. emergency legislation, border closures, crisis funds) beyond normal political channels?"
  - "Does the article invoke state security or national survival as justification for urgent policy action?"
  - "Does the article treat the issue as one that transcends ordinary democratic deliberation or legal constraint?"
Presence rule: present if any indicator = yes
Special fields: none

OTHERING
Definition: The article constructs or reinforces a boundary between an in-group ("us")
and an out-group ("them") defined by nationality, ethnicity, religion, or migration
status, and positions that out-group as different, inferior, or threatening.
Scope exclusions — the following are NOT othering:
  - Economic or geopolitical contrasts between states, firms, or institutions
    (e.g. Dutch industry versus foreign competitors, the EU versus the UK,
    national economies competing for investment).
  - Neutral demographic, statistical, or administrative description of a migrant
    group (population counts, housing quotas, capacity figures, policy
    categories), unless the article attaches evaluative weight to the group itself.
JSON key: "othering"
Indicators:
  - "Does the article use language that distinguishes between an in-group (e.g. Dutch, European, native) and an out-group (e.g. migrants, foreigners)?"
  - "Does the article attribute collective characteristics, behaviours, or motivations to the out-group as a whole?"
  - "Does the article position the out-group implicitly or explicitly as a problem, burden, or threat?"
  - "Does the article rely on stereotypes or generalisations about a group defined by nationality, religion, or ethnicity?"
Presence rule: present if indicator 1 = yes AND at least one of indicators 2-4 = yes.
  Indicator 1 alone is necessary but NOT sufficient.
Special fields:
  othering_type — "hostile" (the article's own voice characterises the out-group as
                  a burden, threat, or morally deficient; includes derogatory or
                  dehumanising vocabulary),
                  "institutional" (the article proposes, reports, or endorses
                  differential legal or social rights for the out-group — restricted
                  benefits, separate housing regimes, conditional access to services —
                  without necessarily hostile language),
                  "reported" (the article reports another actor's othering framing and
                  neither endorses nor challenges it),
                  "contested" (the article reports an othering framing in order to
                  challenge, rebut, or debunk it);
                  set to null if not present.
  othering_evidence — ONE short sentence (max 25 words) quoting or paraphrasing the
                  specific passage that justifies the othering_type assigned. This
                  exists so the type call can be audited after the fact; keep it
                  minimal. Set to null if the frame is not present.

AGENCY
Definition: The article implicitly or explicitly attributes agency — the capacity to act,
decide, or change outcomes — to a particular class of actors.
JSON key: "agency"
Indicators:
  - "Does the article emphasise the actions or choices of individuals as the primary determinant of outcomes?"
  - "Does the article present collective action (social movements, communities, civil society) as the primary driver of change?"
  - "Does the article focus on state or institutional actors as having primary responsibility and capacity to address the issue?"
  - "Does the article focus on a company, firm, or industry actor as the primary agent driving the situation?"
Presence rule: agency_type is the PRIMARY judgement for this frame. Decide agency_type
  first, then set present = "yes" for any agency_type other than "none", and
  present = "no" if agency_type is "none". Do NOT derive presence from the
  indicators: the indicators record which class of agent is emphasised, and are not
  evidence that any agency is attributed at all.
Special fields:
  agency_type — "individual" (individuals are the primary agents),
                "collective" (a non-corporate group is the primary agent — the public,
                activists, residents, social movements, communities, civil society),
                "state" (governments or institutions are primary agents),
                "corporations" (a company, firm, or corporate/industry actor — e.g. an
                oil major, utility, airline, or other named commercial entity — is the
                identified agent driving the article's narrative. Distinct from
                "collective", which is a non-corporate group such as the public,
                activists, or residents),
                "none" (the article attributes no meaningful agency to anyone — the
                situation is presented as structurally determined, driven by systemic
                or economic forces with no specific named actor, beyond anyone's
                meaningful control, or agency is simply not at issue).
                Choose "none" whenever no class of actor is presented as able to
                affect the outcome; it is a substantive category, not a fallback.
                There is no separate "structural" value: structurally determined
                situations belong under "none".
"""

# ── system prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPT_A = """\
You are an expert media analyst trained in news framing analysis. You will be given a Dutch \
newspaper article and a set of frame definitions. Your task is to determine which frames are \
present in the article by working through each indicator systematically.

The article is written in Dutch. Analyse the content, not the language — assess what is being \
communicated, not how it is phrased.

For each frame below, evaluate every indicator and return a structured JSON output. Do not add \
explanation or commentary outside the JSON.\
"""

SYSTEM_PROMPT_B = """\
You are an expert media analyst trained in news framing analysis. You will be given a Dutch \
newspaper article and a set of frame definitions. Your task is to classify which frames are \
present by reasoning through the article step by step before reaching a conclusion.

The article is written in Dutch. Analyse the content, not the language — assess what is being \
communicated, not how it is phrased.

Follow this structure exactly:

PART 1 — REASONING:
Step 1. Summarise the core message of the article in 2-3 sentences, focusing on the central \
issue, the actors involved, and the tone.
Step 2. For each frame definition provided, consider whether the article contains evidence of \
that frame. Identify the two or three frames most likely to be present before evaluating all of them.
Step 3. For each frame, work through every indicator explicitly (yes/no) and state which specific \
part of the article supports your answer. Where a frame is absent, briefly explain why.

PART 2 — CONCLUSION:
Return your final classification as a JSON object in the format specified below. Do not include \
any text after the JSON.\
"""


# ── helpers ───────────────────────────────────────────────────────────────────


def sep(label: str) -> None:
    width = 68
    print(f"\n{'=' * width}\n  {label}\n{'=' * width}")


def build_json_format(frame_slugs: list[str]) -> str:
    frame_block_lines = []
    for slug in frame_slugs:
        extras = FRAME_EXTRA_FIELDS.get(slug, [])
        lines = [
            f'    "{slug}": {{',
            '      "indicators": {"[indicator_text]": "[yes / no]"},',
            '      "present": "[yes / no]",',
        ]
        for field in extras:
            if field == "direction":
                lines.append(
                    '      "direction": "[deserving / undeserving / contested / null]",'
                )
            elif field == "responsible_actor":
                lines.append('      "responsible_actor": "[actor name or null]",')
            elif field == "agency_type":
                lines.append(
                    '      "agency_type": "[individual / collective / state / '
                    'corporations / none / null]",'
                )
            elif field == "othering_type":
                lines.append(
                    '      "othering_type": "[hostile / institutional / reported / contested / null]",'
                )
            elif field == "othering_evidence":
                lines.append(
                    '      "othering_evidence": "[one short sentence, max 25 words, or null]",'
                )
        lines[-1] = lines[-1].rstrip(",")  # remove trailing comma on last field
        lines.append("    }")
        frame_block_lines.append("\n".join(lines))

    frames_content = ",\n".join(frame_block_lines)
    return "{\n" + '  "frames": {\n' + frames_content + "\n  }\n}"


def build_user_message(
    frame_defs: str, article_body: str, frame_slugs: list[str]
) -> str:
    json_fmt = build_json_format(frame_slugs)
    return (
        f"FRAME DEFINITIONS:\n{frame_defs}\n\n"
        f"ARTICLE:\n{article_body}\n\n"
        f"Return your response in the following JSON format exactly:\n{json_fmt}"
    )


def extract_json(text: str) -> dict:
    """Extract the outermost (largest-spanning) valid JSON object from the response."""
    decoder = json.JSONDecoder()
    best: Optional[dict] = None
    best_end: int = -1
    for match in re.finditer(r"\{", text):
        try:
            obj, end = decoder.raw_decode(text, match.start())
            if isinstance(obj, dict) and end > best_end:
                best = obj
                best_end = end
        except json.JSONDecodeError:
            continue
    if best is None:
        raise ValueError(f"No valid JSON object found in response:\n{text[:500]}")
    return best


def call_api(client: openai.OpenAI, system: str, user: str, max_tokens: int) -> str:
    """Call the OpenAI API with exponential-backoff retry on rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return response.choices[0].message.content or ""
        except openai.RateLimitError as exc:
            wait = 2 ** (attempt + 1)
            print(
                f"     [rate limit] waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}"
            )
            time.sleep(wait)
            if attempt == MAX_RETRIES - 1:
                raise
        except openai.APIError as exc:
            wait = 2 ** (attempt + 1)
            print(
                f"     [API error] {exc} — waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}"
            )
            time.sleep(wait)
            if attempt == MAX_RETRIES - 1:
                raise
    raise RuntimeError("Exceeded max retries")  # unreachable but satisfies type checker


def null_frame_row(frame_slugs: list[str]) -> dict:
    """Return all columns for the given frames set to None (for skipped passes)."""
    row: dict[str, Any] = {}
    for slug in frame_slugs:
        row[f"{slug}_present"] = None
        row[f"{slug}_indicators"] = None
        for field in FRAME_EXTRA_FIELDS.get(slug, []):
            row[f"{slug}_{field}"] = None
    return row


def _yes(value: Any) -> bool:
    return str(value).strip().lower() == "yes"


def derive_presence(slug: str, data: dict) -> tuple[Any, bool]:
    """Apply frame-specific presence rules deterministically.

    Two frames no longer use the plain OR over indicators, so their presence is
    computed here rather than taken from the model's own "present" field. This
    makes the rule auditable and independent of model compliance.

      othering — present iff indicator 1 = yes AND at least one of 2-4 = yes.
                 Indicator 1 (in-group/out-group language) is necessary but not
                 sufficient; on its own it fired on neutral demographic and
                 administrative description.
      agency   — present iff agency_type != "none". The indicators record which
                 class of agent is emphasised and are jointly exhaustive, so an
                 OR over them made present=yes unconditional. Valid types:
                 individual / collective / state / corporations / none.
                 Structurally determined situations fall under "none"; there is
                 deliberately no separate "structural" value.

    Every other frame keeps the model's "present" value (plain OR).
    Returns (present, overridden) where `overridden` flags that the derived
    value differs from what the model returned.
    """
    stated = data.get("present")

    if slug == "othering":
        indicators = data.get("indicators") or {}
        values = list(indicators.values())
        if len(values) != 4:
            return stated, False  # unexpected shape: keep model's answer
        derived = (
            "yes" if (_yes(values[0]) and any(_yes(v) for v in values[1:])) else "no"
        )
    elif slug == "agency":
        atype = str(data.get("agency_type") or "").strip().lower()
        if atype not in {"individual", "collective", "state", "corporations", "none"}:
            return stated, False  # unusable agency_type: keep model's answer
        derived = "no" if atype == "none" else "yes"
    else:
        return stated, False

    return derived, str(stated).strip().lower() != derived


def flatten_frame_result(result: dict, frame_slugs: list[str]) -> dict:
    """Flatten the 'frames' dict from an API response into per-column values."""
    row: dict[str, Any] = {}
    frames = result.get("frames", {})
    for slug in frame_slugs:
        data = frames.get(slug, {}) or {}
        present, overridden = derive_presence(slug, data)
        row[f"{slug}_present"] = present
        if overridden:
            row[f"{slug}_present_model_stated"] = data.get("present")
        indicators = data.get("indicators")
        row[f"{slug}_indicators"] = json.dumps(indicators) if indicators else None
        for field in FRAME_EXTRA_FIELDS.get(slug, []):
            row[f"{slug}_{field}"] = data.get(field)
    return row


# ── annotation pipeline for a single article ─────────────────────────────────


def derive_corpus_type(row: pd.Series) -> str:
    """Derive corpus_type from dataset metadata rather than LLM inference."""
    corpus = str(row.get("corpus", "")).strip().lower()
    overlap = bool(row.get("corpus_overlap", False))
    return "intersection" if overlap else corpus


def _debug_pass(label: str, raw: str, parsed: dict) -> None:
    print(f"\n  [DEBUG] ── {label} ──────────────────────────────────")
    print(f"  [DEBUG] raw response ({len(raw)} chars):")
    print(raw)
    print(f"  [DEBUG] parsed JSON:")
    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    print(f"  [DEBUG] ────────────────────────────────────────────")


def annotate_article(
    idx: int,
    row: pd.Series,
    client: openai.OpenAI,
    system_prompt: str,
    api_call_counter: list[int],
    max_tokens: int = MAX_TOKENS_A,
    debug: bool = False,
) -> tuple[dict, str]:
    """
    Run all four annotation passes for one article.
    Returns (flat_row_dict, corpus_type).
    Raises on unrecoverable API failure.
    """
    body = str(row.get("body", "") or "")
    result: dict[str, Any] = {}

    # Corpus type comes from dataset metadata, not LLM inference.
    corpus_type = derive_corpus_type(row)
    result["corpus_type"] = corpus_type

    # ── Pass 1: generic frames ────────────────────────────────────────────────
    user_msg = build_user_message(PASS1_DEFS, body, PASS1_FRAMES)
    raw = call_api(client, system_prompt, user_msg, max_tokens)
    api_call_counter[0] += 1
    parsed = extract_json(raw)
    if debug:
        _debug_pass("Pass 1 — generic frames", raw, parsed)
    result.update(flatten_frame_result(parsed, PASS1_FRAMES))

    # ── Pass 2: migration-specific frames ─────────────────────────────────────
    run_pass2 = corpus_type in ("migration", "intersection")
    if run_pass2:
        user_msg = build_user_message(PASS2_DEFS, body, PASS2_FRAMES)
        raw = call_api(client, system_prompt, user_msg, max_tokens)
        api_call_counter[0] += 1
        p2 = extract_json(raw)
        if debug:
            _debug_pass("Pass 2 — migration frames", raw, p2)
        result.update(flatten_frame_result(p2, PASS2_FRAMES))
    else:
        result.update(null_frame_row(PASS2_FRAMES))

    # ── Pass 3: climate-specific frames ───────────────────────────────────────
    run_pass3 = corpus_type in ("climate", "intersection")
    if run_pass3:
        user_msg = build_user_message(PASS3_DEFS, body, PASS3_FRAMES)
        raw = call_api(client, system_prompt, user_msg, max_tokens)
        api_call_counter[0] += 1
        p3 = extract_json(raw)
        if debug:
            _debug_pass("Pass 3 — climate frames", raw, p3)
        result.update(flatten_frame_result(p3, PASS3_FRAMES))
    else:
        result.update(null_frame_row(PASS3_FRAMES))

    # ── Pass 4: cross-crisis frames (always run) ───────────────────────────────
    user_msg = build_user_message(PASS4_DEFS, body, PASS4_FRAMES)
    raw = call_api(client, system_prompt, user_msg, max_tokens)
    api_call_counter[0] += 1
    parsed = extract_json(raw)
    if debug:
        _debug_pass("Pass 4 — cross-crisis frames", raw, parsed)
    result.update(flatten_frame_result(parsed, PASS4_FRAMES))

    # Log which passes ran
    skip_msgs = []
    if not run_pass2:
        skip_msgs.append("Pass 2 skipped (migration frames)")
    if not run_pass3:
        skip_msgs.append("Pass 3 skipped (climate frames)")
    skip_str = " | " + " | ".join(skip_msgs) if skip_msgs else ""
    print(f"     [{idx}] corpus_type={corpus_type!r}{skip_str}")

    return result, corpus_type


# ── output helpers ────────────────────────────────────────────────────────────


def build_output_row(
    article_idx: int,
    input_row: pd.Series,
    annotation: dict,
    prompt_variant: str,
    run_number: int,
) -> dict:
    out: dict[str, Any] = {"article_idx": article_idx}
    for col in CARRY_COLS:
        out[col] = input_row.get(col)
    out.update(annotation)
    out["skipped_reason"] = None
    out["prompt_variant"] = prompt_variant
    out["run_number"] = run_number
    return out


def build_skipped_row(
    article_idx: int,
    input_row: pd.Series,
    prompt_variant: str,
    run_number: int,
) -> dict:
    """Row written when an article is skipped due to on_topic_flag=False."""
    out: dict[str, Any] = {"article_idx": article_idx}
    for col in CARRY_COLS:
        out[col] = input_row.get(col)
    out["corpus_type"] = derive_corpus_type(input_row)
    out.update(null_frame_row(ALL_FRAMES))
    out["skipped_reason"] = "off_topic_keyword_flag"
    out["prompt_variant"] = prompt_variant
    out["run_number"] = run_number
    return out


def append_row(output_path: Path, row_dict: dict, write_header: bool) -> None:
    pd.DataFrame([row_dict]).to_csv(
        output_path,
        mode="a",
        index=False,
        header=write_header,
    )


# ── main ──────────────────────────────────────────────────────────────────────


def print_sample_results(article_idx: int, row: pd.Series, annotation: dict) -> None:
    body = str(row.get("body", "") or "")
    print(f"\n{'─' * 68}")
    print(
        f"  ARTICLE {article_idx}  |  {row.get('outlet_clean', 'unknown')}  |  {row.get('date', '')}"
    )
    print(f"{'─' * 68}")
    excerpt = body[:600].replace("\n", " ")
    if len(body) > 600:
        excerpt += " [...]"
    print(f"\n  {excerpt}\n")
    print(f"  corpus_type        : {annotation.get('corpus_type', '?')}")
    print()
    for slug in ALL_FRAMES:
        present = annotation.get(f"{slug}_present")
        if present is None:
            continue
        tag = "[YES]" if str(present).lower() == "yes" else "[ no]"
        extras = ""
        for field in FRAME_EXTRA_FIELDS.get(slug, []):
            val = annotation.get(f"{slug}_{field}")
            if val:
                extras += f"  {field}={val}"
        indicators_raw = annotation.get(f"{slug}_indicators")
        ind_summary = ""
        if indicators_raw:
            try:
                ind = json.loads(indicators_raw)
                yes_inds = [k for k, v in ind.items() if str(v).lower() == "yes"]
                if yes_inds:
                    ind_summary = "\n       -> " + "\n       -> ".join(yes_inds)
            except (json.JSONDecodeError, AttributeError):
                pass
        print(f"  {tag} {slug}{extras}{ind_summary}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM annotation pipeline for framing analysis."
    )
    parser.add_argument("--corpus", required=True, choices=["migration", "climate"])
    parser.add_argument("--prompt", required=True, choices=["A", "B"])
    parser.add_argument(
        "--run", required=True, type=int, choices=range(1, 6), metavar="1-5"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Dry-run qualitative preview: annotate the first N articles and print results "
        "without writing any output files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw API responses and parsed JSON for every pass (use with --sample).",
    )
    args = parser.parse_args()

    corpus_name = args.corpus
    prompt_variant = args.prompt
    run_number = args.run

    system_prompt = SYSTEM_PROMPT_A if prompt_variant == "A" else SYSTEM_PROMPT_B
    max_tokens = MAX_TOKENS_A if prompt_variant == "A" else MAX_TOKENS_B

    input_path = SAMPLED_DIR / SAMPLE_FILES[corpus_name]
    output_path = (
        ANNOTATED_DIR / f"{corpus_name}_prompt{prompt_variant}_run{run_number}.csv"
    )
    failed_path = (
        ANNOTATED_DIR
        / f"failed_{corpus_name}_prompt{prompt_variant}_run{run_number}.txt"
    )

    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    # ── load and validate input ───────────────────────────────────────────────
    sep("ANNOTATE — LLM framing annotation pipeline")
    print(f"  Corpus  : {corpus_name}")
    print(f"  Prompt  : {prompt_variant}")
    print(f"  Run     : {run_number}")
    print(f"  Input   : {input_path}")
    print(f"  Output  : {output_path}")

    if not input_path.exists():
        sys.exit(f"ERROR: input file not found: {input_path}")

    # Read both flagged sample files and confirm row counts before doing anything else
    sep("Loading flagged sample datasets")
    for corpus, fname in SAMPLE_FILES.items():
        path = SAMPLED_DIR / fname
        if not path.exists():
            sys.exit(f"ERROR: expected sample file not found: {path}")
        n = len(pd.read_csv(path, usecols=["outlet_clean"]))
        print(f"  {fname}: {n:,} rows confirmed")

    df = pd.read_csv(input_path)
    df["article_idx"] = range(len(df))
    total = len(df)
    n_on_topic = int(df["on_topic_flag"].sum())
    n_off_topic = total - n_on_topic
    print(
        f"\n  Active corpus ({corpus_name}): {total:,} articles total  "
        f"({n_on_topic:,} on-topic, {n_off_topic:,} will be skipped)"
    )

    # ── resume: find already-completed articles ───────────────────────────────
    completed_indices: set[int] = set()
    write_header = True
    if output_path.exists():
        existing = pd.read_csv(output_path, usecols=["article_idx"])
        completed_indices = set(existing["article_idx"].dropna().astype(int))
        write_header = False
        print(
            f"  Resuming: {len(completed_indices):,} articles already in output, "
            f"{total - len(completed_indices):,} remaining"
        )

    # ── set up OpenAI client ──────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY environment variable is not set")
    client = openai.OpenAI(api_key=api_key)

    # ── sample / qualitative preview mode ────────────────────────────────────
    if args.sample > 0:
        sep(f"SAMPLE PREVIEW — first {args.sample} articles (no output written)")
        preview_df = df.head(args.sample)
        api_call_counter = [0]
        skipped_preview = 0
        for _, row in preview_df.iterrows():
            idx = int(row["article_idx"])
            on_topic = bool(row.get("on_topic_flag", True))
            if not on_topic:
                print(
                    f"\n  Article {idx + 1}/{args.sample}  [SKIPPED — off-topic]"
                    f"  meta={row.get('meta', '?')!r}"
                )
                skipped_preview += 1
                continue
            print(f"\n  Annotating article {idx + 1}/{args.sample} ...")
            try:
                annotation, _ = annotate_article(
                    idx,
                    row,
                    client,
                    system_prompt,
                    api_call_counter,
                    max_tokens=max_tokens,
                    debug=args.debug,
                )
                print_sample_results(idx, row, annotation)
            except Exception as exc:
                print(f"  ERROR on article {idx}: {exc}")
        sep(
            f"PREVIEW COMPLETE — {api_call_counter[0]} API calls made  |  "
            f"{skipped_preview} skipped (off-topic)  |  no files written"
        )
        print()
        return

    # ── run annotation ────────────────────────────────────────────────────────
    sep(
        f"Running annotation — {corpus_name} / prompt {prompt_variant} / run {run_number}"
    )

    api_call_counter = [0]
    completed_count = len(completed_indices)
    skipped_count = 0
    failed_indices: list[int] = []

    for article_idx, row in df.iterrows():
        idx = int(row["article_idx"])
        if idx in completed_indices:
            continue

        on_topic = bool(row.get("on_topic_flag", True))
        if not on_topic:
            skipped_row = build_skipped_row(idx, row, prompt_variant, run_number)
            append_row(
                output_path,
                skipped_row,
                write_header=(write_header and completed_count == 0),
            )
            write_header = False
            skipped_count += 1
            print(
                f"  Article {idx + 1}/{total} (idx={idx})  SKIPPED — off_topic_keyword_flag"
                f"  meta={row.get('meta', '?')!r}"
            )
            continue

        print(f"\n  Article {idx + 1}/{total} (idx={idx})")
        try:
            annotation, _ = annotate_article(
                idx,
                row,
                client,
                system_prompt,
                api_call_counter,
                max_tokens=max_tokens,
            )
            out_row = build_output_row(idx, row, annotation, prompt_variant, run_number)
            append_row(
                output_path,
                out_row,
                write_header=(write_header and completed_count == 0),
            )
            write_header = False
            completed_count += 1
            print(
                f"     annotated {completed_count}/{n_on_topic}  "
                f"skipped so far: {skipped_count}  "
                f"(API calls: {api_call_counter[0]})"
            )
        except Exception as exc:
            print(f"     FAILED after {MAX_RETRIES} retries: {exc}")
            failed_indices.append(idx)

    # ── final summary ─────────────────────────────────────────────────────────
    sep("ANNOTATION COMPLETE")
    print(f"  On-topic articles annotated : {completed_count:,}")
    print(f"  Off-topic articles skipped  : {skipped_count:,}")
    print(f"  Total API calls made        : {api_call_counter[0]:,}")
    print(f"  Failed articles             : {len(failed_indices):,}")
    print(f"  Output written to           : {output_path}")

    if failed_indices:
        failed_path.write_text(
            f"Failed article indices for {corpus_name} prompt{prompt_variant} run{run_number}:\n"
            + "\n".join(str(i) for i in failed_indices)
            + "\n"
        )
        print(f"  Failed indices logged to : {failed_path}")
    print()


if __name__ == "__main__":
    main()
