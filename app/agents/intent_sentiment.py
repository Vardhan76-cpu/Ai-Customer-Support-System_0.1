import json
import os
import re
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from app.models.schemas import IntentSentimentAnalysis


load_dotenv()


HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv(
    "HF_MODEL",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
)


client = None
_hf_quota_depleted = False


if HF_TOKEN:
    try:
        client = InferenceClient(
            api_key=HF_TOKEN,
            provider="auto",
        )
    except Exception as exc:
        print(
            f"InferenceClient init note: {exc}"
        )


# ============================================================
# RESPONSE CONTENT EXTRACTION
# ============================================================

def _extract_content(response) -> str:
    if not response:
        return ""

    choices = getattr(
        response,
        "choices",
        None,
    )

    if not choices:
        return ""

    message = getattr(
        choices[0],
        "message",
        None,
    )

    if not message:
        return ""

    content = getattr(
        message,
        "content",
        None,
    )

    if content:
        return str(content).strip()

    return ""


# ============================================================
# MAIN ANALYSIS ENTRY POINT
# ============================================================

def analyze_customer_message(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> IntentSentimentAnalysis:
    """
    Analyze a customer message for:

    - Intent
    - Emotion
    - Sentiment
    - Frustration Level
    - Satisfaction Trend
    - Escalation Risk
    - Confidence
    - Suggested Action
    - Suggested Response
    - Response Evaluation
    - Communication Tips

    Uses the LLM when available.

    If the LLM is unavailable because of authentication,
    quota, credits, or similar API problems, the system
    automatically switches to the deterministic NLP engine.
    """

    global _hf_quota_depleted

    cleaned_msg = message.strip()

    # --------------------------------------------------------
    # Empty message handling
    # --------------------------------------------------------

    if not cleaned_msg:
        return IntentSentimentAnalysis(
            intent="general_inquiry",
            emotion="neutral",
            sentiment="neutral",
            frustration_level=0,
            satisfaction_trend="stable",
            escalation_risk="low",
            confidence=0.90,
        )

    # --------------------------------------------------------
    # LLM ANALYSIS
    # --------------------------------------------------------

    if client and not _hf_quota_depleted:

        try:
            llm_result = _analyze_with_llm(
                cleaned_msg,
                history,
            )

            if llm_result:
                return llm_result

        except Exception as exc:

            err_str = str(exc).lower()

            # ------------------------------------------------
            # TASK 8 FIX
            #
            # Stop repeated external API attempts when
            # authentication, authorization, quota,
            # credit, or payment errors occur.
            # ------------------------------------------------

            if (
                "401" in err_str
                or "unauthorized" in err_str
                or "authentication" in err_str
                or "invalid api" in err_str
                or "invalid token" in err_str
                or "402" in err_str
                or "quota" in err_str
                or "credits" in err_str
                or "credit" in err_str
                or "payment required" in err_str
            ):
                _hf_quota_depleted = True

            print(
                "INTENT/SENTIMENT LLM NOTICE:",
                exc,
                "Using deterministic engine.",
            )

    # --------------------------------------------------------
    # DETERMINISTIC NLP FALLBACK
    # --------------------------------------------------------

    return _analyze_with_heuristics(
        cleaned_msg,
        history,
    )


# ============================================================
# LLM ANALYSIS METHOD
# ============================================================

def _analyze_with_llm(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Optional[IntentSentimentAnalysis]:

    recent_history = history[-6:] if history else []

    history_text = (
        "\n".join(
            f"{'Customer' if m.get('role') == 'user' else 'Agent'}: "
            f"{m.get('content', '')}"
            for m in recent_history
        )
        if recent_history
        else "No previous conversation."
    )

    system_prompt = """
You are an expert customer support Intent & Sentiment Analysis Agent.

Analyze the customer's latest message in context of the conversation history.

You must return a JSON object with EXACTLY the following keys:

- "intent": detected customer intent
  One of:
  "refund",
  "cancellation",
  "delivery_issue",
  "payment_issue",
  "account_issue",
  "complaint",
  "return_exchange",
  "general_inquiry"

- "emotion":
  "happy",
  "neutral",
  "confused",
  "worried",
  "frustrated",
  "angry",
  "satisfied"

- "sentiment":
  "positive",
  "neutral",
  "negative"

- "frustration_level":
  Integer from 0 to 10

- "satisfaction_trend":
  "improving",
  "declining",
  "stable"

- "escalation_risk":
  "low",
  "medium",
  "high",
  "critical"

- "escalation_reasoning":
  String explaining why this risk level was assigned.

- "confidence":
  Float between 0.0 and 1.0

- "suggested_action":
  Specific actionable coaching recommendation.

- "suggested_response":
  Recommended response template for the agent.

- "response_evaluation":
  JSON object with:
  "tone",
  "clarity",
  "empathy",
  "professionalism"

  Each must be an integer from 1 to 5.

- "communication_tips":
  List of actionable communication improvement tips.

Return ONLY the raw JSON object.
Do not use markdown.
"""

    user_prompt = f"""
CONVERSATION HISTORY:

{history_text}

LATEST CUSTOMER MESSAGE:

"{message}"

Provide JSON analysis:
"""

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ],
        max_tokens=800,
        temperature=0.1,
    )

    content = _extract_content(response)

    if not content:
        return None

    json_str = content

    # --------------------------------------------------------
    # Remove markdown code fences if the model adds them
    # --------------------------------------------------------

    if "```json" in json_str:

        match = re.search(
            r"```json(.*?)```",
            json_str,
            re.DOTALL,
        )

        if match:
            json_str = match.group(1).strip()

    elif "```" in json_str:

        match = re.search(
            r"```(.*?)```",
            json_str,
            re.DOTALL,
        )

        if match:
            json_str = match.group(1).strip()

    data = json.loads(json_str)

    frustration = max(
        0,
        min(
            10,
            int(
                data.get(
                    "frustration_level",
                    5,
                )
            ),
        ),
    )

    confidence = max(
        0.0,
        min(
            1.0,
            float(
                data.get(
                    "confidence",
                    0.9,
                )
            ),
        ),
    )

    # --------------------------------------------------------
    # Dynamic coaching fallback
    # --------------------------------------------------------

    suggested_action = str(
        data.get(
            "suggested_action",
            "",
        )
    ).strip()

    suggested_response = str(
        data.get(
            "suggested_response",
            "",
        )
    ).strip()

    escalation_reasoning = str(
        data.get(
            "escalation_reasoning",
            "",
        )
    ).strip()

    eval_dict = data.get(
        "response_evaluation",
        {},
    )

    comm_tips = data.get(
        "communication_tips",
        [],
    )

    if not suggested_action or not suggested_response:

        def_action, def_resp = _generate_coaching_guidance(
            message,
            str(
                data.get(
                    "intent",
                    "general_inquiry",
                )
            ).lower(),
            str(
                data.get(
                    "emotion",
                    "neutral",
                )
            ).lower(),
            frustration,
            str(
                data.get(
                    "escalation_risk",
                    "low",
                )
            ).lower(),
            history,
        )

        suggested_action = (
            suggested_action
            or def_action
        )

        suggested_response = (
            suggested_response
            or def_resp
        )

    from app.models.schemas import ResponseEvaluation

    evaluation = ResponseEvaluation(
        tone=(
            eval_dict.get("tone", 5)
            if isinstance(eval_dict, dict)
            else 5
        ),
        clarity=(
            eval_dict.get("clarity", 5)
            if isinstance(eval_dict, dict)
            else 5
        ),
        empathy=(
            eval_dict.get("empathy", 5)
            if isinstance(eval_dict, dict)
            else 5
        ),
        professionalism=(
            eval_dict.get(
                "professionalism",
                5,
            )
            if isinstance(eval_dict, dict)
            else 5
        ),
    )

    # --------------------------------------------------------
    # Escalation reasoning fallback
    # --------------------------------------------------------

    if not escalation_reasoning:

        risk = str(
            data.get(
                "escalation_risk",
                "low",
            )
        ).lower()

        if risk == "critical":

            escalation_reasoning = (
                "Customer exhibits severe frustration, "
                "potential legal or manager threats, or "
                "extreme dissatisfaction requiring immediate "
                "intervention."
            )

        elif risk == "high":

            escalation_reasoning = (
                "High frustration and negative sentiment "
                "observed. Immediate de-escalation needed."
            )

        elif risk == "medium":

            escalation_reasoning = (
                "Customer is showing signs of impatience "
                "or confusion. Clear communication required."
            )

        else:

            escalation_reasoning = (
                "Standard inquiry. No immediate escalation "
                "risks detected."
            )

    # --------------------------------------------------------
    # Communication tips fallback
    # --------------------------------------------------------

    if not comm_tips:

        if frustration > 7:

            comm_tips = [
                "Acknowledge the customer's frustration immediately.",
                "Avoid using policies as an excuse.",
                "Provide a direct, concrete resolution.",
            ]

        else:

            comm_tips = [
                "Maintain a polite and professional tone.",
                "Be clear and concise with next steps.",
            ]

    return IntentSentimentAnalysis(
        intent=str(
            data.get(
                "intent",
                "general_inquiry",
            )
        ).lower(),

        emotion=str(
            data.get(
                "emotion",
                "neutral",
            )
        ).lower(),

        sentiment=str(
            data.get(
                "sentiment",
                "neutral",
            )
        ).lower(),

        frustration_level=frustration,

        satisfaction_trend=str(
            data.get(
                "satisfaction_trend",
                "stable",
            )
        ).lower(),

        escalation_risk=str(
            data.get(
                "escalation_risk",
                "low",
            )
        ).lower(),

        escalation_reasoning=escalation_reasoning,

        confidence=confidence,

        suggested_action=suggested_action,

        suggested_response=suggested_response,

        response_evaluation=evaluation,

        communication_tips=comm_tips,
    )


# ============================================================
# DETERMINISTIC COMPREHENSIVE NLP ENGINE
# ============================================================

def _analyze_with_heuristics(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> IntentSentimentAnalysis:

    msg_lower = message.lower()

    raw_words = re.findall(
        r"\b\w+\b",
        message,
    )

    upper_words = [
        word
        for word in raw_words
        if word.isupper() and len(word) > 1
    ]

    caps_ratio = (
        len(upper_words)
        / max(len(raw_words), 1)
    )

    exclamations = message.count("!")

    # ========================================================
    # 1. ESCALATION TRIGGERS
    # ========================================================

    escalation_patterns = [

        r"\b(speak|talk|transfer|escalate|demand|get)\b.*?"
        r"\b(manager|supervisor)\b",

        r"\b(someone|anyone)\b.*?"
        r"\b(authorize|approve|override)\b",

        r"\b(lawyer|attorney|lawsuit|legal action|"
        r"consumer court|better business bureau|\bbbb\b|police)\b",

        r"\b(report you|sue you|take you to court)\b",

        r"\b(escalate this|escalate to|transfer me)\b",

        r"\b(file an official complaint|official complaint)\b",
    ]

    has_escalation_phrase = any(
        re.search(
            pattern,
            msg_lower,
        )
        for pattern in escalation_patterns
    )

    # ========================================================
    # 2. INTENT DETECTION
    # ========================================================

    intent = "general_inquiry"

    intent_confidence = 0.88

    refund_kw = [
        "refund",
        "money back",
        "reimburse",
        "reimbursement",
        "charge back",
        "chargeback",
    ]

    cancel_kw = [
        "cancel",
        "cancellation",
        "stop subscription",
        "terminate account",
        "unsubscribe",
    ]

    delivery_kw = [
        "delivery",
        "delivered",
        "shipping",
        "shipment",
        "package",
        "order",
        "arrived",
        "transit",
        "carrier",
        "courier",
        "tracking",
        "late",
        "delay",
        "delayed",
        "where is",
    ]

    payment_kw = [
        "payment",
        "charged",
        "billing",
        "credit card",
        "debit card",
        "transaction",
        "invoice",
        "double charge",
        "charged twice",
        "declined",
        "pay",
    ]

    account_kw = [
        "account",
        "password",
        "login",
        "logged out",
        "sign in",
        "reset",
        "hacked",
        "unauthorized",
        "profile",
        "locked out",
        "access",
    ]

    return_kw = [
        "return",
        "exchange",
        "replace",
        "replacement",
        "defective",
        "broken",
        "wrong size",
        "damaged item",
        "faulty",
        "shattered",
    ]

    complaint_kw = [
        "complaint",
        "rude",
        "unacceptable",
        "terrible",
        "awful",
        "horrible",
        "worst service",
        "incompetent",
        "bad experience",
        "poor service",
    ]

    if any(
        keyword in msg_lower
        for keyword in refund_kw
    ):

        intent = "refund"
        intent_confidence = 0.95

    elif any(
        keyword in msg_lower
        for keyword in cancel_kw
    ):

        intent = "cancellation"
        intent_confidence = 0.94

    elif (
        any(
            keyword in msg_lower
            for keyword in payment_kw
        )
        and (
            "double" in msg_lower
            or "charged" in msg_lower
            or "twice" in msg_lower
            or "declined" in msg_lower
        )
    ):

        intent = "payment_issue"
        intent_confidence = 0.93

    elif any(
        keyword in msg_lower
        for keyword in account_kw
    ):

        intent = "account_issue"
        intent_confidence = 0.92

    elif any(
        keyword in msg_lower
        for keyword in return_kw
    ):

        intent = "return_exchange"
        intent_confidence = 0.91

    elif any(
        keyword in msg_lower
        for keyword in delivery_kw
    ):

        intent = "delivery_issue"
        intent_confidence = 0.92

    elif any(
        keyword in msg_lower
        for keyword in complaint_kw
    ):

        intent = "complaint"
        intent_confidence = 0.90

    # ========================================================
    # 3. FRUSTRATION LEVEL
    # ========================================================

    frustration_score = 1

    furious_kw = [
        "furious",
        "outraged",
        "disgusted",
        "livid",
        "scam",
        "fraud",
        "thieves",
        "stealing",
        "rip off",
        "bullshit",
        "had enough",
    ]

    angry_kw = [
        "angry",
        "unacceptable",
        "terrible",
        "horrible",
        "ridiculous",
        "fed up",
        "sick and tired",
        "waste of time",
        "pissed",
        "garbage",
        "trash",
        "awful",
        "rude",
        "unhelpful",
    ]

    annoyed_kw = [
        "frustrated",
        "annoyed",
        "irritated",
        "disappointed",
        "waiting forever",
        "still not",
        "how hard can it be",
        "too long",
        "runaround",
    ]

    confused_kw = [
        "confused",
        "don't understand",
        "not sure",
        "why was i",
        "what happened",
        "unclear",
        "makes no sense",
    ]

    worried_kw = [
        "worried",
        "concerned",
        "anxious",
        "scared",
        "nervous",
        "security emergency",
        "unauthorized",
        "hacked",
    ]

    desires_resolution = any(
        phrase in msg_lower
        for phrase in [
            "need this resolved",
            "need it resolved",
            "get this resolved",
            "want this resolved",
            "hope this is resolved",
            "to be resolved",
            "get it resolved",
            "resolve this today",
            "resolved today",
            "need to resolve",
            "trying to resolve",
        ]
    )

    positive_resolution_kw = [
        "thank you so much",
        "thank you very much",
        "problem is solved",
        "issue is solved",
        "fixed it",
        "that worked",
        "that fixed it",
        "such a relief",
        "huge relief",
        "perfect, thank you",
        "wonderful, thank you",
        "amazing day",
        "great help",
        "solved now",
        "fixed now",
        "resolved now",
        "all sorted out",
    ]

    negated_resolution = any(
        phrase in msg_lower
        for phrase in [
            "not resolved",
            "isn't resolved",
            "not solved",
            "unresolved",
            "not working",
            "still not",
        ]
    )

    impatient_kw = [
        "delayed",
        "delay",
        "late",
        "urgent",
        "urgently",
        "asap",
        "where is my",
        "needed it",
        "waiting for",
        "how much longer",
    ]

    if any(
        keyword in msg_lower
        for keyword in furious_kw
    ):

        frustration_score += 6

    elif any(
        keyword in msg_lower
        for keyword in angry_kw
    ):

        frustration_score += 4

    elif any(
        keyword in msg_lower
        for keyword in annoyed_kw
    ):

        frustration_score += 3

    elif any(
        keyword in msg_lower
        for keyword in impatient_kw
    ):

        frustration_score += 3

    elif any(
        keyword in msg_lower
        for keyword in worried_kw
    ):

        frustration_score += 2

    elif any(
        keyword in msg_lower
        for keyword in confused_kw
    ):

        frustration_score += 2

    if desires_resolution:
        frustration_score = max(
            frustration_score,
            4,
        )

    if (
        caps_ratio > 0.25
        and len(raw_words) >= 3
    ):
        frustration_score += 2

    if exclamations >= 2:
        frustration_score += 1

    if exclamations >= 4:
        frustration_score += 1

    if has_escalation_phrase:
        frustration_score = max(
            frustration_score,
            8,
        )
        frustration_score += 2

    if negated_resolution:
        frustration_score = max(
            frustration_score,
            6,
        )

    past_customer_msgs = [
        m.get("content", "")
        for m in (history or [])
        if m.get("role") == "user"
    ]

    if (
        len(past_customer_msgs) >= 2
        and any(
            keyword in msg_lower
            for keyword in (
                angry_kw
                + annoyed_kw
            )
        )
    ):
        frustration_score += 1

    if len(past_customer_msgs) >= 3:
        frustration_score += 1

    if (
        any(
            keyword in msg_lower
            for keyword in positive_resolution_kw
        )
        and not any(
            keyword in msg_lower
            for keyword in (
                angry_kw
                + furious_kw
            )
        )
        and not has_escalation_phrase
        and not desires_resolution
        and not negated_resolution
    ):

        frustration_score = min(
            frustration_score,
            1,
        )

    frustration_level = max(
        0,
        min(
            10,
            frustration_score,
        ),
    )

    # ========================================================
    # 4. EMOTION & SENTIMENT
    # ========================================================

    sentiment = "neutral"
    emotion = "neutral"

    if (
        any(
            keyword in msg_lower
            for keyword in positive_resolution_kw
        )
        and not has_escalation_phrase
        and not any(
            keyword in msg_lower
            for keyword in (
                angry_kw
                + furious_kw
            )
        )
        and not desires_resolution
        and not negated_resolution
    ):

        sentiment = "positive"

        emotion = (
            "happy"
            if any(
                keyword in msg_lower
                for keyword in [
                    "great",
                    "awesome",
                    "wonderful",
                    "amazing",
                    "love",
                ]
            )
            else "satisfied"
        )

        frustration_level = min(
            frustration_level,
            2,
        )

    elif (
        frustration_level >= 7
        or has_escalation_phrase
        or any(
            keyword in msg_lower
            for keyword in furious_kw
        )
    ):

        sentiment = "negative"
        emotion = "angry"

    elif (
        frustration_level >= 4
        or any(
            keyword in msg_lower
            for keyword in (
                angry_kw
                + annoyed_kw
            )
        )
        or desires_resolution
    ):

        sentiment = "negative"

        emotion = (
            "impatient"
            if any(
                keyword in msg_lower
                for keyword in impatient_kw
            )
            else "frustrated"
        )

    elif any(
        keyword in msg_lower
        for keyword in worried_kw
    ):

        sentiment = (
            "negative"
            if frustration_level >= 3
            else "neutral"
        )

        emotion = "worried"

    elif any(
        keyword in msg_lower
        for keyword in confused_kw
    ):

        sentiment = (
            "neutral"
            if frustration_level <= 2
            else "negative"
        )

        emotion = "confused"

    elif any(
        keyword in msg_lower
        for keyword in [
            "please",
            "could you",
            "hello",
            "hi",
            "inquiry",
            "how to",
            "ok",
            "got it",
        ]
    ):

        sentiment = "neutral"
        emotion = "neutral"

    # ========================================================
    # 5. SATISFACTION TREND
    # ========================================================

    satisfaction_trend = "stable"

    if past_customer_msgs:

        prev_msg = past_customer_msgs[-1].lower()

        prev_was_angry = (
            any(
                keyword in prev_msg
                for keyword in (
                    angry_kw
                    + annoyed_kw
                    + furious_kw
                )
            )
            or "!" in prev_msg
        )

        if (
            prev_was_angry
            and (
                sentiment == "positive"
                or emotion in [
                    "satisfied",
                    "happy",
                ]
                or any(
                    keyword in msg_lower
                    for keyword in positive_resolution_kw
                )
            )
        ):

            satisfaction_trend = "improving"

        elif (
            frustration_level >= 6
            and (
                has_escalation_phrase
                or len(past_customer_msgs) >= 2
                or prev_was_angry
            )
        ):

            satisfaction_trend = "declining"

        elif (
            sentiment == "positive"
            and prev_was_angry
        ):

            satisfaction_trend = "improving"

        elif (
            sentiment == "negative"
            and len(past_customer_msgs) >= 2
        ):

            satisfaction_trend = "declining"

        else:

            satisfaction_trend = "stable"

    else:

        if sentiment == "positive":

            satisfaction_trend = "improving"

        elif (
            frustration_level >= 7
            or has_escalation_phrase
        ):

            satisfaction_trend = "declining"

        else:

            satisfaction_trend = "stable"

    # ========================================================
    # 6. ESCALATION RISK
    # ========================================================

    if (
        has_escalation_phrase
        and frustration_level >= 8
        and len(past_customer_msgs) >= 2
    ):

        escalation_risk = "critical"

        escalation_reasoning = (
            "Customer exhibits severe frustration, "
            "potential legal/manager threats, or extreme "
            "dissatisfaction requiring immediate intervention."
        )

    elif (
        has_escalation_phrase
        or frustration_level >= 8
    ):

        escalation_risk = "high"

        escalation_reasoning = (
            "High frustration and negative sentiment "
            "observed. Immediate de-escalation needed."
        )

    elif (
        frustration_level >= 4
        or satisfaction_trend == "declining"
    ):

        escalation_risk = "medium"

        escalation_reasoning = (
            "Customer is showing signs of impatience "
            "or confusion. Clear communication required."
        )

    else:

        escalation_risk = "low"

        escalation_reasoning = (
            "Standard inquiry. No immediate escalation "
            "risks detected."
        )

    confidence = round(
        min(
            0.98,
            max(
                0.85,
                intent_confidence
                + (
                    0.04
                    if has_escalation_phrase
                    else 0.0
                ),
            ),
        ),
        2,
    )

    # ========================================================
    # 7. DYNAMIC COACHING
    # ========================================================

    suggested_action, suggested_response = (
        _generate_coaching_guidance(
            message=message,
            intent=intent,
            emotion=emotion,
            frustration_level=frustration_level,
            escalation_risk=escalation_risk,
            history=history,
        )
    )

    from app.models.schemas import ResponseEvaluation

    evaluation = ResponseEvaluation(
        tone=(
            3
            if frustration_level > 7
            else 5
        ),
        clarity=(
            4
            if "confused" in emotion
            else 5
        ),
        empathy=(
            3
            if escalation_risk in [
                "high",
                "critical",
            ]
            else 5
        ),
        professionalism=4,
    )

    if frustration_level >= 8:

        comm_tips = [
            "Acknowledge the customer's frustration immediately.",
            "Avoid using policies as an excuse.",
            "Provide a direct, concrete resolution.",
        ]

    elif escalation_risk == "medium":

        comm_tips = [
            "Address the confusion or impatience directly.",
            "Clearly outline the next steps and expected timeframe.",
        ]

    else:

        comm_tips = [
            "Maintain a polite and professional tone.",
            "Be clear and concise with next steps.",
        ]

    return IntentSentimentAnalysis(
        intent=intent,
        emotion=emotion,
        sentiment=sentiment,
        frustration_level=frustration_level,
        satisfaction_trend=satisfaction_trend,
        escalation_risk=escalation_risk,
        escalation_reasoning=escalation_reasoning,
        confidence=confidence,
        suggested_action=suggested_action,
        suggested_response=suggested_response,
        response_evaluation=evaluation,
        communication_tips=comm_tips,
    )


# ============================================================
# DYNAMIC PER-MESSAGE COACHING GUIDANCE
# ============================================================

def _generate_coaching_guidance(
    message: str,
    intent: str,
    emotion: str,
    frustration_level: int,
    escalation_risk: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> tuple[str, str]:

    msg_lower = message.lower()

    history = history or []

    # --------------------------------------------------------
    # Conversation Stage
    # --------------------------------------------------------

    customer_turns = [
        m
        for m in history
        if m.get("role") == "user"
    ]

    agent_turns = [
        m
        for m in history
        if m.get("role") == "assistant"
    ]

    turn_number = (
        len(customer_turns) + 1
    )

    last_agent_msg = (
        agent_turns[-1]
        .get("content", "")
        .lower()
        if agent_turns
        else ""
    )

    # --------------------------------------------------------
    # Agent state flags
    # --------------------------------------------------------

    agent_has_asked_for_info = any(
        keyword in last_agent_msg
        for keyword in [
            "order number",
            "order id",
            "email",
            "could you share",
            "could you provide",
            "verify",
            "may i have",
            "please provide",
            "your account",
            "reference",
        ]
    )

    agent_has_confirmed_action = any(
        keyword in last_agent_msg
        for keyword in [
            "initiated",
            "processed",
            "cancelled",
            "refund",
            "issued",
            "dispatched",
            "shipped",
            "sent to your email",
            "released",
            "reversed",
            "resolved",
            "waived",
            "expedited",
            "upgraded",
            "locked",
            "reset link",
            "credited",
        ]
    )

    agent_is_investigating = any(
        keyword in last_agent_msg
        for keyword in [
            "checking",
            "looking into",
            "investigating",
            "will check",
            "let me look",
            "pulling up",
            "reviewing",
            "i am looking",
            "coordinating",
            "one moment",
        ]
    )

    # --------------------------------------------------------
    # Customer state flags
    # --------------------------------------------------------

    customer_has_given_info = bool(
        re.search(
            r"[\w\.-]+@[\w\.-]+\.\w+",
            msg_lower,
        )
        or re.search(
            r"(order|ref|ticket|#)\s*[:#]?"
            r"\s*[a-zA-Z0-9\-]{4,}",
            msg_lower,
        )
        or "my email" in msg_lower
        or "registered email" in msg_lower
        or "order number is" in msg_lower
        or "reference number" in msg_lower
    )

    customer_is_still_waiting = any(
        keyword in msg_lower
        for keyword in [
            "still waiting",
            "nothing yet",
            "no update",
            "not received",
            "still not",
            "any update",
            "what's happening",
            "how much longer",
            "any progress",
            "haven't heard",
            "checked yet",
        ]
    )

    customer_is_escalating = (
        escalation_risk == "high"
        or any(
            keyword in msg_lower
            for keyword in [
                "manager",
                "supervisor",
                "legal",
                "lawsuit",
                "complaint",
                "bbb",
                "authorize",
                "someone who can",
                "report you",
                "consumer court",
            ]
        )
    )

    customer_is_satisfied = (
        any(
            keyword in msg_lower
            for keyword in [
                "thank you so much",
                "thank you very much",
                "problem is solved",
                "issue is solved",
                "such a relief",
                "appreciate your help",
                "solved now",
                "fixed now",
                "resolved now",
                "all sorted out",
                "that worked",
                "great help",
                "wonderful",
            ]
        )
        and frustration_level <= 2
    )

    # ========================================================
    # PRIORITY 1 - RESOLUTION CONFIRMED
    # ========================================================

    if customer_is_satisfied and not any(
        keyword in msg_lower
        for keyword in [
            "not resolved",
            "still",
            "unresolved",
        ]
    ):

        action = (
            "Resolution confirmed. Warmly acknowledge "
            "the customer's satisfaction and offer help "
            "for any future needs."
        )

        response = (
            "You're very welcome! I'm glad we were able "
            "to get this sorted out for you. Please don't "
            "hesitate to reach out if there's anything "
            "else we can help with."
        )

        return action, response

    # ========================================================
    # PRIORITY 2 - ESCALATION
    # ========================================================

    if customer_is_escalating:

        if (
            turn_number >= 3
            and agent_is_investigating
        ):

            action = (
                "Customer has escalated after repeated "
                "stalls. Stop investigating and take "
                "immediate ownership. Provide a direct "
                "resolution or supervisor escalation."
            )

            response = (
                "I completely understand your frustration, "
                "and I sincerely apologize that this has "
                "taken longer than it should. I am taking "
                "ownership of your case now and will make "
                "sure it receives immediate attention."
            )

        else:

            action = (
                "De-escalate immediately. Acknowledge "
                "the customer's frustration, validate "
                "their concern, and provide a clear next step."
            )

            response = (
                "I sincerely apologize for the frustration "
                "this has caused. I am personally taking "
                "ownership of your case right now and will "
                "make sure we work toward a clear resolution."
            )

        return action, response

    # ========================================================
    # PRIORITY 3 - CUSTOMER STILL WAITING
    # ========================================================

    if (
        customer_is_still_waiting
        and agent_has_confirmed_action
    ):

        action = (
            "Customer is following up on a previously "
            "confirmed action. Provide the current status, "
            "timestamp if available, and a clear ETA."
        )

        response = (
            "I can see the action was recorded on our end. "
            "Let me confirm the current processing status "
            "and provide you with the latest update and "
            "expected completion time."
        )

        return action, response

    if (
        customer_is_still_waiting
        and agent_is_investigating
    ):

        action = (
            "Customer is growing impatient with an open "
            "investigation. Provide the best available "
            "status and a specific next step instead of "
            "giving another vague waiting message."
        )

        response = (
            "I apologize for the wait. I have your case "
            "open and I am reviewing the current status. "
            "Here is the latest update: [insert update]. "
            "The next step is [specific action], with an "
            "expected timeframe of [timeframe]."
        )

        return action, response

    # ========================================================
    # PRIORITY 4 - CUSTOMER PROVIDED INFORMATION
    # ========================================================

    if customer_has_given_info:

        action = (
            "Customer has provided an identifier. Verify "
            "the account or order details and provide a "
            "concrete status update without requesting "
            "the same information again."
        )

        response = (
            f"Thank you for providing that information. "
            f"I'll use it to review your "
            f"{intent.replace('_', ' ')} details and "
            f"confirm the current status and next step."
        )

        return action, response

    # ========================================================
    # PRIORITY 5 - INFORMATION REQUESTED PREVIOUSLY
    # ========================================================

    if (
        agent_has_asked_for_info
        and not customer_has_given_info
        and turn_number >= 2
    ):

        action = (
            "The customer has not yet provided the "
            "requested information. Re-ask clearly and "
            "explain why it is needed to resolve the issue."
        )

        response = (
            "To resolve this as quickly as possible, "
            "I just need your order number or the email "
            "address on your account. This will allow me "
            "to check the exact details and take the next "
            "step."
        )

        return action, response

    # ========================================================
    # REFUND
    # ========================================================

    if intent == "refund":

        if turn_number == 1:

            if (
                frustration_level >= 6
                or emotion in [
                    "angry",
                    "frustrated",
                ]
            ):

                action = (
                    "First contact for an angry refund "
                    "request. Acknowledge the customer's "
                    "dissatisfaction and request the order "
                    "reference needed to investigate."
                )

                response = (
                    "I'm sorry to hear about this experience. "
                    "I understand why you want this resolved "
                    "quickly. Please share your order number "
                    "so I can check the refund details."
                )

            else:

                action = (
                    "Acknowledge the refund request and "
                    "request the order reference needed "
                    "to review the transaction."
                )

                response = (
                    "I'd be happy to help with your refund "
                    "request. Please share your order number "
                    "so I can check the transaction and "
                    "guide you through the next step."
                )

        elif turn_number == 2:

            if agent_has_asked_for_info:

                action = (
                    "The refund case is in progress. "
                    "Confirm the information needed and "
                    "explain the expected processing timeline."
                )

                response = (
                    "Once I have the order number, I can "
                    "check the transaction and confirm the "
                    "next step for your refund request."
                )

            else:

                action = (
                    "Follow up on the refund request by "
                    "confirming eligibility and providing "
                    "a clear processing timeline."
                )

                response = (
                    "I'm reviewing your refund request now. "
                    "I'll confirm the eligibility and "
                    "processing timeline based on the "
                    "transaction details."
                )

        else:

            action = (
                "The refund case has continued across "
                "multiple turns. Provide a concrete status "
                "and avoid repeatedly asking for information."
            )

            response = (
                "I understand this has taken longer than "
                "expected. I'm reviewing the current refund "
                "status and will provide the confirmed "
                "outcome and next step."
            )

        return action, response

    # ========================================================
    # CANCELLATION
    # ========================================================

    if intent == "cancellation":

        if turn_number == 1:

            action = (
                "Confirm the cancellation request and "
                "check the current order or subscription "
                "status before taking the next action."
            )

            response = (
                "I can help with that cancellation. I'll "
                "check the current order or subscription "
                "status and confirm the next step for you."
            )

        elif agent_has_confirmed_action:

            action = (
                "Cancellation has already been confirmed. "
                "Provide the cancellation reference and "
                "explain any remaining processing time."
            )

            response = (
                "Your cancellation has been processed. "
                "I'll provide the confirmation reference "
                "and any remaining processing details."
            )

        else:

            action = (
                "Follow up on the cancellation request "
                "and confirm whether it has been completed."
            )

            response = (
                "I'm checking the cancellation status now "
                "and will confirm whether it has been "
                "fully processed."
            )

        return action, response

    # ========================================================
    # DELIVERY
    # ========================================================

    if intent == "delivery_issue":

        if turn_number == 1:

            action = (
                "Check the delivery or tracking status "
                "and provide the latest available update."
            )

            response = (
                "I'm sorry to hear your delivery is delayed. "
                "Please share your order number so I can "
                "check the latest tracking status and "
                "expected delivery time."
            )

        elif turn_number == 2:

            if any(
                keyword in msg_lower
                for keyword in [
                    "tracking",
                    "where",
                    "status",
                ]
            ):

                action = (
                    "Customer is requesting another tracking "
                    "update. Provide the latest available "
                    "location and delivery estimate."
                )

                response = (
                    "I'll check the latest tracking status "
                    "and provide the current location and "
                    "expected delivery window."
                )

            else:

                action = (
                    "Acknowledge the delivery delay and "
                    "provide a concrete update or next step."
                )

                response = (
                    "I understand how frustrating a delayed "
                    "delivery can be. I'll check the current "
                    "shipment status and provide the next "
                    "available update."
                )

        else:

            action = (
                "The delivery issue remains unresolved after "
                "multiple turns. Provide a definitive status "
                "or escalate the case for resolution."
            )

            response = (
                "I understand the delay has been frustrating. "
                "I'm escalating the delivery issue for a "
                "definitive status and next resolution step."
            )

        return action, response

    # ========================================================
    # PAYMENT
    # ========================================================

    if intent == "payment_issue":

        if turn_number == 1:

            if (
                "twice" in msg_lower
                or "double" in msg_lower
            ):

                action = (
                    "Review the payment records to determine "
                    "whether the customer experienced a "
                    "duplicate charge or temporary hold."
                )

                response = (
                    "I understand why seeing two charges "
                    "would be concerning. I'll review the "
                    "payment records and confirm whether "
                    "this is a duplicate charge or a "
                    "temporary authorization."
                )

            elif "declined" in msg_lower:

                action = (
                    "Review the payment failure and provide "
                    "clear guidance on the likely cause or "
                    "alternative payment option."
                )

                response = (
                    "I'm sorry your payment didn't go through. "
                    "I'll help identify the reason for the "
                    "decline and guide you through the next "
                    "available option."
                )

            else:

                action = (
                    "Review the payment or billing details "
                    "and explain the charge clearly."
                )

                response = (
                    "I'll review the billing details and "
                    "explain the charge clearly so we can "
                    "identify the correct next step."
                )

        elif turn_number == 2:

            action = (
                "Provide the result of the payment review "
                "and clearly explain the confirmed outcome."
            )

            response = (
                "I've reviewed the payment information. "
                "Here is the confirmed status: "
                "[insert payment status and next step]."
            )

        else:

            action = (
                "Resolve the payment issue after multiple "
                "turns by confirming the final outcome and "
                "any applicable reference number."
            )

            response = (
                "I understand this needs to be resolved. "
                "I'll provide the confirmed payment outcome "
                "and the next step for completing the correction."
            )

        return action, response

    # ========================================================
    # ACCOUNT / SECURITY
    # ========================================================

    if intent == "account_issue":

        if any(
            keyword in msg_lower
            for keyword in [
                "unauthorized",
                "hacked",
                "security",
                "stolen",
                "breach",
            ]
        ):

            action = (
                "Treat the account issue as a security priority. "
                "Verify the customer and guide them through "
                "secure account recovery steps."
            )

            response = (
                "I'm treating this as an urgent account "
                "security matter. Please follow the secure "
                "verification and recovery process, and do "
                "not share passwords or verification codes."
            )

        elif turn_number == 1:

            action = (
                "Help the customer regain account access "
                "using the standard secure recovery process."
            )

            response = (
                "I can help you regain access to your account. "
                "I'll guide you through the secure password "
                "reset and account recovery steps."
            )

        else:

            action = (
                "Follow up on the account recovery process "
                "and confirm whether the customer regained access."
            )

            response = (
                "Have you been able to access your account "
                "using the recovery steps? If not, I'll help "
                "you with the next recovery option."
            )

        return action, response

    # ========================================================
    # RETURN / EXCHANGE
    # ========================================================

    if intent == "return_exchange":

        if turn_number == 1:

            if any(
                keyword in msg_lower
                for keyword in [
                    "broken",
                    "defective",
                    "damaged",
                    "shattered",
                ]
            ):

                action = (
                    "Acknowledge the damaged or defective "
                    "item and guide the customer through "
                    "the appropriate replacement or return process."
                )

                response = (
                    "I'm sorry the item arrived damaged. "
                    "I'll help you with the return or "
                    "replacement process and explain the "
                    "next step clearly."
                )

            else:

                action = (
                    "Confirm the return or exchange request "
                    "and explain the applicable next step."
                )

                response = (
                    "I'd be happy to help with your return "
                    "or exchange. I'll check the request "
                    "details and explain the next step."
                )

        else:

            action = (
                "Follow up on the return or exchange request "
                "and provide the latest status and timeline."
            )

            response = (
                "I'll check the current return or exchange "
                "status and provide the latest update and "
                "expected timeline."
            )

        return action, response

    # ========================================================
    # COMPLAINT
    # ========================================================

    if intent == "complaint":

        if turn_number == 1:

            action = (
                "Acknowledge the complaint without becoming "
                "defensive, document the concern, and provide "
                "a concrete next step."
            )

            response = (
                "I sincerely apologize for the experience "
                "you've had. I understand your concern, "
                "and I'll help identify the appropriate "
                "next step to address it."
            )

        else:

            action = (
                "Follow up on the complaint by focusing "
                "on the concrete remedy rather than repeating "
                "the apology."
            )

            response = (
                "I understand your concern. I'll focus on "
                "the specific action needed to address the "
                "issue and provide a clear update."
            )

        return action, response

    # ========================================================
    # IMPATIENT / FRUSTRATED
    # ========================================================

    if (
        emotion in [
            "impatient",
            "frustrated",
        ]
        or customer_is_still_waiting
    ):

        if turn_number >= 3:

            action = (
                "Customer remains frustrated after multiple "
                "turns. Stop repeating generic assurances "
                "and provide a concrete resolution or escalation."
            )

            response = (
                "I understand your frustration, and I don't "
                "want to keep you waiting without a clear "
                "answer. I'll provide the current status and "
                "the specific next step now."
            )

        else:

            action = (
                "Acknowledge the customer's frustration, "
                "provide a clear next step, and avoid vague "
                "waiting statements."
            )

            response = (
                "I understand your frustration and apologize "
                "for the delay. I'll focus on the specific "
                "next step and provide a clear update."
            )

        return action, response

    # ========================================================
    # CONFUSED
    # ========================================================

    if emotion == "confused":

        action = (
            "The customer appears confused. Explain the "
            "situation using simple language and clear steps."
        )

        response = (
            "Let me explain this clearly in simple steps:\n"
            "1. [First thing that happened]\n"
            "2. [Current status]\n"
            "3. [What happens next]\n"
            "Please let me know if any step needs clarification."
        )

        return action, response

    # ========================================================
    # DEFAULT - GENERAL INQUIRY
    # ========================================================

    if turn_number == 1:

        action = (
            "Provide a direct and accurate answer to the "
            "customer's question and offer the appropriate "
            "next step."
        )

        response = (
            "I'd be happy to help with that. "
            "Here's the information you need: "
            "[specific answer]. "
            "Please let me know if you need any further help."
        )

    else:

        action = (
            "Follow up on the previous answer and check "
            "whether the customer's remaining question "
            "has been addressed."
        )

        response = (
            "I hope that clarified the issue. "
            "To summarize: [brief recap]. "
            "Please let me know if you need any further clarification."
        )

    return action, response