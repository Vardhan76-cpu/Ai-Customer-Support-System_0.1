"""
Task 8 - Post-Interaction Summary Agent

Creates a structured summary of a completed customer-support interaction.

The module uses the existing conversation database and the existing
Intent/Sentiment analysis agent. It does not create a new database.
"""

from typing import Any, Dict, List

from app.agents.intent_sentiment import analyze_customer_message
from app.database.database import get_conversation, get_messages


# ============================================================
# Helper: safely convert SQLite rows to dictionaries
# ============================================================

def _row_to_dict(row: Any) -> Dict[str, Any]:
    """
    Convert a SQLite Row or dictionary into a normal dictionary.

    The existing database returns sqlite3.Row objects.
    sqlite3.Row does not support the .get() method, so we
    convert it before using dictionary operations.
    """

    if row is None:
        return {}

    if isinstance(row, dict):
        return row

    try:
        return dict(row)
    except (TypeError, ValueError):
        return {}


# ============================================================
# Helper: safe text conversion
# ============================================================

def _safe_text(value: Any) -> str:
    """Convert a value into safe display text."""

    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# Helper: read analysis fields
# ============================================================

def _get_analysis_value(
    analysis: Any,
    field: str,
    default: Any = None,
) -> Any:
    """
    Read a field from either a Pydantic model or a dictionary.
    """

    if analysis is None:
        return default

    if hasattr(analysis, field):
        return getattr(analysis, field)

    if isinstance(analysis, dict):
        return analysis.get(field, default)

    return default


# ============================================================
# Resolution Quality Score
# ============================================================

def _calculate_resolution_score(
    resolution_quality: float,
    communication_quality: float,
    empathy_quality: float,
    guideline_adherence: float,
) -> Dict[str, Any]:
    """
    Calculate the overall resolution quality score.

    Each component is scored from 1 to 5.
    The final score is converted to a 100-point scale.
    """

    values = [
        resolution_quality,
        communication_quality,
        empathy_quality,
        guideline_adherence,
    ]

    average_score = sum(values) / len(values)

    overall_score = round(
        (average_score / 5) * 100,
        2,
    )

    return {
        "resolution_quality": round(
            resolution_quality,
            2,
        ),
        "communication_quality": round(
            communication_quality,
            2,
        ),
        "empathy": round(
            empathy_quality,
            2,
        ),
        "guideline_adherence": round(
            guideline_adherence,
            2,
        ),
        "average_score": round(
            average_score,
            2,
        ),
        "overall_score": overall_score,
    }


# ============================================================
# Score interaction
# ============================================================

def _score_interaction(
    messages: List[Dict[str, Any]],
    analyses: List[Any],
) -> Dict[str, Any]:
    """
    Estimate interaction quality from the available
    conversation data.

    The existing Intent/Sentiment agent provides response
    evaluation values such as tone, clarity, empathy
    and professionalism.
    """

    response_evaluations = []

    for analysis in analyses:

        evaluation = _get_analysis_value(
            analysis,
            "response_evaluation",
            None,
        )

        if evaluation is None:
            continue

        tone = _get_analysis_value(
            evaluation,
            "tone",
            None,
        )

        clarity = _get_analysis_value(
            evaluation,
            "clarity",
            None,
        )

        empathy = _get_analysis_value(
            evaluation,
            "empathy",
            None,
        )

        professionalism = _get_analysis_value(
            evaluation,
            "professionalism",
            None,
        )

        values = [
            tone,
            clarity,
            empathy,
            professionalism,
        ]

        numeric_values = [
            float(value)
            for value in values
            if isinstance(value, (int, float))
        ]

        if numeric_values:

            response_evaluations.append(
                {
                    "tone": tone,
                    "clarity": clarity,
                    "empathy": empathy,
                    "professionalism": professionalism,
                }
            )

    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    resolution_quality = 3.0
    communication_quality = 3.0
    empathy_quality = 3.0
    guideline_adherence = 3.0

    # --------------------------------------------------------
    # Calculate evaluation scores
    # --------------------------------------------------------

    if response_evaluations:

        tone_values = [
            float(item["tone"])
            for item in response_evaluations
            if isinstance(
                item["tone"],
                (int, float),
            )
        ]

        clarity_values = [
            float(item["clarity"])
            for item in response_evaluations
            if isinstance(
                item["clarity"],
                (int, float),
            )
        ]

        empathy_values = [
            float(item["empathy"])
            for item in response_evaluations
            if isinstance(
                item["empathy"],
                (int, float),
            )
        ]

        professionalism_values = [
            float(item["professionalism"])
            for item in response_evaluations
            if isinstance(
                item["professionalism"],
                (int, float),
            )
        ]

        # ----------------------------------------------------
        # Communication quality
        # ----------------------------------------------------

        communication_scores = []

        if tone_values:
            communication_scores.append(
                sum(tone_values) / len(tone_values)
            )

        if clarity_values:
            communication_scores.append(
                sum(clarity_values) / len(clarity_values)
            )

        if communication_scores:
            communication_quality = (
                sum(communication_scores)
                / len(communication_scores)
            )

        # ----------------------------------------------------
        # Empathy
        # ----------------------------------------------------

        if empathy_values:
            empathy_quality = (
                sum(empathy_values)
                / len(empathy_values)
            )

        # ----------------------------------------------------
        # Guideline adherence
        # ----------------------------------------------------

        if professionalism_values:
            guideline_adherence = (
                sum(professionalism_values)
                / len(professionalism_values)
            )

    # --------------------------------------------------------
    # Resolution quality from final customer state
    # --------------------------------------------------------

    if analyses:

        final_analysis = analyses[-1]

        final_sentiment = _safe_text(
            _get_analysis_value(
                final_analysis,
                "sentiment",
                "",
            )
        ).lower()

        final_trend = _safe_text(
            _get_analysis_value(
                final_analysis,
                "satisfaction_trend",
                "",
            )
        ).lower()

        if final_sentiment in {
            "positive",
            "satisfied",
            "happy",
        }:

            resolution_quality = 4.5

        elif final_sentiment in {
            "negative",
            "frustrated",
            "angry",
            "dissatisfied",
        }:

            resolution_quality = 2.0

        elif final_trend in {
            "improving",
            "positive",
            "increasing",
        }:

            resolution_quality = 4.0

        elif final_trend in {
            "declining",
            "negative",
            "decreasing",
        }:

            resolution_quality = 2.5

        else:

            resolution_quality = 3.0

    return _calculate_resolution_score(
        resolution_quality=resolution_quality,
        communication_quality=communication_quality,
        empathy_quality=empathy_quality,
        guideline_adherence=guideline_adherence,
    )


# ============================================================
# Sentiment journey
# ============================================================

def _build_sentiment_timeline(
    analyses: List[Any],
) -> List[Dict[str, Any]]:
    """
    Create a simple customer sentiment journey timeline.
    """

    timeline = []

    for index, analysis in enumerate(
        analyses,
        start=1,
    ):

        timeline.append(
            {
                "turn": index,

                "emotion": _safe_text(
                    _get_analysis_value(
                        analysis,
                        "emotion",
                        "unknown",
                    )
                ),

                "sentiment": _safe_text(
                    _get_analysis_value(
                        analysis,
                        "sentiment",
                        "unknown",
                    )
                ),

                "frustration_level": _get_analysis_value(
                    analysis,
                    "frustration_level",
                    0,
                ),

                "satisfaction_trend": _safe_text(
                    _get_analysis_value(
                        analysis,
                        "satisfaction_trend",
                        "stable",
                    )
                ),

                "escalation_risk": _safe_text(
                    _get_analysis_value(
                        analysis,
                        "escalation_risk",
                        "low",
                    )
                ),
            }
        )

    return timeline


# ============================================================
# Strengths
# ============================================================

def _identify_strengths(
    score: Dict[str, Any],
    analyses: List[Any],
) -> List[str]:
    """
    Identify areas where the interaction performed well.
    """

    strengths = []

    if score["communication_quality"] >= 4:
        strengths.append(
            "Clear and professional communication."
        )

    if score["empathy"] >= 4:
        strengths.append(
            "Good empathy and customer acknowledgement."
        )

    if score["guideline_adherence"] >= 4:
        strengths.append(
            "Responses were aligned with professional support guidelines."
        )

    if score["resolution_quality"] >= 4:
        strengths.append(
            "The interaction moved toward a satisfactory resolution."
        )

    if not strengths and analyses:
        strengths.append(
            "The interaction was actively analyzed across customer turns."
        )

    return strengths


# ============================================================
# Weaknesses
# ============================================================

def _identify_weaknesses(
    score: Dict[str, Any],
    analyses: List[Any],
) -> List[str]:
    """
    Identify areas that may need improvement.
    """

    weaknesses = []

    if score["communication_quality"] < 3:
        weaknesses.append(
            "Communication clarity or tone needs improvement."
        )

    if score["empathy"] < 3:
        weaknesses.append(
            "Customer empathy and acknowledgement need improvement."
        )

    if score["guideline_adherence"] < 3:
        weaknesses.append(
            "Responses may need better adherence to support guidelines."
        )

    if score["resolution_quality"] < 3:
        weaknesses.append(
            "The customer issue may not have been fully resolved."
        )

    if not weaknesses and analyses:
        weaknesses.append(
            "Continue monitoring the interaction for consistency and resolution quality."
        )

    return weaknesses


# ============================================================
# Coaching recommendations
# ============================================================

def _build_coaching_recommendations(
    score: Dict[str, Any],
    weaknesses: List[str],
) -> List[str]:
    """
    Generate simple personalized coaching recommendations.
    """

    recommendations = []

    if score["communication_quality"] < 4:
        recommendations.append(
            "Use shorter, clearer responses and explain the next action explicitly."
        )

    if score["empathy"] < 4:
        recommendations.append(
            "Acknowledge the customer's concern before giving the solution."
        )

    if score["guideline_adherence"] < 4:
        recommendations.append(
            "Follow the recommended support workflow and verify important details."
        )

    if score["resolution_quality"] < 4:
        recommendations.append(
            "Confirm whether the customer's issue is actually resolved before closing."
        )

    if not recommendations:
        recommendations.append(
            "Maintain the current communication quality and continue confirming resolution."
        )

    return recommendations


# ============================================================
# Find primary issue
# ============================================================

def _find_primary_issue(
    analyses: List[Any],
) -> str:
    """
    Find the most common detected customer intent.
    """

    intents = []

    for analysis in analyses:

        intent = _safe_text(
            _get_analysis_value(
                analysis,
                "intent",
                "",
            )
        )

        if intent:
            intents.append(intent)

    if not intents:
        return "General customer support issue"

    counts = {}

    for intent in intents:

        normalized = intent.lower()

        counts[normalized] = (
            counts.get(normalized, 0) + 1
        )

    primary = max(
        counts,
        key=counts.get,
    )

    for intent in intents:

        if intent.lower() == primary:
            return intent

    return primary


# ============================================================
# Find final resolution
# ============================================================

def _find_final_resolution(
    messages: List[Dict[str, Any]],
    analyses: List[Any],
) -> str:
    """
    Describe the final resolution state.
    """

    if not analyses:
        return (
            "Resolution status could not be determined."
        )

    final_analysis = analyses[-1]

    suggested_action = _safe_text(
        _get_analysis_value(
            final_analysis,
            "suggested_action",
            "",
        )
    )

    final_sentiment = _safe_text(
        _get_analysis_value(
            final_analysis,
            "sentiment",
            "",
        )
    )

    final_trend = _safe_text(
        _get_analysis_value(
            final_analysis,
            "satisfaction_trend",
            "",
        )
    )

    if suggested_action:

        if final_sentiment:

            return (
                f"Recommended final action: "
                f"{suggested_action}. "
                f"Final customer sentiment: "
                f"{final_sentiment}."
            )

        return (
            f"Recommended final action: "
            f"{suggested_action}."
        )

    if final_sentiment:

        return (
            f"Final customer sentiment was "
            f"{final_sentiment}. "
            f"Satisfaction trend: "
            f"{final_trend or 'stable'}."
        )

    return (
        "Resolution status could not be determined."
    )


# ============================================================
# Main Task 8 function
# ============================================================

def build_post_interaction_summary(
    conversation_id: int,
) -> Dict[str, Any]:
    """
    Build the complete post-interaction report for one conversation.

    This function reads the existing conversation and messages,
    analyzes customer turns using the existing Intent/Sentiment
    agent, and returns a structured report for the UI/API.
    """

    # --------------------------------------------------------
    # Get conversation
    # --------------------------------------------------------

    conversation = get_conversation(
        conversation_id
    )

    if conversation is None:

        raise ValueError(
            f"Conversation {conversation_id} was not found."
        )

    # IMPORTANT:
    # SQLite returns sqlite3.Row.
    # Convert it into a normal dictionary.
    conversation = _row_to_dict(
        conversation
    )

    # --------------------------------------------------------
    # Get messages
    # --------------------------------------------------------

    messages = get_messages(
        conversation_id
    )

    if messages is None:
        messages = []

    # IMPORTANT:
    # Convert every SQLite Row into a dictionary.
    messages = [
        _row_to_dict(message)
        for message in messages
    ]

    # --------------------------------------------------------
    # Get customer messages
    # --------------------------------------------------------

    customer_messages = [
        message
        for message in messages
        if _safe_text(
            message.get("role")
        ).lower() == "user"
    ]

    # --------------------------------------------------------
    # Analyze customer turns
    # --------------------------------------------------------

    analyses = []

    history = []

    for message in customer_messages:

        customer_text = _safe_text(
            message.get("content")
        )

        if not customer_text:
            continue

        try:

            analysis = analyze_customer_message(
                message=customer_text,
                history=history,
            )

            analyses.append(
                analysis
            )

        except Exception as exc:

            # One failed analysis should not stop
            # the complete report.

            print(
                "TASK 8 CUSTOMER ANALYSIS ERROR:",
                repr(exc),
            )

            continue

        history.append(
            {
                "role": "user",
                "content": customer_text,
            }
        )

    # --------------------------------------------------------
    # Build report sections
    # --------------------------------------------------------

    primary_issue = _find_primary_issue(
        analyses
    )

    final_resolution = _find_final_resolution(
        messages=messages,
        analyses=analyses,
    )

    score = _score_interaction(
        messages=messages,
        analyses=analyses,
    )

    strengths = _identify_strengths(
        score=score,
        analyses=analyses,
    )

    weaknesses = _identify_weaknesses(
        score=score,
        analyses=analyses,
    )

    coaching = _build_coaching_recommendations(
        score=score,
        weaknesses=weaknesses,
    )

    sentiment_timeline = _build_sentiment_timeline(
        analyses
    )

    # --------------------------------------------------------
    # Final structured response
    # --------------------------------------------------------

    return {
        "success": True,

        "conversation_id": conversation_id,

        "conversation_title": _safe_text(
            conversation.get("title")
        ),

        "interaction_summary": (
            f"The interaction contained "
            f"{len(messages)} messages with "
            f"{len(customer_messages)} customer turns. "
            f"The primary issue was identified as "
            f"'{primary_issue}'."
        ),

        "primary_issue": primary_issue,

        "final_resolution": final_resolution,

        "customer_turn_count": len(
            customer_messages
        ),

        "total_message_count": len(
            messages
        ),

        "sentiment_journey": (
            sentiment_timeline
        ),

        "resolution_quality": score,

        "strengths": strengths,

        "weaknesses": weaknesses,

        "coaching_recommendations": coaching,
    }