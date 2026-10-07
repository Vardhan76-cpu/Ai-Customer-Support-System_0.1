from collections import Counter
from statistics import mean

from app.database.database import get_connection
from app.task8.post_interaction_summary import build_post_interaction_summary


def _row_to_dict(row):
    if row is None:
        return None

    if isinstance(row, dict):
        return dict(row)

    return {key: row[key] for key in row.keys()}


def get_all_conversations():
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT id, title, summary, created_at, updated_at
            FROM conversations
            ORDER BY id ASC
            """
        )

        rows = cursor.fetchall()

        return [_row_to_dict(row) for row in rows]

    finally:
        connection.close()


def get_message_count(conversation_id):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT COUNT(*) AS count
            FROM messages
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        )

        row = cursor.fetchone()

        return int(row["count"])

    finally:
        connection.close()


def build_performance_analytics():
    conversations = get_all_conversations()

    if not conversations:
        return {
            "success": True,
            "total_sessions": 0,
            "total_messages": 0,
            "average_frustration": 0,
            "average_resolution_quality": 0,
            "average_communication_quality": 0,
            "average_empathy": 0,
            "average_guideline_adherence": 0,
            "escalation_rate": 0,
            "common_issues": [],
            "escalation_triggers": [],
            "session_insights": [],
            "improvement_indicators": {
                "sentiment_handling": "No data available",
                "response_quality": "No data available",
                "resolution_rate": "No data available",
                "escalation_frequency": "No data available",
            },
            "recommendations": [
                "Complete customer interactions to generate performance analytics."
            ],
        }

    total_messages = 0

    frustration_scores = []
    resolution_scores = []
    communication_scores = []
    empathy_scores = []
    guideline_scores = []

    primary_issues = Counter()
    escalation_triggers = Counter()

    session_insights = []

    successful_sessions = 0
    escalated_sessions = 0

    for conversation in conversations:
        conversation_id = conversation["id"]

        total_messages += get_message_count(conversation_id)

        try:
            report = build_post_interaction_summary(
                conversation_id=conversation_id
            )

            if not report.get("success"):
                continue

            resolution_quality = report.get("resolution_quality", {})

            resolution_score = float(
                resolution_quality.get("resolution_quality", 0)
            )

            communication_score = float(
                resolution_quality.get("communication_quality", 0)
            )

            empathy_score = float(
                resolution_quality.get("empathy", 0)
            )

            guideline_score = float(
                resolution_quality.get("guideline_adherence", 0)
            )

            resolution_scores.append(resolution_score)
            communication_scores.append(communication_score)
            empathy_scores.append(empathy_score)
            guideline_scores.append(guideline_score)

            primary_issue = report.get("primary_issue")

            if primary_issue:
                primary_issues[primary_issue] += 1

            sentiment_journey = report.get("sentiment_journey", [])

            session_frustrations = []

            session_escalated = False

            for turn in sentiment_journey:
                frustration = turn.get("frustration_level")

                if frustration is not None:
                    frustration = float(frustration)

                    frustration_scores.append(frustration)
                    session_frustrations.append(frustration)

                risk = str(
                    turn.get("escalation_risk", "")
                ).lower()

                if risk in {
                    "high",
                    "critical",
                }:
                    session_escalated = True
                    escalated_sessions += 1

            if session_escalated:
                escalation_triggers["High/Critical escalation risk"] += 1

            if resolution_score >= 4:
                successful_sessions += 1

            average_session_frustration = (
                mean(session_frustrations)
                if session_frustrations
                else 0
            )

            session_insights.append(
                {
                    "conversation_id": conversation_id,
                    "conversation_title": conversation.get(
                        "title",
                        "Untitled Conversation",
                    ),
                    "primary_issue": primary_issue,
                    "resolution_quality": resolution_score,
                    "communication_quality": communication_score,
                    "empathy": empathy_score,
                    "guideline_adherence": guideline_score,
                    "average_frustration": round(
                        average_session_frustration,
                        2,
                    ),
                    "customer_turn_count": report.get(
                        "customer_turn_count",
                        0,
                    ),
                    "escalation_risk": (
                        "High/Critical"
                        if session_escalated
                        else "Low/Medium"
                    ),
                }
            )

        except Exception as exc:
            session_insights.append(
                {
                    "conversation_id": conversation_id,
                    "conversation_title": conversation.get(
                        "title",
                        "Untitled Conversation",
                    ),
                    "error": str(exc),
                }
            )

    total_sessions = len(conversations)

    average_frustration = (
        mean(frustration_scores)
        if frustration_scores
        else 0
    )

    average_resolution_quality = (
        mean(resolution_scores)
        if resolution_scores
        else 0
    )

    average_communication_quality = (
        mean(communication_scores)
        if communication_scores
        else 0
    )

    average_empathy = (
        mean(empathy_scores)
        if empathy_scores
        else 0
    )

    average_guideline_adherence = (
        mean(guideline_scores)
        if guideline_scores
        else 0
    )

    resolution_rate = (
        (successful_sessions / total_sessions) * 100
        if total_sessions
        else 0
    )

    escalation_rate = (
        (escalated_sessions / total_sessions) * 100
        if total_sessions
        else 0
    )

    if average_frustration <= 3:
        sentiment_handling = "Good"
    elif average_frustration <= 6:
        sentiment_handling = "Needs Improvement"
    else:
        sentiment_handling = "Needs Strong Improvement"

    if average_communication_quality >= 4:
        response_quality = "Good"
    elif average_communication_quality >= 3:
        response_quality = "Moderate"
    else:
        response_quality = "Needs Improvement"

    if resolution_rate >= 80:
        resolution_indicator = "Good"
    elif resolution_rate >= 60:
        resolution_indicator = "Moderate"
    else:
        resolution_indicator = "Needs Improvement"

    if escalation_rate <= 20:
        escalation_indicator = "Low"
    elif escalation_rate <= 40:
        escalation_indicator = "Moderate"
    else:
        escalation_indicator = "High"

    recommendations = []

    if average_frustration > 5:
        recommendations.append(
            "Improve frustration handling by acknowledging customer concerns earlier."
        )

    if average_communication_quality < 4:
        recommendations.append(
            "Improve response clarity and provide more direct next steps."
        )

    if average_empathy < 4:
        recommendations.append(
            "Use stronger empathy and customer acknowledgement during difficult interactions."
        )

    if resolution_rate < 80:
        recommendations.append(
            "Confirm the customer's issue is fully resolved before closing the interaction."
        )

    if escalation_rate > 20:
        recommendations.append(
            "Review repeated escalation triggers and improve first-response resolution."
        )

    if not recommendations:
        recommendations.append(
            "Maintain the current support quality and continue monitoring performance trends."
        )

    return {
        "success": True,
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "average_frustration": round(
            average_frustration,
            2,
        ),
        "average_resolution_quality": round(
            average_resolution_quality,
            2,
        ),
        "average_communication_quality": round(
            average_communication_quality,
            2,
        ),
        "average_empathy": round(
            average_empathy,
            2,
        ),
        "average_guideline_adherence": round(
            average_guideline_adherence,
            2,
        ),
        "resolution_rate": round(
            resolution_rate,
            2,
        ),
        "escalation_rate": round(
            escalation_rate,
            2,
        ),
        "common_issues": [
            {
                "issue": issue,
                "count": count,
            }
            for issue, count in primary_issues.most_common()
        ],
        "escalation_triggers": [
            {
                "trigger": trigger,
                "count": count,
            }
            for trigger, count in escalation_triggers.most_common()
        ],
        "session_insights": session_insights,
        "improvement_indicators": {
            "sentiment_handling": sentiment_handling,
            "response_quality": response_quality,
            "resolution_rate": resolution_indicator,
            "escalation_frequency": escalation_indicator,
        },
        "recommendations": recommendations,
    }