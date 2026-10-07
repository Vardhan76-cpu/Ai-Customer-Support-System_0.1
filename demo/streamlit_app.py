import streamlit as st
import requests
import json
import os
from app.models.schemas import SimulatorConfig

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ==================== TASK 8 ====================

if "task8_report" not in st.session_state:
    st.session_state.task8_report = None


st.set_page_config(
    page_title="AI Customer Support Coaching",
    page_icon="🎧",
    layout="wide",
)


st.sidebar.title("🎛️ Navigation & Modes")

app_mode = st.sidebar.radio(
    "Select Operating Mode:",
    [
        "🏛️ Live Support Console",
        "📚 Support Knowledge Base & Ingestion",
        "📊 Post-Interaction Report",
        "📈 Performance Analytics",
    ],
    index=0,
)

st.sidebar.divider()


# ============================================================
# LIVE SUPPORT - PANEL 2
# ============================================================

def render_panel_2(analysis, escalation_threshold):

    st.subheader("📊 Panel 2: Live Coaching Feed")

    st.caption(
        "Real-time sentiment, frustration scoring & intervention tips"
    )

    if not analysis:

        st.info(
            "📊 Coaching analytics will appear here once "
            "a customer turn is generated."
        )

        return

    risk = analysis.get(
        "escalation_risk",
        "low",
    ).lower()

    reasoning = analysis.get(
        "escalation_reasoning",
        "",
    )

    threshold = escalation_threshold.lower()

    risk_levels = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    if risk_levels.get(
        risk,
        1,
    ) >= risk_levels.get(
        threshold,
        3,
    ):

        st.toast(
            f"🚨 ALERT: Escalation Risk is {risk.upper()}!",
            icon="🚨",
        )

        st.error(
            f"**🚨 GLOBAL ESCALATION ALERT "
            f"({risk.upper()}):** {reasoning}"
        )

    if risk == "critical":

        st.error(
            f"💥 **ESCALATION RISK: CRITICAL**\n"
            f"{reasoning}\n"
            f"*Immediate supervisor intervention required!*"
        )

    elif risk == "high":

        st.error(
            f"🚨 **ESCALATION RISK: HIGH**\n"
            f"{reasoning}\n"
            f"*Empathize and offer direct resolution "
            f"or supervisor queue.*"
        )

    elif risk == "medium":

        st.warning(
            f"⚠️ **ESCALATION RISK: MEDIUM**\n"
            f"{reasoning}\n"
            f"*Provide clear timelines and concrete next actions.*"
        )

    else:

        st.success(
            f"🛡️ **ESCALATION RISK: LOW**\n"
            f"{reasoning}\n"
            f"*Maintain polite, efficient communication.*"
        )

    m1, m2 = st.columns(2)

    with m1:

        st.metric(
            "Detected Intent",
            analysis.get(
                "intent",
                "N/A",
            ).upper().replace(
                "_",
                " ",
            ),
        )

    with m2:

        sentiment = analysis.get(
            "sentiment",
            "neutral",
        ).upper()

        sentiment_icon = (
            "🟢"
            if sentiment == "POSITIVE"
            else (
                "🔴"
                if sentiment == "NEGATIVE"
                else "⚪"
            )
        )

        st.metric(
            "Sentiment",
            f"{sentiment_icon} {sentiment}",
        )

    m3, m4 = st.columns(2)

    with m3:

        emotion_icon = {
            "happy": "😊",
            "satisfied": "✨",
            "neutral": "😐",
            "confused": "😕",
            "worried": "😟",
            "impatient": "😤",
            "frustrated": "😠",
            "angry": "😡",
            "relieved": "😌",
        }.get(
            analysis.get(
                "emotion",
                "neutral",
            ),
            "👤",
        )

        st.metric(
            "Customer Emotion",
            f"{emotion_icon} "
            f"{analysis.get('emotion', 'N/A').title()}",
        )

    with m4:

        trend = analysis.get(
            "satisfaction_trend",
            "stable",
        ).lower()

        trend_icon = (
            "📈"
            if trend == "improving"
            else (
                "📉"
                if trend == "declining"
                else "➡️"
            )
        )

        st.metric(
            "Satisfaction Trend",
            f"{trend_icon} {trend.title()}",
        )

    frustration = analysis.get(
        "frustration_level",
        0,
    )

    st.markdown(
        f"**Frustration Level: {frustration} / 10**"
    )

    st.progress(
        min(
            1.0,
            max(
                0.0,
                frustration / 10.0,
            ),
        )
    )

    st.caption(
        f"Classification Confidence: "
        f"{int(analysis.get('confidence', 0.9) * 100)}%"
    )

    st.divider()

    st.markdown(
        "💡 **Suggested Agent Action** "
        "*(Turn Specific — Changes Each Message)*"
    )

    action_text = analysis.get(
        "suggested_action"
    ) or ""

    if action_text:

        st.info(
            action_text
        )

    else:

        st.info(
            "Acknowledge customer concerns empathetically "
            "and provide prompt assistance."
        )

    comm_tips = analysis.get(
        "communication_tips",
        [],
    )

    if comm_tips:

        st.markdown(
            "🎯 **Actionable Communication Tips**"
        )

        for tip in comm_tips:

            st.markdown(
                f"- {tip}"
            )

    eval_data = analysis.get(
        "response_evaluation",
        {},
    )

    if eval_data:

        st.markdown(
            "📝 **Response Evaluation**"
        )

        e1, e2, e3, e4 = st.columns(4)

        with e1:

            st.metric(
                "Tone",
                f"{eval_data.get('tone', 5)}/5",
            )

        with e2:

            st.metric(
                "Clarity",
                f"{eval_data.get('clarity', 5)}/5",
            )

        with e3:

            st.metric(
                "Empathy",
                f"{eval_data.get('empathy', 5)}/5",
            )

        with e4:

            st.metric(
                "Professional",
                f"{eval_data.get('professionalism', 5)}/5",
            )

    suggested_resp = analysis.get(
        "suggested_response"
    ) or ""

    if suggested_resp:

        st.markdown(
            "💬 **Recommended Agent Response Template**"
        )

        st.code(
            suggested_resp,
            language="markdown",
        )

    with st.expander(
        "🔍 View Raw Analysis JSON"
    ):

        st.json(
            analysis
        )


# ============================================================
# LIVE SUPPORT - PANEL 3
# ============================================================

def render_panel_3(recs):

    st.subheader(
        "📚 Panel 3: Knowledge Recommendations"
    )

    st.caption(
        "Contextually retrieved support articles, FAQs & policies (RAG)"
    )

    if not recs:

        st.info(
            "📚 Knowledge recommendations will appear here."
        )

        return

    st.markdown(
        f"**Surfaced {len(recs)} Knowledge Assets:**"
    )

    for i, item in enumerate(
        recs,
        1,
    ):

        category = item.get(
            "category",
            "Support Article",
        )

        badge = {
            "Policy": "⚖️ Policy",
            "FAQ": "❓ FAQ",
            "Troubleshooting": "🔧 Troubleshooting",
            "Support Article": "📄 Article",
        }.get(
            category,
            "📄 Article",
        )

        with st.expander(
            f"{i}. {badge} | "
            f"{item.get('title', 'Document')}"
        ):

            st.markdown(
                f"**Relevance Score:** "
                f"`{round(item.get('score', 0.0), 3)}`"
            )

            actionable = item.get(
                "actionable_summary",
                "",
            )

            if actionable:

                st.markdown(
                    "**Actionable Summary:**"
                )

                st.success(
                    actionable
                )

            st.markdown(
                "**Excerpt:**"
            )

            st.info(
                item.get("snippet")
                or item.get("text", "")
            )

            st.caption(
                f"Source: "
                f"`{item.get('document_name', 'Unknown')}`"
            )


# ============================================================
# CHAT HISTORY
# ============================================================

def render_chat_history(messages):

    customer_turn_idx = 0

    for msg in messages:

        role = msg["role"]

        is_agent = role == "assistant"

        if not is_agent:

            customer_turn_idx += 1

        with st.chat_message(
            role,
            avatar="🧑‍💼" if is_agent else "👤",
        ):

            if is_agent:

                st.markdown(
                    "**Support Agent:**"
                )

            else:

                st.markdown(
                    f"**Customer** "
                    f"*(Turn {customer_turn_idx})*:"
                )

            st.write(
                msg["content"]
            )

            if (
                not is_agent
                and msg.get("analysis")
            ):

                a = msg["analysis"]

                sentiment_color = (
                    "🔴"
                    if a.get("sentiment") == "negative"
                    else (
                        "🟢"
                        if a.get("sentiment") == "positive"
                        else "⚪"
                    )
                )

                st.caption(
                    f"🏷️ "
                    f"`{a.get('intent', '').replace('_', ' ').upper()}` | "
                    f"{sentiment_color} "
                    f"{a.get('sentiment', '').title()} | "
                    f"Emotion: *{a.get('emotion', '').title()}* | "
                    f"Frustration: "
                    f"**{a.get('frustration_level', 0)}/10** | "
                    f"Risk: "
                    f"**{a.get('escalation_risk', 'low').upper()}**"
                )


# ============================================================
# TASK 8 - PERFORMANCE ANALYTICS DASHBOARD
# ============================================================

def render_performance_analytics():

    st.title(
        "📈 Performance Analytics Dashboard"
    )

    st.caption(
        "Task 8 - Overall interaction, resolution, "
        "escalation, and improvement analytics."
    )

    st.divider()

    # --------------------------------------------------------
    # LOAD ANALYTICS
    # --------------------------------------------------------

    with st.spinner(
        "Loading performance analytics..."
    ):

        try:

            response = requests.get(
                f"{API_URL}/task8/performance-analytics",
                timeout=60,
            )

            if response.status_code != 200:

                st.error(
                    f"Unable to load performance analytics. "
                    f"HTTP {response.status_code}"
                )

                try:

                    st.json(
                        response.json()
                    )

                except Exception:

                    st.code(
                        response.text
                    )

                return

            data = response.json()

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the FastAPI server. "
                "Make sure the API is running on "
                "http://127.0.0.1:8000"
            )

            return

        except requests.exceptions.Timeout:

            st.error(
                "The performance analytics request timed out."
            )

            return

        except Exception as exc:

            st.error(
                f"Unexpected error while loading analytics: {exc}"
            )

            return

    if not data.get(
        "success",
        False,
    ):

        st.error(
            "Performance analytics could not be generated."
        )

        with st.expander(
            "🔍 View API Response"
        ):

            st.json(
                data
            )

        return

    # ========================================================
    # OVERALL PERFORMANCE
    # ========================================================

    st.header(
        "📊 Overall Performance"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Sessions",
            data.get(
                "total_sessions",
                0,
            ),
        )

    with col2:

        st.metric(
            "Total Messages",
            data.get(
                "total_messages",
                0,
            ),
        )

    with col3:

        st.metric(
            "Resolution Rate",
            f"{float(data.get('resolution_rate', 0)):.1f}%",
        )

    with col4:

        st.metric(
            "Escalation Rate",
            f"{float(data.get('escalation_rate', 0)):.1f}%",
        )

    st.divider()

    # ========================================================
    # QUALITY METRICS
    # ========================================================

    st.header(
        "⭐ Response Quality Metrics"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Avg. Frustration",
            f"{float(data.get('average_frustration', 0)):.2f}/10",
        )

    with col2:

        st.metric(
            "Resolution Quality",
            f"{float(data.get('average_resolution_quality', 0)):.2f}/5",
        )

    with col3:

        st.metric(
            "Communication Quality",
            f"{float(data.get('average_communication_quality', 0)):.2f}/5",
        )

    with col4:

        st.metric(
            "Empathy",
            f"{float(data.get('average_empathy', 0)):.2f}/5",
        )

    # --------------------------------------------------------
    # GUIDELINE ADHERENCE
    # --------------------------------------------------------

    guideline_score = float(
        data.get(
            "average_guideline_adherence",
            0,
        )
    )

    st.subheader(
        "📘 Guideline Adherence"
    )

    st.progress(
        min(
            1.0,
            max(
                0.0,
                guideline_score / 5.0,
            ),
        )
    )

    st.write(
        f"Average guideline adherence: "
        f"**{guideline_score:.2f}/5**"
    )

    st.divider()

    # ========================================================
    # COMMON CUSTOMER ISSUES
    # ========================================================

    st.header(
        "🎯 Common Customer Issues"
    )

    common_issues = data.get(
        "common_issues",
        [],
    )

    if common_issues:

        if all(
            isinstance(item, dict)
            for item in common_issues
        ):

            st.dataframe(
                common_issues,
                use_container_width=True,
                hide_index=True,
            )

        else:

            issue_rows = []

            for item in common_issues:

                issue_rows.append(
                    {
                        "Issue": str(item),
                    }
                )

            st.dataframe(
                issue_rows,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "No recurring customer issues "
            "have been identified yet."
        )

    st.divider()

    # ========================================================
    # ESCALATION TRIGGERS
    # ========================================================

    st.header(
        "🚨 Common Escalation Triggers"
    )

    escalation_triggers = data.get(
        "escalation_triggers",
        [],
    )

    if escalation_triggers:

        if all(
            isinstance(item, dict)
            for item in escalation_triggers
        ):

            st.dataframe(
                escalation_triggers,
                use_container_width=True,
                hide_index=True,
            )

        else:

            trigger_rows = []

            for item in escalation_triggers:

                trigger_rows.append(
                    {
                        "Trigger": str(item),
                    }
                )

            st.dataframe(
                trigger_rows,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "No major escalation triggers identified."
        )

    st.divider()

    # ========================================================
    # SESSION LEVEL INSIGHTS
    # ========================================================

    st.header(
        "🔎 Session-Level Insights"
    )

    session_insights = data.get(
        "session_insights",
        [],
    )

    if session_insights:

        for index, insight in enumerate(
            session_insights,
            start=1,
        ):

            with st.expander(
                f"Session Insight {index}"
            ):

                if isinstance(
                    insight,
                    dict,
                ):

                    st.json(
                        insight
                    )

                else:

                    st.write(
                        insight
                    )

    else:

        st.info(
            "No session-level insights available."
        )

    st.divider()

    # ========================================================
    # IMPROVEMENT INDICATORS
    # ========================================================

    st.header(
        "📈 Improvement Indicators"
    )

    indicators = data.get(
        "improvement_indicators",
        {},
    )

    if indicators:

        indicator_rows = []

        for name, value in indicators.items():

            indicator_rows.append(
                {
                    "Indicator": name,
                    "Value": value,
                }
            )

        st.dataframe(
            indicator_rows,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No improvement indicators available."
        )

    st.divider()

    # ========================================================
    # ACTIONABLE RECOMMENDATIONS
    # ========================================================

    st.header(
        "🎓 Actionable Recommendations"
    )

    recommendations = data.get(
        "recommendations",
        [],
    )

    if recommendations:

        for index, recommendation in enumerate(
            recommendations,
            start=1,
        ):

            st.info(
                f"**Recommendation {index}:** "
                f"{recommendation}"
            )

    else:

        st.info(
            "No additional recommendations available."
        )

    st.divider()

    # ========================================================
    # RAW ANALYTICS JSON
    # ========================================================

    with st.expander(
        "🔍 View Complete Analytics JSON"
    ):

        st.json(
            data
        )


# ============================================================
# LIVE SUPPORT CONSOLE
# ============================================================

if app_mode == "🏛️ Live Support Console":

    st.title(
        "🏛️ Live Support Console"
    )

    st.caption(
        "Proactive real-time coaching platform with "
        "Simulator, Manual, and Replay modes."
    )

    tab_sim, tab_manual, tab_replay = st.tabs(
        [
            "🤖 Simulator Mode",
            "✍️ Manual Mode",
            "⏪ Replay Mode",
        ]
    )

    with st.sidebar:

        st.header(
            "⚙️ Escalation Alert Config"
        )

        escalation_threshold = st.selectbox(
            "🚨 Escalation Alert Threshold:",
            options=[
                "critical",
                "high",
                "medium",
            ],
            index=1,
        )

        st.divider()

    # ========================================================
    # SIMULATOR MODE
    # ========================================================

    with tab_sim:

        if "sim_conversation_id" not in st.session_state:

            st.session_state.sim_conversation_id = None

        if "sim_messages" not in st.session_state:

            st.session_state.sim_messages = []

        if "sim_current_emotion" not in st.session_state:

            st.session_state.sim_current_emotion = "frustrated"

        if "sim_patience" not in st.session_state:

            st.session_state.sim_patience = "low"

        if "sim_latest_analysis" not in st.session_state:

            st.session_state.sim_latest_analysis = None

        if "sim_latest_knowledge_recs" not in st.session_state:

            st.session_state.sim_latest_knowledge_recs = []

        if "last_scenario_preset" not in st.session_state:

            st.session_state.last_scenario_preset = None

        with st.sidebar:

            st.header(
                "⚙️ Simulator Persona Setup"
            )

            persona = st.selectbox(
                "Customer Persona:",
                [
                    "angry",
                    "frustrated",
                    "impatient",
                    "confused",
                    "calm",
                    "polite",
                ],
                index=0,
            )

            scenario_presets = {

                "Refund Request": {
                    "desc": (
                        "Customer received a defective item 2 days ago "
                        "and demands an immediate full refund without restocking fees."
                    ),
                    "emotion": "angry",
                    "severity": "high",
                    "patience": "low",
                    "resolution": (
                        "immediate full refund with all fees waived"
                    ),
                },

                "Delayed Order Delivery": {
                    "desc": (
                        "Birthday gift order delayed 4 days past "
                        "guaranteed date with no tracking updates."
                    ),
                    "emotion": "impatient",
                    "severity": "high",
                    "patience": "low",
                    "resolution": (
                        "priority dispatch trace and shipping fee refund"
                    ),
                },

                "Payment Failure / Double Charge": {
                    "desc": (
                        "Customer noticed duplicate charges during checkout "
                        "on credit card statement, status pending."
                    ),
                    "emotion": "confused",
                    "severity": "medium",
                    "patience": "medium",
                    "resolution": (
                        "release of pre-authorization hold "
                        "and single charge confirmation"
                    ),
                },

                "Order Cancellation": {
                    "desc": (
                        "Customer accidentally placed duplicate order "
                        "15 minutes ago and wants immediate cancellation."
                    ),
                    "emotion": "calm",
                    "severity": "low",
                    "patience": "high",
                    "resolution": (
                        "cancellation confirmation email "
                        "and hold release"
                    ),
                },

                "Product Replacement (Damaged)": {
                    "desc": (
                        "Item arrived with glass screen completely shattered; "
                        "customer asks for replacement."
                    ),
                    "emotion": "frustrated",
                    "severity": "high",
                    "patience": "medium",
                    "resolution": (
                        "complimentary replacement shipped overnight "
                        "without returning broken glass"
                    ),
                },

                "Subscription Renewal Issue": {
                    "desc": (
                        "Customer billed for annual subscription renewal "
                        "unexpectedly, inquires about cancellation."
                    ),
                    "emotion": "worried",
                    "severity": "medium",
                    "patience": "medium",
                    "resolution": (
                        "full refund within 14 days "
                        "and cancellation of auto-renewal"
                    ),
                },

                "Account Lockout & Security": {
                    "desc": (
                        "Customer locked out of account after password reset "
                        "attempts, reports security concern."
                    ),
                    "emotion": "worried",
                    "severity": "high",
                    "patience": "low",
                    "resolution": (
                        "emergency session lock "
                        "and secure verification link"
                    ),
                },

                "Supervisor Escalation Demand": {
                    "desc": (
                        "Customer experienced multiple delays and explicitly "
                        "demands to speak with a supervisor immediately."
                    ),
                    "emotion": "angry",
                    "severity": "high",
                    "patience": "low",
                    "resolution": (
                        "empathetic de-escalation and warm supervisor "
                        "queue transfer with ticket ID"
                    ),
                },
            }

            selected_preset = st.selectbox(
                "Scenario Category:",
                list(
                    scenario_presets.keys()
                ),
                index=0,
            )

            preset_data = scenario_presets[
                selected_preset
            ]

            scenario_desc = st.text_area(
                "Scenario Context:",
                value=preset_data["desc"],
                height=80,
            )

            c_s1, c_s2 = st.columns(2)

            with c_s1:

                initial_emotion = st.selectbox(
                    "Starting Emotion:",
                    [
                        "angry",
                        "frustrated",
                        "impatient",
                        "confused",
                        "worried",
                        "calm",
                        "neutral",
                    ],
                    index=[
                        "angry",
                        "frustrated",
                        "impatient",
                        "confused",
                        "worried",
                        "calm",
                        "neutral",
                    ].index(
                        preset_data["emotion"]
                    ),
                )

            with c_s2:

                issue_severity = st.selectbox(
                    "Severity:",
                    [
                        "critical",
                        "high",
                        "medium",
                        "low",
                    ],
                    index=[
                        "critical",
                        "high",
                        "medium",
                        "low",
                    ].index(
                        preset_data["severity"]
                    ),
                )

            initial_patience = st.select_slider(
                "Starting Patience Level:",
                options=[
                    "low",
                    "medium",
                    "high",
                ],
                value=preset_data["patience"],
            )

            expected_resolution = st.text_input(
                "Expected Customer Resolution:",
                value=preset_data["resolution"],
            )

            st.divider()

            reset_triggered = st.button(
                "🔄 Start / Reset Simulation",
                use_container_width=True,
                type="primary",
            )

        def initialize_session():

            st.session_state.sim_messages = []

            st.session_state.sim_latest_analysis = None

            st.session_state.sim_latest_knowledge_recs = []

            cfg_payload = {

                "persona": persona,

                "scenario": scenario_desc,

                "initial_emotion": initial_emotion,

                "current_emotion": initial_emotion,

                "issue_severity": issue_severity,

                "patience_level": initial_patience,

                "expected_resolution": expected_resolution,
            }

            opening_text = (
                f"I am {initial_emotion} "
                f"about my issue: "
                f"{scenario_desc}"
            )

            try:

                res = requests.post(
                    f"{API_URL}/simulator/chat",
                    json={
                        "conversation_id": None,
                        "agent_message": "",
                        "config": cfg_payload,
                    },
                    timeout=60,
                )

                if res.status_code == 200:

                    data = res.json()

                    st.session_state.sim_conversation_id = (
                        data.get(
                            "conversation_id"
                        )
                    )

                    st.session_state.sim_latest_analysis = (
                        data.get(
                            "analysis"
                        )
                    )

                    st.session_state.sim_latest_knowledge_recs = (
                        data.get(
                            "knowledge_recommendations",
                            [],
                        )
                    )

                    if data.get(
                        "customer_message"
                    ):

                        opening_text = data.get(
                            "customer_message"
                        )

            except Exception:

                pass

            st.session_state.sim_messages = [
                {
                    "role": "user",
                    "content": opening_text,
                    "analysis": (
                        st.session_state.sim_latest_analysis
                    ),
                    "knowledge_recommendations": (
                        st.session_state.sim_latest_knowledge_recs
                    ),
                }
            ]

            st.session_state.sim_current_emotion = (
                initial_emotion
            )

            st.session_state.sim_patience = (
                initial_patience
            )

            st.session_state.last_scenario_preset = (
                selected_preset
            )

        if (
            reset_triggered
            or not st.session_state.sim_messages
            or st.session_state.last_scenario_preset
            != selected_preset
        ):

            initialize_session()

        col_chat_sim, col_coaching_sim, col_knowledge_sim = st.columns(
            [5, 4, 4]
        )

        with col_chat_sim:

            st.subheader(
                "💬 Panel 1: Conversation Window"
            )

            render_chat_history(
                st.session_state.sim_messages
            )

            agent_input = st.chat_input(
                "Type your support response to the customer...",
                key="sim_input",
            )

            if agent_input:

                st.session_state.sim_messages.append(
                    {
                        "role": "assistant",
                        "content": agent_input,
                    }
                )

                with st.spinner(
                    "Simulated customer is typing..."
                ):

                    try:

                        cfg_payload = {

                            "persona": persona,

                            "scenario": scenario_desc,

                            "initial_emotion": initial_emotion,

                            "current_emotion": (
                                st.session_state.sim_current_emotion
                            ),

                            "issue_severity": issue_severity,

                            "patience_level": (
                                st.session_state.sim_patience
                            ),

                            "expected_resolution": (
                                expected_resolution
                            ),
                        }

                        sim_req = {

                            "conversation_id": (
                                st.session_state.sim_conversation_id
                            ),

                            "agent_message": agent_input,

                            "config": cfg_payload,
                        }

                        res = requests.post(
                            f"{API_URL}/simulator/chat",
                            json=sim_req,
                            timeout=60,
                        )

                        if res.status_code == 200:

                            data = res.json()

                            st.session_state.sim_conversation_id = (
                                data.get(
                                    "conversation_id",
                                    st.session_state.sim_conversation_id,
                                )
                            )

                            st.session_state.sim_messages.append(
                                {
                                    "role": "user",
                                    "content": data.get(
                                        "customer_message",
                                        "",
                                    ),
                                    "analysis": data.get(
                                        "analysis"
                                    ),
                                    "knowledge_recommendations": data.get(
                                        "knowledge_recommendations"
                                    ),
                                }
                            )

                            st.session_state.sim_current_emotion = (
                                data.get(
                                    "current_emotion",
                                    st.session_state.sim_current_emotion,
                                )
                            )

                            st.session_state.sim_patience = (
                                data.get(
                                    "patience_level",
                                    st.session_state.sim_patience,
                                )
                            )

                            st.session_state.sim_latest_analysis = (
                                data.get(
                                    "analysis"
                                )
                            )

                            st.session_state.sim_latest_knowledge_recs = (
                                data.get(
                                    "knowledge_recommendations",
                                    [],
                                )
                            )

                        st.rerun()

                    except Exception as e:

                        st.error(
                            f"Simulator request failed: {e}"
                        )

        with col_coaching_sim:

            render_panel_2(
                st.session_state.sim_latest_analysis,
                escalation_threshold,
            )

        with col_knowledge_sim:

            render_panel_3(
                st.session_state.sim_latest_knowledge_recs
            )

    # ========================================================
    # MANUAL MODE
    # ========================================================

    with tab_manual:

        if "man_messages" not in st.session_state:

            st.session_state.man_messages = []

        if "man_latest_analysis" not in st.session_state:

            st.session_state.man_latest_analysis = None

        if "man_latest_recs" not in st.session_state:

            st.session_state.man_latest_recs = []

        c1, c2 = st.columns(
            [1, 8]
        )

        with c1:

            if st.button(
                "🔄 Clear",
                key="man_clear",
            ):

                st.session_state.man_messages = []

                st.session_state.man_latest_analysis = None

                st.session_state.man_latest_recs = []

                st.rerun()

        with c2:

            input_type = st.radio(
                "Next speaker:",
                [
                    "Customer",
                    "Support Agent",
                ],
                horizontal=True,
                key="man_speaker",
            )

        if input_type == "Support Agent":

            col_chat_man, col_coaching_man, col_knowledge_man = st.columns(
                [5, 4, 4]
            )

            with col_chat_man:

                st.subheader(
                    "💬 Panel 1: Conversation Window (Manual)"
                )

                render_chat_history(
                    st.session_state.man_messages
                )

                man_input = st.chat_input(
                    "Enter the agent's response...",
                    key="man_input",
                )

                if man_input:

                    msg_obj = {
                        "role": "assistant",
                        "content": man_input,
                    }

                    st.session_state.man_messages.append(
                        msg_obj
                    )

                    st.rerun()

            with col_coaching_man:

                render_panel_2(
                    st.session_state.man_latest_analysis,
                    escalation_threshold,
                )

            with col_knowledge_man:

                render_panel_3(
                    st.session_state.man_latest_recs
                )

        else:

            st.subheader(
                "💬 Panel 1: Conversation Window (Manual)"
            )

            render_chat_history(
                st.session_state.man_messages
            )

            man_input = st.chat_input(
                "Enter the customer's message...",
                key="man_input",
            )

            if man_input:

                msg_obj = {
                    "role": "user",
                    "content": man_input,
                }

                with st.spinner(
                    "Analyzing customer message..."
                ):

                    hist = [
                        {
                            "role": m["role"],
                            "content": m["content"],
                        }
                        for m in st.session_state.man_messages
                    ]

                    try:

                        a_res = requests.post(
                            f"{API_URL}/analysis/intent-sentiment",
                            json={
                                "message": man_input,
                                "history": hist,
                            },
                            timeout=10,
                        )

                        if a_res.status_code == 200:

                            msg_obj["analysis"] = (
                                a_res.json().get(
                                    "analysis"
                                )
                            )

                            st.session_state.man_latest_analysis = (
                                msg_obj["analysis"]
                            )

                    except:

                        pass

                    try:

                        k_res = requests.post(
                            f"{API_URL}/recommendations/knowledge",
                            json={
                                "message": man_input,
                                "history": hist,
                                "top_k": 4,
                            },
                            timeout=10,
                        )

                        if k_res.status_code == 200:

                            msg_obj[
                                "knowledge_recommendations"
                            ] = (
                                k_res.json().get(
                                    "recommendations",
                                    [],
                                )
                            )

                            st.session_state.man_latest_recs = (
                                msg_obj[
                                    "knowledge_recommendations"
                                ]
                            )

                    except:

                        pass

                st.session_state.man_messages.append(
                    msg_obj
                )

                st.rerun()

    # ========================================================
    # REPLAY MODE
    # ========================================================

    with tab_replay:

        if "rep_messages" not in st.session_state:

            st.session_state.rep_messages = []

        if "rep_current_idx" not in st.session_state:

            st.session_state.rep_current_idx = -1

        uploaded_transcript = st.file_uploader(
            "Upload Conversation Transcript (JSON format)",
            type=[
                "json"
            ],
            key="rep_upload",
        )

        if uploaded_transcript:

            if st.button(
                "Load Transcript",
                key="load_transcript",
            ):

                try:

                    data = json.load(
                        uploaded_transcript
                    )

                    st.session_state.rep_messages = data

                    st.session_state.rep_current_idx = 0

                except Exception as e:

                    st.error(
                        f"Failed to load JSON: {e}"
                    )

        if (
            st.session_state.rep_messages
            and st.session_state.rep_current_idx >= 0
        ):

            c1, c2, c3 = st.columns(3)

            with c1:

                if (
                    st.button("⏪ Previous")
                    and st.session_state.rep_current_idx > 0
                ):

                    st.session_state.rep_current_idx -= 1

                    st.rerun()

            with c2:

                if st.button(
                    "🔄 Restart"
                ):

                    st.session_state.rep_current_idx = 0

                    st.rerun()

            with c3:

                if (
                    st.button("Next ⏩")
                    and st.session_state.rep_current_idx
                    < len(
                        st.session_state.rep_messages
                    ) - 1
                ):

                    st.session_state.rep_current_idx += 1

                    st.rerun()

            current_msgs = (
                st.session_state.rep_messages[
                    : st.session_state.rep_current_idx + 1
                ]
            )

            latest_msg = current_msgs[-1]

            if (
                latest_msg.get("role") == "user"
                and "analysis" not in latest_msg
            ):

                with st.spinner(
                    "Analyzing historical customer message..."
                ):

                    hist = [
                        {
                            "role": m["role"],
                            "content": m["content"],
                        }
                        for m in current_msgs[:-1]
                    ]

                    try:

                        a_res = requests.post(
                            f"{API_URL}/analysis/intent-sentiment",
                            json={
                                "message": latest_msg["content"],
                                "history": hist,
                            },
                            timeout=10,
                        )

                        if a_res.status_code == 200:

                            latest_msg["analysis"] = (
                                a_res.json().get(
                                    "analysis"
                                )
                            )

                        k_res = requests.post(
                            f"{API_URL}/recommendations/knowledge",
                            json={
                                "message": latest_msg["content"],
                                "history": hist,
                                "top_k": 4,
                            },
                            timeout=10,
                        )

                        if k_res.status_code == 200:

                            latest_msg[
                                "knowledge_recommendations"
                            ] = (
                                k_res.json().get(
                                    "recommendations",
                                    [],
                                )
                            )

                    except:

                        pass

            latest_analysis = None

            latest_recs = []

            for m in reversed(
                current_msgs
            ):

                if m.get(
                    "role"
                ) == "user":

                    latest_analysis = m.get(
                        "analysis"
                    )

                    latest_recs = m.get(
                        "knowledge_recommendations",
                        [],
                    )

                    break

            col_chat_rep, col_coaching_rep, col_knowledge_rep = st.columns(
                [5, 4, 4]
            )

            with col_chat_rep:

                st.subheader(
                    "💬 Panel 1: Conversation Replay"
                )

                render_chat_history(
                    current_msgs
                )

                if (
                    st.session_state.rep_current_idx
                    == len(
                        st.session_state.rep_messages
                    ) - 1
                ):

                    st.success(
                        "🏁 Replay Complete!"
                    )

                    max_frustration = max(
                        [
                            m.get(
                                "analysis",
                                {}
                            ).get(
                                "frustration_level",
                                0,
                            )
                            for m in st.session_state.rep_messages
                            if (
                                m.get("role") == "user"
                                and m.get("analysis")
                            )
                        ],
                        default=0,
                    )

                    trends = [
                        m.get(
                            "analysis",
                            {}
                        ).get(
                            "satisfaction_trend",
                            "stable",
                        )
                        for m in st.session_state.rep_messages
                        if (
                            m.get("role") == "user"
                            and m.get("analysis")
                        )
                    ]

                    final_trend = (
                        trends[-1]
                        if trends
                        else "stable"
                    )

                    st.info(
                        f"Summary: Max Frustration was "
                        f"{max_frustration}/10. "
                        f"Final Satisfaction Trend was "
                        f"'{final_trend}'."
                    )

            with col_coaching_rep:

                render_panel_2(
                    latest_analysis,
                    escalation_threshold,
                )

            with col_knowledge_rep:

                render_panel_3(
                    latest_recs
                )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

elif app_mode == "📚 Support Knowledge Base & Ingestion":

    st.title(
        "📚 Support Knowledge Base & Document Ingestion"
    )

    st.info(
        "Ingest documents to ChromaDB."
    )

    col_upload, col_list = st.columns(2)

    with col_upload:

        st.subheader(
            "⬆️ Upload Document"
        )

        uploaded_file = st.file_uploader(
            "Upload a support document",
            type=[
                "pdf",
                "txt",
                "md",
            ],
        )

        if uploaded_file is not None:

            if st.button(
                "⬆️ Upload & Index Document",
                type="primary",
            ):

                with st.spinner(
                    "Processing document..."
                ):

                    try:

                        response = requests.post(
                            f"{API_URL}/upload",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type,
                                )
                            },
                            timeout=120,
                        )

                        if response.status_code == 200:

                            data = response.json()

                            st.success(
                                "Document indexed successfully into ChromaDB!"
                            )

                            st.write(
                                f"**Filename:** "
                                f"{data.get('filename')}"
                            )

                            st.write(
                                f"**Chunks Stored:** "
                                f"{data.get('chunks', 0)}"
                            )

                    except Exception as exc:

                        st.error(
                            f"Upload failed: {exc}"
                        )

    with col_list:

        st.subheader(
            "📚 Knowledge Base Status"
        )

        st.info(
            "Core Knowledge Corpus Indexed:\n"
            "- `delivery_shipping_faq.md`\n"
            "- `order_cancellation_policy.md`\n"
            "- `product_warranty_replacement.md`\n"
            "- `subscription_management_faq.md`\n"
            "- `escalation_dispute_policy.md`\n"
            "- `refund_policy.pdf`\n"
            "- `payment_faq.pdf`\n"
            "- `account_support.pdf`"
        )


# ============================================================
# TASK 8: POST-INTERACTION REPORT
# ============================================================

elif app_mode == "📊 Post-Interaction Report":

    st.title(
        "📊 Post-Interaction Report"
    )

    st.caption(
        "Review the completed customer interaction, sentiment journey, "
        "resolution quality, strengths, weaknesses, and coaching recommendations."
    )

    st.divider()

    # --------------------------------------------------------
    # Conversation Selection
    # --------------------------------------------------------

    st.subheader(
        "🔎 Select Conversation"
    )

    conversation_id = st.number_input(
        "Conversation ID",
        min_value=1,
        value=1,
        step=1,
        help=(
            "Enter the conversation ID generated "
            "by the support system."
        ),
    )

    generate_report = st.button(
        "📊 Generate Post-Interaction Report",
        type="primary",
        use_container_width=True,
    )

    if generate_report:

        with st.spinner(
            "Generating post-interaction report..."
        ):

            try:

                response = requests.get(
                    f"{API_URL}/task8/post-interaction-summary/"
                    f"{int(conversation_id)}",
                    timeout=60,
                )

                if response.status_code == 200:

                    report = response.json()

                    st.session_state.task8_report = report

                    st.success(
                        "Post-interaction report generated successfully."
                    )

                elif response.status_code == 404:

                    st.error(
                        f"Conversation {conversation_id} "
                        f"was not found."
                    )

                else:

                    st.error(
                        f"Report generation failed. "
                        f"HTTP Status: {response.status_code}"
                    )

                    try:

                        st.json(
                            response.json()
                        )

                    except Exception:

                        st.write(
                            response.text
                        )

            except requests.exceptions.ConnectionError:

                st.error(
                    "Could not connect to the FastAPI server. "
                    "Make sure the API is running on "
                    "http://127.0.0.1:8000"
                )

            except Exception as exc:

                st.error(
                    f"Unexpected error: {exc}"
                )

    # --------------------------------------------------------
    # DISPLAY REPORT
    # --------------------------------------------------------

    report = st.session_state.get(
        "task8_report"
    )

    if report:

        st.divider()

        # ====================================================
        # INTERACTION OVERVIEW
        # ====================================================

        st.header(
            "📋 Interaction Overview"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Conversation ID",
                report.get(
                    "conversation_id",
                    "N/A",
                ),
            )

        with col2:

            st.metric(
                "Customer Turns",
                report.get(
                    "customer_turn_count",
                    0,
                ),
            )

        with col3:

            st.metric(
                "Total Messages",
                report.get(
                    "total_message_count",
                    0,
                ),
            )

        st.markdown(
            f"**Conversation:** "
            f"{report.get('conversation_title', 'Untitled Conversation')}"
        )

        # ====================================================
        # INTERACTION SUMMARY
        # ====================================================

        st.subheader(
            "📝 Interaction Summary"
        )

        st.info(
            report.get(
                "interaction_summary",
                "No interaction summary available.",
            )
        )

        # ====================================================
        # PRIMARY ISSUE & FINAL RESOLUTION
        # ====================================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "🎯 Primary Issue"
            )

            primary_issue = report.get(
                "primary_issue",
                "Not identified",
            )

            st.success(
                str(
                    primary_issue
                )
                .replace(
                    "_",
                    " ",
                )
                .title()
            )

        with col2:

            st.subheader(
                "✅ Final Resolution"
            )

            st.write(
                report.get(
                    "final_resolution",
                    "No final resolution available.",
                )
            )

        st.divider()

        # ====================================================
        # CUSTOMER SENTIMENT JOURNEY
        # ====================================================

        st.subheader(
            "📈 Customer Sentiment Journey"
        )

        sentiment_journey = report.get(
            "sentiment_journey",
            [],
        )

        if sentiment_journey:

            for item in sentiment_journey:

                col1, col2, col3, col4, col5, col6 = st.columns(
                    6
                )

                with col1:

                    st.metric(
                        "Turn",
                        item.get(
                            "turn",
                            "N/A",
                        ),
                    )

                with col2:

                    st.write(
                        "**Emotion**"
                    )

                    st.write(
                        str(
                            item.get(
                                "emotion",
                                "neutral",
                            )
                        ).title()
                    )

                with col3:

                    st.write(
                        "**Sentiment**"
                    )

                    st.write(
                        str(
                            item.get(
                                "sentiment",
                                "neutral",
                            )
                        ).title()
                    )

                with col4:

                    st.write(
                        "**Frustration**"
                    )

                    st.write(
                        f"{item.get('frustration_level', 0)}/10"
                    )

                with col5:

                    st.write(
                        "**Satisfaction**"
                    )

                    st.write(
                        str(
                            item.get(
                                "satisfaction_trend",
                                "stable",
                            )
                        ).title()
                    )

                with col6:

                    st.write(
                        "**Risk**"
                    )

                    st.write(
                        str(
                            item.get(
                                "escalation_risk",
                                "low",
                            )
                        ).upper()
                    )

                st.divider()

        else:

            st.info(
                "No sentiment journey data available."
            )

        # ====================================================
        # RESOLUTION QUALITY
        # ====================================================

        st.subheader(
            "⭐ Resolution Quality"
        )

        quality = report.get(
            "resolution_quality",
            {},
        )

        overall_score = quality.get(
            "overall_score",
            0,
        )

        average_score = quality.get(
            "average_score",
            0,
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Overall Score",
                f"{overall_score}/100",
            )

        with col2:

            st.metric(
                "Average Score",
                f"{average_score}/5",
            )

        with col3:

            st.metric(
                "Resolution Quality",
                f"{quality.get('resolution_quality', 0)}/5",
            )

        st.progress(
            min(
                1.0,
                max(
                    0.0,
                    float(overall_score) / 100,
                ),
            )
        )

        st.markdown(
            "### Quality Breakdown"
        )

        q1, q2, q3, q4 = st.columns(4)

        with q1:

            st.metric(
                "Resolution",
                f"{quality.get('resolution_quality', 0)}/5",
            )

        with q2:

            st.metric(
                "Communication",
                f"{quality.get('communication_quality', 0)}/5",
            )

        with q3:

            st.metric(
                "Empathy",
                f"{quality.get('empathy', 0)}/5",
            )

        with q4:

            st.metric(
                "Guideline Adherence",
                f"{quality.get('guideline_adherence', 0)}/5",
            )

        st.divider()

        # ====================================================
        # STRENGTHS & WEAKNESSES
        # ====================================================

        st.subheader(
            "💪 Strengths & Areas for Improvement"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### 💪 Strengths"
            )

            strengths = report.get(
                "strengths",
                [],
            )

            if strengths:

                for strength in strengths:

                    st.success(
                        strength
                    )

            else:

                st.info(
                    "No strengths identified."
                )

        with col2:

            st.markdown(
                "### ⚠️ Weaknesses"
            )

            weaknesses = report.get(
                "weaknesses",
                [],
            )

            if weaknesses:

                for weakness in weaknesses:

                    st.warning(
                        weakness
                    )

            else:

                st.info(
                    "No weaknesses identified."
                )

        st.divider()

        # ====================================================
        # COACHING RECOMMENDATIONS
        # ====================================================

        st.subheader(
            "🎓 Personalized Coaching Recommendations"
        )

        recommendations = report.get(
            "coaching_recommendations",
            [],
        )

        if recommendations:

            for index, recommendation in enumerate(
                recommendations,
                start=1,
            ):

                st.info(
                    f"**Recommendation {index}:** "
                    f"{recommendation}"
                )

        else:

            st.info(
                "No coaching recommendations were generated."
            )

        st.divider()

        # ====================================================
        # COMPLETE REPORT JSON
        # ====================================================

        with st.expander(
            "🔍 View Complete Report JSON"
        ):

            st.json(
                report
            )


# ============================================================
# TASK 8: PERFORMANCE ANALYTICS
# ============================================================

elif app_mode == "📈 Performance Analytics":

    render_performance_analytics()