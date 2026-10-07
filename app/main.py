from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import UPLOAD_DIR, TOP_K

from app.database.database import (
    create_conversation,
    get_conversation,
    add_message,
    get_recent_messages,
    count_messages,
    update_conversation_summary,
)

from app.ingestion.loader import load_document
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages

from app.embeddings.embedder import embed_texts

from app.vectorstore.chroma_store import add_chunks

from app.retrieval.search import semantic_search

from app.generation.answer import (
    generate_chat_answer,
    summarize_conversation,
)

from app.models.schemas import (
    ChatRequest,
    SimulatorTurnRequest,
    SimulatorTurnResponse,
    AnalyzeMessageRequest,
    AnalyzeMessageResponse,
    IntentSentimentAnalysis,
    KnowledgeRecommendationItem,
    KnowledgeRecommendationRequest,
    KnowledgeRecommendationResponse,
)

from app.agents.simulator import generate_customer_turn
from app.agents.intent_sentiment import analyze_customer_message
from app.agents.knowledge import recommend_knowledge

# ---------------------------------------------------------
# Task 8
# ---------------------------------------------------------

from app.task8.post_interaction_summary import (
    build_post_interaction_summary,
)

from app.task8.performance_analytics import (
    build_performance_analytics,
)


app = FastAPI(title="Support RAG API")


# =========================================================
# Health Check
# =========================================================

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Support RAG API is running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# =========================================================
# Upload Document
# =========================================================

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected.",
            )

        original_name = file.filename
        suffix = Path(original_name).suffix.lower()

        allowed_extensions = {
            ".pdf",
            ".txt",
            ".md",
        }

        if suffix not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}",
            )

        # -------------------------------------------------
        # Make sure upload directory exists
        # -------------------------------------------------

        upload_dir = Path(UPLOAD_DIR)
        upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------------------
        # Create short physical filename
        # -------------------------------------------------

        document_id = uuid4().hex

        saved_path = (
            upload_dir
            / f"{document_id}{suffix}"
        )

        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        saved_path.write_bytes(file_bytes)

        print(
            f"UPLOAD: saved file -> {saved_path}"
        )

        # -------------------------------------------------
        # Load document
        # -------------------------------------------------

        print(
            "UPLOAD STEP 1: loading document"
        )

        pages = load_document(
            saved_path
        )

        print(
            f"UPLOAD STEP 2: loaded {len(pages)} pages"
        )

        # -------------------------------------------------
        # Clean document
        # -------------------------------------------------

        print(
            "UPLOAD STEP 3: cleaning document"
        )

        cleaned_pages = clean_pages(
            pages
        )

        print(
            f"UPLOAD STEP 4: cleaned "
            f"{len(cleaned_pages)} pages"
        )

        # -------------------------------------------------
        # Chunk document
        # -------------------------------------------------

        print(
            "UPLOAD STEP 5: creating chunks"
        )

        chunks = chunk_pages(
            cleaned_pages
        )

        print(
            f"UPLOAD STEP 6: created "
            f"{len(chunks)} chunks"
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No usable text was found "
                    "in the document."
                ),
            )

        # -------------------------------------------------
        # Prepare chunk IDs + metadata
        # -------------------------------------------------

        for chunk in chunks:

            chunk["chunk_id"] = (
                f"{document_id}_"
                f"page_{chunk['page_number']}_"
                f"chunk_{chunk['chunk_number']}"
            )

            chunk["metadata"] = {
                "document_id": document_id,
                "filename": original_name,
                "page_number": chunk["page_number"],
                "chunk_number": chunk["chunk_number"],
            }

        # -------------------------------------------------
        # Create embeddings
        # -------------------------------------------------

        print(
            "UPLOAD STEP 7: creating embeddings"
        )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embed_texts(
            texts
        )

        print(
            f"UPLOAD STEP 8: created "
            f"{len(embeddings)} embeddings"
        )

        # -------------------------------------------------
        # Store in Chroma
        # -------------------------------------------------

        print(
            "UPLOAD STEP 9: storing vectors"
        )

        stored_count = add_chunks(
            chunks,
            embeddings,
        )

        print(
            f"UPLOAD STEP 10: stored "
            f"{stored_count} chunks"
        )

        return {
            "success": True,
            "document_id": document_id,
            "filename": original_name,
            "chunks": stored_count,
            "message": (
                "Document uploaded and indexed "
                "successfully."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:

        print(
            "UPLOAD ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Upload failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# =========================================================
# Chat
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        print(
            "\n================ CHAT REQUEST ================"
        )

        # -------------------------------------------------
        # Validate query
        # -------------------------------------------------

        query = request.query.strip()

        if not query:

            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty.",
            )

        print(
            f"CHAT QUERY: {query}"
        )

        # -------------------------------------------------
        # Create conversation if needed
        # -------------------------------------------------

        conversation_id = request.conversation_id

        if conversation_id is None:

            print(
                "CHAT: creating new conversation"
            )

            conversation_id = create_conversation()

            print(
                f"CHAT: created conversation "
                f"{conversation_id}"
            )

        # -------------------------------------------------
        # Get conversation
        # -------------------------------------------------

        print(
            "CHAT STEP 1: loading conversation"
        )

        conversation = get_conversation(
            conversation_id
        )

        if conversation is None:

            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        # -------------------------------------------------
        # Get previous conversation history
        # -------------------------------------------------

        print(
            "CHAT STEP 2: loading recent history"
        )

        history = get_recent_messages(
            conversation_id,
            limit=10,
        )

        if history is None:
            history = []

        print(
            f"CHAT: loaded "
            f"{len(history)} previous messages"
        )

        # -------------------------------------------------
        # Get existing summary
        # -------------------------------------------------

        summary = ""

        if isinstance(
            conversation,
            dict,
        ):

            summary = conversation.get(
                "summary",
                "",
            ) or ""

        else:

            try:

                summary = (
                    conversation["summary"]
                    or ""
                )

            except Exception:

                summary = ""

        print(
            f"CHAT: summary length = "
            f"{len(summary)}"
        )

        # -------------------------------------------------
        # Save current user message
        # -------------------------------------------------

        print(
            "CHAT STEP 3: saving user message"
        )

        add_message(
            conversation_id=conversation_id,
            role="user",
            content=query,
        )

        # -------------------------------------------------
        # Semantic search
        # -------------------------------------------------

        print(
            "CHAT STEP 4: semantic search"
        )

        top_k = request.top_k or TOP_K

        results = semantic_search(
            query,
            top_k=top_k,
        )

        if results is None:
            results = []

        print(
            f"CHAT STEP 5: retrieved "
            f"{len(results)} results"
        )

        # -------------------------------------------------
        # Generate AI answer
        # -------------------------------------------------

        print(
            "CHAT STEP 6: generating AI answer"
        )

        answer = generate_chat_answer(
            query=query,
            history=history,
            summary=summary,
            results=results,
        )

        print(
            "CHAT STEP 7: AI answer generated"
        )

        if not answer:

            answer = (
                "I could not generate an answer "
                "for that question."
            )

        # -------------------------------------------------
        # Save assistant response
        # -------------------------------------------------

        print(
            "CHAT STEP 8: saving assistant message"
        )

        add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )

        # -------------------------------------------------
        # Count messages
        # -------------------------------------------------

        print(
            "CHAT STEP 9: counting messages"
        )

        message_count = count_messages(
            conversation_id
        )

        print(
            f"CHAT: message count = "
            f"{message_count}"
        )

        # -------------------------------------------------
        # Summarize after enough messages
        # -------------------------------------------------

        if message_count >= 20:

            print(
                "CHAT STEP 10: summarizing conversation"
            )

            try:

                all_messages = get_recent_messages(
                    conversation_id,
                    limit=20,
                )

                # Existing summary is passed to the
                # summarization function.
                new_summary = summarize_conversation(
                    summary,
                    all_messages,
                )

                if new_summary:

                    update_conversation_summary(
                        conversation_id,
                        new_summary,
                    )

                    print(
                        "CHAT: conversation "
                        "summary updated"
                    )

            except Exception as summary_error:

                # Do not fail the entire chat
                # if summary generation fails.

                print(
                    "SUMMARY ERROR:",
                    repr(summary_error),
                )

        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        print(
            "CHAT STEP 11: returning response"
        )

        return {
            "success": True,
            "conversation_id": conversation_id,
            "answer": answer,
            "results": results,
        }

    except HTTPException:
        raise

    except Exception as exc:

        print(
            "\n!!!!!!!!!!!!!!!! CHAT ERROR !!!!!!!!!!!!!!!!"
        )

        print(
            "ERROR TYPE:",
            type(exc).__name__,
        )

        print(
            "ERROR:",
            str(exc),
        )

        print(
            "ERROR REPR:",
            repr(exc),
        )

        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Chat failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# =========================================================
# Simulator Chat
# =========================================================

@app.post(
    "/simulator/chat",
    response_model=SimulatorTurnResponse,
)
def simulator_chat(
    request: SimulatorTurnRequest
):

    try:

        print(
            "\n================ SIMULATOR CHAT REQUEST ================"
        )

        # -------------------------------------------------
        # Create conversation if needed
        # -------------------------------------------------

        conversation_id = (
            request.conversation_id
        )

        if conversation_id is None:

            print(
                "SIMULATOR: creating new conversation"
            )

            conversation_id = create_conversation(
                "Simulator Session"
            )

            print(
                f"SIMULATOR: created conversation "
                f"{conversation_id}"
            )

        # -------------------------------------------------
        # Get existing conversation
        # -------------------------------------------------

        conversation = get_conversation(
            conversation_id
        )

        if conversation is None:

            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        # -------------------------------------------------
        # Save Agent's message
        # -------------------------------------------------

        if request.agent_message.strip():

            print(
                "SIMULATOR: saving agent message"
            )

            add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=request.agent_message.strip(),
            )

        # -------------------------------------------------
        # Get previous history
        # -------------------------------------------------

        history = get_recent_messages(
            conversation_id,
            limit=20,
        )

        if history is None:
            history = []

        # -------------------------------------------------
        # Generate Simulated Customer Response
        # -------------------------------------------------

        print(
            "SIMULATOR: generating customer turn"
        )

        result = generate_customer_turn(
            agent_message=(
                request.agent_message.strip()
            ),
            history=history,
            config=request.config,
        )

        customer_message = result.get(
            "customer_message",
            "Error generating response.",
        )

        current_emotion = result.get(
            "current_emotion",
            request.config.current_emotion,
        )

        patience_level = result.get(
            "patience_level",
            request.config.patience_level,
        )

        # -------------------------------------------------
        # Save Customer's message
        # -------------------------------------------------

        print(
            "SIMULATOR: saving customer message"
        )

        add_message(
            conversation_id=conversation_id,
            role="user",
            content=customer_message,
        )

        # -------------------------------------------------
        # Real-time Intent & Sentiment Analysis
        # -------------------------------------------------

        print(
            "SIMULATOR: running intent "
            "& sentiment analysis"
        )

        analysis = analyze_customer_message(
            message=customer_message,
            history=history,
        )

        # -------------------------------------------------
        # Knowledge Recommendations
        # -------------------------------------------------

        print(
            "SIMULATOR: retrieving "
            "knowledge recommendations"
        )

        knowledge_recs = recommend_knowledge(
            message=customer_message,
            history=history,
            top_k=4,
        )

        print(
            "SIMULATOR: returning response "
            "with analysis and knowledge recommendations"
        )

        return SimulatorTurnResponse(
            success=True,
            conversation_id=conversation_id,
            customer_message=customer_message,
            current_emotion=current_emotion,
            patience_level=patience_level,
            analysis=analysis,
            knowledge_recommendations=knowledge_recs,
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            "\n!!!!!!!!!!!!!!!! SIMULATOR ERROR !!!!!!!!!!!!!!!!"
        )

        print(
            "ERROR TYPE:",
            type(exc).__name__,
        )

        print(
            "ERROR:",
            str(exc),
        )

        print(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Simulation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# =========================================================
# Intent & Sentiment Analysis Endpoint
# =========================================================

@app.post(
    "/analysis/intent-sentiment",
    response_model=AnalyzeMessageResponse,
)
def analyze_intent_and_sentiment(
    request: AnalyzeMessageRequest
):

    try:

        print(
            "\n================ INTENT & SENTIMENT REQUEST ================"
        )

        history = request.history or []

        if (
            request.conversation_id is not None
            and not history
        ):

            db_history = get_recent_messages(
                request.conversation_id,
                limit=10,
            )

            if db_history:
                history = db_history

        analysis = analyze_customer_message(
            message=request.message,
            history=history,
        )

        return AnalyzeMessageResponse(
            success=True,
            analysis=analysis,
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            "ANALYSIS ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# =========================================================
# Knowledge Recommendation Endpoint
# =========================================================

@app.post(
    "/recommendations/knowledge",
    response_model=KnowledgeRecommendationResponse,
)
def get_knowledge_recommendations(
    request: KnowledgeRecommendationRequest
):

    try:

        print(
            "\n================ KNOWLEDGE RECOMMENDATION REQUEST ================"
        )

        history = request.history or []

        if (
            request.conversation_id is not None
            and not history
        ):

            db_history = get_recent_messages(
                request.conversation_id,
                limit=10,
            )

            if db_history:
                history = db_history

        recs = recommend_knowledge(
            message=request.message,
            history=history,
            top_k=request.top_k,
        )

        return KnowledgeRecommendationResponse(
            success=True,
            query=request.message,
            recommendations=recs,
            count=len(recs),
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            "KNOWLEDGE RECOMMENDATION ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Recommendation retrieval failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# =========================================================
# Task 8 - Post-Interaction Summary
# =========================================================

@app.get(
    "/task8/post-interaction-summary/{conversation_id}"
)
def get_post_interaction_summary(
    conversation_id: int
):

    try:

        print(
            "\n================ TASK 8 POST-INTERACTION SUMMARY ================"
        )

        # -------------------------------------------------
        # Build Task 8 report
        # -------------------------------------------------

        report = build_post_interaction_summary(
            conversation_id=conversation_id
        )

        print(
            "TASK 8: Summary generated successfully"
        )

        return report

    except ValueError as exc:

        print(
            "TASK 8 SUMMARY ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except HTTPException:
        raise

    except Exception as exc:

        print(
            "TASK 8 SUMMARY ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Post-interaction summary failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )


# =========================================================
# Task 8 - Performance Analytics
# =========================================================

@app.get(
    "/task8/performance-analytics"
)
def get_performance_analytics():

    try:

        print(
            "\n================ TASK 8 PERFORMANCE ANALYTICS ================"
        )

        # -------------------------------------------------
        # Build Task 8 analytics
        # -------------------------------------------------

        analytics = build_performance_analytics()

        print(
            "TASK 8: Performance analytics generated successfully"
        )

        return analytics

    except Exception as exc:

        print(
            "TASK 8 PERFORMANCE ANALYTICS ERROR:",
            repr(exc),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"Performance analytics failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )