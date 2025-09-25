import warnings
import os
import time
import re
from datetime import timedelta

import yaml
import markdown
from flask import Flask, render_template, request, session, redirect, url_for, jsonify

from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.vectorstores import Chroma

warnings.filterwarnings("ignore", category=DeprecationWarning)

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

OLLAMA_HOST = config['ollama']['host']
LLM_MODEL = config['ollama']['llm_model']
EMBEDDING_MODEL = config['ollama']['embedding_model']
VECTOR_DIR = config['data_ingestion']['vector_store']['persist_directory']
COLLECTION_NAME = config['data_ingestion']['vector_store']['collection_name']
RETRIEVAL_K = config['rag']['retrieval_k']
CHAIN_TYPE = config['rag'].get('chain_type', 'stuff')

llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_HOST, temperature=0.3)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST)

if not os.path.exists(VECTOR_DIR):
    raise FileNotFoundError(f"Chroma directory not found: {VECTOR_DIR}. Run your data ingestion first.")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

chat_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

app = Flask(__name__)
app.secret_key = "Oxygen200373#"
app.permanent_session_lifetime = timedelta(minutes=15)
app_name = "SecuritasBotMM"

HISTORY_WINDOW_SECONDS = 5 * 60

def filter_recent_messages(history):
    cutoff = time.time() - HISTORY_WINDOW_SECONDS
    return [msg for msg in history if msg.get("timestamp", 0) >= cutoff]

@app.route("/")
def index():
    if "messages" not in session:
        session["messages"] = []
    if "histories" not in session:
        session["histories"] = {}
    return render_template("Home.html", app_name=app_name, messages=session["messages"])

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()
    session_id = str(data.get("session_id", "default"))

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        if "histories" not in session:
            session["histories"] = {}
        if session_id not in session["histories"]:
            session["histories"][session_id] = []
        history = session["histories"][session_id]

        history = filter_recent_messages(history)

        last_quiz = next((msg for msg in reversed(history) if msg.get("type") == "quiz"), None)
        quiz_answer_match = None

        if last_quiz:
            quiz_answer_match = re.search(
                r"(?:the answer of\s*(\d+)\s*is\s*([a-dA-D1-5]))|"
                r"(?:the answer\s*is\s*([a-dA-D1-5]))|"
                r"(?:answer[:\s]+([a-dA-D1-5]))",
                prompt, re.IGNORECASE
            )

        if last_quiz and quiz_answer_match:
            q_num = quiz_answer_match.group(1) or 1
            user_ans = quiz_answer_match.group(2) or quiz_answer_match.group(3) or quiz_answer_match.group(4)
            try:
                q_num = int(q_num)
            except:
                q_num = 1

            answer_key = last_quiz.get("answer_key", {})
            correct_ans = answer_key.get(q_num)

            if correct_ans:
                if str(user_ans).strip().lower() == str(correct_ans).strip().lower():
                    result_text = f"✅ Correct! Question {q_num} answer is <b>{correct_ans}</b>."
                else:
                    result_text = f"❌ Wrong! Question {q_num} answer is <b>{correct_ans}</b>."
            else:
                result_text = f"⚠️ Could not find the correct answer for Question {q_num}."

            history.append({"role": "assistant", "content": result_text, "timestamp": time.time()})
            session["histories"][session_id] = history
            session.modified = True
            return jsonify({ "answer": f"<p>{result_text}</p>"})

        if "quiz" in prompt.lower():
            quiz_text = llm(
                f"Generate a single multiple-choice question on cybersecurity. "
                f"Provide **only the question text**. "
                f"Do NOT include the options, correct answer, hints, or any extra notes."

            )

            answer_key = {}
            for line in quiz_text.splitlines():
                m = re.match(r"(\d+)\.\s*([a-dA-D1-5])", line.strip())
                if m:
                    answer_key[int(m.group(1))] = m.group(2)

            answer_html = markdown.markdown(quiz_text)
            answer_html = f"<div class='quiz-block'>{answer_html}</div>"

            history.append({
                "role": "assistant",
                "content": answer_html,
                "type": "quiz",
                "answer_key": answer_key
            })
            session["histories"][session_id] = history
            session.modified = True
            return jsonify({ "answer": answer_html})

        chat_history = [(m["role"], m["content"]) for m in history if m["role"] in ["user", "assistant"]]
        result = chat_rag_chain({"question": prompt, "chat_history": chat_history})
        answer_content = result.get("answer", "No answer.")
        answer_html = markdown.markdown(answer_content)

        history.append({"role": "assistant", "content": answer_html, "timestamp": time.time()})
        session["histories"][session_id] = history
        session.modified = True

        return jsonify({ "answer": answer_html})
    except Exception as e:
        think_block = f"🤖 <think> Error: {e} </think>"
        return jsonify({"thinking": think_block, "answer": "<p>No answer.</p>"}), 500

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
