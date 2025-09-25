import warnings
import os
import time
import re
import json
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
        
        if last_quiz:

            user_answer_match = re.search(r'\b([a-dA-D])\b', prompt)
            
            if user_answer_match:
                user_answer = user_answer_match.group(1).lower()
                correct_answer = last_quiz.get('correct_answer', '').lower()
                
                if user_answer == correct_answer:
                    response = f"✅ Correct! Well done!"
                else:
                    response = f"❌ Incorrect. The correct answer was {correct_answer.upper()}."
                
                history.append({"role": "assistant", "content": response, "timestamp": time.time()})
                session["histories"][session_id] = history
                session.modified = True
                return jsonify({"answer": f"<p>{response}</p>"})


        if "quiz" in prompt.lower():

            quiz_json_text = llm(
                "Generate a single multiple-choice question on cybersecurity. "
                "Return in valid JSON only, format: "
                "{'question': '...', 'options': {'a':'...', 'b':'...', 'c':'...', 'd':'...'}, 'answer':'a'}. "
                "DO NOT include the correct answer in the question text or options."
            )
            quiz_json_text = quiz_json_text.replace("'", '"').strip()

            try:
                quiz_json = json.loads(quiz_json_text)
                question_text = quiz_json["question"]
                options = quiz_json["options"]
                correct_answer = quiz_json["answer"].lower()

                options_html = "".join([f"<p>{k.upper()}: {v}</p>" for k, v in options.items()])
                instructions = "<p>Please respond with the letter of your answer (a, b, c, or d).</p>"
                quiz_html = f"<div class='quiz-block'><p>{question_text}</p>{options_html}{instructions}</div>"

                history.append({
                    "role": "assistant",
                    "content": quiz_html,
                    "type": "quiz",
                    "correct_answer": correct_answer,
                    "timestamp": time.time()
                })
                session["histories"][session_id] = history
                session.modified = True
                return jsonify({"answer": quiz_html})

            except json.JSONDecodeError:
                quiz_text = llm(
                    "Generate a single multiple-choice question on cybersecurity with exactly 4 options labeled a), b), c), d). "
                    "Provide only the question and options. "
                    "DO NOT include the correct answer in the response. "
                    "At the end, indicate the correct answer with 'Correct answer: X' on a separate line where X is a, b, c, or d."
                )
                
                correct_match = re.search(r'Correct answer:\s*([a-d])', quiz_text, re.IGNORECASE)
                if not correct_match:
                    correct_match = re.search(r'Answer:\s*([a-d])', quiz_text, re.IGNORECASE)
                if not correct_match:
                    correct_match = re.search(r'\(([a-d])\)\s*$', quiz_text, re.IGNORECASE)
                
                correct_answer = correct_match.group(1).lower() if correct_match else 'a'
                
                quiz_display = re.sub(r'(Correct answer|Answer):\s*[a-d]', '', quiz_text, flags=re.IGNORECASE)
                quiz_display = re.sub(r'\(([a-d])\)\s*$', '', quiz_display, flags=re.IGNORECASE)
                
                quiz_html = markdown.markdown(quiz_display)
                quiz_html = f"<div class='quiz-block'>{quiz_html}<p>Please respond with the letter of your answer (a, b, c, or d).</p></div>"
                
                history.append({
                    "role": "assistant",
                    "content": quiz_html,
                    "type": "quiz",
                    "correct_answer": correct_answer,
                    "timestamp": time.time()
                })
                session["histories"][session_id] = history
                session.modified = True
                return jsonify({"answer": quiz_html})

        chat_history = [(m["role"], m["content"]) for m in history if m["role"] in ["user", "assistant"]]
        result = chat_rag_chain({"question": prompt, "chat_history": chat_history})
        answer_content = result.get("answer", "No answer.")
        answer_html = markdown.markdown(answer_content)

        history.append({"role": "assistant", "content": answer_html, "timestamp": time.time()})
        session["histories"][session_id] = history
        session.modified = True

        return jsonify({"answer": answer_html})
    except Exception as e:
        think_block = f"🤖 <think> Error: {e} </think>"
        return jsonify({"thinking": think_block, "answer": "<p>No answer.</p>"}), 500

@app.route("/reset")
def reset():
    session.clear()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)