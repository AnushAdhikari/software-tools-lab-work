import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import pipeline

FAQS = [
    ("When is the library open?", "The main library is open 8 AM to 10 PM Monday to Thursday, 8 AM to 6 PM on Friday, and 10 AM to 4 PM on weekends. During exam weeks it stays open until midnight."),
    ("How do I book a study room?", "Study rooms are booked through the library portal up to 7 days in advance. Each booking is 2 hours, and you can hold one booking at a time."),
    ("Where do I get my student ID card?", "Student ID cards are issued at the Student Services desk in Building A, room 104. Bring a government photo ID and your acceptance letter."),
    ("How do I connect to campus Wi-Fi?", "Connect to the network called NB-Secure and sign in with your student email and password. Guests can use NB-Guest, which needs no password but is slower."),
    ("What should I do if I forget my password?", "Reset it at the IT self-service portal using your recovery email. If that fails, visit the IT Help Desk in Building C, room 210."),
    ("Where can I print?", "Printers are in the library and in Building C. Black-and-white pages cost 10 cents and colour pages cost 50 cents, charged to your student card."),
    ("Is there free tutoring?", "Yes. The Learning Centre offers free drop-in tutoring for math, writing and programming, Monday to Friday from 10 AM to 4 PM in Building B, room 120."),
    ("How do I see a counsellor?", "Counselling is free and confidential. Book an appointment online through Student Wellness or call the front desk. Same-day crisis appointments are available."),
    ("Where can I park?", "Students park in Lots 3 and 4 with a semester permit, which costs $180 and is bought online. Lots 1 and 2 are for staff and visitors."),
    ("What is the deadline to drop a course without a failing grade?", "You can drop a course without academic penalty until the end of week 8. After that, the course shows as a failing grade on your transcript."),
    ("Does the cafeteria have vegetarian options?", "Yes. The main cafeteria in Building A has a vegetarian and vegan station every weekday from 11 AM to 3 PM, and halal options are labelled at every counter."),
    ("How do I contact Accessibility Services?", "Email Accessibility Services or visit Building A, room 112. Register early in the term so accommodations such as extra exam time are in place before your first test."),
]
THRESHOLD = 0.40
SYSTEM_PROMPT = (
    "You are the campus help assistant for Northbridge College. "
    "Answer the student's question using ONLY the FAQ context provided. "
    "If the context does not contain the answer, say you don't know and suggest "
    "contacting Student Services. Answer in one to three sentences."
)

@st.cache_resource
def load():
    chunks = [f"Q: {q}\nA: {a}" for q, a in FAQS]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(chunks, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    return chunks, model, index, llm

def retrieve(question, chunks, model, index, k=3):
    q = model.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0]) if s >= THRESHOLD]

def generate(question, context, llm):
    prompt = "<context>\n" + "\n\n".join(c for _, c in context) + f"\n</context>\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    output = llm(messages, max_new_tokens=150, do_sample=False)
    return output[0]["generated_text"][-1]["content"].strip()

st.title("Campus FAQ Chatbot")
st.caption("Northbridge College (sample data) - answers come only from the FAQ.")

with st.spinner("Loading models (first run downloads about 1 GB)..."):
    chunks, model, index, llm = load()
question = st.text_input("Ask a question about campus:")

if question:
    context = retrieve(question, chunks, model, index)
    if not context:
        st.warning("Sorry, I don't know that one. Please contact Student Services.")
    else:
        with st.spinner("Thinking..."):
            st.write(generate(question, context, llm))
        with st.expander("Sources used"):
            for score, chunk in context:
                st.markdown(f"**{score:.2f}**  \n{chunk}")
