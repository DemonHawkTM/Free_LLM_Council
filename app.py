# --- PASSWORD PROTECTION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input
        st.text_input(
            "Security Clearance Required", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input again
        st.text_input(
            "Access Denied. Try Again", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Do not run the rest of the app if password is wrong

import streamlit as st
import os
import time
import g4f
from groq import Groq
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# --- PAGE CONFIG (Must be first) ---
st.set_page_config(
    page_title="LLM Council",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS STYLING (The "Polish") ---
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Card-like containers for opinions */
    div.stExpander {
        background-color: #1a1c24;
        border: 1px solid #30333d;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Header Styling */
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Custom Button Styling */
    .stButton button {
        background: linear-gradient(45deg, #4b6cb7, #182848);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px rgba(75, 108, 183, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
os.environ["GOOGLE_API_KEY"] = st.secrets.get("GOOGLE_API_KEY", "")

if not os.environ["GROQ_API_KEY"] or not os.environ["GOOGLE_API_KEY"]:
    st.error("⚠️ Keys missing! Please check your .streamlit/secrets.toml")
    st.stop()

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# --- DEFINITIONS ---
COUNCIL_MEMBERS = [
    {"type": "groq", "id": "llama-3.3-70b-versatile", "name": "Llama 3.3 (Logic)", "icon": "🧠"},
    {"type": "groq", "id": "llama-3.1-8b-instant", "name": "Llama 3.1 (Speed)", "icon": "⚡"}, 
    {"type": "g4f",  "id": "gpt-4o", "name": "GPT-4o (Wildcard)", "icon": "🔮"} 
]

# --- FUNCTIONS ---
def get_groq_opinion(model_id, prompt, system_role="expert council member"):
    try:
        completion = groq_client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": f"You are an {system_role}. Be concise."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"

def get_chatgpt_opinion(prompt):
    """Bulletproof Failover Rotator"""
    providers = [
        g4f.Provider.PollinationsAI, 
        g4f.Provider.DuckDuckGo, 
        g4f.Provider.Blackbox
    ]
    
    for provider in providers:
        try:
            client = g4f.client.Client(provider=provider)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            if response.choices[0].message.content:
                return f"[{provider.__name__}] {response.choices[0].message.content}"
        except Exception:
            continue

    # Fallback to Groq if all hacks fail
    try:
        return "[System Fallback] " + get_groq_opinion("llama-3.3-70b-versatile", prompt)
    except:
        return "Critical Error: All APIs Failed."

def get_council_opinion(member, prompt):
    if member["type"] == "groq":
        return get_groq_opinion(member["id"], prompt)
    elif member["type"] == "g4f":
        return get_chatgpt_opinion(prompt)

def get_chairman_consensus(prompt, council_results):
    try:
        chairman = genai.GenerativeModel('gemini-2.5-flash')
    except:
        chairman = genai.GenerativeModel('gemini-1.5-flash')
        
    consensus_prompt = f"""
    You are the Chairman. User Query: "{prompt}"
    Council Opinions:
    """
    for res in council_results:
        consensus_prompt += f"\n--- {res['model_name']} ---\n{res['opinion']}\n"
        
    consensus_prompt += "\nSynthesize a single perfect 10/10 answer."
    
    try:
        time.sleep(1) # Rate limit safety
        response = chairman.generate_content(consensus_prompt)
        return response.text
    except Exception as e:
        return f"Chairman Error: {e}"

# --- MAIN UI ---
st.title("🧙‍♂️ The LLM Council")
st.caption("A multi-model reasoning system powered by Groq, G4F, & Gemini")

# Input Area
with st.container():
    user_query = st.text_area("Agenda for the Council:", height=100, placeholder="e.g. Help me debug this Python code...")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        run_btn = st.button("Summon Council", use_container_width=True)

if run_btn and user_query:
    st.toast("The Council has been summoned...", icon="🔔")
    
    # 1. THE COUNCIL DELIBERATES (Status Container)
    council_results = []
    
    # This creates a visually pleasing expandable box that logs steps
    with st.status("The Council is deliberating...", expanded=True) as status:
        
        # Create placeholders for the cards immediately so layout is stable
        cols = st.columns(len(COUNCIL_MEMBERS))
        placeholders = [col.empty() for col in cols]
        
        # Parallel Execution
        with ThreadPoolExecutor() as executor:
            future_map = {executor.submit(get_council_opinion, m, user_query): m for m in COUNCIL_MEMBERS}
            
            for i, future in enumerate(future_map):
                member = future_map[future]
                
                # Update status log
                status.write(f"⏳ {member['name']} is thinking...")
                
                # Get result
                opinion = future.result()
                council_results.append({"model_name": member["name"], "opinion": opinion})
                
                # Render the card immediately in the column
                with cols[i]:
                    with st.expander(f"{member['icon']} {member['name']}", expanded=True):
                        st.markdown(opinion)
                        st.caption("✅ Opinion Submitted")
        
        status.update(label="Council Deliberation Complete!", state="complete", expanded=False)

    # 2. THE CHAIRMAN SYNTHESIZES
    st.divider()
    
    # A cleaner looking spinner for the final step
    with st.spinner("🧠 The Chairman (Gemini) is synthesizing the final verdict..."):
        final_verdict = get_chairman_consensus(user_query, council_results)
    
    # Final Result Card
    st.markdown("### 🏛️ The Final Verdict")
    st.success(final_verdict)
    
    st.toast("Verdict Reached!", icon="✅")
