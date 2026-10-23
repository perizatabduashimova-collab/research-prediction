from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Оқу нәтижесін болжау", layout="centered")
st.title("1-Зерттеу сұрағы: Сегізінші аптадағы тест балын болжау")
st.write("Алғашқы төрт аптадағы көрсеткіштер негізінде студенттің үлгерімін анықтау.")

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "research_model.joblib"
    return joblib.load(path)

bundle = load_model()
model = bundle["model"]
features = bundle["features"]

with st.form("prediction_form"):
    pretest_score = st.slider("Бастапқы тест балы", 0, 100, 60)
    attendance_pct = st.slider("Сабаққа қатысу үлесі (%)", 0, 100, 80)
    submitted_tasks = st.number_input("Орындалған жұмыстар саны (3-10)", min_value=0, max_value=10, value=7)
    late_tasks = st.number_input("Кеш тапсырылған жұмыстар саны", min_value=0, max_value=10, value=1)
    
    submitted = st.form_submit_button("Болжам жасау")

if submitted:
    if late_tasks > submitted_tasks:
        st.error("Қате: Кеш тапсырылған жұмыстар саны орындалған жұмыстардан артық болмауы тиіс!")
    else:
        row = pd.DataFrame([{
            "pretest_score": pretest_score,
            "attendance_pct": attendance_pct,
            "submitted_tasks": submitted_tasks,
            "late_tasks": late_tasks
        }])
        
        prediction = float(model.predict(row[features])[0])
        st.metric("Болжамды тест балы (8-ші апта)", f"{prediction:.1f} / 100")
        st.caption("Бұл болжам тек оқу үдерісін жоспарлауға көмектеседі және түпкілікті баға болып саналмайды.")
