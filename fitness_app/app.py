"""
FitAI — AI-powered fitness tracker (EN / FR)
Free  : workout logging, history, static templates, streak, meal advice
Premium : AI program generation, adaptation, progress charts, AI Coach, AI meal plan
"""

import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

# On Streamlit Cloud secrets come from st.secrets; locally from .env
try:
    for key in ("GEMINI_API_KEY", "DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

import pandas as pd
from datetime import date

from database import (
    init_db, get_user, update_user,
    log_session, get_sessions, log_exercise, get_session_exercises,
    get_exercise_history, save_program, get_programs, get_streak,
    register_user, login_user, create_auth_token, validate_auth_token, delete_auth_token,
)
from translations import (
    t, goals_display, levels_display, goal_to_en, level_to_en,
    goal_to_display, meal_advice, GOALS_EN, LEVELS_EN,
)

# ── Config ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="FitAI 💪", page_icon="💪", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
  .block-container { padding-top: 0.75rem; padding-bottom: 4rem; }
  .stTabs [data-baseweb="tab-list"] { gap: 0; }
  .stTabs [data-baseweb="tab"] { font-size: 0.78rem; padding: 0.45rem 0.6rem; }
  div[data-testid="metric-container"] { background: #1e1e2e; border-radius: 10px; padding: 0.6rem; }
  @media (max-width: 480px) {
    .stTabs [data-baseweb="tab"] { font-size: 0.68rem; padding: 0.35rem 0.4rem; }
  }
</style>
""", unsafe_allow_html=True)

init_db()

# ── Session state ──────────────────────────────────────────────────────────────

for key, default in [("uid", None), ("chat", []), ("lang", "en")]:
    if key not in st.session_state:
        st.session_state[key] = default

lang = st.session_state.lang

# ── Restore session from URL query param ───────────────────────────────────────

if not st.session_state.uid:
    token = st.query_params.get("session")
    if token:
        user_from_token = validate_auth_token(token)
        if user_from_token:
            st.session_state.uid = user_from_token["id"]

# ── Auth gate ──────────────────────────────────────────────────────────────────

if not st.session_state.uid:
    # Language toggle on auth page
    col_title, col_lang = st.columns([5, 1])
    with col_title:
        st.title("💪 FitAI")
        st.markdown(t("ai_powered", lang))
    with col_lang:
        chosen = st.radio("🌐", ["EN", "FR"], horizontal=True,
                          index=0 if lang == "en" else 1, label_visibility="collapsed")
        if chosen.lower() != lang:
            st.session_state.lang = chosen.lower()
            st.rerun()

    st.divider()
    tab_login, tab_register = st.tabs([t("login", lang), t("signup", lang)])

    with tab_login:
        with st.form("login_form"):
            email    = st.text_input(t("email", lang))
            password = st.text_input(t("password", lang), type="password")
            if st.form_submit_button(t("login", lang), type="primary", use_container_width=True):
                user = login_user(email, password)
                if user:
                    token = create_auth_token(user["id"])
                    st.query_params["session"] = token
                    st.session_state.uid = user["id"]
                    st.session_state.chat = []
                    st.rerun()
                else:
                    st.error(t("invalid_credentials", lang))

    with tab_register:
        with st.form("register_form"):
            name = st.text_input(t("full_name", lang))
            email = st.text_input(t("email", lang), key="reg_email")
            pw    = st.text_input(t("password", lang), type="password", key="reg_pw")
            pw2   = st.text_input(t("confirm_password", lang), type="password")
            c1, c2 = st.columns(2)
            age    = c1.number_input(t("age", lang), 10, 100, 25)
            weight = c2.number_input(t("weight_kg", lang), 30.0, 300.0, 70.0, step=0.5)
            c3, c4 = st.columns(2)
            height = c3.number_input(t("height_cm", lang), 100, 250, 170)
            level_disp = c4.selectbox(t("fitness_level_label", lang), levels_display(lang))
            goal_disp  = st.selectbox(t("primary_goal", lang), goals_display(lang))
            if st.form_submit_button(t("create_account", lang), type="primary", use_container_width=True):
                if not name.strip() or not email.strip() or not pw:
                    st.error(t("fill_all_fields", lang))
                elif pw != pw2:
                    st.error(t("passwords_no_match", lang))
                elif len(pw) < 6:
                    st.error(t("password_too_short", lang))
                else:
                    uid = register_user(email, pw, name,
                                        age, weight, height,
                                        goal_to_en(goal_disp, lang),
                                        level_to_en(level_disp, lang))
                    if uid:
                        token = create_auth_token(uid)
                        st.query_params["session"] = token
                        st.session_state.uid = uid
                        st.session_state.chat = []
                        st.rerun()
                    else:
                        st.error(t("email_exists", lang))
    st.stop()

# ── Logged-in: sidebar ─────────────────────────────────────────────────────────

user = get_user(st.session_state.uid)

with st.sidebar:
    # Language toggle
    chosen = st.radio("🌐 Language", ["EN", "FR"], horizontal=True,
                      index=0 if lang == "en" else 1)
    if chosen.lower() != lang:
        st.session_state.lang = chosen.lower()
        lang = chosen.lower()
        st.rerun()

    st.divider()
    st.markdown(f"**{user['name']}**")
    st.caption(user.get("email") or "")
    if st.button(t("logout", lang), use_container_width=True):
        token = st.query_params.get("session")
        if token:
            delete_auth_token(token)
        st.query_params.clear()
        st.session_state.uid = None
        st.session_state.chat = []
        st.rerun()

st.divider()

# ── Free templates ─────────────────────────────────────────────────────────────

FREE_TEMPLATES = {
    "StrongLifts 5×5  (3-day, beginner)": {
        "description": "The most proven beginner strength program. Add 2.5 kg every session.",
        "days_per_week": 3,
        "days": [
            {"label": "Workout A — Mon/Fri", "exercises": [
                {"name": "Squat",           "sets": 5, "reps": "5"},
                {"name": "Bench Press",     "sets": 5, "reps": "5"},
                {"name": "Barbell Row",     "sets": 5, "reps": "5"},
                {"name": "Dips",            "sets": 3, "reps": "8"},
                {"name": "Face Pull",       "sets": 3, "reps": "15"},
            ]},
            {"label": "Workout B — Wed", "exercises": [
                {"name": "Squat",           "sets": 5, "reps": "5"},
                {"name": "Overhead Press",  "sets": 5, "reps": "5"},
                {"name": "Deadlift",        "sets": 1, "reps": "5"},
                {"name": "Pull-ups",        "sets": 3, "reps": "8"},
                {"name": "Barbell Curl",    "sets": 3, "reps": "10"},
            ]},
        ],
    },
    "Push / Pull / Legs  (6-day, intermediate)": {
        "description": "High-frequency 6-day split for hypertrophy. Alternate A/B weeks.",
        "days_per_week": 6,
        "days": [
            {"label": "Monday — Push (Chest · Shoulders · Triceps)", "exercises": [
                {"name": "Bench Press",      "sets": 4, "reps": "6–10"},
                {"name": "Overhead Press",   "sets": 3, "reps": "8–12"},
                {"name": "Incline DB Press", "sets": 3, "reps": "10–12"},
                {"name": "Lateral Raises",   "sets": 4, "reps": "15–20"},
                {"name": "Tricep Pushdown",  "sets": 3, "reps": "12–15"},
            ]},
            {"label": "Tuesday — Pull (Back · Biceps)", "exercises": [
                {"name": "Pull-ups",         "sets": 4, "reps": "6–10"},
                {"name": "Barbell Row",      "sets": 4, "reps": "8–10"},
                {"name": "Face Pull",        "sets": 3, "reps": "15–20"},
                {"name": "Barbell Curl",     "sets": 3, "reps": "10–12"},
                {"name": "Hammer Curl",      "sets": 3, "reps": "12–15"},
            ]},
            {"label": "Wednesday — Legs", "exercises": [
                {"name": "Squat",                "sets": 4, "reps": "6–10"},
                {"name": "Romanian Deadlift",    "sets": 3, "reps": "8–12"},
                {"name": "Leg Press",            "sets": 3, "reps": "12–15"},
                {"name": "Leg Curl",             "sets": 3, "reps": "12–15"},
                {"name": "Calf Raises",          "sets": 4, "reps": "15–20"},
            ]},
            {"label": "Thu–Sat: repeat Push/Pull/Legs", "exercises": []},
        ],
    },
    "Fat Loss  (4-day, any level)": {
        "description": "2 strength + 2 cardio days. Maximises caloric burn while preserving muscle.",
        "days_per_week": 4,
        "days": [
            {"label": "Monday — Upper Strength", "exercises": [
                {"name": "DB Bench Press",   "sets": 3, "reps": "12–15"},
                {"name": "Cable Row",        "sets": 3, "reps": "12–15"},
                {"name": "Overhead Press",   "sets": 3, "reps": "12–15"},
                {"name": "Lateral Raises",   "sets": 3, "reps": "15–20"},
                {"name": "Tricep Pushdown",  "sets": 3, "reps": "12–15"},
            ]},
            {"label": "Tuesday — LISS Cardio", "exercises": [
                {"name": "Walk / Bike / Elliptical", "duration_min": 45, "type": "cardio"},
            ]},
            {"label": "Thursday — Lower Strength", "exercises": [
                {"name": "Goblet Squat",     "sets": 3, "reps": "15"},
                {"name": "Hip Thrust",       "sets": 3, "reps": "12–15"},
                {"name": "Walking Lunges",   "sets": 3, "reps": "12 each"},
                {"name": "Leg Curl",         "sets": 3, "reps": "12–15"},
                {"name": "Calf Raises",      "sets": 3, "reps": "20"},
            ]},
            {"label": "Saturday — HIIT", "exercises": [
                {"name": "HIIT Circuit (bike / sprints / burpees)", "duration_min": 25, "type": "cardio"},
            ]},
        ],
    },
    "Endurance Base  (5-day, beginner/intermediate)": {
        "description": "Build your aerobic base with structured run/bike sessions + 2 strength days.",
        "days_per_week": 5,
        "days": [
            {"label": "Monday — Easy Run", "exercises": [
                {"name": "Easy jog / zone 2", "duration_min": 35, "type": "cardio"},
            ]},
            {"label": "Tuesday — Strength A (Full Body)", "exercises": [
                {"name": "Squat",        "sets": 3, "reps": "10"},
                {"name": "Pull-ups",     "sets": 3, "reps": "8"},
                {"name": "Push-ups",     "sets": 3, "reps": "15"},
                {"name": "Dumbbell Row", "sets": 3, "reps": "10"},
                {"name": "Plank",        "sets": 3, "reps": "45s"},
            ]},
            {"label": "Wednesday — Tempo Run", "exercises": [
                {"name": "Tempo run (zone 3–4)", "duration_min": 30, "type": "cardio"},
            ]},
            {"label": "Friday — Strength B (Full Body)", "exercises": [
                {"name": "Deadlift",       "sets": 3, "reps": "8"},
                {"name": "Overhead Press", "sets": 3, "reps": "10"},
                {"name": "Lunges",         "sets": 3, "reps": "12"},
                {"name": "Push-ups",       "sets": 3, "reps": "15"},
                {"name": "Plank",          "sets": 3, "reps": "45s"},
            ]},
            {"label": "Sunday — Long Easy Run", "exercises": [
                {"name": "Long run / bike (zone 2)", "duration_min": 60, "type": "cardio"},
            ]},
        ],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN NAVIGATION TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_home, tab_log, tab_prog, tab_stats, tab_ai, tab_nutrition = st.tabs([
    t("tab_home", lang), t("tab_log", lang), t("tab_programs", lang),
    t("tab_progress", lang), t("tab_ai", lang), t("tab_nutrition", lang),
])

# ═══════════════════════════════════════════════════════
# TAB 1 — HOME
# ═══════════════════════════════════════════════════════

with tab_home:
    streak       = get_streak(st.session_state.uid)
    all_sessions = get_sessions(st.session_state.uid, limit=100)
    today        = str(date.today())
    today_done   = any(s["date"] == today for s in all_sessions[:7])

    st.markdown(f"### {t('hey', lang)} {user['name']}! 👋")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t("streak", lang),   f"{streak} d")
    c2.metric(t("workouts", lang), len(all_sessions))
    c3.metric(t("goal_metric", lang), goal_to_display(user["goal"], lang).split("/")[0].strip()[:12])
    c4.metric(t("level_metric", lang), user["fitness_level"][:5])

    st.markdown(f"#### {t('today_section', lang)}")
    if today_done:
        st.success(t("workout_logged_today", lang))
    else:
        st.warning(t("no_workout_today", lang))

    st.markdown(f"#### {t('recent_workouts', lang)}")
    recent = all_sessions[:5]
    if recent:
        for s in recent:
            d        = date.fromisoformat(s["date"])
            days_ago = (date.today() - d).days
            if days_ago == 0:
                label = t("today_label", lang)
            elif days_ago == 1:
                label = t("yesterday", lang)
            else:
                label = f"{days_ago}{t('days_ago', lang)}"
            with st.container(border=True):
                cc1, cc2 = st.columns([3, 1])
                cc1.markdown(f"**{label}** — {s['ex_count']} {t('exercises_label', lang)}")
                cc2.caption(s["date"])
                if s.get("notes"):
                    st.caption(s["notes"])
    else:
        st.info(t("no_workouts_yet", lang))

    if not user["is_premium"]:
        st.divider()
        with st.container(border=True):
            st.markdown(t("upgrade_title", lang))
            st.markdown(t("upgrade_desc", lang))
            col_free, col_paid = st.columns(2)
            with col_free:
                st.markdown(t("free_label", lang))
                st.markdown(t("free_features", lang))
            with col_paid:
                st.markdown(t("premium_label", lang))
                st.markdown(t("premium_features", lang))
            if st.button(t("activate_premium", lang), type="primary", use_container_width=True):
                update_user(st.session_state.uid, is_premium=1)
                st.success(t("premium_activated", lang))
                st.rerun()
    else:
        st.success(t("premium_active", lang))

# ═══════════════════════════════════════════════════════
# TAB 2 — LOG WORKOUT
# ═══════════════════════════════════════════════════════

with tab_log:
    st.markdown(t("log_title", lang))

    with st.form("workout_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        session_date = c1.date_input(t("date", lang), date.today())
        notes        = c2.text_input(t("notes", lang), placeholder=t("notes_placeholder", lang))

        n_ex = st.slider(t("num_exercises", lang), 1, 12, 4)
        exercises_data = []

        for i in range(n_ex):
            st.markdown(f"**{t('exercise_n', lang)} {i + 1}**")
            cn, ct = st.columns([3, 1])
            ex_name = cn.text_input(t("name", lang), key=f"en{i}")
            ex_type = ct.selectbox(t("type", lang), [t("strength", lang), t("cardio", lang)], key=f"et{i}")

            if ex_type == t("strength", lang):
                cs, cr, cw = st.columns(3)
                sets   = cs.number_input(t("sets", lang),  1, 20,    3,    key=f"es{i}")
                reps   = cr.number_input(t("reps", lang),  1, 100,  10,    key=f"er{i}")
                weight = cw.number_input(t("kg", lang),    0.0, 500.0, 0.0, key=f"ew{i}", step=2.5)
                exercises_data.append({"name": ex_name, "type": "strength",
                                       "sets": sets, "reps": reps, "weight": weight})
            else:
                cd, ci = st.columns(2)
                duration  = cd.number_input(t("minutes", lang), 1, 300, 30, key=f"ed{i}")
                intensity = ci.selectbox(t("intensity", lang),
                    [t("low", lang), t("moderate", lang), t("high", lang), t("hiit", lang)], key=f"ei{i}")
                exercises_data.append({"name": ex_name, "type": "cardio",
                                       "duration": duration, "intensity": intensity})

            if i < n_ex - 1:
                st.markdown("---")

        submitted = st.form_submit_button(t("save_workout", lang), type="primary", use_container_width=True)

    if submitted:
        sid    = log_session(st.session_state.uid, session_date, notes)
        logged = 0
        for ex in exercises_data:
            if ex["name"].strip():
                if ex["type"] == "strength":
                    log_exercise(sid, ex["name"], ex["sets"], ex["reps"], ex["weight"], ex_type="strength")
                else:
                    log_exercise(sid, ex["name"], duration=ex["duration"], ex_type="cardio")
                logged += 1
        if logged > 0:
            st.success(f"✅ {logged} {t('exercises_saved_msg', lang)} {session_date}!")
            st.balloons()
        else:
            st.warning(t("no_exercises_saved", lang))

# ═══════════════════════════════════════════════════════
# TAB 3 — PROGRAMS
# ═══════════════════════════════════════════════════════

with tab_prog:
    st.markdown(t("programs_title", lang))
    pt_free, pt_mine = st.tabs([t("free_templates_tab", lang), t("my_programs_tab", lang)])

    with pt_free:
        tmpl_name = st.selectbox(t("choose_template", lang), list(FREE_TEMPLATES.keys()))
        tmpl      = FREE_TEMPLATES[tmpl_name]
        st.caption(f"📅 {tmpl['days_per_week']} {t('days_week', lang)} — {tmpl['description']}")

        for day in tmpl["days"]:
            with st.expander(day["label"]):
                exs = day["exercises"]
                if not exs:
                    st.caption("(same pattern — see above days)")
                for ex in exs:
                    if "duration_min" in ex:
                        st.write(f"• **{ex['name']}** — {ex['duration_min']} min cardio")
                    else:
                        st.write(f"• **{ex['name']}** — {ex['sets']} × {ex['reps']}")

        if st.button(t("save_template_btn", lang), use_container_width=True):
            save_program(st.session_state.uid, tmpl_name, tmpl, is_ai=False)
            st.success(f'"{tmpl_name}" {t("template_saved", lang)}')

    with pt_mine:
        my_programs = get_programs(st.session_state.uid)
        if not my_programs:
            st.info(t("no_programs", lang))
        else:
            for prog in my_programs:
                badge   = "🤖 AI" if prog["is_ai"] else "📋"
                content = prog["content"]
                with st.expander(f"{badge} {prog['name']}  ·  {prog['created_at'][:10]}"):
                    if content.get("description"):
                        st.caption(content["description"])
                    days = content.get("days", content.get("schedule", []))
                    for day in days[:5]:
                        label = day.get("label", day.get("day", "Day"))
                        st.markdown(f"**{label}**")
                        for ex in day.get("exercises", [])[:6]:
                            if "duration_min" in ex:
                                st.write(f"  • {ex['name']}: {ex.get('duration_min')} min")
                            else:
                                st.write(f"  • {ex['name']}: {ex.get('sets')} × {ex.get('reps')}")
                    for week in content.get("weeks", [])[:1]:
                        st.markdown(f"*Week {week['week']}* — {week.get('notes', '')}")
                        days_w = week.get("days", [])
                        if isinstance(days_w, str):
                            st.caption(days_w)
                        else:
                            for day in days_w:
                                st.markdown(f"**{day.get('day', '')}** — {day.get('focus', '')}")
                                for ex in day.get("exercises", [])[:5]:
                                    if "duration_min" in ex:
                                        st.write(f"  • {ex['name']}: {ex.get('duration_min')} min")
                                    else:
                                        st.write(f"  • {ex['name']}: {ex.get('sets')} × {ex.get('reps')}")

# ═══════════════════════════════════════════════════════
# TAB 4 — PROGRESS
# ═══════════════════════════════════════════════════════

with tab_stats:
    st.markdown(t("progress_title", lang))
    all_sessions = get_sessions(st.session_state.uid, limit=100)

    if not all_sessions:
        st.info(t("no_sessions", lang))
    else:
        st.subheader(t("workout_history", lang))
        df = pd.DataFrame(all_sessions)
        df["date"] = pd.to_datetime(df["date"])
        st.dataframe(
            df[["date", "ex_count", "notes"]].rename(columns={
                "date": t("date", lang), "ex_count": t("exercises_label", lang), "notes": t("notes", lang)
            }),
            hide_index=True, use_container_width=True,
        )

        if user["is_premium"]:
            import plotly.express as px

            st.divider()
            st.subheader(t("exercise_progression", lang))

            all_ex_names = []
            for s in all_sessions:
                all_ex_names.extend(
                    e["name"] for e in get_session_exercises(s["id"]) if e["ex_type"] == "strength"
                )
            unique_ex = sorted(set(all_ex_names))

            if unique_ex:
                sel_ex  = st.selectbox(t("track_exercise", lang), unique_ex)
                history = get_exercise_history(st.session_state.uid, sel_ex)

                if len(history) > 1:
                    df_h           = pd.DataFrame(history)
                    df_h["date"]   = pd.to_datetime(df_h["date"])
                    df_h           = df_h.sort_values("date")
                    df_h["volume"] = df_h["sets"] * df_h["reps"] * df_h["weight"]

                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.line(df_h, x="date", y="weight",
                                      title=f"{sel_ex} — Weight (kg)", markers=True)
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig2 = px.bar(df_h, x="date", y="volume",
                                      title=f"{sel_ex} — Volume (sets×reps×kg)")
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info(t("log_twice", lang))

            st.divider()
            st.subheader(t("weekly_freq", lang))
            df_freq          = pd.DataFrame(all_sessions)
            df_freq["date"]  = pd.to_datetime(df_freq["date"])
            df_freq["isoyr"] = df_freq["date"].dt.isocalendar().year.astype(str)
            df_freq["isoWk"] = df_freq["date"].dt.isocalendar().week.astype(str).str.zfill(2)
            df_freq["week"]  = df_freq["isoyr"] + "-W" + df_freq["isoWk"]
            weekly = df_freq.groupby("week").size().reset_index(name="sessions")
            fig3   = px.bar(weekly.tail(16), x="week", y="sessions",
                            title="Sessions per Week (last 16 weeks)")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.divider()
            st.info(f"💎 **Premium**: {t('exercise_progression', lang)}, volume tracking, weekly analysis.")

# ═══════════════════════════════════════════════════════
# TAB 5 — AI COACH
# ═══════════════════════════════════════════════════════

with tab_ai:
    st.markdown(t("ai_coach_title", lang))

    if not user["is_premium"]:
        st.warning(t("premium_feature_msg", lang))
        if st.button(t("activate_now", lang), type="primary", use_container_width=True):
            update_user(st.session_state.uid, is_premium=1)
            st.success(t("premium_activated", lang))
            st.rerun()
        st.stop()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found.")
        st.stop()

    from ai_coach import generate_program, adapt_program, ask_coach, generate_meal_plan

    ai_gen, ai_adapt, ai_chat, ai_meal = st.tabs([
        t("gen_program_tab", lang), t("adapt_tab", lang),
        t("chat_tab", lang), t("meal_plan_tab", lang),
    ])

    # ── Generate Program ──────────────────────────────────────────────────────

    with ai_gen:
        st.markdown(
            f"**Profile:** {user['name']} · {goal_to_display(user['goal'], lang)} · "
            f"{user['fitness_level']} · {user['age']} y · {user['weight']} kg"
        )
        c1, c2 = st.columns(2)
        weeks        = c1.slider(t("duration_weeks", lang), 4, 16, 8)
        model_choice = c2.selectbox(t("ai_model_label", lang),
            ["gemini-2.0-flash — fast & free ✅", "gemini-1.5-pro — best quality"])
        model = "gemini-2.0-flash" if "flash" in model_choice else "gemini-1.5-pro"

        if st.button(t("generate_btn", lang), type="primary", use_container_width=True):
            with st.spinner(t("analysing_profile", lang)):
                try:
                    prog = generate_program(user, weeks, model)
                    save_program(st.session_state.uid, prog.get("program_name", "AI Program"), prog, is_ai=True)
                    st.success(f"✅ **{prog.get('program_name')}** generated!")
                    st.caption(prog.get("description", ""))
                    if prog.get("tips"):
                        with st.expander(t("coach_tips", lang)):
                            for tip in prog["tips"]:
                                st.write(f"• {tip}")
                    for week in prog.get("weeks", [])[:2]:
                        with st.expander(f"Week {week['week']} — {week.get('notes', '')}"):
                            days = week.get("days", [])
                            if isinstance(days, str):
                                st.caption(days)
                            else:
                                for day in days:
                                    st.markdown(f"**{day.get('day', '')}** — {day.get('focus', '')}")
                                    for ex in day.get("exercises", []):
                                        if "duration_min" in ex:
                                            st.write(f"  • {ex['name']}: {ex.get('duration_min')} min")
                                        else:
                                            st.write(f"  • {ex['name']}: {ex.get('sets')} × {ex.get('reps')} — rest {ex.get('rest_sec', 60)} s")
                except Exception as e:
                    st.error(f"Error generating program: {e}")

    # ── Adapt Program ─────────────────────────────────────────────────────────

    with ai_adapt:
        st.markdown(t("adapt_title", lang))
        st.caption(t("adapt_desc", lang))
        ai_programs = [p for p in get_programs(st.session_state.uid) if p["is_ai"]]

        if not ai_programs:
            st.info(t("no_ai_programs", lang))
        else:
            prog_map = {f"{p['name']}  ({p['created_at'][:10]})": p for p in ai_programs}
            sel_name = st.selectbox(t("program_to_adapt", lang), list(prog_map.keys()))
            sel_prog = prog_map[sel_name]
            c1, c2  = st.columns(2)
            missed  = c1.number_input(t("sessions_missed", lang), 1, 10, 1)
            reason  = c2.text_input(t("reason_label", lang), placeholder=t("reason_placeholder", lang))

            if st.button(t("get_adaptation_btn", lang), type="primary", use_container_width=True):
                with st.spinner(t("analysing_adapt", lang)):
                    try:
                        adaptation = adapt_program(user, sel_prog, missed, reason)
                        st.success(adaptation.get("message", ""))
                        if adaptation.get("adjustments"):
                            st.subheader(t("adjustments", lang))
                            for adj in adaptation["adjustments"]:
                                with st.container(border=True):
                                    st.markdown(f"**{adj.get('day', '')}**: {adj.get('change', '')}")
                                    st.caption(adj.get("reason", ""))
                        if adaptation.get("modified_days"):
                            st.subheader(t("modified_schedule", lang))
                            for day in adaptation["modified_days"]:
                                with st.expander(f"{day.get('day', 'Day')} — {day.get('focus', '')}"):
                                    for ex in day.get("exercises", []):
                                        if "duration_min" in ex:
                                            st.write(f"• {ex['name']}: {ex.get('duration_min')} min")
                                        else:
                                            st.write(f"• {ex['name']}: {ex.get('sets')} × {ex.get('reps')}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ── Chat ──────────────────────────────────────────────────────────────────

    with ai_chat:
        st.markdown(t("chat_title", lang))
        st.caption(t("chat_desc", lang))

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if question := st.chat_input(t("chat_input", lang)):
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner(t("thinking", lang)):
                    try:
                        recent   = get_sessions(st.session_state.uid, limit=5)
                        response = ask_coach(user, question, st.session_state.chat, recent)
                        st.markdown(response)
                        st.session_state.chat.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error: {e}")

        if st.session_state.chat:
            if st.button(t("clear_chat", lang)):
                st.session_state.chat = []
                st.rerun()

    # ── AI Meal Plan ──────────────────────────────────────────────────────────

    with ai_meal:
        st.markdown(t("meal_plan_title", lang))
        st.caption(t("meal_plan_desc", lang))

        c1, c2 = st.columns(2)
        meal_days    = c1.slider(t("meal_plan_days", lang), 3, 7, 7)
        model_choice = c2.selectbox(t("ai_model_label", lang),
            ["gemini-2.0-flash — fast & free ✅", "gemini-1.5-pro — best quality"],
            key="meal_model")
        model = "gemini-2.0-flash" if "flash" in model_choice else "gemini-1.5-pro"

        if st.button(t("gen_meal_btn", lang), type="primary", use_container_width=True):
            with st.spinner(t("generating_meal", lang)):
                try:
                    plan = generate_meal_plan(user, meal_days, lang, model)

                    # Daily targets
                    macros = plan.get("macros", {})
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(t("calories", lang), plan.get("daily_calories", "—"))
                    m2.metric(t("protein", lang),  f"{macros.get('protein_g', '—')}g")
                    m3.metric(t("carbs", lang),     f"{macros.get('carbs_g', '—')}g")
                    m4.metric(t("fat", lang),       f"{macros.get('fat_g', '—')}g")

                    # Days
                    for day in plan.get("days", []):
                        with st.expander(f"📅 {day.get('day', '')}"):
                            for meal in day.get("meals", []):
                                st.markdown(f"**{meal.get('meal', '')}** — {meal.get('calories', '')} kcal · {meal.get('protein_g', '')}g protein")
                                st.write("  " + ", ".join(meal.get("foods", [])))

                    if plan.get("tips"):
                        with st.expander(t("meal_tips", lang)):
                            for tip in plan["tips"]:
                                st.write(f"• {tip}")
                except Exception as e:
                    st.error(f"Error: {e}")

# ═══════════════════════════════════════════════════════
# TAB 6 — NUTRITION (free advice)
# ═══════════════════════════════════════════════════════

with tab_nutrition:
    st.markdown(t("nutrition_title", lang))
    nt_free, nt_ai = st.tabs([t("free_advice_tab", lang), t("ai_meal_plan_tab", lang)])

    with nt_free:
        advice = meal_advice(user["goal"], lang)
        st.markdown(advice["title"])
        st.caption(f"{t('advice_for_goal', lang)} **{goal_to_display(user['goal'], lang)}**")

        st.subheader("📋 Key Principles" if lang == "en" else "📋 Principes clés")
        for tip in advice["tips"]:
            st.markdown(f"- {tip}")

        st.divider()
        col_food, col_sample = st.columns([1, 2])
        with col_food:
            st.subheader("✅ Top Foods" if lang == "en" else "✅ Aliments clés")
            for food in advice["foods"]:
                st.write(f"• {food}")
        with col_sample:
            if advice["sample"]:
                st.subheader("🍽️ Sample Day" if lang == "en" else "🍽️ Exemple de journée")
                st.markdown(advice["sample"])

    with nt_ai:
        if not user["is_premium"]:
            st.warning(t("premium_feature_msg", lang))
            if st.button(t("activate_now", lang), type="primary",
                         use_container_width=True, key="nu_premium"):
                update_user(st.session_state.uid, is_premium=1)
                st.success(t("premium_activated", lang))
                st.rerun()
        else:
            st.info("👈 " + ("Generate your AI meal plan in the AI Coach tab → Meal Plan." if lang == "en"
                             else "Générez votre plan repas IA dans l'onglet Coach IA → Plan repas."))
