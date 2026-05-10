"""
AI Coach — powered by Google Gemini Flash.
Premium features: program generation, session adaptation, coach chat.
"""
import json
import os
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

SYSTEM = (
    "You are FitAI, an expert fitness coach with deep knowledge in exercise science, "
    "strength & hypertrophy training, cardio, recovery, and evidence-based nutrition. "
    "You design programs based on current sports science (progressive overload, "
    "periodization, RIR/RPE, frequency-volume relationship). "
    "Be concise, practical, and motivating. Always return valid JSON when asked."
)


def _chat(prompt: str, model: str = "gemini-2.0-flash", json_mode: bool = False) -> str:
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        max_output_tokens=8192,
        **({"response_mime_type": "application/json"} if json_mode else {}),
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text


def generate_program(user: dict, weeks: int = 8, model: str = "gemini-2.0-flash") -> dict:
    prompt = f"""Create a {weeks}-week fitness program for:
- Age: {user['age']}, Weight: {user['weight']}kg, Height: {user['height']}cm
- Goal: {user['goal']}
- Fitness level: {user['fitness_level']}

Rules:
- Week 1: include full day/exercise details (max 5 exercises per day, max 4 training days)
- Weeks 2+: include only "week", "notes" (progression note), and "days" as a SHORT summary string, NO exercise lists
- Every 4th week: deload note

Return ONLY valid compact JSON:
{{
  "program_name": "...",
  "description": "2 sentences max",
  "weeks": [
    {{
      "week": 1,
      "notes": "base volume",
      "days": [
        {{"day": "Monday", "focus": "Push", "exercises": [
          {{"name": "Bench Press", "sets": 4, "reps": "6-8", "rest_sec": 120}},
          {{"name": "Running", "duration_min": 20, "intensity": "moderate", "type": "cardio"}}
        ]}},
        {{"day": "Tuesday", "focus": "Rest", "exercises": []}}
      ]
    }},
    {{"week": 2, "notes": "add 1 set per compound lift", "days": "Same structure as week 1"}},
    {{"week": 3, "notes": "+2.5kg on main lifts", "days": "Same structure as week 1"}},
    {{"week": 4, "notes": "DELOAD — reduce volume by 40%", "days": "Same structure, half the sets"}}
  ],
  "tips": ["tip1", "tip2", "tip3"]
}}

Include all {weeks} weeks following that pattern."""

    return json.loads(_chat(prompt, model, json_mode=True))


def adapt_program(user: dict, program: dict, missed: int, reason: str = "", model: str = "gemini-2.0-flash") -> dict:
    content_preview = json.dumps(
        program.get("content", {}).get("weeks", [{}])[0], indent=2
    )[:900]

    prompt = f"""A user missed {missed} workout session(s).
Reason: {reason or "not specified"}
Profile: Goal = {user['goal']}, Level = {user['fitness_level']}

Current program (week 1 preview):
{content_preview}

Create a smart 7-day adaptation. Redistribute volume, prioritize compound lifts,
adjust intensity if fatigued, keep progressive overload on track.

Return ONLY valid JSON:
{{
  "message": "brief encouraging message (1-2 sentences)",
  "adjustments": [
    {{"day": "Monday", "change": "description", "reason": "why"}}
  ],
  "modified_days": [
    {{
      "day": "Day name",
      "focus": "...",
      "exercises": [
        {{"name": "...", "sets": 3, "reps": "8-10", "rest_sec": 90}}
      ]
    }}
  ]
}}"""

    return json.loads(_chat(prompt, model, json_mode=True))


def generate_meal_plan(user: dict, days: int = 7, lang: str = "en", model: str = "gemini-2.0-flash") -> dict:
    if lang == "fr":
        prompt = f"""Crée un plan de repas de {days} jours pour :
- Objectif : {user['goal']}
- Poids : {user['weight']} kg, Âge : {user['age']} ans, Taille : {user['height']} cm
- Niveau : {user['fitness_level']}

Calcule les besoins caloriques et les macros adaptés. Max 4 repas par jour. Réponds UNIQUEMENT en JSON valide :
{{
  "daily_calories": 2500,
  "macros": {{"protein_g": 180, "carbs_g": 280, "fat_g": 80}},
  "days": [
    {{
      "day": "Lundi",
      "meals": [
        {{"meal": "Petit-déjeuner", "foods": ["Flocons d'avoine 80g", "3 œufs", "Banane"], "calories": 650, "protein_g": 35}},
        {{"meal": "Déjeuner", "foods": ["Poulet 180g", "Riz 150g", "Brocoli"], "calories": 700, "protein_g": 55}},
        {{"meal": "Collation", "foods": ["Yaourt grec 200g"], "calories": 200, "protein_g": 20}},
        {{"meal": "Dîner", "foods": ["Saumon 200g", "Patate douce", "Épinards"], "calories": 750, "protein_g": 50}}
      ]
    }}
  ],
  "tips": ["conseil1", "conseil2", "conseil3"]
}}
Inclus les {days} jours."""
    else:
        prompt = f"""Create a {days}-day meal plan for:
- Goal: {user['goal']}
- Weight: {user['weight']}kg, Age: {user['age']}, Height: {user['height']}cm
- Level: {user['fitness_level']}

Calculate appropriate daily calories and macros. Max 4 meals per day. Return ONLY valid JSON:
{{
  "daily_calories": 2500,
  "macros": {{"protein_g": 180, "carbs_g": 280, "fat_g": 80}},
  "days": [
    {{
      "day": "Monday",
      "meals": [
        {{"meal": "Breakfast", "foods": ["Oats 80g", "3 eggs", "Banana"], "calories": 650, "protein_g": 35}},
        {{"meal": "Lunch", "foods": ["Chicken 180g", "Rice 150g", "Broccoli"], "calories": 700, "protein_g": 55}},
        {{"meal": "Snack", "foods": ["Greek yogurt 200g"], "calories": 200, "protein_g": 20}},
        {{"meal": "Dinner", "foods": ["Salmon 200g", "Sweet potato", "Spinach"], "calories": 750, "protein_g": 50}}
      ]
    }}
  ],
  "tips": ["tip1", "tip2", "tip3"]
}}
Include all {days} days."""

    return json.loads(_chat(prompt, model, json_mode=True))


def ask_coach(user: dict, question: str, history: list = None, recent_sessions: list = None,
              model: str = "gemini-2.0-flash") -> str:
    profile = (
        f"User: {user['name']}, {user['age']}y, {user['weight']}kg | "
        f"Goal: {user['goal']} | Level: {user['fitness_level']}"
    )
    if recent_sessions:
        profile += f" | {len(recent_sessions)} recent sessions logged"

    history_text = ""
    for msg in (history or [])[-4:]:
        if msg.get("role") in ("user", "assistant"):
            role = "User" if msg["role"] == "user" else "Coach"
            history_text += f"{role}: {msg['content']}\n"

    prompt = f"""Profile: {profile}

{history_text}User: {question}"""

    return _chat(prompt, model)
