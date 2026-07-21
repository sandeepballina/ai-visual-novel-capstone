
import streamlit as st
import google.generativeai as genai
import requests
import json
import os
import uuid
from gtts import gTTS
st.set_page_config(page_title="AI Visual Novel", page_icon="📖", layout="centered")


@st.cache_resource
def get_gemini_client(api_key: str):
    """Cache the Gemini client/model so we don't re-init it on every rerun."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-flash-latest")

with st.sidebar:
    st.title("Story Settings")

    api_key = st.text_input("Gemini API Key", type="password")

    genre = st.selectbox(
        "Story Genre",
        ["Fantasy Adventure", "Sci-Fi Thriller", "Mystery Noir", "Cyberpunk", "Horror"],
    )

    art_style = st.selectbox(
        "Art Style",
        ["Anime", "Watercolor", "Pixel Art", "Photorealistic", "Comic Book"],
    )

    start_button = st.button("🎬 Start New Story", use_container_width=True)

st.title("📖 AI Visual Novel Engine")

if "chat" not in st.session_state:
    st.session_state.chat = None
if "history" not in st.session_state:
    st.session_state.history = []          # list of dicts: {story_text, image_path, options}
if "current_scene" not in st.session_state:
    st.session_state.current_scene = None  # the parsed JSON dict of the current scene

SYSTEM_PROMPT = f"""
You are the narrative engine for an interactive visual novel.
Genre: {genre}
Art style for image prompts: {art_style}

RULES:
1. You must ALWAYS respond with ONLY a valid JSON object. No markdown fences, no
   preamble, no explanation outside the JSON.
2. The JSON object must have exactly these three keys:
   - "story_text": a vivid narrative paragraph (3-6 sentences) continuing the story.
   - "image_prompt": a heavily detailed, descriptive prompt (mention the {art_style}
     art style explicitly) suitable for an AI image generator, describing the current scene.
   - "options": a list of 2 to 3 short, distinct strings describing what the player can do next.

Example of a valid response:
{{
  "story_text": "...",
  "image_prompt": "...",
  "options": ["Open the door", "Search the desk", "Leave the room"]
}}

Begin the story now with an opening scene.
"""

def parse_scene_json(raw_text: str):
    """
    PHASE 2: Parse Gemini's string response into a Python dict.
    Gemini sometimes wraps JSON in ```json ... ``` fences even when told not to,
    so we strip those defensively before parsing.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    try:
        scene = json.loads(cleaned)
        # Basic shape validation
        if not all(k in scene for k in ("story_text", "image_prompt", "options")):
            raise ValueError("Missing required keys in JSON response.")
        return scene
    except (json.JSONDecodeError, ValueError) as e:
        st.toast(f"⚠️ Couldn't parse the AI's response as JSON: {e}")
        return {
            "story_text": raw_text,
            "image_prompt": None,
            "options": ["Continue"],
        }


def generate_image(prompt: str):
    """
    PHASE 4: Send the image_prompt to Pollinations and download the image.
    PHASE 5: Wrapped in try/except so a slow/down image API never crashes the app.
    """
    if not prompt:
        return None

    try:
        safe_prompt = requests.utils.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=768&height=512&nologo=true"
        response = requests.get(url, timeout=25)
        response.raise_for_status()

        os.makedirs("generated_images", exist_ok=True)
        image_path = os.path.join("generated_images", f"{uuid.uuid4().hex}.png")
        with open(image_path, "wb") as f:
            f.write(response.content)
        return image_path

    except requests.exceptions.RequestException:
        st.toast("🖼️ Image server is busy, skipping visual...")
        return None


def generate_narration(story_text: str):
    """
    PHASE 4: Convert story_text to speech using gTTS.
    PHASE 5: Wrapped in try/except — TTS failure shouldn't crash the app either.
    """
    if not story_text:
        return None

    try:
        os.makedirs("generated_audio", exist_ok=True)
        audio_path = os.path.join("generated_audio", f"{uuid.uuid4().hex}.mp3")
        tts = gTTS(text=story_text, lang="en")
        tts.save(audio_path)
        return audio_path

    except Exception:
        st.toast("🔇 Narration engine is busy, skipping audio...")
        return None


def advance_story(user_message: str):
    """
    Send a message to the Gemini chat, parse the JSON reply, generate the
    matching image + narration, and store everything in session_state.
    """
    try:
        response = st.session_state.chat.send_message(user_message)
        raw_text = response.text
    except Exception as e:
        st.toast(f"🤖 The story engine is having trouble: {e}")
        return

    scene = parse_scene_json(raw_text)
    scene["image_path"] = generate_image(scene.get("image_prompt"))
    scene["audio_path"] = generate_narration(scene.get("story_text"))

    st.session_state.current_scene = scene
    st.session_state.history.append(scene)

if start_button:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar first.")
    else:
        model = get_gemini_client(api_key)
        st.session_state.chat = model.start_chat(history=[])
        st.session_state.history = []
        st.session_state.current_scene = None
        with st.spinner("Writing the opening scene..."):
            advance_story(SYSTEM_PROMPT)

scene = st.session_state.current_scene

if scene:
    if scene.get("image_path"):
        st.image(scene["image_path"], use_container_width=True)

    st.write(scene.get("story_text", ""))

    if scene.get("audio_path"):
        st.audio(scene["audio_path"], format="audio/mp3")

    st.divider()
    st.subheader("What do you do?")
    options = scene.get("options", [])
    cols = st.columns(len(options)) if options else []

    for i, option_text in enumerate(options):
        with cols[i]:
            if st.button(option_text, key=f"option_{len(st.session_state.history)}_{i}", use_container_width=True):
                with st.spinner("The story continues..."):
                    advance_story(option_text)
                st.rerun()

else:
    st.info("👈 Enter your Gemini API key, pick a genre and art style, then click **Start New Story**.")