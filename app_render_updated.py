import os
import json
import re

import gradio as gr
from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.5-flash-lite"

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing. Add it in Render > Environment.")

client = genai.Client(api_key=api_key)


# Confidence Calibration

# Adjust confidence based on visible evidence
def calibrate_confidence(result):
    visible_count = len(result.get("visible", []))
    confidence = result.get("confidence", 0.0)

    if visible_count == 0:
        confidence = min(confidence, 0.30)

    elif visible_count == 1:
        confidence = min(confidence, 0.50)

    elif visible_count == 2:
        confidence = min(confidence, 0.70)

    result["confidence"] = confidence

    return result


# Improve safety wording
SYSTEM_PROMPT = """
You are a mushroom expert chatbot.

Your role:
- Answer questions about mushrooms and fungi.
- Help with mushroom identification from images.
- Keep answers short, natural, and factual.
- Clearly mention uncertainty when identification is not reliable.

Safety:
- Never claim that a wild mushroom is safe to eat based only on an image.
- Toxic mushrooms may be discussed factually, including general effects and historical use.
- Do not give step-by-step preparation, detoxification, or consumption instructions.
- If a toxic mushroom has historically been consumed after special processing,
  you may mention that only in general terms without explaining the method.
- Make clear that misidentification or incorrect handling can be dangerous.
- Avoid inventing medical, chemical, or species information.

Topic control:
- Stay focused on mushrooms and fungi.
- If the user asks about an unrelated topic, politely redirect the conversation back to mushrooms.
"""


def analyze_mushroom(image):
    prompt = """
Carefully analyze this mushroom image.

First inspect the visible cap, hymenium, stipe, color,
surface texture, and other useful visual features.

Then make the most likely identification.

Return ONLY valid JSON:

{
  "common_name": "string",
  "genus": "string",
  "confidence": 0.0,
  "visible": [],
  "color": "string",
  "edible": false
}

Rules:
- confidence must be between 0 and 1
- visible may only contain "cap", "hymenium", and "stipe"
- include only parts clearly visible in the image
- do not assume hidden features
- distinguish look-alike mushrooms carefully
- lower confidence when important features are missing
- use "Unknown" when the visual evidence is too weak
- color should describe the main visible color
- edible must be true or false based on general knowledge of the predicted species
- never use the edible field as advice that the photographed mushroom is safe to eat
- return JSON only
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[image, prompt],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0
        )
    )

    text = response.text.strip()

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            result = json.loads(match.group())

            allowed_parts = {"cap", "hymenium", "stipe"}
            result["visible"] = [
                part for part in result.get("visible", [])
                if part in allowed_parts
            ]

            if not isinstance(result.get("confidence"), (int, float)):
                result["confidence"] = 0.0

            if not isinstance(result.get("edible"), bool):
                result["edible"] = False

            result["common_name"] = result.get("common_name", "Unknown")
            result["genus"] = result.get("genus", "Unknown")
            result["color"] = result.get("color", "Unknown")

            result = calibrate_confidence(result)

            return result

        except json.JSONDecodeError:
            pass

    return {
        "common_name": "Unknown",
        "genus": "Unknown",
        "confidence": 0.0,
        "visible": [],
        "color": "Unknown",
        "edible": False
    }


# Image only
def summarize_mushroom(info):
    prompt = f"""
Here is structured information from a mushroom image:

{json.dumps(info)}

Write a short natural summary for the user.

Rules:
- use 2 or 3 short sentences
- sound conversational, not robotic
- mention the likely identification if available
- mention uncertainty when confidence is low
- do not show the JSON
- do not add facts that are not in the structured information
- do not say that the mushroom is safe to eat
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2
        )
    )

    return response.text.strip()


# Lookes good, so final stream fuction
def stream_mushroom_answer(question, mushroom_info=None):

    if not is_mushroom_related(question, mushroom_info):
        yield (
            "I'm focused on mushrooms and fungi. "
            "You can ask about identification, safety, habitats, "
            "appearance, or other mushroom-related topics."
        )
        return

    image_context = ""

    if mushroom_info is not None:
        image_context = f"""
Information from the latest mushroom image:

{json.dumps(mushroom_info)}
"""

    prompt = f"""
{image_context}

User question:
{question}

Answer naturally and directly.

Rules:
- keep the answer short, usually 2 to 4 sentences
- use the image information when relevant
- do not show the raw JSON
- do not invent missing visual details
- mention uncertainty when it matters
- avoid repeating safety warnings for simple questions like color or visible parts
- if the question is about eating, toxicity, or safety, clearly say that image-based identification cannot confirm a wild mushroom is safe to eat
"""

    response = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2
        )
    )

    full_answer = ""

    for chunk in response:
        if chunk.text:
            full_answer += chunk.text
            yield full_answer


def is_mushroom_related(question, mushroom_info=None, has_image=False):

    prompt = f"""
Decide whether the user's request is related to mushrooms or fungi.

Question:
{question}

Image attached: {has_image}

Important:
- If an image is attached, requests to describe, identify, read,
  or transcribe that image are allowed.
- Follow-up questions such as "What color is it?",
  "Can I eat it?" or "Why are you not confident?"
  are also allowed when mushroom context exists.
- Clearly unrelated topics such as football, politics,
  movies, etc. should return NO.

Answer only YES or NO.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0
        )
    )

    return response.text.strip().upper().startswith("YES")


saved_mushroom_info = None

def chat_function(message, image, history):
    global saved_mushroom_info

    history = history or []
    message = (message or "").strip()

    if not message and image is None:
        yield history, "", None
        return

    transcription_words = [
        "transcribe",
        "read the text",
        "extract the text",
        "what does the text say"
    ]

    is_transcription = (
        image is not None and
        any(word in message.lower() for word in transcription_words)
    )

    if is_transcription:
        history.append({
            "role": "user",
            "content": message
        })

        history.append({
            "role": "assistant",
            "content": ""
        })

        prompt = """
Transcribe the text visible in this image as accurately as possible.
Keep the original language.
Do not summarize or translate it.
Do not add information that is not visible.
Preserve headings and paragraphs when possible.
"""

        response = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=[image, prompt],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0
            )
        )

        full_answer = ""

        for chunk in response:
            if chunk.text:
                full_answer += chunk.text
                history[-1]["content"] = full_answer
                yield history, "", None

        return

    related = is_mushroom_related(
        message,
        saved_mushroom_info,
        has_image=image is not None
    )

    if not related:
        if message:
            history.append({
                "role": "user",
                "content": message
            })

        history.append({
            "role": "assistant",
            "content":
                "I'm focused on mushrooms and fungi. "
                "You can ask about identification, safety, habitats, "
                "appearance, or other mushroom-related topics."
        })

        yield history, "", None
        return

    if image is not None:
        saved_mushroom_info = analyze_mushroom(image)

        print("\nMushroom JSON:")
        print(json.dumps(saved_mushroom_info, indent=2))

        if not message:
            history.append({
                "role": "user",
                "content": "[Mushroom image uploaded]"
            })

            history.append({
                "role": "assistant",
                "content": ""
            })

            summary = summarize_mushroom(saved_mushroom_info)
            history[-1]["content"] = summary

            yield history, "", None
            return

        history.append({
            "role": "user",
            "content": message
        })

        history.append({
            "role": "assistant",
            "content": ""
        })

        for answer in stream_mushroom_answer(
            message,
            saved_mushroom_info
        ):
            history[-1]["content"] = answer
            yield history, "", None

        return

    if message:
        history.append({
            "role": "user",
            "content": message
        })

        history.append({
            "role": "assistant",
            "content": ""
        })

        for answer in stream_mushroom_answer(
            message,
            saved_mushroom_info
        ):
            history[-1]["content"] = answer
            yield history, "", None


# Final Mushroom.ai interface

css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --bg: #f6f6f2;
    --panel: #ffffff;
    --soft: #eef0ea;
    --border: #d8ddd5;
    --text: #20251f;
    --muted: #737a72;
    --accent: #6d7f67;
    --accent-dark: #55614f;
}

body {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: 0 auto !important;
    background: var(--bg) !important;
}

.app-shell {
    padding: 22px 24px 18px 24px;
}

/* Header */
.header-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 24px 28px;
    margin-bottom: 14px;
}

.app-title {
    font-family: 'Playfair Display', serif;
    font-size: 34px;
    font-weight: 700;
    margin: 0;
    color: var(--text);
}

.app-subtitle {
    margin-top: 6px;
    font-size: 14px;
    color: var(--muted);
}

/* Main panels */
.side-panel,
.chat-panel,
.info-panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 16px;
}

.panel-title {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text);
    margin-bottom: 8px;
}

.panel-text {
    font-size: 13px;
    color: var(--muted);
    line-height: 1.55;
}

/* Chat area */
#chatbot {
    background: #fbfbf8 !important;
    border: 1px solid var(--border) !important;
    border-radius: 0 !important;
    min-height: 560px;
}

/* Image */
.image-box {
    border-radius: 0 !important;
    overflow: hidden;
}

/* Information cards */
.mini-card {
    background: var(--soft);
    border: 1px solid var(--border);
    border-radius: 0;
    padding: 14px;
    margin-top: 12px;
}

/* Input composer */
#composer {
    background: #ffffff;
    border: 1px solid #cfd5cc;
    border-radius: 0;
    padding: 8px 10px;
    margin-top: 10px;
    box-shadow: 0 3px 10px rgba(20, 30, 20, 0.05);
}

/* Text box */
#message_box textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 14px !important;
    padding: 12px 10px !important;
    color: var(--text) !important;
    min-height: 52px !important;
}

#message_box textarea:focus {
    outline: none !important;
    border: none !important;
    box-shadow: none !important;
}

#message_box > div {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* Send button */
#send_button {
    background: var(--text) !important;
    color: white !important;
    border: none !important;
    border-radius: 0 !important;
    min-height: 52px !important;
    font-weight: 600 !important;
    padding: 0 22px !important;
}

#send_button:hover {
    background: var(--accent-dark) !important;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 14px;
    color: #8c938b;
    font-size: 12px;
}
"""


with gr.Blocks() as demo:

    with gr.Column(elem_classes="app-shell"):

        # Header
        gr.HTML(
            """
            <div class="header-card">
                <div class="app-title">MUSHROOM.ai</div>
                <div class="app-subtitle">
                    Multimodal mushroom analysis and conversational guidance
                </div>
            </div>
            """
        )

        with gr.Row(equal_height=True):

            # Left panel
            with gr.Column(
                scale=1,
                min_width=280,
                elem_classes="side-panel"
            ):

                gr.HTML(
                    """
                    <div class="panel-title">Image</div>
                    <div class="panel-text">
                        Upload a mushroom image for visual analysis.
                    </div>
                    """
                )

                image_input = gr.Image(
                    type="pil",
                    label=None,
                    elem_classes="image-box"
                )

                gr.HTML(
                    """
                    <div class="mini-card">
                        <div class="panel-title">Usage</div>
                        <div class="panel-text">
                            Upload an image by itself for a short summary,
                            or combine it with a question for a specific answer.
                        </div>
                    </div>
                    """
                )

            # Main chat area
            with gr.Column(
                scale=3,
                min_width=650,
                elem_classes="chat-panel"
            ):

                chatbot = gr.Chatbot(
                    height=560,
                    elem_id="chatbot"
                )

                with gr.Row(elem_id="composer"):

                    message_input = gr.Textbox(
                        placeholder="Ask about this mushroom...",
                        show_label=False,
                        lines=1,
                        max_lines=4,
                        scale=8,
                        elem_id="message_box"
                    )

                    send_button = gr.Button(
                        "Send",
                        scale=1,
                        elem_id="send_button"
                    )

            # Right information panel
            with gr.Column(
                scale=1,
                min_width=260,
                elem_classes="info-panel"
            ):

                gr.HTML(
                    """
                    <div class="panel-title">About</div>
                    <div class="panel-text">
                        Ask about visible features, identification,
                        confidence, habitat, or mushroom safety.
                    </div>

                    <div class="mini-card">
                        <div class="panel-title">Safety</div>
                        <div class="panel-text">
                            Image-based identification can be uncertain.
                            Do not use this assistant alone to decide whether
                            a wild mushroom is safe to eat.
                        </div>
                    </div>

                    <div class="mini-card">
                        <div class="panel-title">Model</div>
                        <div class="panel-text">
                            Gemini 3.5 Flash-Lite
                        </div>
                    </div>
                    """
                )

        gr.HTML(
            """
            <div class="footer">
                Developed by Nusrat Khan
            </div>
            """
        )


    # Send button
    send_button.click(
        fn=chat_function,
        inputs=[
            message_input,
            image_input,
            chatbot
        ],
        outputs=[
            chatbot,
            message_input,
            image_input
        ]
    )

    # Press Enter to send
    message_input.submit(
        fn=chat_function,
        inputs=[
            message_input,
            image_input,
            chatbot
        ],
        outputs=[
            chatbot,
            message_input,
            image_input
        ]
    )


port = int(os.environ.get("PORT", 10000))

demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    css=css
)
