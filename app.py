    border: 1px solid var(--border);
    border-radius: 0;
    padding: 14px;
    margin-top: 12px;
}
#composer {
    background: #ffffff;
    border: 1px solid #cfd5cc;
    border-radius: 0;
    padding: 8px 10px;
    margin-top: 10px;
    box-shadow: 0 3px 10px rgba(20, 30, 20, 0.05);
}
#message_box textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 14px !important;
    padding: 12px 10px !important;
    color: var(--text) !important;
    min-height: 52px !important;
}
#message_box textarea:focus { outline: none !important; border: none !important; box-shadow: none !important; }
#message_box > div { border: none !important; box-shadow: none !important; background: transparent !important; }
#send_button {
    background: var(--text) !important;
    color: white !important;
    border: none !important;
    border-radius: 0 !important;
    min-height: 52px !important;
    font-weight: 600 !important;
    padding: 0 22px !important;
}
#send_button:hover { background: var(--accent-dark) !important; }
.footer { text-align: center; margin-top: 14px; color: #8c938b; font-size: 12px; }
"""


with gr.Blocks() as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.HTML("""
        <div class="header-card">
            <div class="app-title">Mushroom.ai</div>
            <div class="app-subtitle">Multimodal mushroom analysis and conversational guidance</div>
        </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=280, elem_classes="side-panel"):
                gr.HTML("""
                <div class="panel-title">Image</div>
                <div class="panel-text">Upload a mushroom image for visual analysis.</div>
                """)
                image_input = gr.Image(type="pil", label=None, elem_classes="image-box")
                gr.HTML("""
                <div class="mini-card">
                    <div class="panel-title">Usage</div>
                    <div class="panel-text">
                        Upload an image by itself for a short summary,
                        or combine it with a question for a specific answer.
                    </div>
                </div>
                """)

            with gr.Column(scale=3, min_width=650, elem_classes="chat-panel"):
                chatbot = gr.Chatbot(height=560, elem_id="chatbot")
                with gr.Row(elem_id="composer"):
                    message_input = gr.Textbox(
                        placeholder="Ask about this mushroom...",
                        show_label=False,
                        lines=1,
                        max_lines=4,
                        scale=8,
                        elem_id="message_box"
                    )
                    send_button = gr.Button("Send", scale=1, elem_id="send_button")

            with gr.Column(scale=1, min_width=260, elem_classes="info-panel"):
                gr.HTML("""
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
                    <div class="panel-text">Gemini 3.5 Flash-Lite</div>
                </div>
                """)

        gr.HTML('<div class="footer">Developed by Nusrat Khan</div>')

    send_button.click(
        fn=chat_function,
        inputs=[message_input, image_input, chatbot],
        outputs=[chatbot, message_input, image_input]
    )

    message_input.submit(
        fn=chat_function,
        inputs=[message_input, image_input, chatbot],
        outputs=[chatbot, message_input, image_input]
    )


port = int(os.environ.get("PORT", 10000))

demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=port,
    css=css
)
