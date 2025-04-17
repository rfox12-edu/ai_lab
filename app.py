import streamlit as st
from PIL import Image, ImageOps
import io
import base64
import requests
from streamlit_drawable_canvas import st_canvas
import json
import ollama

# Sidebar Configuration
st.sidebar.header("Configuration")
MODEL_NAME = st.sidebar.text_input("Ollama Model", value="qwen2.5:0.5b")

# Main Streamlit UI
st.title("Sketch Classifier with Ollama")
st.header("(AI Only)")
st.markdown("Draw a digit, and ask the LLM figure it out!")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    # Convert image to 28x28 grayscale PNG
    img = Image.fromarray(canvas_result.image_data.astype("uint8"))
    img = ImageOps.grayscale(img)
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    st.image(img, caption="Processed Image (28x28)", width=100)
    promptText = st.text_area("Prompt", value=f"What is this image (encoded in base64) a sketch of?: {img_base64}", height=200)

    if st.button("Prompt AI"):
        with st.spinner("Running Ollama with tool calling..."):
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "user",
                            "content": promptText
                        }
                    ],
                    #tools=[classify_image]
                )

                st.success("LLM Response:")
                st.write(response.message.content)

            except Exception as e:
                st.error("Ollama model failed to run:")
                st.text(str(e))

else:
    st.info("Use your mouse to draw above.")
