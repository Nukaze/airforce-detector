import time
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from ultralytics import YOLO
import tensorflow as tf

def pre_config() -> None:
    print("\nstreamlit version: ",st.__version__)
    
    city_sunrise_icon = ":city_sunrise:"
    graph_icon = "📈"
    # ref: https://fonts.google.com/icons?icon.query=graph
    graph_monitoring_google_icon = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#FFFF55"><path d="M160-120q-17 0-28.5-11.5T120-160v-40q0-17 11.5-28.5T160-240q17 0 28.5 11.5T200-200v40q0 17-11.5 28.5T160-120Zm160 0q-17 0-28.5-11.5T280-160v-220q0-17 11.5-28.5T320-420q17 0 28.5 11.5T360-380v220q0 17-11.5 28.5T320-120Zm160 0q-17 0-28.5-11.5T440-160v-140q0-17 11.5-28.5T480-340q17 0 28.5 11.5T520-300v140q0 17-11.5 28.5T480-120Zm160 0q-17 0-28.5-11.5T600-160v-200q0-17 11.5-28.5T640-400q17 0 28.5 11.5T680-360v200q0 17-11.5 28.5T640-120Zm160 0q-17 0-28.5-11.5T760-160v-360q0-17 11.5-28.5T800-560q17 0 28.5 11.5T840-520v360q0 17-11.5 28.5T800-120ZM560-481q-16 0-30.5-6T503-504L400-607 188-395q-12 12-28.5 11.5T131-396q-11-12-10.5-28.5T132-452l211-211q12-12 26.5-17.5T400-686q16 0 31 5.5t26 17.5l103 103 212-212q12-12 28.5-11.5T829-771q11 12 10.5 28.5T828-715L617-504q-11 11-26 17t-31 6Z"/></svg>"""
    analytics_google_icon = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#FFFF55"><path d="M280-280h80v-200h-80v200Zm320 0h80v-400h-80v400Zm-160 0h80v-120h-80v120Zm0-200h80v-80h-80v80ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z"/></svg>"""
    
    st.set_page_config(
        page_title=f"Obj Det App",
        page_icon=graph_monitoring_google_icon,
        layout="centered",
        initial_sidebar_state="expanded",
    )
    return

def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img
    
# Function to perform object detection on the image
def yolo_detect_objects(image):
    model = YOLO("../model/yolov8m_0883_best.pt").to("cpu")
    results = model(image, conf=.25, show=False, stream=False)  # Perform detection
    annotated_image_bgr = results[0].plot()  # Get annotated image
    # convert the annotated image to PIL format bgr to rgb
    annotated_image_rgb = Image.fromarray(cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB))
    return [annotated_image_rgb, results]

def yolo_extract_classes_and_count(results):
    res = results[0]
    class_ids = res.boxes.cls.cpu().numpy().astype(int)
    class_names = [res.names[c] for c in class_ids]
    class_count = {}
    for name in class_names:
        if name not in class_count:
            class_count[name] = 0
        class_count[name] += 1
        
    return class_count


def tensorflow_detect_objects(image):
    model = tf.keras.models.load_model("../model/warden_0896.h5")
    # Preprocess the image
    image_array = np.array(image.resize((224, 224))) / 255.0        # Resize and normalize the image
    image_array = np.expand_dims(image_array, axis=0)               # Add batch dimension 

    predictions = model.predict(image_array)                        # Perform prediction
    return predictions



def main():
    st.title("Aircraft Detection App")
    
    ai_choice = ["TensorFlow", "YOLO"]
    selected_ai = st.selectbox("Select AI model", ai_choice)
    
    # Option to upload an image or input an image URL
    option_choice = ["Upload Image", "Image URL"]
    option = st.segmented_control("Choose input method", option_choice, default=option_choice[0])
    
    
    
    text_style = ["Normal", "Code"]

    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_container_width=True)
            if st.button("Detect Aircraft"):
                with st.spinner("Detecting objects..."):
                    if selected_ai == "YOLO":
                        annotated_img, results = yolo_detect_objects(image)
                        class_count = yolo_extract_classes_and_count(results)
                    else:
                        predictions = tensorflow_detect_objects(image)
                        annotated_img = image  # Placeholder, as TensorFlow model might not provide bounding boxes
                        class_count = {"Predictions": predictions.tolist()}
                    
                    st.session_state["annotated_img"] = annotated_img
                    st.session_state["class_count"] = class_count
                    
                    st.image(annotated_img, caption="Detection Results with Bounding Boxes.", use_container_width=True)
                    st.write("Detected objects:")
                    result_text_style = st.segmented_control("Select result text style", text_style, default=text_style[0])
                    if result_text_style == "Normal":
                        st.subheader(f"`{'`, `'.join([f'{name}: {count}' for name, count in class_count.items()])}`")
                    elif result_text_style == "Code":
                        st.code(class_count)
                    else:
                        st.info("Please select a result text style.")
                    
    elif option == "Image URL":
        image_url = st.text_input("Enter image URL")
        if st.button("Detect Aircraft"):
            if image_url:
                image = load_image_from_url(image_url)
                st.image(image, caption="Image from URL.", use_container_width=True)
                with st.spinner("Detecting objects..."):
                    if selected_ai == "YOLO":
                        annotated_img, results = yolo_detect_objects(image)
                        class_count = yolo_extract_classes_and_count(results)
                    else:
                        predictions = tensorflow_detect_objects(image)
                        annotated_img = image  # Placeholder, as TensorFlow model might not provide bounding boxes
                        class_count = {"Predictions": predictions.tolist()}
                    
                    st.session_state["annotated_img"] = annotated_img
                    st.session_state["class_count"] = class_count
                    
                    st.image(annotated_img, caption="Detection Results with Bounding Boxes.", use_container_width=True)
                    st.header("Detected objects:")
                    result_text_style = st.segmented_control("Select result text style", text_style, default=text_style[0])
                    if result_text_style == "Normal":
                        st.subheader(f"`{'`, `'.join([f'{name}: {count}' for name, count in class_count.items()])}`")
                    elif result_text_style == "Code":
                        st.code(class_count)
                    else:
                        st.info("Please select a result text style.")
            else:
                st.warning("Please enter an image URL.")
    else:
        st.warning("Please select an input method.")
        
    if "annotated_img" in st.session_state and "class_count" in st.session_state:
        st.image(st.session_state["annotated_img"], caption="Detection Results with Bounding Boxes.", use_container_width=True)
        st.header("Detected objects:")
        result_text_style = st.segmented_control("Select result text style", text_style, default=text_style[1])
        if result_text_style == "Normal":
            st.subheader(f"`{'`, `'.join([f'{name}: {count}' for name, count in st.session_state['class_count'].items()])}`")
        elif result_text_style == "Code":
            st.code(st.session_state["class_count"])


if __name__ == "__main__":
    pre_config()
    main()