import os
import time
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO

from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model

ROOT = os.getcwd()

def pre_config() -> None:
    print("\nstreamlit version: ",st.__version__)
    
    # ref: https://fonts.google.com/icons?icon.query=graph
    graph_monitoring_google_icon = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#FFFF55"><path d="M160-120q-17 0-28.5-11.5T120-160v-40q0-17 11.5-28.5T160-240q17 0 28.5 11.5T200-200v40q0 17-11.5 28.5T160-120Zm160 0q-17 0-28.5-11.5T280-160v-220q0-17 11.5-28.5T320-420q17 0 28.5 11.5T360-380v220q0 17-11.5 28.5T320-120Zm160 0q-17 0-28.5-11.5T440-160v-140q0-17 11.5-28.5T480-340q17 0 28.5 11.5T520-300v140q0 17-11.5 28.5T480-120Zm160 0q-17 0-28.5-11.5T600-160v-200q0-17 11.5-28.5T640-400q17 0 28.5 11.5T680-360v200q0 17-11.5 28.5T640-120Zm160 0q-17 0-28.5-11.5T760-160v-360q0-17 11.5-28.5T800-560q17 0 28.5 11.5T840-520v360q0 17-11.5 28.5T800-120ZM560-481q-16 0-30.5-6T503-504L400-607 188-395q-12 12-28.5 11.5T131-396q-11-12-10.5-28.5T132-452l211-211q12-12 26.5-17.5T400-686q16 0 31 5.5t26 17.5l103 103 212-212q12-12 28.5-11.5T829-771q11 12 10.5 28.5T828-715L617-504q-11 11-26 17t-31 6Z"/></svg>"""
    analytics_google_icon = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#FFFF55"><path d="M280-280h80v-200h-80v200Zm320 0h80v-400h-80v400Zm-160 0h80v-120h-80v120Zm0-200h80v-80h-80v80ZM200-120q-33 0-56.5-23.5T120-200v-560q0-33 23.5-56.5T200-840h560q33 0 56.5 23.5T840-760v560q0 33-23.5 56.5T760-120H200Zm0-80h560v-560H200v560Zm0-560v560-560Z"/></svg>"""
    air_ticket_icon = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#00ffcc"><path d="m354-334 356-94q15-4 22.5-18.5T736-476q-4-15-17.5-22.5T690-502l-98 26-160-150-56 14 96 168-96 24-50-38-38 10 66 114Zm446 174H160q-33 0-56.5-23.5T80-240v-160q33 0 56.5-23.5T160-480q0-33-23.5-56.5T80-560v-160q0-33 23.5-56.5T160-800h640q33 0 56.5 23.5T880-720v480q0 33-23.5 56.5T800-160Zm0-80v-480H160v102q37 22 58.5 58.5T240-480q0 43-21.5 79.5T160-342v102h640ZM480-480Z"/></svg>"""
    
    flare_icon = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#00ffcc"><path d="M40-440v-80h240v80H40Zm270-154-84-84 56-56 84 84-56 56Zm130-86v-240h80v240h-80Zm210 86-56-56 84-84 56 56-84 84Zm30 154v-80h240v80H680Zm-200 80q-50 0-85-35t-35-85q0-50 35-85t85-35q50 0 85 35t35 85q0 50-35 85t-85 35Zm198 134-84-84 56-56 84 84-56 56Zm-396 0-56-56 84-84 56 56-84 84ZM440-40v-240h80v240h-80Z"/></svg>"""
    airport_icon = """<svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#00ffcc"><path d="m330-380 70-20 70 20v-30l-40-30v-100l130 40v-40l-130-80v-88q0-12-9-22t-21-10q-13 0-21.5 10t-8.5 22v88l-130 80v40l130-40v100l-40 30v30ZM606-80q-15 0-26-10.5T569-116q0-8 3-16t9-14q86-92 152.5-196T800-552q0-79-24-147t-66-115q-5-5-7.5-12t-2.5-15q0-17 11-28t28-11q8 0 15.5 3t12.5 9q56 60 84.5 142T880-552q0 121-74.5 242T634-92q-5 6-12.5 9T606-80ZM400-186q122-112 181-203.5T640-552q0-109-69.5-178.5T400-800q-101 0-170.5 69.5T160-552q0 71 59 162.5T400-186Zm0 106Q239-217 159.5-334.5T80-552q0-150 96.5-239T400-880q127 0 223.5 89T720-552q0 100-79.5 217.5T400-80Zm0-472Z"/></svg>"""
    st.set_page_config(
        page_title=f"AFDet | Air Force Detection",
        page_icon=air_ticket_icon,
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


def tensorflow_aircraft_class_names():
    class_names = [
        "A10", "A400M", "AG600", "AH64", "An124", "An22", "An225", "An72", 
        "AV8B", "B1", "B2", "B21", "B52", "Be200", "C130", "C17", "C2", 
        "C390", "C5", "CH47", "CL415", "E2", "E7", "EF2000", "F117", 
        "F14", "F15", "F16", "F18", "F22", "F35", "F4", "H6", "J10", 
        "J20", "JAS39", "JF17", "JH7", "Ka27", "Ka52", "KC135", "KF21", 
        "KJ600", "Mi24", "Mi26", "Mi28", "Mig29", "Mig31", "Mirage2000", 
        "MQ9", "P3", "Rafale", "RQ4", "SR71", "Su24", "Su25", "Su34", 
        "Su57", "TB001", "TB2", "Tornado", "Tu160", "Tu22M", "Tu95", 
        "U2", "UH60", "US2", "V22", "Vulcan", "WZ7", "XB70", "Y20", 
        "YF23", "Z19"
    ]
    return class_names


def tensorflow_detect_objects(image, top_k=10):
    try:
        # model_path_relative = "model/warden_0896.h5"
        model_path_relative = "model/warden_0726.h5"
        model_path_full = os.path.join(ROOT, model_path_relative)
        model = load_model(model_path_full)
    except Exception as e:
        st.warning(f"Please try again.")
        st.error(f"Error loading TensorFlow model: {e}")
        st.stop()
        return None
        
    
    if image.mode != "RGB":
        image = image.convert("RGB")                                 # Convert to RGB mode
        
    # Preprocess the image
    image_array = np.array(image.resize((224, 224))) / 255.0        # Resize and normalize the image
    image_array = np.expand_dims(image_array, axis=0)               # Add batch dimension 

    predictions = model.predict(image_array)                        # Perform prediction
    class_probs = predictions[0]                                    # Get the class probabilities
    
    top_indices = np.argsort(class_probs)[::-1][:top_k]              # Get the indices of the top-k classes
    top_k_class_probs = [(index, class_probs[index]) for index in top_indices]                      # Get the probabilities of the top-k classes

       
    return predictions, top_k_class_probs



def main():
    st.title("Aircraft Detection App")
    
    ai_choice = ["TensorFlow (Classification 1 class)", "YOLO (Detection multiple classes)"]
    selected_ai = st.selectbox("Select AI model", ai_choice)
    
    if "selected_ai" in st.session_state and st.session_state["selected_ai"] != selected_ai:
        st.session_state.pop("original_img", None)
        st.session_state.pop("annotated_img", None)
        st.session_state.pop("yolo_class_count", None)
        st.session_state.pop("tensorflow_predictions", None)
        st.session_state.pop("top_k_class_probs", None)
        # st.session_state.clear()      # careful with this, it will clear all session state variables including the user preferences

    st.session_state["selected_ai"] = selected_ai
    
    
    # Option to upload an image or input an image URL
    option_choice = ["Upload Image", "Image URL"]
    selected_option = st.segmented_control("Choose input method", option_choice, default=option_choice[0], key="option")
    
    text_style = ["Normal", "Code"]

    # Ensure original image is only stored once
    if selected_option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Store the original image only once
            if "original_img" not in st.session_state:
                st.session_state["original_img"] = image
            # st.image(image, caption="Uploaded Image.", use_container_width=True)
            
            if st.button("Detect Aircraft"):
                with st.spinner("Detecting objects..."):
                    if selected_ai == ai_choice[1]:
                        annotated_img, results = yolo_detect_objects(image)
                        class_count = yolo_extract_classes_and_count(results)
                        st.session_state["annotated_img"] = annotated_img
                        st.session_state["yolo_class_count"] = class_count
                    else:
                        top_k = 5
                        predictions, top_k_class_probs = tensorflow_detect_objects(image, top_k)
                        st.session_state["tensorflow_predictions"] = predictions
                        st.session_state["top_k_class_probs"] = top_k_class_probs
                        st.write(f"Top {top_k} predictions:")
                      
                      
    elif selected_option == "Image URL":
        image_url = st.text_input("Enter image URL")
        if st.button("Detect Aircraft") and image_url:
            image = load_image_from_url(image_url)
            # Store the original image only once
            if "original_img" not in st.session_state:
                st.session_state["original_img"] = image
            # st.image(image, caption="Image from URL.", use_container_width=True)
            with st.spinner("Detecting objects..."):
                if selected_ai == ai_choice[1]:
                    annotated_img, results = yolo_detect_objects(image)
                    class_count = yolo_extract_classes_and_count(results)
                    st.session_state["annotated_img"] = annotated_img
                    st.session_state["yolo_class_count"] = class_count
                else:
                    top_k = 5
                    predictions, top_k_class_probs = tensorflow_detect_objects(image, top_k)
                    st.session_state["tensorflow_predictions"] = predictions
                    st.session_state["top_k_class_probs"] = top_k_class_probs
                    st.write(f"Top {top_k} predictions:")
                    
                
        elif not image_url:
            st.warning("Please enter an image URL.")

    # Display original image if available
    if "original_img" in st.session_state:
        st.image(st.session_state["original_img"], caption="Original Image.", use_container_width=True)
    
    # Render results only if detection was performed
    if "annotated_img" in st.session_state and ai_choice[1] in selected_ai:
        st.image(st.session_state["annotated_img"], caption="Detection Results with Bounding Boxes.", use_container_width=True)
        st.write("Detected objects:")

    if ai_choice[1] in selected_ai and "yolo_class_count" in st.session_state:
        # Unique key based on option and AI selection
        result_text_style = st.segmented_control(
            "Select result text style", 
            text_style, 
            default=text_style[0],
            key=f"result_text_style_{selected_option}_{selected_ai}"
        )
    
        if result_text_style == "Normal":
            st.subheader(f"`{'`, `'.join([f'{name}: {count}' for name, count in st.session_state['yolo_class_count'].items()])}`")
        elif result_text_style == "Code":
            st.code(st.session_state["yolo_class_count"])
            
    if ai_choice[0] in selected_ai and "tensorflow_predictions" in st.session_state and "top_k_class_probs" in st.session_state:
        result_text_style = st.segmented_control(
            "Select result text style", 
            text_style, 
            default=text_style[0], 
            key=f"result_text_style_{selected_option}_{selected_ai}"
        )
    
        if result_text_style == "Normal":    
            st.write("TensorFlow predictions (74 classes):")
            st.write(st.session_state["tensorflow_predictions"])
            
            for rank, (index, prob) in enumerate(st.session_state["top_k_class_probs"]):
                st.subheader(f"#{rank+1} | {tensorflow_aircraft_class_names()[index]} (`{prob:.2%}`)") 
                
        elif result_text_style == "Code":
            tensorflow_predictions = list(st.session_state["tensorflow_predictions"])
            st.code(tensorflow_predictions)
            
        
    # end of main()


if __name__ == "__main__":
    pre_config()
    main()