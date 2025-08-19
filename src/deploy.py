# %%
import main
import tensorflow as tf
import os
from tensorflow.keras.models import load_model # type: ignore
import mediapipe as mp
import cv2
import numpy as np
import streamlit as st
import os, queue, threading
import time

# %% [markdown]
# ## Hardware optimizations

# %%
os.environ["OMP_NUM_THREADS"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
cv2.setNumThreads(2)

# %% [markdown]
# ## Main variables

# %%
IMG_SIZE = main.IMG_SIZE
MODEL_TO_BE_USED_H5 = "models/USINGTHIS.h5"
TF_MODEL_TO_BE_USED = "models/USINGTHIS.tflite"
TARGET_W, TARGET_H = 640, 480
PRED_EVERY_N = 3

# %% [markdown]
# ## TfLite class

# %%
class TFLITEModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=2)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.in_dtype = self.input_details[0]['dtype']

    def predict(self, x_np: np.ndarray) -> np.ndarray:
        x_in = x_np.astype(np.float16 if self.in_dtype == np.float16 else np.float32, copy=False)
        self.interpreter.set_tensor(self.input_details[0]['index'], x_in)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

# %% [markdown]
# ## Code for loading model data

# %%
@st.cache_data
def load_labels():
    train_gen, _ = main.cnnDataPrep()    
    return list(train_gen.class_indices.keys())
labels = load_labels()

# %%
@st.cache_resource
def load_asl_model():
    if os.path.exists(TF_MODEL_TO_BE_USED):
        return TFLITEModel(TF_MODEL_TO_BE_USED)
    else:
        return ("keras",load_model(MODEL_TO_BE_USED_H5))
model_kind, model = load_asl_model()

# %% [markdown]
# ## Streamlit config setups

# %%
st.set_page_config(page_title="ASL Translator", page_icon= ":ok_hand:", layout= "centered")
st.title("ASL Alphabet Translation")
st.subheader("Real-time sign language alphabet detection with webcam using deep learning!")

# %%
st.toggle("Start Camera", key="run_cam", value=False)
conf_thresh = st.slider("Confidence Threshold", 0.0, 1.0, 0.4, 0.05)
status = st.empty()
frame_placeholder = st.empty()

# %% [markdown]
# ## Code for running website and camera operations

# %%
# Main Loop
cap = None
frame_q:"queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
stop_flag = threading.Event()

# Init mediapipe hands once
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )


if st.session_state.get("run_cam", False):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        status.info("Camera is running.")
    else:
        st.error("Error: Could not open webcam.")
        cap = None
        
        
while (st.session_state.get("run_cam", False)) and (cap is not None) and (model is not None):
    ok, frame = cap.read()
    if (not ok) or (frame is None) or (getattr(frame, "size", 0) == 0):
        time.sleep(0.03)
        continue
    
    frame = cv2.flip(frame, 1) 
    
    try:
        processed = main.process_frame(
            frame, 
            model if model_kind=="keras" else model,
            hands,
            labels) 
        
    except Exception as e:
        processed = None
    
    if (processed is None) or (getattr(processed, "size", 0) == 0):
        processed = frame
    
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
    time.sleep(0.01)     
    
    
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
if not st.session_state.get("run_cam", False):
    status.success("Camera is stopped.")


