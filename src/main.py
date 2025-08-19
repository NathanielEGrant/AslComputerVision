# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras import layers, Input, Model # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.applications import MobileNetV2 # type: ignore
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
import cv2
import numpy as np
import streamlit as st

# %%
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATAPATH = "Data/ASL_Alphabet_Dataset/asl_alphabet_train"
EPOCHS = 5
random_seed = 67
np.random.seed(random_seed)

# %% [markdown]
# ## Data prep code

# %%
@st.cache_resource
def cnnDataPrep():

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        DATAPATH,
        target_size = IMG_SIZE,
        class_mode = "categorical",
        subset = "training"
    )

    val_gen = train_datagen.flow_from_directory(
        DATAPATH,
        target_size = IMG_SIZE,
        class_mode = "categorical",
        subset = "validation"
    )
    return train_gen, val_gen

# %% [markdown]
# ## Run this to prepare data!

# %%
train_gen, val_gen = cnnDataPrep()

# %% [markdown]
# ## Model training Code

# %%
def modelTrain():
    data = np.load("asl_combined_data.npz")
    
    image_input = Input(shape=(224, 224, 3), name="img_input")
    base_model = MobileNetV2(weights='imagenet',include_top=False, input_shape=(224,224,3))(image_input)
    x1 = layers.GlobalAveragePooling2D()(base_model)
    x1= layers.Dense(128, activation='relu')(x1)
    x1 = layers.Dropout(0.35)(x1)
    output = layers.Dense(29, activation='softmax')(x1)
    

    model = Model(inputs=image_input, outputs = output)
    
    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics =['accuracy'])
    model.fit(train_gen, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data = val_gen)
    model.save("models/trained_model.h5")
    model.summary()
    return model

# %%
#TF LiteConversion
def convertToTFLite(model_path, tflite_path):
    model = load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.float16]
    tflite_model = converter.convert()
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted and saved to {tflite_path}")

# %% [markdown]
# ## Run this to train model and convert to TFLite for better runtime

# %%
modelTrain()
convertToTFLite("models/trained_model.h5", "models/mainmodel4now.tflite") #Change model name as needed

# %% [markdown]
# ## Used to deploy mobile window, version with no mediapipe

# %%
def camDeploy(model_to_be_used):
    model = load_model(model_to_be_used)
    labels = list(train_gen.class_indices.keys())

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        roi = cv2.resize(frame, IMG_SIZE)
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)
    
        pred = model.predict(roi, verbose=0)
        idx = np.argmax(pred[0])
    
        if idx < len(labels):
            letter = labels[idx]
        else:
            letter = "unknown"    

        mirror_frame = cv2.flip(frame, 1)
        cv2.putText(mirror_frame, f"Predicted: {letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        display_frame = cv2.resize(mirror_frame, None, fy=1.5, fx=1.5)
        cv2.imshow('ASL Recognition', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    pass

# %% [markdown]
# ## Run this to open CV window

# %%
camDeploy("models/mainmodel4now.tflite") 

# %% [markdown]
# ## Helper code with mediapipe to track and box hand in frame

# %%
def as_landmark_list(lms):
    if hasattr(lms, 'landmark'):
        return lms.landmark
    return lms

def bbox_from_landmarks(landmarks, img_shape, pad=0.5):
    
    H,W = img_shape[:2]
    points = as_landmark_list(landmarks)
    xs = np.array([lm.x for lm in points], dtype=float) * W
    ys = np.array([lm.y for lm in points], dtype=float) * H
    
    x1, y1 = np.min(xs), np.min(ys)
    x2, y2 = np.max(xs), np.max(ys)
    
    w = x2 - x1
    h = y2 - y1
    x1 -= w * pad
    y1 -= h * pad
    x2 += w * pad
    y2 += h * pad
    
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(W, int(np.ceil(x2)))
    y2 = min(H, int(np.ceil(y2)))
    return (x1, y1, x2, y2)

def draw_bbox(image, bbox, label=None, thickness=2):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
    if label:
        cv2.putText(image, label, (x1, max(0, y1 - 8)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness, cv2.LINE_AA)

def predict_in_bbox(frame, box):
    
    if frame is None or box is None:
        return None
    
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    
    if crop.ndim == 2:
        crop = crop[..., None]
    elif crop.ndim != 3:
        return None
    
    crop = cv2.resize(crop, IMG_SIZE, interpolation=cv2.INTER_AREA)
    
    crop = crop.astype('float32') / 255.0
    crop = np.expand_dims(crop, axis=0)
    return crop
    

# %%
def process_frame(frame, model, hands, labels):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        box = bbox_from_landmarks(hand_landmarks, frame.shape)
        draw_bbox(frame, box, label="Hand")
        
        x = predict_in_bbox(frame, box)
        if x is not None:
            preds = model.predict(x, verbose=0)
            idx = int(np.argmax(preds[0]))
            confidence = float(preds[0][idx])
            label = labels[idx] if 0 <= idx < len(labels) else "Unknown"
            
            if confidence < 0.4:
                label = "Unknown"
                
            cv2.putText(frame, f"Predicted: {label} ({confidence:.2f})", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        return frame

# %% [markdown]
# ## Code to run camera using mediapipe framing/boxing

# %%
def camMPDeploy(model_to_be_used):
    model = load_model(model_to_be_used)
    labels = list(train_gen.class_indices.keys())

    cap = cv2.VideoCapture(0) 
    
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while True:
        
            success, frame = cap.read()
        
            if not success:
                print("Camera fram not found")
            
            frame = cv2.flip(frame, 1)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            hands_detected = hands.process(frame)
        
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if hands_detected.multi_hand_landmarks:
                for hand_landmarks in hands_detected.multi_hand_landmarks:
                    None
                box = bbox_from_landmarks(hand_landmarks, frame.shape)
                draw_bbox(frame, box, label="Hand") 
                 
                x = predict_in_bbox(frame, box)
                
                if x is None:
                    continue  
               
                preds = model.predict(x, verbose=0)
                idx = int(np.argmax(preds[0]))
                confidence = float(preds[0][idx])
                label = labels[idx]
                    
                if confidence < 0.4:
                    label = "Unknown"
                    
                cv2.putText(frame, f"Predicted: {label} ({confidence:.2f})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
                     
            cv2.imshow('Hand Recognition', frame)
                    
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    pass

# %% [markdown]
# ## Run this to open CV camera window

# %%
camMPDeploy("models/mainmodel4now.tflite")  # Change model name as needed

# %% [markdown]
# ## Mute the import main call

# %%
if __name__ == "__main__":
    train_gen, val_gen = cnnDataPrep()
    model = modelTrain()
    camDeploy()
    camMPDeploy()
    modelTrain()
    camMPDeploy()
    process_frame()
    as_landmark_list()
    bbox_from_landmarks()
    draw_bbox()
    predict_in_bbox() 
    cnnDataPrep()


