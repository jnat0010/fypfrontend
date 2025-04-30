# ml/inference.py
import io
import cv2
import numpy as np
import tensorflow as tf

from ml.modelCRRNew import ColorCastRemoval

model = ColorCastRemoval()
model.load("./checkpoints")

def remove_colour_cast(image_bytes: bytes) -> bytes:
    print(f"[inference] Got {len(image_bytes)} bytes of image data")

    # Decode to BGR array
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(f"[inference] Decoded to array of shape {bgr.shape}")

    # Convert to LAB and normalize
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    inp = np.expand_dims(lab, axis=0)

    # Call the model
    print("[inference] Running model...")
    _, corrected_lab, _, _, _ = model(tf.convert_to_tensor(inp), training=False)
    print("[inference] Model finished")

    # Convert back and encode
    out_lab = (corrected_lab[0].numpy() * 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    success, buf = cv2.imencode('.png', out_bgr)
    print(f"[inference] Encoded output to {len(buf.tobytes())} bytes")
    return buf.tobytes()
