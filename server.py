# server.py
import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ml.inference import remove_colour_cast

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/restore")
async def restore(
    image: UploadFile = File(...),
    brightness: float = Form(100.0),
    noise:      float = Form(0.0),
    contrast:   float = Form(100.0),
):
    # 1) Read the raw bytes
    img_bytes = await image.read()
    print(f"[server] Received upload: filename={image.filename}, size={len(img_bytes)} bytes")

    # 2) Run inference, passing through your three sliders
    output_bytes = remove_colour_cast(
        img_bytes,
        brightness_pct=brightness,
        noise_pct=noise,
        contrast_pct=contrast,
    )
    print(f"[server] Model returned {len(output_bytes)} bytes")

    # 3) Stream back the result
    return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")
