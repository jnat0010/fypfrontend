# server.py
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ml.inference import remove_colour_cast

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/api/restore")
async def restore(image: UploadFile = File(...)):
    # 1) Read the raw bytes
    img_bytes = await image.read()
    print(f"[server] Received upload: filename={image.filename}, size={len(img_bytes)} bytes")

    # 2) Run inference
    output_bytes = remove_colour_cast(img_bytes)
    print(f"[server] Model returned {len(output_bytes)} bytes")

    # 3) Stream back the result
    return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")
