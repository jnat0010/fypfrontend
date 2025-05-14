// src/services/api.js
import axios from 'axios';
const BASE = process.env.REACT_APP_API_BASE_URL;

export async function restoreImage(file, paramsOrProgress, maybeProgress) {
  // Determine which signature was used:
  let params = {};
  let onProgress = () => {};

  if (typeof paramsOrProgress === 'function') {
    // old signature: (file, onProgress)
    onProgress = paramsOrProgress;
  } else {
    // new signature: (file, params, onProgress)
    params = paramsOrProgress || {};
    if (typeof maybeProgress === 'function') {
      onProgress = maybeProgress;
    }
  }

  // Destructure with defaults
  const {
    brightness = 100,
    noise      = 0,
    contrast   = 100,
  } = params;

  const form = new FormData();
  form.append('image', file);
  form.append('brightness', brightness);
  form.append('noise',      noise);
  form.append('contrast',   contrast);

  const response = await axios.post(
    `${BASE}/api/restore`,
    form,
    {
      responseType: 'blob',
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: ({ loaded, total }) => {
        const pct = Math.round((loaded / total) * 100);
        onProgress(pct);
      }
    }
  );

  return response.data;
}
