import axios from 'axios';

const API_BASE =
  process.env.REACT_APP_API_BASE_URL?.replace(/\/$/, '') || ''; // e.g. "http://localhost:8000" or "" if using CRA proxy

/**
 * Send the user’s file to the ML endpoint,
 * report upload progress, and return the processed image blob.
 */
export async function restoreImage(file, onProgress) {
  const form = new FormData();
  form.append('image', file);

  const response = await axios.post(
    `${API_BASE}/api/restore`,
    form, 
    {
      responseType: 'blob',     // we expect an image back
      onUploadProgress: ({ loaded, total }) => {
        const percent = Math.round((loaded / total) * 100);
        onProgress(percent);
      }
    }
  );

  return response.data;  // a Blob

  
}
