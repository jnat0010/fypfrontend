// src/components/UploadPanel.jsx
import React, { useRef } from 'react';

export default function UploadPanel({ file, preview, onFileSelect, onUpload }) {
  const fileInputRef = useRef(null);

  const handleSelectClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className="upload-panel">
      {/* 1) Hidden native file input */}
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        style={{ display: 'none' }}
        onChange={e => onFileSelect(e.target.files[0] || null)}
      />

      {/* 2) Preview box */}
      <div className="upload-preview">
        {preview
          ? <img src={preview} alt="Preview" className="preview-img" />
          : <span>No image selected</span>
        }
      </div>

      {/* 3a) Show “Select Image” if no file chosen */}
      {!file ? (
        <button onClick={handleSelectClick}>
          Select Image
        </button>
      ) : (
      /* 3b) Once a file is chosen: Change + Start buttons side by side */
        <div className="upload-actions">
          <button onClick={handleSelectClick}>
            Change Image
          </button>
          <button onClick={onUpload}>
            Start Restoring
          </button>
        </div>
      )}
    </div>
  );
}
