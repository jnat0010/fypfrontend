// src/components/ResultPanel.jsx
import React, { useState } from 'react';

export default function ResultPanel({
  src,
  original,    // URL of the uploaded image preview
  progress,
  onRedo,
  onFeedback
}) {
  const [showOriginal, setShowOriginal] = useState(false);

  // Decide which URL to display
  const displaySrc = showOriginal ? original : src;

  return (
    <>
      <div className="result-image">
        {displaySrc ? (
          <img
            src={displaySrc}
            alt={showOriginal ? "Original" : "Restored"}
            style={{ maxWidth: '100%' }}
          />
        ) : (
          <span>No image yet</span>
        )}
      </div>

      <div className="footer">
        <progress value={progress} max="100" style={{ width: '100%' }} />

        <div className="footer-actions">
          {/* Feedback button */}
          <button onClick={onFeedback} disabled={!src || !onFeedback}>
            Leave Feedback
          </button>

          {/* Redo button */}
          <button onClick={onRedo} disabled={!src}>
            Redo
          </button>

          {/* Toggle Original/Restored */}
          <button
            onClick={() => setShowOriginal(!showOriginal)}
            disabled={!src || !original}
          >
            {showOriginal ? "View Restored" : "View Original"}
          </button>

          {/* Download always downloads the restored image */}
          {src ? (
            <a href={src} download="restored.png">
              <button>Download Result</button>
            </a>
          ) : (
            <button disabled>Download Result</button>
          )}
        </div>
      </div>
    </>
  );
}
