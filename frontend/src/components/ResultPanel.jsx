// src/components/ResultPanel.jsx
import React from 'react';

export default function ResultPanel({
  src,
  original,
  progress,
  onRedo,
  onToggleOriginal,
  showOriginal
}) {
  // Which URL is currently displayed?
  const displaySrc = showOriginal ? original : src;

  return (
    <>
      {/* Preview box */}
      <div className="result-preview">
        { displaySrc
          ? <img src={displaySrc} alt="" className="preview-img" />
          : <span>No image yet</span>
        }
      </div>

      {/* Progress + buttons */}
      <div className="footer">
        <progress value={progress} max="100" style={{ width: '100%' }} />
        <div className="footer-actions">
          <button onClick={onRedo} disabled={!src}>
            Redo
          </button>
          <button
            onClick={onToggleOriginal}
            disabled={!src}
          >
            {showOriginal ? 'View Restored' : 'View Original'}
          </button>
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
