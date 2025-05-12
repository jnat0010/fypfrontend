// src/App.js
import React, { useState } from 'react';
import './App.css';

import UploadPanel      from './components/UploadPanel';
import ResultPanel      from './components/ResultPanel';
import ColourHistogram  from './components/ColourHistogram';
import SettingsPanel    from './components/SettingsPanel';  // ← new
import { restoreImage } from './services/api';

export default function App() {
  const [file, setFile]               = useState(null);
  const [preview, setPreview]         = useState(null);
  const [progress, setProgress]       = useState(0);
  const [resultSrc, setResultSrc]     = useState(null);
  const [showOriginal, setShowOriginal] = useState(false);

  // NEW: slider params
  const [params, setParams] = useState({
    brightness: 100,
    noise:       0,
    contrast:   100,
  });
 
  const handleParamChange = (name, value) => {
    setParams(prev => ({ ...prev, [name]: value }));
  };

  // When the user picks a file:
  const handleFileSelect = f => {
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
    setResultSrc(null);
    setProgress(0);
    setShowOriginal(false);
  };

  // Send to ML service (now passes params)
  const runRestore = async () => {
    if (!file) return;
    setProgress(0);
    setResultSrc(null);
    const blob = await restoreImage(file, params, setProgress);
    setResultSrc(URL.createObjectURL(blob));
    setShowOriginal(false);
  };

  // “Redo” just re‐runs the same restore:
  const handleRedo = () => runRestore();

  // Flip between original / restored:
  const toggleOriginal = () => {
    setShowOriginal(v => !v);
  };

  // Which URL are we showing right now?
  const displaySrc = showOriginal ? preview : resultSrc;

  return (
    <div className="app-root">
      <header className="app-header">Colour Cast Remover</header>

      <div className="app-container">
        {/* ─── UPLOAD COLUMN ─────────────────────────── */}
        <div className="left-panel">
          <div className="card">
            <div className="card-header">
              <h2>Upload Original Image</h2>
            </div>
            <div className="card-body">
              <UploadPanel
                file={file}
                preview={preview}
                onFileSelect={handleFileSelect}
                onUpload={runRestore}
              />

              {/* ─── SETTINGS PANEL ──────────────────────── */}
              <SettingsPanel
                brightness={params.brightness}
                noise={params.noise}
                contrast={params.contrast}
                onChange={handleParamChange}
              />
            </div>
          </div>

          {/* ─── HISTOGRAM CARD ───────────────────────── */}
          <div className="card histogram-card" style={{ marginTop: 24 }}>
            <div className="card-header">
              <h2>Colour Histogram</h2>
            </div>
            <div className="card-body">
              {displaySrc ? (
                <ColourHistogram imageSrc={displaySrc} />
              ) : (
                <div className="histogram-preview">
                  <span>No histogram data yet</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ─── RESULT COLUMN ─────────────────────────── */}
        <div className="right-panel">
          <div className="card">
            <div className="card-header">
              <h2>Result</h2>
            </div>
            <div className="card-body">
              <ResultPanel
                src={resultSrc}
                original={preview}
                progress={progress}
                onRedo={handleRedo}
                showOriginal={showOriginal}
                onToggleOriginal={toggleOriginal}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
