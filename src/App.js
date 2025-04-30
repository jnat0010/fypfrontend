// src/App.js
import React, { useState } from 'react';
import './App.css';
import UploadPanel from './components/UploadPanel';
import ResultPanel from './components/ResultPanel';
import { restoreImage } from './services/api';

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [progress, setProgress] = useState(0);
  const [resultSrc, setResultSrc] = useState(null);

  // Called when user selects/choses a file
  const handleFileSelect = selectedFile => {
    setFile(selectedFile);
    setResultSrc(null);
    setProgress(0);
    setPreview(URL.createObjectURL(selectedFile));
  };

  // Upload + inference logic
  const runRestore = async () => {
    if (!file) return;
    setProgress(0);
    setResultSrc(null);
    const blob = await restoreImage(file, setProgress);
    setResultSrc(URL.createObjectURL(blob));
  };

  // Redo is just re-running the same restore
  const handleRedo = () => {
    runRestore();
  };

  return (
    <>
      <header className="app-header">Colour Cast Remover</header>
      <div className="app-container">
        <div className="left-panel">
          <div className="header-bar">
            <h2>Upload original Image</h2>
          </div>
          <UploadPanel
            file={file}
            preview={preview}
            onFileSelect={handleFileSelect}
            onUpload={runRestore}
          />
        </div>
        <div className="right-panel">
          <div className="header-bar">
            <h2>Result</h2>
          </div>
          <ResultPanel
            src={resultSrc}
            original={preview}       /* <-- add this */
            progress={progress}
            onRedo={handleRedo}
          />
        </div>
      </div>
    </>
  );
}
