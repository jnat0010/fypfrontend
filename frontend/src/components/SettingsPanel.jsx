// src/components/SettingsPanel.jsx
import React from 'react';

export default function SettingsPanel({ brightness, noise, contrast, onChange }) {
  return (
    <div className="settings-panel">
      <div className="slider-group">
        <label>
          Brightness: {brightness}%
          <input
            type="range"
            min="0"
            max="200"
            step="1"
            value={brightness}
            onChange={e => onChange('brightness', Number(e.target.value))}
          />
        </label>
      </div>
      <div className="slider-group">
        <label>
          Noise Level: {noise}%
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={noise}
            onChange={e => onChange('noise', Number(e.target.value))}
          />
        </label>
      </div>
      <div className="slider-group">
        <label>
          Contrast: {contrast}%
          <input
            type="range"
            min="0"
            max="200"
            step="1"
            value={contrast}
            onChange={e => onChange('contrast', Number(e.target.value))}
          />
        </label>
      </div>
    </div>
  );
}
