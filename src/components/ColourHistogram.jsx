import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip
} from 'recharts';

/**
 * ColourHistogram
 * Renders an RGB histogram with overlapping translucent areas on a dark background.
 * @param {{ imageSrc: string }} props
 */
export default function ColourHistogram({ imageSrc }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    if (!imageSrc) {
      setData([]);
      return;
    }
    const img = new Image();
    img.crossOrigin = '';
    img.src = imageSrc;
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const { data: px } = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const binsR = new Array(256).fill(0);
      const binsG = new Array(256).fill(0);
      const binsB = new Array(256).fill(0);
      for (let i = 0; i < px.length; i += 4) {
        binsR[px[i]]++;
        binsG[px[i + 1]]++;
        binsB[px[i + 2]]++;
      }
      const chartData = binsR.map((r, idx) => ({
        value: idx,
        red: r,
        green: binsG[idx],
        blue: binsB[idx]
      }));
      setData(chartData);
    };
    img.onerror = () => setData([]);
  }, [imageSrc]);

  if (!data.length) {
    return null;
  }

  return (
    <div className="histogram-container">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <XAxis dataKey="value" hide />
          <YAxis hide />
          <Tooltip
            contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }}
            labelFormatter={val => `Intensity: ${val}`}
            formatter={(value, name) => [value, name.charAt(0).toUpperCase() + name.slice(1)]}
          />
          <Area
            type="monotone"
            dataKey="blue"
            stroke="#3399FF"
            fill="#3399FF"
            fillOpacity={0.4}
            isAnimationActive={false}
          />
          <Area
            type="monotone"
            dataKey="green"
            stroke="#33FF99"
            fill="#33FF99"
            fillOpacity={0.4}
            isAnimationActive={false}
          /> 
          <Area
            type="monotone"
            dataKey="red"
            stroke="#FF3366"
            fill="#FF3366"
            fillOpacity={0.4}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
