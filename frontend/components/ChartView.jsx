import React from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';

export default function ChartView({ data, companyName }) {
  if (!data || data.length === 0) {
    return <div style={{ color: 'var(--text-secondary)' }}>No price history data available.</div>;
  }

  // Calculate domain min/max so the chart line isn't squeezed at the top
  const closes = data.map(d => d.close).filter(v => v !== null && v !== undefined);
  const minClose = Math.min(...closes);
  const maxClose = Math.max(...closes);
  const padding = (maxClose - minClose) * 0.1 || 10;
  
  const yDomain = [
    Math.max(0, Math.floor(minClose - padding)),
    Math.ceil(maxClose + padding)
  ];

  return (
    <div className="glass-card" style={{ height: '320px', padding: '20px' }}>
      <h3 style={{ fontSize: '16px', fontWeight: 600, marginBottom: '16px', color: '#FFFFFF' }}>
        📈 90-Day Closing Price Trend: {companyName}
      </h3>
      <div style={{ width: '100%', height: '220px' }}>
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
            <defs>
              <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10B981" stopOpacity={0.2}/>
                <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis 
              dataKey="date" 
              stroke="#475569" 
              fontSize={11}
              tickLine={false}
              axisLine={false}
              dy={10}
            />
            <YAxis 
              domain={yDomain}
              stroke="#475569" 
              fontSize={11}
              tickLine={false}
              axisLine={false}
              dx={-5}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: '#1E293B', 
                border: '1px solid #334155',
                borderRadius: '6px',
                color: '#FFFFFF'
              }}
              labelStyle={{ color: '#94A3B8', fontSize: '11px' }}
            />
            <Area 
              type="monotone" 
              dataKey="close" 
              stroke="#10B981" 
              strokeWidth={2}
              fillOpacity={1} 
              fill="url(#colorClose)" 
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
