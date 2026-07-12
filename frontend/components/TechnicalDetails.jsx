import React from 'react';

export default function TechnicalDetails({ technicals, currentPrice }) {
  const rsi = technicals.RSI || 50.0;
  const ma50 = technicals.MA50 || currentPrice;
  const ma200 = technicals.MA200 || currentPrice;
  const macd = technicals.MACD || 0.0;

  // status helpers
  const rsiStatus = rsi > 70 ? "Overbought ⚠️" : rsi < 30 ? "Oversold 📈" : "Neutral ⚖️";
  const ma50Status = currentPrice > ma50 ? "Above (Uptrend) ✅" : "Below (Downtrend) ⚠️";
  const ma200Status = currentPrice > ma200 ? "Above (Long-term Up) ✅" : "Below (Long-term Down) ⚠️";
  const macdStatus = macd > 0 ? "Bullish momentum 📈" : "Bearish momentum 📉";

  const rows = [
    { name: "RSI (14-day)", value: rsi.toFixed(2), status: rsiStatus },
    { name: "50-Day Moving Avg", value: `₹${ma50.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, status: ma50Status },
    { name: "200-Day Moving Avg", value: `₹${ma200.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, status: ma200Status },
    { name: "MACD Histogram", value: `${macd >= 0 ? '+' : ''}${macd.toFixed(2)}`, status: macdStatus }
  ];

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF' }}>
        📐 Technical Details
      </h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', textAlign: 'left' }}>
            <th style={{ padding: '10px 0', color: 'var(--text-secondary)', fontWeight: 500 }}>Indicator</th>
            <th style={{ padding: '10px 0', color: 'var(--text-secondary)', fontWeight: 500, textAlign: 'right' }}>Value</th>
            <th style={{ padding: '10px 0 10px 16px', color: 'var(--text-secondary)', fontWeight: 500 }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <td style={{ padding: '12px 0', fontWeight: 500, color: '#E2E8F0' }}>{row.name}</td>
              <td style={{ padding: '12px 0', textAlign: 'right', fontWeight: 600, color: '#FFFFFF' }}>{row.value}</td>
              <td style={{ padding: '12px 0 12px 16px', color: '#CBD5E1' }}>{row.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
