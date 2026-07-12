import React, { useState } from 'react';

const SUPPORTED_LIST = [
  "tcs", "reliance", "wipro", "infosys", "sbi", "sbin", "hdfc", "hdfcbank",
  "adani", "adaniports", "tata steel", "tatasteel", "tata power", "tatapower",
  "maruti", "maruti suzuki", "l&t", "lt", "airtel", "bhartiartl", "tata motors",
  "tatamotors", "titan", "wipro", "hcltech", "techm", "itc"
];

export default function Sidebar({ activeMode, setActiveMode }) {
  const [checkQuery, setCheckQuery] = useState("");
  const [checkResult, setCheckResult] = useState(null);

  const runCheck = (val) => {
    if (!val.trim()) {
      setCheckResult(null);
      return;
    }
    const clean = val.toLowerCase().trim();
    const isSupported = SUPPORTED_LIST.some(item =>
      item.includes(clean) || clean.includes(item)
    );
    if (isSupported) {
      setCheckResult({ success: true, text: `✅ Supported! You can search and analyze this stock.` });
    } else {
      setCheckResult({ success: false, text: `❌ Currently not available in the model database. Try another.` });
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      runCheck(checkQuery);
    }
  };

  const modes = [
    { id: "analysis", label: "Stock Analysis & Chat", icon: "🔍" },
    { id: "compare", label: "Side-by-Side Comparison", icon: "📊" },
    { id: "backtest", label: "Strategy Backtesting", icon: "📈" },
    { id: "learn", label: "Learn Basics", icon: "📚" }
  ];

  return (
    <div style={{
      width: '280px',
      padding: '24px 20px',
      borderRight: '1px solid var(--card-border)',
      backgroundColor: '#0E1321',
      display: 'flex',
      flexDirection: 'column',
      gap: '24px',
      flexShrink: 0
    }}>
      {/* Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <img src="https://img.icons8.com/nolan/96/combo-chart.png" style={{ width: '40px' }} alt="Graph" />
        <div>
          <h2 style={{ fontSize: '18px', fontWeight: 700 }}>StockChat AI</h2>
          <p style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Multi-Signal Indian Stock Advisor</p>
        </div>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid rgba(255,255,255,0.08)' }} />

      {/* Mode List */}
      <div>
        <p style={{
          fontSize: '11px',
          fontWeight: 700,
          color: 'var(--text-secondary)',
          textTransform: 'uppercase',
          letterSpacing: '0.8px',
          marginBottom: '12px'
        }}>
          Choose Mode
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {modes.map(m => {
            const isActive = activeMode === m.id;
            return (
              <button
                key={m.id}
                onClick={() => setActiveMode(m.id)}
                style={{
                  width: '100%',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  border: isActive ? '1px solid var(--primary-emerald)' : '1px solid #2D3748',
                  background: isActive ? 'linear-gradient(90deg, #059669 0%, #10B981 100%)' : '#141B2D',
                  color: '#FFFFFF',
                  fontSize: '14px',
                  fontWeight: 600,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px',
                  boxShadow: isActive ? '0 4px 12px rgba(16, 185, 129, 0.2)' : 'none',
                  transition: 'all 0.2s ease',
                  textAlign: 'left'
                }}
              >
                <span>{m.icon}</span>
                <span>{m.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid rgba(255,255,255,0.08)' }} />

      {/* Supported Stocks checker */}
      <div>
        <p style={{
          fontSize: '11px',
          fontWeight: 700,
          color: 'var(--text-secondary)',
          textTransform: 'uppercase',
          letterSpacing: '0.8px',
          marginBottom: '8px'
        }}>
          Supported Stocks Check
        </p>
        <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
          Verify stock availability:
        </p>
        <input
          type="text"
          value={checkQuery}
          onChange={(e) => {
            setCheckQuery(e.target.value);
            if (!e.target.value.trim()) setCheckResult(null);
          }}
          onKeyDown={handleKeyDown}
          placeholder="e.g. Reliance, TCS, SBI... then press Enter"
          className="search-input"
          style={{ padding: '8px 12px', fontSize: '13px' }}
        />
        {checkResult && (
          <div style={{
            marginTop: '10px',
            padding: '10px',
            borderRadius: '6px',
            fontSize: '12px',
            backgroundColor: checkResult.success ? 'rgba(16,185,129,0.1)' : 'rgba(239,68,68,0.1)',
            border: checkResult.success ? '1px solid rgba(16,185,129,0.2)' : '1px solid rgba(239,68,68,0.2)',
            color: checkResult.success ? '#A7F3D0' : '#FECACA'
          }}>
            {checkResult.text}
          </div>
        )}
        <div style={{ marginTop: '12px', fontSize: '11px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
          <strong>Quick Examples:</strong><br />
          TCS, Reliance, Wipro, Infosys, SBI, HDFC Bank, Adani Ports, Tata Steel, Tata Power, Maruti Suzuki.
        </div>
      </div>
    </div>
  );
}
