import React from 'react';

export default function Fundamentals({ data }) {
  if (!data || Object.keys(data).length === 0) {
    return (
      <div className="glass-card">
        <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF', marginBottom: '12px' }}>
          📊 Fundamental Valuation (Long-Term Health)
        </h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>Fundamentals metrics are currently unavailable for this stock.</p>
      </div>
    );
  }

  const pe = data.PE_Ratio;
  const eps = data.EPS;
  const pb = data.PB_Ratio;
  const debt = data.Debt_to_Equity;
  const roe = data.ROE;
  const mcap = data.Market_Cap;

  // Formatting helpers
  const peStr = pe !== null && pe !== undefined ? pe.toFixed(2) : "N/A";
  const pbStr = pb !== null && pb !== undefined ? pb.toFixed(2) : "N/A";
  const epsStr = eps !== null && eps !== undefined ? `₹${eps.toFixed(2)}` : "N/A";
  const debtStr = debt !== null && debt !== undefined ? `${debt.toFixed(2)}%` : "N/A";
  const roeStr = roe !== null && roe !== undefined ? `${roe.toFixed(2)}%` : "N/A";
  const mcapStr = mcap !== null && mcap !== undefined ? `₹${mcap.toLocaleString('en-IN')} Cr` : "N/A";

  // Interpretation helpers
  const peInterpretation = pe === null || pe === undefined ? "N/A" : pe > 40 ? "Premium / High Value (> 40)" : pe > 25 ? "Moderate Value (25-40)" : "Undervalued / Healthy (< 25)";
  const pbInterpretation = pb === null || pb === undefined ? "N/A" : pb > 6.0 ? "Growth Premium (> 6.0)" : pb > 3.0 ? "Fair Value (3.0-6.0)" : "Value Stock (< 3.0)";
  const epsInterpretation = eps === null || eps === undefined ? "N/A" : eps > 0 ? "Profitable ✅" : "Negative Earnings ⚠️";
  const debtInterpretation = debt === null || debt === undefined ? "N/A" : debt > 100 ? "Highly Leveraged (> 100%)" : debt > 50 ? "Leveraged (50-100%)" : "Low Debt / Low Risk (< 50%)";
  const roeInterpretation = roe === null || roe === undefined ? "N/A" : roe > 15 ? "High Return / Efficient (> 15%)" : roe > 8 ? "Moderate Return (8-15%)" : "Low Efficiency (< 8%)";
  const mcapInterpretation = mcap === null || mcap === undefined ? "N/A" : mcap > 100000 ? "Mega Cap (> 1,00,000 Cr)" : mcap > 20000 ? "Large Cap (20k-100k Cr)" : "Mid/Small Cap (< 20k Cr)";

  const rows = [
    { label: "P/E Ratio (Price/Earnings)", value: peStr, benchmark: peInterpretation },
    { label: "EPS (Earnings Per Share)", value: epsStr, benchmark: epsInterpretation },
    { label: "P/B Ratio (Price/Book)", value: pbStr, benchmark: pbInterpretation },
    { label: "Debt to Equity Ratio", value: debtStr, benchmark: debtInterpretation },
    { label: "Return on Equity (ROE)", value: roeStr, benchmark: roeInterpretation },
    { label: "Market Capitalization", value: mcapStr, benchmark: mcapInterpretation }
  ];

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF' }}>
        📊 Fundamental Valuation (Long-Term Health)
      </h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', textAlign: 'left' }}>
            <th style={{ padding: '10px 0', color: 'var(--text-secondary)', fontWeight: 500 }}>Key Metric</th>
            <th style={{ padding: '10px 0', color: 'var(--text-secondary)', fontWeight: 500, textAlign: 'right' }}>Value</th>
            <th style={{ padding: '10px 0 10px 16px', color: 'var(--text-secondary)', fontWeight: 500 }}>Interpretation</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
              <td style={{ padding: '12px 0', fontWeight: 500, color: '#E2E8F0' }}>{row.label}</td>
              <td style={{ padding: '12px 0', textAlign: 'right', fontWeight: 600, color: '#FFFFFF' }}>{row.value}</td>
              <td style={{ padding: '12px 0 12px 16px', color: '#CBD5E1' }}>{row.benchmark}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
