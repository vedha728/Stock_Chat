import React from 'react';

export default function SignalsSummary({ technicals, sentiment, institutional }) {
  // 1. Technical Outlook
  const p50 = technicals.Price_Above_MA50;
  const p200 = technicals.Price_Above_MA200;
  let techText = "Price moving DOWN";
  let techClass = "card-danger";
  let techIcon = "📉";
  if (p50 && p200) {
    techText = "Price moving UP strongly";
    techClass = "card-success";
    techIcon = "📈";
  } else if (p50 || p200) {
    techText = "Price moving UP weakly";
    techClass = "card-warning";
    techIcon = "⚠️";
  }

  // 2. News Outlook
  const score = sentiment.score || 0.0;
  let newsText = `Neutral / Flat (Score: ${score >= 0 ? '+' : ''}${score.toFixed(2)})`;
  let newsClass = "card-warning";
  let newsIcon = "⚖️";
  if (score > 0.15) {
    newsText = `Positive (Score: ${score >= 0 ? '+' : ''}${score.toFixed(2)})`;
    newsClass = "card-success";
    newsIcon = "📰";
  } else if (score < -0.15) {
    newsText = `Negative (Score: ${score.toFixed(2)})`;
    newsClass = "card-danger";
    newsIcon = "📰";
  }

  // 3. FII/DII Outlook
  const fii = institutional.FII_10d_Net || 0;
  const dii = institutional.DII_10d_Net || 0;
  let smartText = "Diverging institutional flow";
  let smartClass = "card-warning";
  let smartIcon = "💼";
  if (fii > 0 && dii > 0) {
    smartText = "Big investors buying heavily";
    smartClass = "card-success";
    smartIcon = "💼";
  } else if (fii < 0 && dii < 0) {
    smartText = "Big investors pulling money out";
    smartClass = "card-danger";
    smartIcon = "💼";
  } else if (fii < 0 && dii > 0) {
    smartText = "Foreign investors out, Indian investors buying";
    smartClass = "card-warning";
    smartIcon = "💼";
  } else {
    smartText = "Foreign investors buying, Indian investors reducing";
    smartClass = "card-success";
    smartIcon = "💼";
  }

  const signalStyles = {
    card: {
      padding: '16px 20px',
      borderRadius: '8px',
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      fontWeight: 500,
      fontSize: '14px',
      border: '1px solid'
    },
    success: {
      backgroundColor: 'rgba(16, 185, 129, 0.08)',
      borderColor: 'rgba(16, 185, 129, 0.2)',
      color: '#A7F3D0'
    },
    warning: {
      backgroundColor: 'rgba(245, 158, 11, 0.08)',
      borderColor: 'rgba(245, 158, 11, 0.2)',
      color: '#FDE68A'
    },
    danger: {
      backgroundColor: 'rgba(239, 68, 68, 0.08)',
      borderColor: 'rgba(239, 68, 68, 0.2)',
      color: '#FECACA'
    }
  };

  const getStyle = (type) => {
    if (type === "card-success") return { ...signalStyles.card, ...signalStyles.success };
    if (type === "card-warning") return { ...signalStyles.card, ...signalStyles.warning };
    return { ...signalStyles.card, ...signalStyles.danger };
  };

  return (
    <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF' }}>
        🔍 Core Signals Summary
      </h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={getStyle(techClass)}>
          <span style={{ fontSize: '18px' }}>{techIcon}</span>
          <div>
            <strong style={{ opacity: 0.7, fontSize: '11px', textTransform: 'uppercase', display: 'block', letterSpacing: '0.5px' }}>Technical Chart</strong>
            <span>{techText}</span>
          </div>
        </div>

        <div style={getStyle(newsClass)}>
          <span style={{ fontSize: '18px' }}>{newsIcon}</span>
          <div>
            <strong style={{ opacity: 0.7, fontSize: '11px', textTransform: 'uppercase', display: 'block', letterSpacing: '0.5px' }}>News Sentiment</strong>
            <span>{newsText}</span>
          </div>
        </div>

        <div style={getStyle(smartClass)}>
          <span style={{ fontSize: '18px' }}>{smartIcon}</span>
          <div>
            <strong style={{ opacity: 0.7, fontSize: '11px', textTransform: 'uppercase', display: 'block', letterSpacing: '0.5px' }}>Smart Money</strong>
            <span>{smartText}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
