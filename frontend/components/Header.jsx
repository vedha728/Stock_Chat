import React from 'react';

export default function Header() {
  return (
    <div className="header-banner">
      <h1 style={{
        margin: 0, 
        fontSize: '32px', 
        fontWeight: 700, 
        letterSpacing: '-0.5px',
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        <img 
          src="https://img.icons8.com/fluency/96/stocks.png" 
          style={{
            width: '42px', 
            height: '42px', 
            marginRight: '12px', 
            verticalAlign: 'middle',
            filter: 'drop-shadow(0 2px 8px rgba(16, 185, 129, 0.35))'
          }} 
          alt="StockChat Logo"
        />
        <span style={{
          background: 'linear-gradient(90deg, #10B981, #34D399)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent'
        }}>
          Multi-Signal Stock Advisory
        </span>
      </h1>
      <p style={{
        margin: '8px 0 0 0', 
        color: '#94A3B8', 
        fontSize: '15px', 
        fontWeight: 400, 
        letterSpacing: '0.2px'
      }}>
        Leveraging <span style={{ color: '#6EE7B7', fontWeight: 500 }}>Technical Charts</span>, <span style={{ color: '#6EE7B7', fontWeight: 500 }}>News Sentiment</span>, <span style={{ color: '#6EE7B7', fontWeight: 500 }}>Institutional Flows</span>, and <span style={{ color: '#6EE7B7', fontWeight: 500 }}>Fundamentals</span>
      </p>
    </div>
  );
}
