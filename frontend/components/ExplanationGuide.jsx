import React from 'react';

export default function ExplanationGuide({ modelAnalysis, beginnerExplanation }) {
  
  // Clean markdown headings & render lines nicely
  const renderFormattedText = (text) => {
    if (!text) return null;
    const lines = text.split('\n');
    return lines.map((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) return <div key={idx} style={{ height: '8px' }}></div>;
      
      // Check for headings
      if (
        trimmed.startsWith('Factors Favouring') || 
        trimmed.startsWith('Factors Against') || 
        trimmed.startsWith('Strategic Advice:') ||
        trimmed.includes('SECTION A') ||
        trimmed.includes('SECTION B') ||
        trimmed.includes('Business & Industry Context') ||
        trimmed.includes('The Big Picture') ||
        trimmed.includes('What You Should Do')
      ) {
        let headingColor = '#FFFFFF';
        if (trimmed.includes('+')) headingColor = '#A7F3D0';
        else if (trimmed.includes('-')) headingColor = '#FECACA';
        
        return (
          <h4 key={idx} style={{ 
            fontSize: '15px', 
            fontWeight: 600, 
            marginTop: '16px', 
            marginBottom: '10px',
            color: headingColor
          }}>
            {trimmed.replace(/===/g, '').trim()}
          </h4>
        );
      }
      
      // If it's a bullet point
      if (trimmed.startsWith('•') || trimmed.startsWith('-') || trimmed.startsWith('*')) {
        const content = trimmed.replace(/^[•\-*]\s*/, '').trim();
        return (
          <li key={idx} style={{ 
            marginLeft: '20px', 
            marginBottom: '8px', 
            fontSize: '14px', 
            color: '#E2E8F0',
            lineHeight: '1.6'
          }}>
            {content}
          </li>
        );
      }
      
      // Default paragraph
      return (
        <p key={idx} style={{ 
          fontSize: '14px', 
          color: '#CBD5E1', 
          lineHeight: '1.6', 
          marginBottom: '8px' 
        }}>
          {trimmed}
        </p>
      );
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      {/* Segment 1: Model Analysis & Strategy */}
      <div className="glass-card" style={{ borderLeft: '4px solid var(--primary-emerald)' }}>
        <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF', marginBottom: '8px' }}>
          🤖 Model Analysis & Strategy
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {renderFormattedText(modelAnalysis)}
        </div>
      </div>

      {/* Segment 2: Simple Explanation Guide */}
      <div className="glass-card">
        <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF', marginBottom: '12px' }}>
          🧠 Simple Explanation Guide
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {renderFormattedText(beginnerExplanation)}
        </div>
      </div>
    </div>
  );
}
