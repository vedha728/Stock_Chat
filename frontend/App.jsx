import React, { useState } from 'react';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ChartView from './components/ChartView';
import SignalsSummary from './components/SignalsSummary';
import TechnicalDetails from './components/TechnicalDetails';
import Fundamentals from './components/Fundamentals';
import ExplanationGuide from './components/ExplanationGuide';

const LOADING_STEPS = [
  "Analyzing technical charts & indicators...",
  "Fetching latest news sentiment...",
  "Calculating institutional flows (FII/DII)...",
  "Running predictive models...",
  "Generating AI strategic explanation..."
];

const COMPARE_LOADING_STEPS = [
  "Resolving stock symbols for comparison...",
  "Fetching technical charts & indicators for all target stocks...",
  "Retrieving news headlines & sentiment trends...",
  "Comparing institutional flow signals (FII/DII)...",
  "Running multi-signal prediction models...",
  "Generating AI side-by-side comparison report..."
];

const BACKTEST_LOADING_STEPS = [
  "Retrieving 1-year historical price data...",
  "Generating backtest window FII/DII capital flow cache...",
  "Synthesizing technical indicators day-by-day...",
  "Simulating trade execution strategies...",
  "Calculating benchmark buy & hold performance...",
  "Finalizing performance report metrics..."
];

export default function App() {
  const [activeMode, setActiveMode] = useState("analysis");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Loader states
  const [loadingStep, setLoadingStep] = useState(0);

  React.useEffect(() => {
    let interval;
    if (loading) {
      setLoadingStep(0);
      interval = setInterval(() => {
        setLoadingStep((prev) => (prev + 1) % LOADING_STEPS.length);
      }, 2000);
    } else {
      setLoadingStep(0);
    }
    return () => clearInterval(interval);
  }, [loading]);
  
  // Data states
  const [analysisData, setAnalysisData] = useState(null);
  
  // Comparison states
  const [compareQuery, setCompareQuery] = useState("");
  const [compareData, setCompareData] = useState(null);
  
  // Backtest states
  const [backtestQuery, setBacktestQuery] = useState("");
  const [backtestData, setBacktestData] = useState(null);
  
  // Learn states
  const [learnQuery, setLearnQuery] = useState("");
  const [learnData, setLearnData] = useState(null);
  
  // UI states
  const [newsExpanded, setNewsExpanded] = useState(false);

  const fetchAnalysis = async (searchQuery) => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    setError(null);
    setAnalysisData(null);
    try {
      const res = await fetch(`/api/analyze?query=${encodeURIComponent(searchQuery)}`);
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.error || "Failed to fetch stock analysis");
      }
      setAnalysisData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchComparison = async (searchQuery) => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    setError(null);
    setCompareData(null);
    try {
      const res = await fetch(`/api/compare?query=${encodeURIComponent(searchQuery)}`);
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.error || "Failed to run comparison");
      }
      setCompareData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchBacktest = async (searchQuery) => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    setError(null);
    setBacktestData(null);
    try {
      const res = await fetch(`/api/backtest?query=${encodeURIComponent(searchQuery)}`);
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.error || "Failed to run backtest");
      }
      setBacktestData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchLearn = async (searchQuery) => {
    if (!searchQuery.trim()) return;
    setLoading(true);
    setError(null);
    setLearnData(null);
    try {
      const res = await fetch(`/api/learn?query=${encodeURIComponent(searchQuery)}`);
      const json = await res.json();
      if (!res.ok) {
        throw new Error(json.error || "Failed to consult glossary");
      }
      setLearnData(json);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Trigger search on enter key
  const handleKeyDown = (e, callback, val) => {
    if (e.key === 'Enter') {
      callback(val);
    }
  };

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      {/* Sidebar navigation */}
      <Sidebar activeMode={activeMode} setActiveMode={(mode) => {
        setActiveMode(mode);
        setError(null);
        setLoading(false);
        setAnalysisData(null);
        setCompareData(null);
        setBacktestData(null);
        setLearnData(null);
      }} />

      {/* Main Content Area */}
      <div style={{ flex: 1, padding: '30px 40px', overflowY: 'auto' }}>
        <Header />



        {/* Global Error Banner */}
        {error && (
          <div style={{
            padding: '16px 20px',
            backgroundColor: 'rgba(239, 68, 68, 0.08)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: '8px',
            color: '#FECACA',
            marginBottom: '24px',
            fontSize: '14px',
            fontWeight: 500
          }}>
            {error}
          </div>
        )}

        {/* MODE 1: STOCK ANALYSIS & CHAT */}
        {activeMode === "analysis" && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            
            {/* Centered Search block */}
            <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
              <div style={{ width: '100%', maxWidth: '650px', display: 'flex', flexDirection: 'column', gap: '16px', textAlign: 'center' }}>
                <h3 style={{ fontSize: '18px', fontWeight: 600, color: '#FFFFFF' }}>
                  🔍 Search Stock or Ask a Question
                </h3>
                <div>
                  <p style={{ fontSize: '12px', color: 'var(--text-secondary)', textAlign: 'left', marginBottom: '6px', fontWeight: 500 }}>
                    Type your query below:
                  </p>
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyDown={(e) => handleKeyDown(e, fetchAnalysis, query)}
                    placeholder="e.g. Should I buy TCS? or analyze Reliance, or what is the news on Wipro?"
                    className="search-input"
                  />
                </div>
                
                {/* Suggestions row */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                  <button onClick={() => { setQuery("Should I buy TCS?"); fetchAnalysis("Should I buy TCS?"); }} className="btn-pill">
                    Should I buy TCS?
                  </button>
                  <button onClick={() => { setQuery("Should I sell Reliance?"); fetchAnalysis("Should I sell Reliance?"); }} className="btn-pill">
                    Should I sell Reliance?
                  </button>
                  <button onClick={() => { setQuery("News about Wipro"); fetchAnalysis("News about Wipro"); }} className="btn-pill">
                    News about Wipro
                  </button>
                </div>
                
                {/* Inline loading progress */}
                {loading && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '16px', alignItems: 'flex-start', textAlign: 'left' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#E2E8F0', fontSize: '13px', fontWeight: 500 }}>
                      <div className="spinner-mini"></div>
                      <span>{LOADING_STEPS[loadingStep]}</span>
                    </div>
                    <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontStyle: 'italic', paddingLeft: '28px' }}>
                      Please wait a few seconds while we synthesize real-time market data...
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Analysis Dashboard Grid */}
            {analysisData && !loading && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                <div style={{
                  padding: '16px 20px',
                  backgroundColor: 'rgba(16, 185, 129, 0.08)',
                  border: '1px solid rgba(16, 185, 129, 0.2)',
                  borderRadius: '8px',
                  color: '#A7F3D0',
                  fontSize: '15px',
                  fontWeight: 600
                }}>
                  📌 Target Stock: {analysisData.company_name} ({analysisData.ticker}) | Price: ₹{analysisData.current_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                </div>

                {/* Verdict Card */}
                <div className="glass-card" style={{ 
                  borderLeft: '4px solid', 
                  borderColor: analysisData.ml_result.Recommendation === 'BUY' ? 'var(--primary-emerald)' : analysisData.ml_result.Recommendation === 'SELL' ? 'var(--danger-red)' : 'var(--warning-amber)'
                }}>
                  <h2 style={{ fontSize: '20px', fontWeight: 700, marginBottom: '8px' }}>
                    Verdict: {analysisData.ml_result.Recommendation} ({(analysisData.ml_result.Confidence * 100).toFixed(1)}% Confidence)
                  </h2>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '14px', lineHeight: '1.5' }}>
                    {analysisData.ml_result.Recommendation === 'BUY' 
                      ? "Strong bullish factors align across technicals, FII/DII purchases, and sentiment triggers. Buying is recommended."
                      : analysisData.ml_result.Recommendation === 'SELL' 
                      ? "Bearish momentum indicators and FII capital outflows suggest immediate downswings. Selling or avoiding new positions is advised."
                      : `Signal chance (${(analysisData.ml_result.Confidence * 100).toFixed(1)}%) is within neutral parameters. Defaulting to HOLD.`
                    }
                  </p>
                  
                  {/* Probability channels */}
                  <div style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                        <span>📈 BUY Chance</span>
                        <span>{(analysisData.ml_result.Breakdown.BUY * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ width: '100%', height: '6px', backgroundColor: '#1E293B', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${analysisData.ml_result.Breakdown.BUY * 100}%`, height: '100%', backgroundColor: 'var(--primary-emerald)' }}></div>
                      </div>
                    </div>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                        <span>⚖️ HOLD Chance</span>
                        <span>{(analysisData.ml_result.Breakdown.HOLD * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ width: '100%', height: '6px', backgroundColor: '#1E293B', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${analysisData.ml_result.Breakdown.HOLD * 100}%`, height: '100%', backgroundColor: 'var(--warning-amber)' }}></div>
                      </div>
                    </div>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                        <span>📉 SELL Chance</span>
                        <span>{(analysisData.ml_result.Breakdown.SELL * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ width: '100%', height: '6px', backgroundColor: '#1E293B', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${analysisData.ml_result.Breakdown.SELL * 100}%`, height: '100%', backgroundColor: 'var(--danger-red)' }}></div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Row 1: Signals Summary & Chart */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '24px', alignItems: 'start' }}>
                  <SignalsSummary 
                    technicals={analysisData.technical_indicators}
                    sentiment={analysisData.sentiment}
                    institutional={analysisData.institutional_flow}
                  />
                  <ChartView 
                    data={analysisData.historical_prices}
                    companyName={analysisData.company_name}
                  />
                </div>

                {/* Row 2: Technical Details & Fundamentals Side-by-Side */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', alignItems: 'start' }}>
                  <TechnicalDetails 
                    technicals={analysisData.technical_indicators}
                    currentPrice={analysisData.current_price}
                  />
                  <Fundamentals data={analysisData.fundamentals} />
                </div>

                {/* Explanation (Full width) */}
                <ExplanationGuide 
                  modelAnalysis={analysisData.model_analysis}
                  beginnerExplanation={analysisData.beginner_explanation}
                />

                {/* News List (Full width & Collapsible dropdown) */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div 
                    onClick={() => setNewsExpanded(!newsExpanded)} 
                    style={{ 
                      display: 'flex', 
                      justifyContent: 'space-between', 
                      alignItems: 'center', 
                      cursor: 'pointer',
                      userSelect: 'none'
                    }}
                  >
                    <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                      📰 Latest News & Sentiment Sources
                    </h3>
                    <span style={{ fontSize: '16px', color: '#94A3B8' }}>
                      {newsExpanded ? '▲' : '▼'}
                    </span>
                  </div>
                  
                  {newsExpanded && (
                    <div style={{ marginTop: '12px' }}>
                      {analysisData.news_headlines.length === 0 ? (
                        <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>No recent news articles found.</p>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                          {analysisData.news_headlines.map((news, idx) => (
                            <div key={idx} style={{ 
                              paddingBottom: '12px', 
                              borderBottom: idx < analysisData.news_headlines.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none' 
                            }}>
                              <a 
                                href={news.url} 
                                target="_blank" 
                                rel="noreferrer"
                                style={{ 
                                  color: '#38BDF8', 
                                  fontSize: '14px', 
                                  fontWeight: 600, 
                                  textDecoration: 'none',
                                  display: 'block',
                                  marginBottom: '4px'
                                }}
                              >
                                {idx + 1}. {news.title}
                              </a>
                              <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Source: {news.source}</span>
                              <p style={{ fontSize: '12px', color: '#94A3B8', marginTop: '6px', fontStyle: 'italic', lineHeight: '1.4' }}>
                                "{news.description}"
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* MODE 2: SIDE-BY-SIDE COMPARISON */}
        {activeMode === "compare" && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div style={{ width: '100%', maxWidth: '650px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: 600, color: '#FFFFFF', textAlign: 'center' }}>
                📊 Compare Multiple Stocks Side-by-Side
              </h3>
              <div>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 500 }}>
                  Ask any comparison query (e.g. 'Compare TCS and Wipro' or 'Which is best to buy TCS or Wipro') or enter stock names:
                </p>
                <input
                  type="text"
                  value={compareQuery}
                  onChange={(e) => setCompareQuery(e.target.value)}
                  onKeyDown={(e) => handleKeyDown(e, fetchComparison, compareQuery)}
                  placeholder="e.g. Compare TCS and Wipro"
                  className="search-input"
                />
              </div>
              
              {/* Inline loading progress */}
              {loading && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '16px', alignItems: 'flex-start', textAlign: 'left' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#E2E8F0', fontSize: '13px', fontWeight: 500 }}>
                    <div className="spinner-mini"></div>
                    <span>{COMPARE_LOADING_STEPS[loadingStep % COMPARE_LOADING_STEPS.length]}</span>
                  </div>
                  <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontStyle: 'italic', paddingLeft: '28px' }}>
                    Please wait a few seconds while we synthesize real-time market data...
                  </span>
                </div>
              )}
            </div>

            {compareData && !loading && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                {/* Comparison Grid Table */}
                <div className="glass-card">
                  <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF', marginBottom: '16px' }}>
                    ⚔️ Side-by-Side Stock Comparison
                  </h3>
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '14px' }}>
                      <thead>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', textAlign: 'left' }}>
                          <th style={{ padding: '12px 10px', color: 'var(--text-secondary)', fontWeight: 500 }}>Metric</th>
                          {compareData.stocks.map((s, idx) => (
                            <th key={idx} style={{ padding: '12px 10px', color: '#FFFFFF', fontWeight: 600 }}>
                              {s.company_name} ({s.ticker})
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 10px', fontWeight: 600, color: 'var(--text-secondary)' }}>Signal Verdict</td>
                          {compareData.stocks.map((s, idx) => (
                            <td key={idx} style={{ padding: '12px 10px', fontWeight: 700, color: s.recommendation === 'BUY' ? 'var(--primary-emerald)' : s.recommendation === 'SELL' ? 'var(--danger-red)' : 'var(--warning-amber)' }}>
                              {s.recommendation} ({s.confidence.toFixed(1)}%)
                            </td>
                          ))}
                        </tr>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 10px', color: 'var(--text-secondary)' }}>Current Price</td>
                          {compareData.stocks.map((s, idx) => (
                            <td key={idx} style={{ padding: '12px 10px', fontWeight: 600 }}>
                              ₹{s.current_price.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                            </td>
                          ))}
                        </tr>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 10px', color: 'var(--text-secondary)' }}>P/E Ratio</td>
                          {compareData.stocks.map((s, idx) => (
                            <td key={idx} style={{ padding: '12px 10px' }}>
                              {s.fundamentals.PE_Ratio ? s.fundamentals.PE_Ratio.toFixed(2) : "N/A"}
                            </td>
                          ))}
                        </tr>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 10px', color: 'var(--text-secondary)' }}>P/B Ratio</td>
                          {compareData.stocks.map((s, idx) => (
                            <td key={idx} style={{ padding: '12px 10px' }}>
                              {s.fundamentals.PB_Ratio ? s.fundamentals.PB_Ratio.toFixed(2) : "N/A"}
                            </td>
                          ))}
                        </tr>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 10px', color: 'var(--text-secondary)' }}>Debt to Equity</td>
                          {compareData.stocks.map((s, idx) => (
                            <td key={idx} style={{ padding: '12px 10px' }}>
                              {s.fundamentals.Debt_to_Equity ? `${s.fundamentals.Debt_to_Equity.toFixed(1)}%` : "N/A"}
                            </td>
                          ))}
                        </tr>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 10px', color: 'var(--text-secondary)' }}>Return on Equity</td>
                          {compareData.stocks.map((s, idx) => (
                            <td key={idx} style={{ padding: '12px 10px' }}>
                              {s.fundamentals.ROE ? `${s.fundamentals.ROE.toFixed(1)}%` : "N/A"}
                            </td>
                          ))}
                        </tr>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 10px', color: 'var(--text-secondary)' }}>Sentiment Score</td>
                          {compareData.stocks.map((s, idx) => (
                            <td key={idx} style={{ padding: '12px 10px' }}>
                              {s.sentiment_score.toFixed(2)}
                            </td>
                          ))}
                        </tr>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 10px', color: 'var(--text-secondary)' }}>Above 50d MA?</td>
                          {compareData.stocks.map((s, idx) => (
                            <td key={idx} style={{ padding: '12px 10px', color: s.tech_above_50 ? '#34D399' : '#F87171' }}>
                              {s.tech_above_50 ? "Yes" : "No"}
                            </td>
                          ))}
                        </tr>
                        <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                          <td style={{ padding: '12px 10px', color: 'var(--text-secondary)' }}>Above 200d MA?</td>
                          {compareData.stocks.map((s, idx) => (
                            <td key={idx} style={{ padding: '12px 10px', color: s.tech_above_200 ? '#34D399' : '#F87171' }}>
                              {s.tech_above_200 ? "Yes" : "No"}
                            </td>
                          ))}
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* AI Summary report */}
                <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  <div>
                    <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF', marginBottom: '12px' }}>
                      🧠 AI Side-by-Side Comparison Summary
                    </h3>
                    <div style={{ 
                      fontSize: '14px', 
                      color: '#E2E8F0', 
                      lineHeight: '1.6', 
                      whiteSpace: 'pre-line' 
                    }}>
                      {compareData.ai_summary}
                    </div>
                  </div>
                  
                  {/* Follow-up button */}
                  <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '16px', display: 'flex', justifyContent: 'flex-start' }}>
                    <button
                      onClick={() => {
                        const tickerNames = compareData.stocks.map(s => s.ticker.split('.')[0]).join(' and ');
                        setQuery(`Based on the comparison of ${tickerNames}, what are the key risks?`);
                        setActiveMode("analysis");
                      }}
                      className="btn-pill"
                      style={{ width: 'auto', padding: '10px 20px', backgroundColor: 'var(--primary-emerald)', borderColor: 'var(--primary-emerald)' }}
                    >
                      💬 Ask a Follow-up Question in Chat
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* MODE 3: STRATEGY BACKTESTING */}
        {activeMode === "backtest" && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div style={{ width: '100%', maxWidth: '650px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: 600, color: '#FFFFFF', textAlign: 'center' }}>
                📈 Run 1-Year Trade Simulation Backtest
              </h3>
              
              {/* Educational Explanation Box */}
              <div style={{
                padding: '16px 20px',
                backgroundColor: 'rgba(56, 189, 248, 0.06)',
                border: '1px solid rgba(56, 189, 248, 0.15)',
                borderRadius: '10px',
                fontSize: '13px',
                lineHeight: '1.6',
                color: '#94A3B8',
                textAlign: 'left'
              }}>
                💡 <strong>What is Backtesting?</strong> It runs a 1-year historical simulation of our machine learning trading model on the selected stock. 
                Starting with <strong>Rs. 100,000</strong>, it automatically enters a trade (BUY) when predictions are above <strong>55% confidence</strong> and exits 
                the trade after <strong>5 days</strong> (or upon a SELL signal). This helps you evaluate if the model successfully avoids downswings 
                and beats a benchmark Buy & Hold strategy.
              </div>

              <div>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 500 }}>
                  Enter stock name or symbol to backtest:
                </p>
                <input
                  type="text"
                  value={backtestQuery}
                  onChange={(e) => setBacktestQuery(e.target.value)}
                  onKeyDown={(e) => handleKeyDown(e, fetchBacktest, backtestQuery)}
                  placeholder="e.g. TCS"
                  className="search-input"
                />
              </div>
              
              {/* Inline loading progress */}
              {loading && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '16px', alignItems: 'flex-start', textAlign: 'left' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', color: '#E2E8F0', fontSize: '13px', fontWeight: 500 }}>
                    <div className="spinner-mini"></div>
                    <span>{BACKTEST_LOADING_STEPS[loadingStep % BACKTEST_LOADING_STEPS.length]}</span>
                  </div>
                  <span style={{ fontSize: '11px', color: 'var(--text-secondary)', fontStyle: 'italic', paddingLeft: '28px' }}>
                    Please wait a few seconds while we run the simulation matrix...
                  </span>
                </div>
              )}
            </div>

            {backtestData && !loading && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
                <h3 style={{ fontSize: '18px', fontWeight: 600, color: '#FFFFFF' }}>
                  📈 1-Year Backtest Report for {backtestData.company_name} ({backtestData.ticker})
                </h3>

                {/* Metrics Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                  <div className="glass-card">
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Initial Capital</div>
                    <div style={{ fontSize: '24px', fontWeight: 700, color: '#FFFFFF', marginTop: '6px' }}>
                      ₹{backtestData.initial_capital.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </div>
                  </div>
                  <div className="glass-card">
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>AI Strategy Return</div>
                    <div style={{ 
                      fontSize: '24px', 
                      fontWeight: 700, 
                      color: backtestData.strategy_return_pct >= 0 ? 'var(--primary-emerald)' : 'var(--danger-red)', 
                      marginTop: '6px' 
                    }}>
                      {backtestData.strategy_return_pct >= 0 ? '+' : ''}{backtestData.strategy_return_pct.toFixed(2)}%
                    </div>
                  </div>
                  <div className="glass-card">
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Total Simulated Trades</div>
                    <div style={{ fontSize: '24px', fontWeight: 700, color: '#FFFFFF', marginTop: '6px' }}>
                      {backtestData.total_trades}
                    </div>
                  </div>
                  <div className="glass-card">
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Final Portfolio Value</div>
                    <div style={{ fontSize: '24px', fontWeight: 700, color: '#FFFFFF', marginTop: '6px' }}>
                      ₹{backtestData.final_value.toLocaleString('en-IN', { minimumFractionDigits: 2 })}
                    </div>
                  </div>
                  <div className="glass-card">
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Benchmark Return (Buy & Hold)</div>
                    <div style={{ 
                      fontSize: '24px', 
                      fontWeight: 700, 
                      color: backtestData.benchmark_return_pct >= 0 ? 'var(--primary-emerald)' : 'var(--danger-red)', 
                      marginTop: '6px' 
                    }}>
                      {backtestData.benchmark_return_pct >= 0 ? '+' : ''}{backtestData.benchmark_return_pct.toFixed(2)}%
                    </div>
                  </div>
                  <div className="glass-card">
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Winning Trades %</div>
                    <div style={{ fontSize: '24px', fontWeight: 700, color: '#FFFFFF', marginTop: '6px' }}>
                      {backtestData.win_rate_pct.toFixed(1)}%
                    </div>
                  </div>
                </div>

                {/* Winner Card */}
                {backtestData.strategy_return_pct > backtestData.benchmark_return_pct ? (
                  <div style={{
                    padding: '16px 20px',
                    backgroundColor: 'rgba(16, 185, 129, 0.08)',
                    border: '1px solid rgba(16, 185, 129, 0.2)',
                    borderRadius: '8px',
                    color: '#A7F3D0',
                    fontSize: '14px',
                    lineHeight: '1.5'
                  }}>
                    🏆 **AI Strategy outperformed the stock's Buy & Hold return by {(backtestData.strategy_return_pct - backtestData.benchmark_return_pct).toFixed(2)}%**! The multi-signal model successfully avoided downswings and entered on key momentum.
                  </div>
                ) : (
                  <div style={{
                    padding: '16px 20px',
                    backgroundColor: 'rgba(245, 158, 11, 0.08)',
                    border: '1px solid rgba(245, 158, 11, 0.2)',
                    borderRadius: '8px',
                    color: '#FDE68A',
                    fontSize: '14px',
                    lineHeight: '1.5'
                  }}>
                    ℹ️ **Benchmark Buy & Hold outperformed the AI strategy by {(backtestData.benchmark_return_pct - backtestData.strategy_return_pct).toFixed(2)}%**. This stock had a persistent, strong uptrend where active trading signals were less effective than simply holding.
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* MODE 4: LEARN BASICS */}
        {activeMode === "learn" && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
            <div style={{ width: '100%', maxWidth: '650px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '16px' }}>
              <h3 style={{ fontSize: '18px', fontWeight: 600, color: '#FFFFFF', textAlign: 'center' }}>
                📚 AI Financial Glossary & Learning Center
              </h3>
              <div>
                <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '6px', fontWeight: 500 }}>
                  Ask any question about financial markets, terms, or how indicators work:
                </p>
                <input
                  type="text"
                  value={learnQuery}
                  onChange={(e) => setQuery(e.target.value) || setLearnQuery(e.target.value)}
                  onKeyDown={(e) => handleKeyDown(e, fetchLearn, learnQuery)}
                  placeholder="e.g. What is RSI? or How does debt-to-equity ratio affect stock safety?"
                  className="search-input"
                />
              </div>
            </div>

            {learnData && !loading && (
              <div className="glass-card">
                <h3 style={{ fontSize: '16px', fontWeight: 600, color: '#FFFFFF', marginBottom: '16px' }}>
                  🧠 AI Financial Guide (For Beginners)
                </h3>
                <div style={{ 
                  fontSize: '14px', 
                  color: '#E2E8F0', 
                  lineHeight: '1.6', 
                  whiteSpace: 'pre-line' 
                }}>
                  {learnData.explanation}
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
