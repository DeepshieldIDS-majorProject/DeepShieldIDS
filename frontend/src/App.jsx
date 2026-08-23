import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const attackInfo = {
  BENIGN: {
    icon: "✓",
    title: "BENIGN",
    subtitle: "Normal Network Traffic",
  },
  BruteForce: {
    icon: "⚠",
    title: "BRUTE FORCE",
    subtitle: "Credential Attack",
  },
  DoS: {
    icon: "ϟ",
    title: "DoS",
    subtitle: "Service Disruption",
  },
  Probe: {
    icon: "⌁",
    title: "PROBE",
    subtitle: "Network Reconnaissance",
  },
  WebAttack: {
    icon: "◆",
    title: "WEB ATTACK",
    subtitle: "Web-Based Attack",
  },
};

function App() {
  const [features, setFeatures] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);

  useEffect(() => {
    checkAPI();
    const timer = setInterval(checkAPI, 5000);

    return () => clearInterval(timer);
  }, []);

  const checkAPI = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);

      if (response.ok) {
        setApiOnline(true);
      } else {
        setApiOnline(false);
      }
    } catch {
      setApiOnline(false);
    }
  };

  const testSample = () => {
    const sample = Array(78).fill(0);
    setFeatures(sample.join(", "));
    setResult(null);
  };

  const clearInput = () => {
    setFeatures("");
    setResult(null);
  };

  const detectAttack = async () => {
    try {
      const values = features
        .split(",")
        .map((value) => Number(value.trim()));

      if (values.length !== 78) {
        alert(
          `Exactly 78 numerical features are required.\n\nCurrent features: ${values.length}`
        );
        return;
      }

      if (values.some((value) => Number.isNaN(value))) {
        alert("Please enter only numerical values.");
        return;
      }

      setLoading(true);
      setResult(null);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          features: values,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      setResult(data);
    } catch (error) {
      alert(error.message);
    } finally {
      setLoading(false);
    }
  };

  const featureCount = features.trim()
    ? features.split(",").filter((x) => x.trim() !== "").length
    : 0;

  const prediction = result?.prediction;
  const predictionInfo = prediction
    ? attackInfo[prediction]
    : attackInfo.BENIGN;

  return (
    <div className="app-shell">
      <div className="background-grid"></div>
      <div className="glow glow-one"></div>
      <div className="glow glow-two"></div>

      {/* TOP NAVBAR */}
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">
            <span>DS</span>
          </div>

          <div>
            <div className="brand-name">DeepShield<span>IDS</span></div>
            <div className="brand-subtitle">
              NETWORK SECURITY PLATFORM
            </div>
          </div>
        </div>

        <div className="system-status">
          <span className={`status-dot ${apiOnline ? "active" : ""}`}></span>

          <div>
            <small>SYSTEM STATUS</small>
            <strong>{apiOnline ? "ONLINE" : "OFFLINE"}</strong>
          </div>
        </div>
      </header>

      <main className="dashboard">

        {/* HERO */}
        <section className="hero-section">
          <div className="hero-left">
            <div className="eyebrow">
              <span></span>
              AI-POWERED THREAT INTELLIGENCE
            </div>

            <h1>
              Network Traffic
              <br />
              <span>Threat Detection</span>
            </h1>

            <p>
              Analyze CIC-IDS2017 network traffic using the trained
              HybridIDSNet intrusion detection model.
            </p>

            <div className="model-badge">
              <div className="model-symbol">AI</div>

              <div>
                <small>ACTIVE MODEL</small>
                <strong>HybridIDSNet</strong>
              </div>

              <div className="model-divider"></div>

              <div>
                <small>CLASSIFICATION</small>
                <strong>5 CLASSES</strong>
              </div>
            </div>
          </div>

          <div className="hero-shield">
            <div className="shield-ring ring-one"></div>
            <div className="shield-ring ring-two"></div>

            <div className="shield">
              <div className="shield-top">DS</div>
              <div className="shield-text">SECURE</div>
            </div>

            <div className="scan-line"></div>
          </div>
        </section>

        {/* METRICS */}
        <section className="metrics">

          <div className="metric-card">
            <div className="metric-number">01</div>
            <div className="metric-content">
              <span>FEATURES</span>
              <strong>78</strong>
              <small>CIC-IDS2017</small>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-number">02</div>
            <div className="metric-content">
              <span>CLASSES</span>
              <strong>05</strong>
              <small>Attack Categories</small>
            </div>
          </div>

          <div className="metric-card highlight">
            <div className="metric-number">03</div>
            <div className="metric-content">
              <span>HELD-OUT ACCURACY</span>
              <strong>97.26%</strong>
              <small>Evaluation Result</small>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-number">04</div>
            <div className="metric-content">
              <span>ROC-AUC</span>
              <strong>99.69%</strong>
              <small>Evaluation Score</small>
            </div>
          </div>

        </section>

        {/* MAIN GRID */}
        <section className="main-grid">

          {/* INPUT PANEL */}
          <div className="panel input-panel">

            <div className="panel-header">
              <div>
                <div className="section-number">01</div>
                <h2>Traffic Analysis</h2>
                <p>Enter CIC-IDS2017 feature values</p>
              </div>

              <div className="feature-counter">
                <strong>{featureCount}</strong>
                <span>/ 78</span>
              </div>
            </div>

            <div className="input-status">
              <span className={featureCount === 78 ? "complete" : ""}></span>

              {featureCount === 78
                ? "78 FEATURES READY FOR ANALYSIS"
                : "78 NUMERICAL FEATURES REQUIRED"}
            </div>

            <textarea
              className="feature-input"
              value={features}
              onChange={(e) => setFeatures(e.target.value)}
              placeholder="Enter feature_1, feature_2, feature_3, ... feature_78"
            />

            <div className="format-box">
              <span>INPUT FORMAT</span>
              <code>
                feature_1, feature_2, ... feature_78
              </code>
            </div>

            <div className="action-buttons">
              <button className="sample-btn" onClick={testSample}>
                <span>◈</span>
                TEST SAMPLE
              </button>

              <button className="clear-btn" onClick={clearInput}>
                CLEAR
              </button>

              <button
                className="detect-btn"
                onClick={detectAttack}
                disabled={loading}
              >
                {loading ? "ANALYZING..." : "DETECT ATTACK"}
                <span>→</span>
              </button>
            </div>
          </div>

          {/* RESULT PANEL */}
          <div className="panel result-panel">

            <div className="panel-header">
              <div>
                <div className="section-number">02</div>
                <h2>Detection Result</h2>
                <p>AI classification output</p>
              </div>

              <div className="live-label">
                <span></span>
                LIVE
              </div>
            </div>

            {!result ? (
              <div className="empty-result">
                <div className="result-orb">
                  <span>DS</span>
                </div>

                <h3>Awaiting Traffic Analysis</h3>

                <p>
                  Submit 78 network traffic features
                  <br />
                  to begin threat detection.
                </p>

                <div className="ready-status">
                  <span></span>
                  SYSTEM READY
                </div>
              </div>
            ) : (
              <div className="result-content">

                <div
                  className={`prediction-banner ${
                    prediction === "BENIGN" ? "safe" : "danger"
                  }`}
                >
                  <div className="prediction-icon">
                    {predictionInfo.icon}
                  </div>

                  <div>
                    <small>THREAT CLASSIFICATION</small>
                    <strong>{predictionInfo.title}</strong>
                    <span>{predictionInfo.subtitle}</span>
                  </div>
                </div>

                <div className="confidence-box">
                  <div className="confidence-heading">
                    <span>MODEL CONFIDENCE</span>
                    <strong>
                      {(result.confidence * 100).toFixed(2)}%
                    </strong>
                  </div>

                  <div className="confidence-track">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${result.confidence * 100}%`,
                      }}
                    ></div>
                  </div>
                </div>

                {result.probabilities && (
                  <div className="probabilities">
                    <div className="probability-title">
                      CLASS PROBABILITIES
                    </div>

                    {Object.entries(result.probabilities).map(
                      ([name, probability]) => (
                        <div className="probability-item" key={name}>
                          <div className="probability-head">
                            <span>{name}</span>
                            <strong>
                              {(probability * 100).toFixed(2)}%
                            </strong>
                          </div>

                          <div className="probability-track">
                            <div
                              className={`probability-fill ${
                                name === prediction
                                  ? "selected"
                                  : ""
                              }`}
                              style={{
                                width: `${Math.max(
                                  probability * 100,
                                  0.5
                                )}%`,
                              }}
                            ></div>
                          </div>
                        </div>
                      )
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        {/* ATTACK CLASSES */}
        <section className="panel classes-panel">

          <div className="panel-header classes-header">
            <div>
              <div className="section-number">03</div>
              <h2>Supported Attack Classes</h2>
              <p>HybridIDSNet classification categories</p>
            </div>
          </div>

          <div className="attack-grid">

            <div className="attack-card benign">
              <div className="attack-icon">✓</div>
              <div>
                <strong>BENIGN</strong>
                <span>Normal Traffic</span>
              </div>
            </div>

            <div className="attack-card">
              <div className="attack-icon">⚠</div>
              <div>
                <strong>BRUTE FORCE</strong>
                <span>Credential Attack</span>
              </div>
            </div>

            <div className="attack-card">
              <div className="attack-icon">ϟ</div>
              <div>
                <strong>DoS</strong>
                <span>Service Disruption</span>
              </div>
            </div>

            <div className="attack-card">
              <div className="attack-icon">⌁</div>
              <div>
                <strong>PROBE</strong>
                <span>Reconnaissance</span>
              </div>
            </div>

            <div className="attack-card">
              <div className="attack-icon">◆</div>
              <div>
                <strong>WEB ATTACK</strong>
                <span>Web-Based Attack</span>
              </div>
            </div>

          </div>
        </section>
      </main>

      <footer className="footer">
        <div>
          <strong>DeepShieldIDS</strong>
          <span>Hybrid AI-Based Network Intrusion Detection System</span>
        </div>

        <div className="footer-meta">
          MODEL: HybridIDSNet
          <i>•</i>
          CIC-IDS2017
          <i>•</i>
          v1.0
        </div>
      </footer>
    </div>
  );
}

export default App;