import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const CLASSES = [
  {
    name: "BENIGN",
    icon: "✓",
    description: "Normal network traffic",
    type: "benign",
  },
  {
    name: "BruteForce",
    icon: "⚠",
    description: "Credential attack",
    type: "danger",
  },
  {
    name: "DoS",
    icon: "ϟ",
    description: "Service disruption",
    type: "danger",
  },
  {
    name: "Probe",
    icon: "⌁",
    description: "Reconnaissance",
    type: "warning",
  },
  {
    name: "WebAttack",
    icon: "◆",
    description: "Web-based attack",
    type: "danger",
  },
];

function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [features, setFeatures] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState("Checking");
  const [error, setError] = useState("");

  const [autoRefresh, setAutoRefresh] = useState(true);
  const [notifications, setNotifications] = useState(true);

  // --------------------------------------------------
  // API HEALTH
  // --------------------------------------------------

  useEffect(() => {
    checkAPI();

    if (!autoRefresh) return;

    const interval = setInterval(checkAPI, 10000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const checkAPI = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);

      if (response.ok) {
        setApiStatus("Online");
      } else {
        setApiStatus("Offline");
      }
    } catch {
      setApiStatus("Offline");
    }
  };

  // --------------------------------------------------
  // NAVIGATION
  // --------------------------------------------------

  const navigate = (page) => {
    setActivePage(page);
    setError("");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  // --------------------------------------------------
  // TEST SAMPLE
  // --------------------------------------------------

  const testSample = () => {
    const sample = Array(78).fill(0);

    setFeatures(sample.join(", "));
    setResult(null);
    setError("");

    setActivePage("traffic");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  // --------------------------------------------------
  // CLEAR
  // --------------------------------------------------

  const clearData = () => {
    setFeatures("");
    setResult(null);
    setError("");
  };

  // --------------------------------------------------
  // DETECT ATTACK
  // --------------------------------------------------

  const detectAttack = async () => {
    try {
      setError("");

      if (!features.trim()) {
        setError("Please enter exactly 78 numerical features.");
        return;
      }

      const values = features
        .split(",")
        .map((value) => value.trim())
        .filter((value) => value !== "")
        .map(Number);

      if (values.length !== 78) {
        setError(
          `Exactly 78 numerical features are required. You entered ${values.length}.`
        );
        return;
      }

      if (values.some((value) => !Number.isFinite(value))) {
        setError("Only valid numerical values are allowed.");
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
        throw new Error(data.detail || "Prediction failed.");
      }

      setResult(data);

      setActivePage("threat");

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to DeepShieldIDS API. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  };

  // --------------------------------------------------
  // PAGE TITLE
  // --------------------------------------------------

  const getPageTitle = () => {
    switch (activePage) {
      case "dashboard":
        return "Network Threat Dashboard";
      case "traffic":
        return "Traffic Analysis";
      case "threat":
        return "Threat Detection";
      case "performance":
        return "Model Performance";
      case "settings":
        return "System Settings";
      case "documentation":
        return "Documentation";
      default:
        return "DeepShieldIDS";
    }
  };

  // --------------------------------------------------
  // DASHBOARD
  // --------------------------------------------------

  const Dashboard = () => (
    <>
      <section className="welcome-section">
        <div>
          <div className="eyebrow">SECURITY MONITORING SYSTEM</div>

          <h1>
            Network Threat <span>Dashboard</span>
          </h1>

          <p>
            Monitor and analyze CIC-IDS2017 network traffic using the trained
            HybridIDSNet intrusion detection model.
          </p>
        </div>

        <div className="system-status-large">
          <span className="status-pulse"></span>

          <div>
            <strong>
              {apiStatus === "Online"
                ? "System Online"
                : "System Offline"}
            </strong>

            <small>
              {apiStatus === "Online"
                ? "FastAPI connected"
                : "API unavailable"}
            </small>
          </div>
        </div>
      </section>

      <section className="metrics-grid">
        <MetricCard
          number="01"
          title="MODEL ACCURACY"
          value="97.26%"
          description="Held-out evaluation"
        />

        <MetricCard
          number="02"
          title="ROC-AUC SCORE"
          value="99.69%"
          description="Evaluation performance"
        />

        <MetricCard
          number="03"
          title="FEATURES"
          value="78"
          description="CIC-IDS2017 features"
        />

        <MetricCard
          number="04"
          title="ATTACK CLASSES"
          value="05"
          description="Multi-class detection"
        />
      </section>

      <section className="dashboard-grid">
        <ActionCard
          number="01"
          icon="◈"
          iconClass="blue-icon"
          title="Traffic Analysis"
          description="Enter 78 CIC-IDS2017 numerical features and send them to HybridIDSNet for classification."
          button="OPEN TRAFFIC ANALYSIS"
          onClick={() => navigate("traffic")}
        />

        <ActionCard
          number="02"
          icon="⚠"
          iconClass="red-icon"
          title="Threat Detection"
          description="View the latest prediction, confidence score and class probability distribution."
          button="VIEW THREAT RESULT"
          onClick={() => navigate("threat")}
        />

        <ActionCard
          number="03"
          icon="▥"
          iconClass="green-icon"
          title="Model Performance"
          description="Review accuracy, precision, recall, F1-score and ROC-AUC evaluation metrics."
          button="VIEW PERFORMANCE"
          onClick={() => navigate("performance")}
        />
      </section>

      <ArchitectureDiagram />
    </>
  );

  // --------------------------------------------------
  // TRAFFIC ANALYSIS
  // --------------------------------------------------

  const TrafficAnalysis = () => {
    const count = features
      .split(",")
      .map((x) => x.trim())
      .filter((x) => x !== "").length;

    return (
      <section className="page-section">
        <PageIntro
          eyebrow="01 / TRAFFIC ANALYSIS"
          title="Network Traffic Inspection"
          description="Enter the 78 numerical CIC-IDS2017 features required by the HybridIDSNet model."
        />

        <div className="analysis-card">
          <div className="analysis-header">
            <div>
              <div className="card-kicker">TRAFFIC FEATURES</div>
              <h2>Feature Input</h2>
            </div>

            <div
              className={
                count === 78
                  ? "input-status complete"
                  : "input-status"
              }
            >
              ●{" "}
              {count === 78
                ? "78 FEATURES READY"
                : `${count}/78 FEATURES`}
            </div>
          </div>

          <textarea
            className="feature-input"
            value={features}
            onChange={(e) => {
              setFeatures(e.target.value);
              setResult(null);
              setError("");
            }}
            placeholder="feature_1, feature_2, feature_3, ... feature_78"
          />

          <div className="format-row">
            <span>INPUT FORMAT</span>
            <code>feature_1, feature_2, ... feature_78</code>
          </div>

          {error && <div className="error-box">⚠ {error}</div>}

          <div className="button-row">
            <button
              className="secondary-button"
              onClick={testSample}
            >
              ◈ TEST SAMPLE
            </button>

            <button
              className="secondary-button"
              onClick={clearData}
            >
              CLEAR
            </button>

            <button
              className="primary-button"
              onClick={detectAttack}
              disabled={loading}
            >
              {loading ? "ANALYZING..." : "DETECT ATTACK →"}
            </button>
          </div>
        </div>

        <div className="info-grid">
          <InfoBox
            title="INPUT DIMENSION"
            value="78"
            description="CIC-IDS2017 numerical features"
          />

          <InfoBox
            title="MODEL"
            value="HybridIDSNet"
            description="5-class intrusion classification"
          />

          <InfoBox
            title="API STATUS"
            value={apiStatus}
            description="FastAPI prediction service"
            green={apiStatus === "Online"}
          />
        </div>
      </section>
    );
  };

  // --------------------------------------------------
  // THREAT DETECTION
  // --------------------------------------------------

  const ThreatDetection = () => (
    <section className="page-section">
      <PageIntro
        eyebrow="02 / THREAT DETECTION"
        title="AI Classification Result"
        description="HybridIDSNet analyzes network traffic and assigns one of five supported security classes."
      />

      {!result ? (
        <div className="empty-result">
          <div className="empty-icon">DS</div>

          <h2>Awaiting Traffic Analysis</h2>

          <p>
            Submit 78 network traffic features from the Traffic Analysis page
            to begin threat detection.
          </p>

          <button
            className="primary-button empty-action"
            onClick={() => navigate("traffic")}
          >
            GO TO TRAFFIC ANALYSIS →
          </button>
        </div>
      ) : (
        <>
          <div className="result-hero">
            <div className="result-main">
              <span className="result-label">PREDICTION</span>

              <div
                className={
                  result.prediction === "BENIGN"
                    ? "prediction benign-result"
                    : "prediction attack-result"
                }
              >
                {result.prediction === "BENIGN" ? "✓" : "⚠"}

                <strong>{result.prediction}</strong>
              </div>
            </div>

            <div className="confidence-box">
              <span>CONFIDENCE</span>

              <strong>
                {(result.confidence * 100).toFixed(2)}%
              </strong>
            </div>
          </div>

          <div className="probability-card">
            <div className="card-kicker">
              CLASS PROBABILITIES
            </div>

            <h2>Confidence Distribution</h2>

            <div className="probability-list">
              {Object.entries(result.probabilities || {}).map(
                ([name, probability]) => (
                  <div className="probability-item" key={name}>
                    <div className="probability-top">
                      <span>{name}</span>

                      <strong>
                        {(probability * 100).toFixed(2)}%
                      </strong>
                    </div>

                    <div className="probability-track">
                      <div
                        className={
                          name === result.prediction
                            ? "probability-fill active"
                            : "probability-fill"
                        }
                        style={{
                          width: `${Math.max(
                            probability * 100,
                            probability > 0 ? 0.5 : 0
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        </>
      )}

      <div className="classification-card">
        <div className="card-kicker">
          THREAT CLASSIFICATION
        </div>

        <h2>Supported Attack Categories</h2>

        <div className="classification-grid">
          {CLASSES.map((item) => (
            <div
              className={`classification-item ${item.type}`}
              key={item.name}
            >
              <div className="class-icon">{item.icon}</div>

              <div>
                <strong>{item.name}</strong>
                <span>{item.description}</span>
              </div>

              <div className="class-arrow">→</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );

  // --------------------------------------------------
  // MODEL PERFORMANCE
  // --------------------------------------------------

  const ModelPerformance = () => (
    <section className="page-section">
      <PageIntro
        eyebrow="03 / MODEL PERFORMANCE"
        title="HybridIDSNet Evaluation"
        description="Final held-out evaluation results obtained from the trained intrusion detection model."
      />

      <div className="performance-grid">
        <PerformanceCard
          title="ACCURACY"
          value="97.26%"
          description="Held-out evaluation"
        />

        <PerformanceCard
          title="PRECISION"
          value="97.94%"
          description="Weighted precision"
        />

        <PerformanceCard
          title="RECALL"
          value="97.26%"
          description="Weighted recall"
        />

        <PerformanceCard
          title="F1 SCORE"
          value="97.50%"
          description="Weighted F1 score"
        />

        <PerformanceCard
          title="ROC-AUC"
          value="99.69%"
          description="Evaluation score"
          highlight
        />
      </div>

      <div className="evaluation-layout">
        <div className="evaluation-card">
          <div className="card-kicker">
            HELD-OUT EVALUATION
          </div>

          <h2>Model Quality</h2>

          <QualityBar label="Accuracy" value={97.26} />
          <QualityBar label="Precision" value={97.94} />
          <QualityBar label="Recall" value={97.26} />
          <QualityBar label="F1 Score" value={97.5} />
          <QualityBar label="ROC-AUC" value={99.69} />
        </div>

        <div className="evaluation-card">
          <div className="card-kicker">
            CLASSIFICATION
          </div>

          <h2>Detection Classes</h2>

          <div className="class-summary">
            {CLASSES.map((item, index) => (
              <div className="class-summary-row" key={item.name}>
                <span>0{index + 1}</span>
                <strong>{item.name}</strong>
                <em>{item.description}</em>
              </div>
            ))}
          </div>
        </div>
      </div>

      <ArchitectureDiagram />
    </section>
  );

  // --------------------------------------------------
  // SETTINGS
  // --------------------------------------------------

  const Settings = () => (
    <section className="page-section">
      <PageIntro
        eyebrow="SYSTEM / SETTINGS"
        title="System Settings"
        description="Configure DeepShieldIDS dashboard behaviour."
      />

      <div className="settings-grid">
        <div className="settings-card">
          <div className="settings-icon">⚙</div>

          <div>
            <h2>API Auto Refresh</h2>
            <p>
              Automatically check the FastAPI health status every 10 seconds.
            </p>
          </div>

          <button
            className={`toggle ${autoRefresh ? "on" : ""}`}
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            <span></span>
            {autoRefresh ? "ON" : "OFF"}
          </button>
        </div>

        <div className="settings-card">
          <div className="settings-icon">♢</div>

          <div>
            <h2>Notifications</h2>
            <p>
              Show notification status indicators in the dashboard.
            </p>
          </div>

          <button
            className={`toggle ${notifications ? "on" : ""}`}
            onClick={() => setNotifications(!notifications)}
          >
            <span></span>
            {notifications ? "ON" : "OFF"}
          </button>
        </div>

        <div className="settings-card">
          <div className="settings-icon">●</div>

          <div>
            <h2>API Connection</h2>
            <p>
              Current FastAPI service status.
            </p>
          </div>

          <strong
            className={
              apiStatus === "Online"
                ? "settings-online"
                : "settings-offline"
            }
          >
            {apiStatus}
          </strong>
        </div>
      </div>
    </section>
  );

  // --------------------------------------------------
  // DOCUMENTATION
  // --------------------------------------------------

  const Documentation = () => (
    <section className="page-section">
      <PageIntro
        eyebrow="SYSTEM / DOCUMENTATION"
        title="DeepShieldIDS Documentation"
        description="Overview of the HybridIDSNet intrusion detection pipeline and API usage."
      />

      <div className="documentation-grid">
        <div className="documentation-card">
          <div className="card-kicker">SYSTEM OVERVIEW</div>

          <h2>How DeepShieldIDS Works</h2>

          <p>
            DeepShieldIDS processes CIC-IDS2017 network traffic features
            and sends the processed 78-feature vector to HybridIDSNet for
            multi-class intrusion detection.
          </p>

          <div className="doc-points">
            <div>
              <strong>01</strong>
              <span>Input 78 numerical traffic features</span>
            </div>

            <div>
              <strong>02</strong>
              <span>HybridIDSNet processes the traffic</span>
            </div>

            <div>
              <strong>03</strong>
              <span>Five-class threat classification</span>
            </div>

            <div>
              <strong>04</strong>
              <span>Prediction and confidence are returned</span>
            </div>
          </div>
        </div>

        <div className="documentation-card">
          <div className="card-kicker">API</div>

          <h2>Prediction Endpoint</h2>

          <div className="code-box">
            <span>POST</span>
            <code>/predict</code>
          </div>

          <p>
            Send a JSON request containing exactly 78 numerical features.
          </p>

          <div className="code-example">
            {
`{
  "features": [78 numerical values]
}`
            }
          </div>

          <div className="code-box health">
            <span>GET</span>
            <code>/health</code>
          </div>
        </div>
      </div>

      <div className="documentation-card architecture-doc">
        <div className="card-kicker">
          HYBRIDIDSNET
        </div>

        <h2>Processing Architecture</h2>

        <div className="doc-pipeline">
          <DocNode number="01" title="INPUT" value="78 Features" />

          <div>→</div>

          <DocNode number="02" title="CNN" value="Feature Extraction" />

          <div>→</div>

          <DocNode number="03" title="LSTM / GRU" value="Temporal Patterns" />

          <div>→</div>

          <DocNode number="04" title="ATTENTION" value="Important Features" />

          <div>→</div>

          <DocNode number="05" title="OUTPUT" value="5 Classes" />
        </div>
      </div>
    </section>
  );

  // --------------------------------------------------
  // MAIN
  // --------------------------------------------------

  return (
    <div className="app-shell">

      {/* SIDEBAR */}

      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">DS</div>

          <div>
            <strong>DeepShieldIDS</strong>
            <span>AI Network Security</span>
          </div>
        </div>

        <div className="environment">
          <span>ENVIRONMENT</span>

          <strong>
            PRODUCTION
            <small>⌄</small>
          </strong>
        </div>

        <nav className="navigation">
          <div className="nav-label">
            MONITORING
          </div>

          <NavButton
            active={activePage === "dashboard"}
            icon="▦"
            text="Dashboard"
            onClick={() => navigate("dashboard")}
          />

          <NavButton
            active={activePage === "traffic"}
            icon="◈"
            text="Traffic Analysis"
            onClick={() => navigate("traffic")}
          />

          <NavButton
            active={activePage === "threat"}
            icon="⚠"
            text="Threat Detection"
            onClick={() => navigate("threat")}
          />

          <NavButton
            active={activePage === "performance"}
            icon="◉"
            text="Model Performance"
            onClick={() => navigate("performance")}
          />

          <div className="nav-label settings-label">
            SYSTEM
          </div>

          <NavButton
            active={activePage === "settings"}
            icon="⚙"
            text="Settings"
            onClick={() => navigate("settings")}
          />

          <NavButton
            active={activePage === "documentation"}
            icon="?"
            text="Documentation"
            onClick={() => navigate("documentation")}
          />
        </nav>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span
              className={
                apiStatus === "Online"
                  ? "online-dot"
                  : "offline-dot"
              }
            ></span>

            <div>
              <strong>
                {apiStatus === "Online"
                  ? "System Online"
                  : "System Offline"}
              </strong>

              <small>
                {apiStatus === "Online"
                  ? "API connected"
                  : "API unavailable"}
              </small>
            </div>
          </div>

          <div className="version">
            DeepShieldIDS v1.0
          </div>
        </div>
      </aside>

      {/* MAIN CONTENT */}

      <main className="main-content">

        <header className="topbar">
          <div>
            <span className="breadcrumb">
              Security <b>/</b> {getPageTitle()}
            </span>

            <h3>{getPageTitle()}</h3>
          </div>

          <div className="topbar-right">
            <div className="api-indicator">
              <span
                className={
                  apiStatus === "Online"
                    ? "online-dot"
                    : "offline-dot"
                }
              ></span>

              API {apiStatus}
            </div>

            {notifications && (
              <div className="notification">
                ♢
                <span>3</span>
              </div>
            )}

            <div className="profile">
              <div className="profile-avatar">
                A
              </div>

              <div>
                <strong>Admin</strong>
                <small>Security Analyst</small>
              </div>
            </div>
          </div>
        </header>

        <div className="content-container">

          {activePage === "dashboard" && <Dashboard />}

          {activePage === "traffic" && <TrafficAnalysis />}

          {activePage === "threat" && <ThreatDetection />}

          {activePage === "performance" && <ModelPerformance />}

          {activePage === "settings" && <Settings />}

          {activePage === "documentation" && <Documentation />}

        </div>

        <footer className="footer">
          <span>DeepShieldIDS</span>
          <span>•</span>
          <span>HybridIDSNet</span>
          <span>•</span>
          <span>CIC-IDS2017</span>
          <span>•</span>
          <span>AI Network Intrusion Detection</span>
          <span>•</span>
          <span>v1.0</span>
        </footer>

      </main>
    </div>
  );
}

// ======================================================
// COMPONENTS
// ======================================================

function NavButton({ active, icon, text, onClick }) {
  return (
    <button
      className={active ? "nav-item active" : "nav-item"}
      onClick={onClick}
    >
      <span>{icon}</span>
      {text}
    </button>
  );
}

function MetricCard({
  number,
  title,
  value,
  description,
}) {
  return (
    <div className="metric-card">
      <div className="metric-number">{number}</div>

      <div className="metric-label">
        {title}
      </div>

      <div className="metric-value">
        {value}
      </div>

      <div className="metric-description">
        {description}
      </div>
    </div>
  );
}

function ActionCard({
  number,
  icon,
  iconClass,
  title,
  description,
  button,
  onClick,
}) {
  return (
    <div className="dashboard-card">
      <div className="section-number">
        {number}
      </div>

      <div className={`card-icon ${iconClass}`}>
        {icon}
      </div>

      <h2>{title}</h2>

      <p>{description}</p>

      <button
        className="main-action-button"
        onClick={onClick}
      >
        {button}
        <span>→</span>
      </button>
    </div>
  );
}

function PageIntro({
  eyebrow,
  title,
  description,
}) {
  return (
    <div className="page-intro">
      <div>
        <div className="eyebrow">{eyebrow}</div>

        <h1>{title}</h1>

        <p>{description}</p>
      </div>
    </div>
  );
}

function InfoBox({
  title,
  value,
  description,
  green,
}) {
  return (
    <div className="info-box">
      <span>{title}</span>

      <strong className={green ? "online-text" : ""}>
        {value}
      </strong>

      <small>{description}</small>
    </div>
  );
}

function PerformanceCard({
  title,
  value,
  description,
  highlight,
}) {
  return (
    <div
      className={
        highlight
          ? "performance-card highlight-performance"
          : "performance-card"
      }
    >
      <span>{title}</span>
      <strong>{value}</strong>
      <small>{description}</small>
    </div>
  );
}

function QualityBar({ label, value }) {
  return (
    <div className="quality-row">
      <span>{label}</span>

      <div className="quality-bar">
        <div style={{ width: `${value}%` }}></div>
      </div>

      <strong>{value.toFixed(2)}%</strong>
    </div>
  );
}

function ArchitectureDiagram() {
  return (
    <section className="architecture-card">
      <div className="section-heading">
        <div>
          <div className="eyebrow">
            SYSTEM ARCHITECTURE
          </div>

          <h2>HybridIDSNet Processing Pipeline</h2>
        </div>
      </div>

      <div className="pipeline">

        <PipelineNode
          number="01"
          title="INPUT"
          value="78 Features"
        />

        <div className="pipeline-line">→</div>

        <PipelineNode
          number="02"
          title="CNN"
          value="Spatial Features"
        />

        <div className="pipeline-line">→</div>

        <PipelineNode
          number="03"
          title="LSTM / GRU"
          value="Temporal Patterns"
        />

        <div className="pipeline-line">→</div>

        <PipelineNode
          number="04"
          title="ATTENTION"
          value="Feature Weighting"
        />

        <div className="pipeline-line">→</div>

        <PipelineNode
          number="05"
          title="OUTPUT"
          value="5 Classes"
          output
        />

      </div>
    </section>
  );
}

function PipelineNode({
  number,
  title,
  value,
  output,
}) {
  return (
    <div
      className={
        output
          ? "pipeline-node output-node"
          : "pipeline-node"
      }
    >
      <div className="pipeline-icon">
        {number}
      </div>

      <strong>{title}</strong>

      <span>{value}</span>
    </div>
  );
}

function DocNode({
  number,
  title,
  value,
}) {
  return (
    <div className="doc-node">
      <span>{number}</span>
      <strong>{title}</strong>
      <small>{value}</small>
    </div>
  );
}

export default App;