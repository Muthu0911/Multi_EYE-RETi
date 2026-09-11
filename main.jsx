import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, ArrowRight, CheckCircle2, FileImage, Info, ShieldCheck, UploadCloud, X } from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const DISCLAIMER = "This prototype is intended for educational and research purposes only. It is not a medical diagnosis and should not replace professional ophthalmic evaluation.";
const STEPS = [
  ["01", "Upload", "Choose a clear retinal fundus image."],
  ["02", "Analyze", "The trained EfficientNet model reviews it."],
  ["03", "View result", "See the top screening signal and probabilities."],
  ["04", "Understand", "Review the optional attention overlay."],
];

function App() {
  const [page, setPage] = useState("home");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const classes = useMemo(() => result ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1]) : [], [result]);

  const selectFile = (candidate) => {
    setError("");
    if (!candidate || !["image/jpeg", "image/png"].includes(candidate.type)) {
      setError("Please select a JPG, JPEG, or PNG fundus image.");
      return;
    }
    if (candidate.size > 12 * 1024 * 1024) {
      setError("Please choose an image smaller than 12 MB.");
      return;
    }
    setFile(candidate);
    setPreview(URL.createObjectURL(candidate));
    setResult(null);
  };

  const analyze = async () => {
    if (!file) return setError("Upload a fundus image before analyzing.");
    setLoading(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "The screening request failed.");
      setResult(data);
      setPage("results");
    } catch (requestError) {
      setError(requestError.message || "Unable to reach the screening service.");
    } finally {
      setLoading(false);
    }
  };

  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => setPage("home")}><span className="brand-mark"><Activity size={20} /></span><span>Retina<span className="brand-soft">Lens</span></span></button>
      <nav><button className={page === "home" ? "active" : ""} onClick={() => setPage("home")}>Home</button><button className={page === "screening" ? "active" : ""} onClick={() => setPage("screening")}>Screening</button><button className={page === "about" ? "active" : ""} onClick={() => setPage("about")}>About model</button></nav>
      <span className="status-pill"><span />Research prototype</span>
    </header>

    {page === "home" && <main>
      <section className="hero container"><div className="hero-copy"><p className="eyebrow">RETINAL HEALTH, MADE CLEAR</p><h1>AI-powered retinal<br /><em>disease screening</em></h1><p className="lead">Upload a retinal fundus image and receive an AI-based screening result in moments, with transparent probability signals to discuss with a qualified clinician.</p><button className="primary-button" onClick={() => setPage("screening")}>Start screening <ArrowRight size={18} /></button><p className="microcopy"><ShieldCheck size={15} /> Image processed in memory and not permanently stored.</p></div><div className="hero-art"><div className="orbit orbit-one" /><div className="orbit orbit-two" /><div className="eye-card"><div className="retina-glow" /><div className="eye-ring" /><div className="eye-pupil" /><span className="scan-label">OPTICAL SCAN <b>● LIVE</b></span></div></div></section>
      <section className="steps-section container"><div className="section-heading"><p className="eyebrow">A SIMPLE FLOW</p><h2>From image to insight</h2></div><div className="steps-grid">{STEPS.map(([number, title, copy]) => <div className="step" key={number}><span>{number}</span><h3>{title}</h3><p>{copy}</p></div>)}</div></section>
      <Disclaimer />
    </main>}

    {page === "screening" && <main className="container page"><div className="page-heading"><p className="eyebrow">SCREENING</p><h1>Upload a fundus image</h1><p>For best results, use a well-lit image with the optic disc and retina visible.</p></div><UploadCard preview={preview} file={file} onSelect={selectFile} onRemove={() => { setFile(null); setPreview(""); setResult(null); }} onAnalyze={analyze} loading={loading} error={error} /></main>}

    {page === "results" && <main className="container page"><div className="page-heading"><p className="eyebrow">SCREENING RESULT</p><h1>Your screening overview</h1><p>This result is an AI screening signal, not a diagnosis.</p></div>{result && <div className="results-layout"><div className="result-primary"><div className="result-icon"><CheckCircle2 size={25} /></div><p className="eyebrow">HIGHEST MODEL SIGNAL</p><h2>{result.predicted_class}</h2><div className="confidence"><strong>{Math.round(result.confidence * 100)}%</strong><span>confidence</span></div><p className="result-note">A higher probability indicates a stronger model signal for this class. Multiple conditions may be present in a single image.</p><button className="secondary-button" onClick={() => setPage("screening")}>Screen another image</button></div><div className="probability-card"><h3>Probability distribution</h3>{classes.map(([label, probability]) => <div className="probability-row" key={label}><div><span>{label}</span><strong>{Math.round(probability * 100)}%</strong></div><div className="bar-track"><div className="bar-fill" style={{ width: `${probability * 100}%` }} /></div></div>)}{result.gradcam_overlay && <div className="explainability"><h3>AI attention overlay</h3><img src={`data:image/png;base64,${result.gradcam_overlay}`} alt="Grad-CAM attention overlay" /><p>Grad-CAM highlights image regions that influenced the highest model signal. It is not a lesion boundary or clinical measurement.</p></div>} {!result.gradcam_overlay && <p className="unavailable"><Info size={16} /> Visual explanation is not available for this model architecture.</p>}</div><div className="image-card"><h3>Uploaded fundus image</h3>{preview && <img src={preview} alt="Uploaded fundus" />}<span>{file?.name}</span></div></div>}<Disclaimer /></main>}

    {page === "about" && <main className="container page"><div className="page-heading"><p className="eyebrow">ABOUT THE MODEL</p><h1>Built for transparent research</h1><p>RetinaLens connects this interface to the existing Multi-Ocular Disease Detection model without replacing its trained weights.</p></div><div className="about-grid"><div className="info-card"><FileImage size={22} /><h3>EfficientNet-B0</h3><p>Transfer-learning architecture with a five-output sigmoid multilabel head.</p></div><div className="info-card"><Activity size={22} /><h3>Original preprocessing</h3><p>224 × 224 resize with ImageNet mean and standard-deviation normalization.</p></div><div className="info-card"><ShieldCheck size={22} /><h3>Five model classes</h3><p>Diabetic Retinopathy, Glaucoma, Cataract, AMD, and Hypertensive Retinopathy.</p></div></div><Disclaimer /></main>}
    <footer><span>RetinaLens · AI research prototype</span><span>Not for clinical diagnosis</span></footer>
  </div>;
}

function UploadCard({ preview, file, onSelect, onRemove, onAnalyze, loading, error }) {
  return <div className="upload-card"><label className={`drop-zone ${preview ? "has-preview" : ""}`} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); onSelect(event.dataTransfer.files[0]); }}>{preview ? <><img src={preview} alt="Fundus preview" /><span className="change-overlay">Change image</span></> : <><UploadCloud size={32} /><strong>Drop your image here</strong><span>or <u>browse files</u></span><small>JPG, JPEG or PNG · max 12 MB</small></>}<input type="file" accept="image/jpeg,image/png" onChange={(event) => onSelect(event.target.files[0])} /></label>{file && <div className="selected-file"><FileImage size={18} /><span>{file.name}</span><button onClick={onRemove} aria-label="Remove image"><X size={17} /></button></div>}{error && <p className="error-message">{error}</p>}<button className="primary-button analyze-button" onClick={onAnalyze} disabled={loading || !file}>{loading ? "Analyzing..." : "Analyze image"} <ArrowRight size={18} /></button></div>;
}

function Disclaimer() {
  return <aside className="disclaimer"><Info size={17} /><p><strong>Medical disclaimer:</strong> {DISCLAIMER}</p></aside>;
}

createRoot(document.getElementById("root")).render(<App />);

