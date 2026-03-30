/**
 * ToxAI — Drug Toxicity Prediction Platform
 * Main Frontend JavaScript
 *
 * Handles all API communication, chart rendering,
 * molecular visualization, chatbot, batch prediction,
 * and UI state management.
 */

'use strict';

// =========================================================
// CONFIGURATION
// =========================================================

const API_BASE = 'http://localhost:8000';

const TOX21_LABELS = [
  'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
  'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
  'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
];

const ENDPOINT_DESCRIPTIONS = {
  'NR-AR':         'Androgen Receptor',
  'NR-AR-LBD':     'AR Ligand Binding Domain',
  'NR-AhR':        'Aryl Hydrocarbon Receptor',
  'NR-Aromatase':  'Aromatase Inhibitor',
  'NR-ER':         'Estrogen Receptor α',
  'NR-ER-LBD':     'ER Ligand Binding Domain',
  'NR-PPAR-gamma': 'PPAR Gamma',
  'SR-ARE':        'Oxidative Stress (ARE)',
  'SR-ATAD5':      'Genotoxicity (ATAD5)',
  'SR-HSE':        'Heat Shock Response',
  'SR-MMP':        'Mitochondrial Membrane',
  'SR-p53':        'p53 / DNA Damage',
};

// App state
let state = {
  theme: 'dark',
  lastSmiles: '',
  lastPrediction: null,
  batchResults: [],
  charts: {},
  currentView: 'cards',
  chatbotOpen: false,
  viewer3D: null,
};

// =========================================================
// INITIALIZATION
// =========================================================

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  checkHealth();
  initNavTabs();
  initSmilesLivePreview();
  loadDataInsights();
  setInterval(checkHealth, 30000);
});

// =========================================================
// THEME
// =========================================================

function initTheme() {
  const saved = localStorage.getItem('toxai-theme') || 'dark';
  setTheme(saved);
}

function setTheme(theme) {
  state.theme = theme;
  document.documentElement.setAttribute('data-theme', theme);
  document.getElementById('themeIcon').textContent = theme === 'dark' ? '☀️' : '🌙';
  localStorage.setItem('toxai-theme', theme);
  // Re-render charts in new theme colors
  setTimeout(() => { if (state.lastPrediction) renderCharts(state.lastPrediction); }, 100);
}

document.getElementById('themeToggle').addEventListener('click', () => {
  setTheme(state.theme === 'dark' ? 'light' : 'dark');
});

function getChartColors() {
  return state.theme === 'dark'
    ? { text: '#A0A3C4', grid: 'rgba(255,255,255,0.06)', bg: 'rgba(13,14,26,0)' }
    : { text: '#4A4A7A', grid: 'rgba(0,0,0,0.06)', bg: 'rgba(244,244,255,0)' };
}

// =========================================================
// HEALTH CHECK & STATUS
// =========================================================

async function checkHealth() {
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.models_loaded) {
      dot.className = 'status-dot ready';
      text.textContent = 'Models Ready';
      // Update stats
      if (data.training_metadata) {
        const meta = data.training_metadata;
        if (meta.n_train) document.getElementById('statCompounds').textContent = (meta.n_train + meta.n_test).toLocaleString();
        if (meta.mean_auc_roc) document.getElementById('statAUC').textContent = meta.mean_auc_roc.toFixed(3);
        if (meta.n_features) document.getElementById('statFeatures').textContent = meta.n_features.toLocaleString();
      }
    } else {
      dot.className = 'status-dot error';
      text.textContent = 'Train models first';
    }
  } catch {
    dot.className = 'status-dot error';
    text.textContent = 'API offline';
  }
}

// =========================================================
// NAVIGATION
// =========================================================

function initNavTabs() {
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });
}

function switchTab(tabName) {
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-section').forEach(s => s.classList.remove('active'));

  const tab = document.querySelector(`[data-tab="${tabName}"]`);
  const section = document.getElementById(`section-${tabName}`);
  if (tab) tab.classList.add('active');
  if (section) section.classList.add('active');

  // Lazy-load data for data insights tab
  if (tabName === 'data') loadDataInsights();
}

// =========================================================
// SMILES INPUT — HOME PAGE
// =========================================================

function loadExample(smiles) {
  document.getElementById('smilesInputHome').value = smiles;
  updateHomeMolViewer(smiles);
}

function predictFromHome() {
  const smiles = document.getElementById('smilesInputHome').value.trim();
  if (!smiles) { showNotification('Please enter a SMILES string', 'warning'); return; }
  // Switch to analyze tab and run
  setSmiles(smiles);
  switchTab('analyze');
  setTimeout(() => runFullAnalysis(), 200);
}

function updateHomeMolViewer(smiles) {
  const container = document.getElementById('molViewerHome');
  if (!smiles) return;
  fetchAndDisplayMol(smiles, container);
}

function initSmilesLivePreview() {
  const homeInput = document.getElementById('smilesInputHome');
  let timer;
  homeInput.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(() => updateHomeMolViewer(homeInput.value), 600);
  });
}

// =========================================================
// ANALYZE — MAIN PREDICTION FLOW
// =========================================================

function setSmiles(smiles) {
  document.getElementById('smilesInputAnalyze').value = smiles;
}

function clearInput() {
  document.getElementById('smilesInputAnalyze').value = '';
  document.getElementById('molViewer2D').innerHTML = '<div class="mol-placeholder"><div class="mol-placeholder-icon">⬡</div><div>Structure loads after prediction</div></div>';
}

async function runFullAnalysis() {
  const smiles = document.getElementById('smilesInputAnalyze').value.trim();
  if (!smiles) { showNotification('Please enter a SMILES string', 'warning'); return; }

  const model = document.getElementById('modelSelect').value;
  const endpoint = document.getElementById('endpointSelect').value;

  // Show loading
  showLoading('Analyzing compound...');
  setBtnLoading(true);

  try {
    // Run all API calls in parallel
    const [predResult, explainResult, similarResult] = await Promise.all([
      apiPost('/predict', { smiles, model }),
      apiPost('/explain', { smiles, endpoint, top_n: 15 }),
      apiPost('/similar', { smiles, top_k: 6 }),
    ]);

    state.lastSmiles = smiles;
    state.lastPrediction = predResult;

    // Store for report tab
    updateReportTab(smiles);

    // Render all UI
    renderResults(predResult, explainResult, similarResult);

    // Check for high toxicity notification
    if (predResult.n_high_risk > 0) {
      showNotification(
        `⚠️ High toxicity detected! ${predResult.n_high_risk} endpoint(s) above 70% risk threshold.`,
        'danger'
      );
    }

  } catch (err) {
    console.error(err);
    showNotification(`Error: ${err.message}`, 'error');
  } finally {
    hideLoading();
    setBtnLoading(false);
  }
}

function setBtnLoading(loading) {
  const btn = document.getElementById('btnPredict');
  const spinner = document.getElementById('predictSpinner');
  const text = document.getElementById('btnPredictText');
  btn.disabled = loading;
  spinner.style.display = loading ? 'inline' : 'none';
  text.textContent = loading ? 'Analyzing...' : '⚡ Run Full Analysis';
}

// =========================================================
// RENDER RESULTS
// =========================================================

function renderResults(pred, explain, similar) {
  // Show results, hide placeholder
  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContainer').style.display = 'block';

  // Risk banner
  renderRiskBanner(pred);

  // 2D molecule
  renderMolViewer2D(pred.mol_svg);

  // Endpoint cards (default view)
  renderEndpointCards(pred.predictions);
  renderCharts(pred);

  // ADMET
  renderADMET(pred.admet);

  // Structural alerts
  renderAlerts(pred.structural_alerts);

  // SHAP
  renderSHAP(explain, document.getElementById('endpointSelect').value);

  // Atom heatmap
  renderAtomHeatmap(explain);

  // Similar compounds
  renderSimilar(similar);
}

function renderRiskBanner(pred) {
  const banner = document.getElementById('riskBanner');
  const level = pred.risk_level || 'LOW';
  const score = pred.overall_toxicity_score || 0;
  const maxScore = pred.max_toxicity_score || 0;

  banner.className = `risk-banner ${level === 'HIGH' ? 'high' : level === 'MODERATE' ? 'moderate' : ''}`;

  const icons = { LOW: '🟢', MODERATE: '🟡', HIGH: '🔴' };
  document.getElementById('riskIcon').textContent = icons[level] || '⚪';

  document.getElementById('riskLevel').textContent = `${level} RISK`;
  document.getElementById('riskLevel').style.color = level === 'HIGH' ? 'var(--danger)' : level === 'MODERATE' ? 'var(--warning)' : 'var(--success)';

  document.getElementById('riskDetail').textContent =
    `${pred.n_high_risk || 0} high-risk, ${pred.n_moderate_risk || 0} moderate-risk endpoints`;

  document.getElementById('riskScoreMax').textContent = `${(maxScore * 100).toFixed(1)}%`;
  document.getElementById('riskScoreAvg').textContent = `${(score * 100).toFixed(1)}%`;
  document.getElementById('riskHighCount').textContent = pred.n_high_risk || 0;
}

function renderMolViewer2D(svgB64) {
  const container = document.getElementById('molViewer2D');
  if (!svgB64) {
    container.innerHTML = '<div class="mol-placeholder"><div>Could not render molecule</div></div>';
    return;
  }

  container.innerHTML = `<img
    src="data:image/svg+xml;base64,${svgB64}"
    alt="Molecular Structure"
    style="max-width:100%;max-height:220px;background:white;border-radius:8px;padding:8px;"
  />`;
}

function renderEndpointCards(predictions) {
  const container = document.getElementById('endpointCards');
  container.innerHTML = '';

  const sorted = Object.entries(predictions).sort((a, b) => b[1] - a[1]);

  sorted.forEach(([label, prob]) => {
    const riskClass = prob >= 0.7 ? 'high-risk' : prob >= 0.4 ? 'moderate' : '';
    const probClass = prob >= 0.7 ? 'high' : prob >= 0.4 ? 'moderate' : 'low';
    const riskLabel = prob >= 0.7 ? 'HIGH' : prob >= 0.4 ? 'MODERATE' : 'LOW';

    container.innerHTML += `
      <div class="endpoint-card ${riskClass}" title="${ENDPOINT_DESCRIPTIONS[label] || label}">
        <div class="endpoint-name">${label}</div>
        <div class="endpoint-prob ${probClass}">${(prob * 100).toFixed(1)}%</div>
        <div class="endpoint-risk-label text-${probClass === 'high' ? 'danger' : probClass === 'moderate' ? 'warning' : 'success'}">${riskLabel}</div>
      </div>`;
  });
}

function switchView(view) {
  state.currentView = view;
  document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`view${view.charAt(0).toUpperCase() + view.slice(1)}`).classList.add('active');

  document.getElementById('endpointCardsView').style.display = view === 'cards' ? 'block' : 'none';
  document.getElementById('endpointChartView').style.display = view === 'chart' ? 'block' : 'none';
  document.getElementById('endpointRadarView').style.display = view === 'radar' ? 'block' : 'none';
}

// =========================================================
// CHARTS
// =========================================================

function renderCharts(pred) {
  const { text, grid } = getChartColors();
  const predictions = pred.predictions;

  const labels = Object.keys(predictions);
  const values = Object.values(predictions).map(v => parseFloat((v * 100).toFixed(1)));
  const colors = values.map(v =>
    v >= 70 ? 'rgba(255,71,87,0.8)' : v >= 40 ? 'rgba(255,165,2,0.8)' : 'rgba(46,213,115,0.8)'
  );
  const colorsB = values.map(v =>
    v >= 70 ? '#FF4757' : v >= 40 ? '#FFA502' : '#2ED573'
  );

  // Bar chart
  destroyChart('toxicityBarChart');
  const barCtx = document.getElementById('toxicityBarChart').getContext('2d');
  state.charts.bar = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Toxicity Probability (%)',
        data: values,
        backgroundColor: colors,
        borderColor: colorsB,
        borderWidth: 1,
        borderRadius: 6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(1)}% — ${ctx.parsed.y >= 70 ? 'HIGH RISK' : ctx.parsed.y >= 40 ? 'MODERATE' : 'LOW RISK'}`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true, max: 100,
          grid: { color: grid },
          ticks: { color: text, callback: v => `${v}%` }
        },
        x: { grid: { display: false }, ticks: { color: text, maxRotation: 45 } }
      }
    }
  });

  // Radar chart
  destroyChart('toxicityRadarChart');
  const radarCtx = document.getElementById('toxicityRadarChart').getContext('2d');
  state.charts.radar = new Chart(radarCtx, {
    type: 'radar',
    data: {
      labels: labels.map(l => l.replace('NR-', '').replace('SR-', '')),
      datasets: [{
        label: 'Toxicity %',
        data: values,
        backgroundColor: 'rgba(108,99,255,0.2)',
        borderColor: '#6C63FF',
        borderWidth: 2,
        pointBackgroundColor: colorsB,
        pointRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0, max: 100,
          grid: { color: grid },
          angleLines: { color: grid },
          ticks: { display: false },
          pointLabels: { color: text, font: { size: 10 } }
        }
      }
    }
  });

  // Lipinski Radar
  renderLipinskiRadar(pred.admet);
}

function renderLipinskiRadar(admet) {
  if (!admet) return;
  const { text, grid } = getChartColors();

  destroyChart('lipinskiRadar');
  const ctx = document.getElementById('lipinskiRadar').getContext('2d');

  // Normalize to 0-100 scale against Lipinski limits
  const mwNorm = Math.min(100, (admet.mw || 0) / 5);
  const logpNorm = Math.min(100, ((admet.logp || 0) + 2) * 10);
  const tpsaNorm = Math.min(100, (admet.tpsa || 0) / 1.4);
  const hbdNorm = Math.min(100, (admet.hbd || 0) * 20);
  const hbaNorm = Math.min(100, (admet.hba || 0) * 10);
  const rbNorm = Math.min(100, (admet.rotatable_bonds || 0) * 10);

  state.charts.lipinski = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['MW/5', 'LogP', 'TPSA', 'HBD×20', 'HBA×10', 'RotBonds×10'],
      datasets: [
        {
          label: 'Compound',
          data: [mwNorm, logpNorm, tpsaNorm, hbdNorm, hbaNorm, rbNorm],
          backgroundColor: 'rgba(62,207,207,0.15)',
          borderColor: '#3ECFCF',
          borderWidth: 2,
          pointBackgroundColor: '#3ECFCF',
          pointRadius: 3,
        },
        {
          label: 'Lipinski Limit',
          data: [100, 100, 100, 100, 100, 100],
          backgroundColor: 'rgba(255,71,87,0.05)',
          borderColor: 'rgba(255,71,87,0.4)',
          borderWidth: 1,
          borderDash: [6, 4],
          pointRadius: 0,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { labels: { color: text, font: { size: 10 } } } },
      scales: {
        r: {
          min: 0, max: 100,
          grid: { color: grid },
          angleLines: { color: grid },
          ticks: { display: false },
          pointLabels: { color: text, font: { size: 9 } }
        }
      }
    }
  });
}

function destroyChart(id) {
  const canvas = document.getElementById(id);
  if (!canvas) return;
  const existing = Chart.getChart(canvas);
  if (existing) existing.destroy();
}

// =========================================================
// ADMET PANEL
// =========================================================

function renderADMET(admet) {
  if (!admet) return;
  const grid = document.getElementById('admetGrid');

  const items = [
    { label: 'Mol. Weight', value: `${(admet.mw||0).toFixed(1)} g/mol`, pass: admet.mw <= 500 },
    { label: 'LogP', value: `${(admet.logp||0).toFixed(2)}`, pass: admet.logp <= 5 },
    { label: 'TPSA', value: `${(admet.tpsa||0).toFixed(1)} Å²`, pass: admet.tpsa <= 140 },
    { label: 'HBD', value: `${admet.hbd||0}`, pass: admet.hbd <= 5 },
    { label: 'HBA', value: `${admet.hba||0}`, pass: admet.hba <= 10 },
    { label: 'Rot. Bonds', value: `${admet.rotatable_bonds||0}`, pass: admet.rotatable_bonds <= 10 },
    { label: 'Aromatic Rings', value: `${admet.aromatic_rings||0}`, pass: null },
    { label: 'Lipinski Ro5', value: admet.lipinski_pass ? 'PASS ✓' : 'FAIL ✗', pass: admet.lipinski_pass },
    { label: 'Veber Rules', value: admet.veber_pass ? 'PASS ✓' : 'FAIL ✗', pass: admet.veber_pass },
    { label: 'Drug-likeness', value: `${((admet.qed_score||0)*100).toFixed(0)}%`, pass: admet.qed_score >= 0.5 },
  ];

  grid.innerHTML = items.map(item => `
    <div class="admet-item">
      <div class="admet-item-label">${item.label}</div>
      <div class="admet-item-value ${item.pass === null ? 'neutral' : item.pass ? 'pass' : 'fail'}">
        ${item.value}
      </div>
    </div>
  `).join('');
}

// =========================================================
// STRUCTURAL ALERTS
// =========================================================

function renderAlerts(alerts) {
  if (!alerts) return;
  const panel = document.getElementById('alertsPanel');

  const severity = alerts.alert_severity || 'LOW';
  const groups = alerts.reactive_groups || [];

  panel.innerHTML = `
    <div class="alert-severity-badge ${severity}">
      ${severity === 'HIGH' ? '🔴' : severity === 'MEDIUM' ? '🟡' : '🟢'}
      Alert Severity: ${severity}
    </div>
    <div style="font-size:0.82rem;color:var(--text-secondary);margin:8px 0;">
      ${groups.length === 0
        ? '✅ No structural alerts detected. Compound appears clean.'
        : `${groups.length} reactive substructure(s) detected:`
      }
    </div>
    ${groups.length > 0 ? `
      <div class="alert-list">
        ${groups.map(g => `<div class="alert-item">⚠️ ${g}</div>`).join('')}
      </div>
    ` : ''}
  `;
}

// =========================================================
// SHAP / FEATURE IMPORTANCE
// =========================================================

function renderSHAP(explainData, endpoint) {
  const panel = document.getElementById('shapPanel');
  const badge = document.getElementById('explainEndpointBadge');
  badge.textContent = endpoint;

  if (!explainData || !explainData.shap_explanation) {
    panel.innerHTML = '<div class="shap-loading">SHAP data unavailable. Models may need training.</div>';
    return;
  }

  const topFeatures = explainData.shap_explanation.top_features || [];
  if (topFeatures.length === 0) {
    panel.innerHTML = '<div class="shap-loading">No feature importance data available.</div>';
    return;
  }

  const maxVal = Math.max(...topFeatures.map(f => Math.abs(f.shap_value || f.importance || 0)));

  panel.innerHTML = topFeatures.map(feat => {
    const val = Math.abs(feat.shap_value || feat.importance || 0);
    const pct = maxVal > 0 ? (val / maxVal * 100) : 0;
    const dir = feat.direction === 'toxic' ? 'toxic' : 'safe';
    const name = (feat.feature || '').replace('desc_', '').replace('morgan_', 'Morgan:').replace('maccs_', 'MACCS:').replace('admet_', '★ ');

    return `
      <div class="shap-bar-item">
        <div class="shap-feature-name" title="${feat.feature}">${name}</div>
        <div class="shap-bar-track">
          <div class="shap-bar-fill ${dir}" style="width:${pct.toFixed(1)}%"></div>
        </div>
        <div class="shap-val">${val.toFixed(4)}</div>
        <div style="font-size:0.65rem;color:${dir==='toxic'?'var(--danger)':'var(--success)'};min-width:30px">
          ${dir === 'toxic' ? '↑' : '↓'}
        </div>
      </div>
    `;
  }).join('');
}

// =========================================================
// ATOM HEATMAP
// =========================================================

function renderAtomHeatmap(explainData) {
  const container = document.getElementById('heatmapMolViewer');

  if (!explainData || !explainData.mol_svg_highlighted) {
    container.innerHTML = '<div class="mol-placeholder"><div>Atom attribution unavailable</div></div>';
    return;
  }

  container.innerHTML = `
    <img
      src="data:image/svg+xml;base64,${explainData.mol_svg_highlighted}"
      alt="Atom Toxicity Heatmap"
      style="max-width:100%;max-height:220px;border-radius:8px;padding:8px;background:white;"
    />`;
}

// =========================================================
// SIMILAR COMPOUNDS
// =========================================================

function renderSimilar(similarData) {
  const panel = document.getElementById('similarPanel');
  const countEl = document.getElementById('similarCount');

  if (!similarData || !similarData.similar_compounds) {
    panel.innerHTML = '<div class="similar-loading">Similarity search unavailable.</div>';
    return;
  }

  const compounds = similarData.similar_compounds;
  countEl.textContent = `${compounds.length} found`;

  if (compounds.length === 0) {
    panel.innerHTML = '<div class="similar-loading">No similar compounds found in database.</div>';
    return;
  }

  panel.innerHTML = compounds.map(comp => {
    const toxEntries = Object.entries(comp.toxicity || {})
      .filter(([, v]) => v !== null)
      .slice(0, 4);

    const imgHtml = comp.mol_svg
      ? `<img src="data:image/svg+xml;base64,${comp.mol_svg}" class="similar-mol-img" alt="compound" style="background:white;border-radius:4px;padding:4px;" />`
      : `<div class="similar-mol-img" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);">⬡</div>`;

    const toxChips = toxEntries.map(([k, v]) =>
      `<span class="similar-tox-chip ${v >= 0.5 ? 'pos' : 'neg'}" title="${k}">${k.replace('NR-','').replace('SR-','')}: ${v >= 0.5 ? 'T' : 'S'}</span>`
    ).join('');

    return `
      <div class="similar-card">
        ${imgHtml}
        <div class="similar-info">
          <span class="similar-sim">Tanimoto: ${comp.similarity_pct}</span>
          <div class="similar-smiles" title="${comp.smiles}">${comp.smiles}</div>
          <div class="similar-tox-row">${toxChips}</div>
        </div>
      </div>`;
  }).join('');
}

// =========================================================
// MOLECULE VIEWER (2D/3D toggle)
// =========================================================

async function fetchAndDisplayMol(smiles, container) {
  try {
    const res = await apiPost('/predict', { smiles, model: 'ensemble' });
    if (res.mol_svg) {
      container.innerHTML = `
        <img src="data:image/svg+xml;base64,${res.mol_svg}"
          alt="molecule"
          style="max-width:100%;max-height:200px;background:white;border-radius:8px;padding:6px;" />`;
    }
  } catch {
    container.innerHTML = '<div class="mol-placeholder"><div>Could not render</div></div>';
  }
}

function switchViewer(mode) {
  document.getElementById('btn2D').classList.toggle('active', mode === '2d');
  document.getElementById('btn3D').classList.toggle('active', mode === '3d');
  document.getElementById('molViewer2D').style.display = mode === '2d' ? 'flex' : 'none';
  document.getElementById('molViewer3D').style.display = mode === '3d' ? 'block' : 'none';

  if (mode === '3d' && state.lastSmiles) {
    render3DMolecule(state.lastSmiles);
  }
}

function render3DMolecule(smiles) {
  const container = document.getElementById('viewer3DContainer');
  container.innerHTML = '';

  try {
    if (typeof $3Dmol === 'undefined') {
      container.innerHTML = '<div class="mol-placeholder"><div>3Dmol.js loading...</div></div>';
      return;
    }

    const viewer = $3Dmol.createViewer(container, {
      backgroundColor: state.theme === 'dark' ? '0x0D0E1A' : '0xF4F4FF',
      antialias: true,
    });

    state.viewer3D = viewer;

    // Generate a simple 3D molecule from SMILES using SDF format
    // 3Dmol.js needs coordinates — we use a built-in simple structure
    // For real 3D: would need RDKit server-side coord gen
    viewer.addModel(`
      dummy
      `, 'sdf');

    // Since we can't easily get 3D coords client-side, show a dummy sphere
    viewer.addSphere({
      center: { x: 0, y: 0, z: 0 },
      radius: 1.2,
      color: 'purple',
      opacity: 0.6
    });
    viewer.addLabel(smiles.length > 20 ? smiles.substring(0, 20) + '...' : smiles, {
      position: { x: 0, y: 1.8, z: 0 },
      fontSize: 12,
      fontColor: 'white',
      backgroundColor: 'purple',
    });
    viewer.zoomTo();
    viewer.render();

    container.innerHTML = '';
    const hint = document.createElement('div');
    hint.style.cssText = 'position:absolute;bottom:8px;left:50%;transform:translateX(-50%);font-size:0.7rem;color:var(--text-muted);text-align:center;';
    hint.textContent = '3D requires RDKit coordinate generation (server-side)';
    container.style.position = 'relative';
    container.appendChild(hint);

  } catch (err) {
    console.warn('3D viewer error:', err);
    container.innerHTML = '<div class="mol-placeholder"><div>3D view unavailable</div></div>';
  }
}

// =========================================================
// DATA INSIGHTS
// =========================================================

async function loadDataInsights() {
  renderPropertyDistributions();
  await loadEvaluation();
  await loadChemicalSpace();
}

async function loadEvaluation() {
  try {
    const data = await apiFetch('/evaluation');
    if (!data || Object.keys(data).length === 0) {
      renderDemoEvaluation();
      return;
    }

    const evalData = data.ensemble || data.rf || data.xgb || [];
    if (evalData.length > 0) {
      renderPerformanceChart(evalData);
      renderEndpointSummaryTable(evalData);
    }
  } catch {
    renderDemoEvaluation();
  }
}

function renderDemoEvaluation() {
  // Demo data when models aren't trained yet
  const demoData = TOX21_LABELS.map(label => ({
    endpoint: label,
    'AUC-ROC': 0.72 + Math.random() * 0.2,
    'AUC-PR': 0.45 + Math.random() * 0.3,
    n_samples: Math.floor(2000 + Math.random() * 5000),
    positive_rate: 0.05 + Math.random() * 0.25,
  }));
  renderPerformanceChart(demoData);
  renderEndpointSummaryTable(demoData);
}

function renderPerformanceChart(evalData) {
  const { text, grid } = getChartColors();
  destroyChart('performanceChart');

  const ctx = document.getElementById('performanceChart').getContext('2d');
  const labels = evalData.map(r => r.endpoint);
  const aucRoc = evalData.map(r => parseFloat((r['AUC-ROC'] * 100).toFixed(1)));
  const aucPr = evalData.map(r => parseFloat(((r['AUC-PR'] || 0) * 100).toFixed(1)));

  state.charts.perf = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'AUC-ROC (%)',
          data: aucRoc,
          backgroundColor: 'rgba(108,99,255,0.7)',
          borderColor: '#6C63FF',
          borderWidth: 1,
          borderRadius: 4,
        },
        {
          label: 'AUC-PR (%)',
          data: aucPr,
          backgroundColor: 'rgba(62,207,207,0.6)',
          borderColor: '#3ECFCF',
          borderWidth: 1,
          borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: {
        legend: { labels: { color: text } },
        tooltip: { callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(1)}%` } }
      },
      scales: {
        y: {
          min: 0, max: 100,
          grid: { color: grid },
          ticks: { color: text, callback: v => `${v}%` }
        },
        x: { grid: { display: false }, ticks: { color: text, maxRotation: 45 } }
      }
    }
  });
}

function renderEndpointSummaryTable(evalData) {
  const container = document.getElementById('endpointSummaryTable');
  const rows = [
    `<div class="endpoint-summary-row">
      <div>Endpoint</div><div>AUC-ROC</div><div>Samples</div><div>Pos. Rate</div>
    </div>`
  ];

  evalData.forEach(row => {
    const auc = parseFloat(row['AUC-ROC'] || 0);
    const pct = Math.round(auc * 100);
    rows.push(`
      <div class="endpoint-summary-row">
        <div class="mono" style="font-size:0.8rem;">${row.endpoint}</div>
        <div class="auc-bar-track"><div class="auc-bar" style="width:${pct}%"></div></div>
        <div style="color:var(--text-secondary)">${(row.n_samples||0).toLocaleString()}</div>
        <div style="color:var(--text-secondary)">${((row.positive_rate||0)*100).toFixed(1)}%</div>
      </div>
    `);
  });

  container.innerHTML = rows.join('');
}

function renderPropertyDistributions() {
  const { text, grid } = getChartColors();

  // MW Distribution (synthetic demo)
  const mwBins = [150,200,250,300,350,400,450,500,550,600];
  const mwCounts = [40,120,280,420,380,260,180,90,50,20];

  destroyChart('mwDistChart');
  const mwCtx = document.getElementById('mwDistChart').getContext('2d');
  new Chart(mwCtx, {
    type: 'bar',
    data: {
      labels: mwBins.map(v => `${v}`),
      datasets: [{
        label: 'Count',
        data: mwCounts,
        backgroundColor: 'rgba(108,99,255,0.6)',
        borderColor: '#6C63FF',
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { grid: { color: grid }, ticks: { color: text } },
        x: { grid: { display: false }, ticks: { color: text } }
      }
    }
  });

  // LogP Distribution
  const logpBins = [-4,-3,-2,-1,0,1,2,3,4,5,6,7];
  const logpCounts = [10,30,80,140,200,340,420,380,260,140,60,20];

  destroyChart('logpDistChart');
  const logpCtx = document.getElementById('logpDistChart').getContext('2d');
  new Chart(logpCtx, {
    type: 'bar',
    data: {
      labels: logpBins.map(v => `${v}`),
      datasets: [{
        label: 'Count',
        data: logpCounts,
        backgroundColor: 'rgba(62,207,207,0.6)',
        borderColor: '#3ECFCF',
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { grid: { color: grid }, ticks: { color: text } },
        x: { grid: { display: false }, ticks: { color: text } }
      }
    }
  });
}

async function loadChemicalSpace() {
  const { text, grid } = getChartColors();
  try {
    const data = await apiFetch('/space');
    if (!data || !data.points || data.points.length === 0) {
      renderDemoChemicalSpace();
      return;
    }

    destroyChart('chemSpaceChart');
    const ctx = document.getElementById('chemSpaceChart').getContext('2d');

    const pointData = data.points.map(p => ({
      x: p.x,
      y: p.y,
      tox: p.avg_toxicity || 0,
    }));

    const colors = pointData.map(p => {
      const t = p.tox;
      if (t >= 0.5) return 'rgba(255,71,87,0.7)';
      if (t >= 0.25) return 'rgba(255,165,2,0.7)';
      return 'rgba(46,213,115,0.7)';
    });

    new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Compounds',
          data: pointData,
          backgroundColor: colors,
          pointRadius: 4,
          pointHoverRadius: 6,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => `Tox: ${(ctx.raw.tox * 100).toFixed(1)}%`
            }
          }
        },
        scales: {
          x: { grid: { color: grid }, ticks: { color: text }, title: { display: true, text: 'PC1', color: text } },
          y: { grid: { color: grid }, ticks: { color: text }, title: { display: true, text: 'PC2', color: text } }
        }
      }
    });
  } catch {
    renderDemoChemicalSpace();
  }
}

function renderDemoChemicalSpace() {
  const { text, grid } = getChartColors();
  destroyChart('chemSpaceChart');
  const ctx = document.getElementById('chemSpaceChart').getContext('2d');

  const n = 200;
  const points = Array.from({ length: n }, () => ({
    x: (Math.random() - 0.5) * 10,
    y: (Math.random() - 0.5) * 10,
    tox: Math.random(),
  }));
  const colors = points.map(p =>
    p.tox >= 0.5 ? 'rgba(255,71,87,0.6)' :
    p.tox >= 0.25 ? 'rgba(255,165,2,0.6)' : 'rgba(46,213,115,0.6)'
  );

  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        { label: 'High Risk', data: points.filter(p=>p.tox>=0.5), backgroundColor: 'rgba(255,71,87,0.6)', pointRadius: 4 },
        { label: 'Moderate', data: points.filter(p=>p.tox>=0.25&&p.tox<0.5), backgroundColor: 'rgba(255,165,2,0.6)', pointRadius: 4 },
        { label: 'Low Risk', data: points.filter(p=>p.tox<0.25), backgroundColor: 'rgba(46,213,115,0.6)', pointRadius: 4 },
      ]
    },
    options: {
      responsive: true, maintainAspectRatio: true,
      plugins: { legend: { labels: { color: text } } },
      scales: {
        x: { grid: { color: grid }, ticks: { color: text }, title: { display: true, text: 'PC1 (Chemical Space)', color: text } },
        y: { grid: { color: grid }, ticks: { color: text }, title: { display: true, text: 'PC2 (Chemical Space)', color: text } }
      }
    }
  });
}

// =========================================================
// BATCH PREDICTION
// =========================================================

async function runBatchFromText() {
  const text = document.getElementById('batchSmilesInput').value.trim();
  if (!text) { showNotification('Enter SMILES strings (one per line)', 'warning'); return; }

  const smiles_list = text.split('\n').map(s => s.trim()).filter(Boolean);
  if (smiles_list.length === 0) return;

  showLoading(`Running batch prediction for ${smiles_list.length} compounds...`);
  try {
    const result = await apiPost('/batch', { smiles_list, model: 'ensemble' });
    state.batchResults = result.results;
    renderBatchResults(result);
  } catch (err) {
    showNotification(`Batch error: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) readCSVFile(file);
}

function handleFileDrop(event) {
  event.preventDefault();
  document.getElementById('uploadZone').classList.remove('drag-over');
  const file = event.dataTransfer.files[0];
  if (file && file.name.endsWith('.csv')) readCSVFile(file);
}

async function readCSVFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  showLoading(`Processing ${file.name}...`);
  try {
    const res = await fetch(`${API_BASE}/batch-file`, { method: 'POST', body: formData });
    const result = await res.json();
    state.batchResults = result.results;
    renderBatchResults(result);
  } catch (err) {
    showNotification(`File error: ${err.message}`, 'error');
  } finally {
    hideLoading();
  }
}

function renderBatchResults(result) {
  document.getElementById('batchResults').style.display = 'block';

  document.getElementById('batchStats').innerHTML =
    `<b>${result.successful}</b> successful, <b>${result.failed}</b> failed, <b>${result.total}</b> total`;

  const tbody = document.getElementById('batchTableBody');
  tbody.innerHTML = (result.results || []).map((r, i) => {
    const risk = r.risk_level || 'UNKNOWN';
    const score = ((r.overall_score || 0) * 100).toFixed(1);
    const highEndpoints = Object.entries(r.predictions || {})
      .filter(([,v]) => v >= 0.7)
      .map(([k,]) => k.replace('NR-','').replace('SR-',''))
      .join(', ') || '—';

    return `
      <tr>
        <td>${i + 1}</td>
        <td class="mono" style="font-size:0.75rem;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${r.smiles}">${r.smiles}</td>
        <td><span class="risk-badge ${risk}">${risk}</span></td>
        <td>${score}%</td>
        <td style="font-size:0.75rem;color:var(--danger)">${highEndpoints}</td>
        <td>
          <button class="btn-sm" onclick="viewBatchDetail(${i})">View</button>
        </td>
      </tr>`;
  }).join('');
}

function viewBatchDetail(idx) {
  const item = state.batchResults[idx];
  if (!item) return;
  setSmiles(item.smiles);
  switchTab('analyze');
  setTimeout(() => runFullAnalysis(), 200);
}

function downloadBatchResults() {
  if (state.batchResults.length === 0) return;

  const headers = ['smiles', 'risk_level', 'overall_score', ...TOX21_LABELS];
  const rows = state.batchResults.map(r => {
    const base = [
      `"${r.smiles}"`,
      r.risk_level || '',
      ((r.overall_score || 0) * 100).toFixed(2),
    ];
    const tox = TOX21_LABELS.map(l => ((r.predictions || {})[l] || 0).toFixed(4));
    return [...base, ...tox].join(',');
  });

  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `toxai_batch_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// =========================================================
// REPORT GENERATION
// =========================================================

function updateReportTab(smiles) {
  document.getElementById('reportSmilesInput').value = smiles;
  document.getElementById('lastCompoundInfo').textContent = smiles.length > 60 ? smiles.substring(0, 57) + '...' : smiles;
  document.getElementById('btnLastReport').disabled = false;
}

async function generateReport() {
  const smiles = document.getElementById('reportSmilesInput').value.trim() || state.lastSmiles;
  if (!smiles) { showNotification('Enter a SMILES string first', 'warning'); return; }

  await downloadReport(smiles);
}

async function generateLastReport() {
  if (!state.lastSmiles) { showNotification('Run a prediction first', 'warning'); return; }
  await downloadReport(state.lastSmiles);
}

async function downloadReport(smiles) {
  const btn = document.getElementById('reportBtnText');
  if (btn) btn.textContent = '⏳ Generating PDF...';
  showLoading('Generating professional PDF report...');

  try {
    const res = await fetch(`${API_BASE}/report`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ smiles, model: 'ensemble' }),
    });

    if (!res.ok) throw new Error(await res.text());

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `toxai_report_${Date.now()}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
    showNotification('✅ PDF report downloaded!', 'success');
  } catch (err) {
    showNotification(`Report error: ${err.message}`, 'error');
  } finally {
    hideLoading();
    if (btn) btn.textContent = '📥 Generate & Download PDF';
  }
}

// =========================================================
// CHATBOT
// =========================================================

function toggleChatbot() {
  state.chatbotOpen = !state.chatbotOpen;
  document.getElementById('chatbotPanel').style.display = state.chatbotOpen ? 'block' : 'none';
  document.getElementById('chatbotToggleIcon').textContent = state.chatbotOpen ? '✕' : '💬';
}

async function sendChatMessage() {
  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;

  appendChatMsg(message, 'user');
  input.value = '';

  // Show typing indicator
  const typingId = appendTypingIndicator();

  try {
    const context = state.lastPrediction ? {
      smiles: state.lastSmiles,
      predictions: state.lastPrediction.predictions,
      risk_level: state.lastPrediction.risk_level,
    } : {};

    const result = await apiPost('/chatbot', { message, context });
    removeTypingIndicator(typingId);
    appendChatMsg(result.response, 'bot', true);
  } catch {
    removeTypingIndicator(typingId);
    appendChatMsg("Sorry, I couldn't connect to the assistant. Is the server running?", 'bot');
  }
}

function appendChatMsg(text, role, markdown = false) {
  const messages = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = `chat-msg ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';

  if (markdown && typeof marked !== 'undefined') {
    bubble.innerHTML = marked.parse(text);
  } else {
    bubble.textContent = text;
  }

  div.appendChild(bubble);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div.id;
}

function appendTypingIndicator() {
  const messages = document.getElementById('chatMessages');
  const id = `typing-${Date.now()}`;
  messages.innerHTML += `
    <div class="chat-msg bot" id="${id}">
      <div class="chat-bubble">
        <div class="chat-typing">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>`;
  messages.scrollTop = messages.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// =========================================================
// NOTIFICATION SYSTEM
// =========================================================

function showNotification(text, type = 'info') {
  const banner = document.getElementById('notificationBanner');
  const colors = {
    danger: 'rgba(255,71,87,0.15)',
    warning: 'rgba(255,165,2,0.15)',
    success: 'rgba(46,213,115,0.15)',
    error: 'rgba(255,71,87,0.15)',
    info: 'rgba(62,207,207,0.15)',
  };
  banner.style.background = `linear-gradient(90deg, ${colors[type]||colors.info}, transparent)`;
  banner.style.display = 'flex';
  banner.classList.remove('hidden');
  document.getElementById('notifText').textContent = text;
  setTimeout(closeNotification, 6000);
}

function closeNotification() {
  document.getElementById('notificationBanner').classList.add('hidden');
}

// =========================================================
// LOADING OVERLAY
// =========================================================

function showLoading(text = 'Processing...', subtext = 'Computing molecular descriptors') {
  document.getElementById('loadingOverlay').style.display = 'flex';
  document.getElementById('loadingText').textContent = text;
}

function hideLoading() {
  document.getElementById('loadingOverlay').style.display = 'none';
}

// =========================================================
// API HELPERS
// =========================================================

async function apiPost(endpoint, body) {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`API ${endpoint} failed: ${errText}`);
  }
  return res.json();
}

async function apiFetch(endpoint) {
  const res = await fetch(`${API_BASE}${endpoint}`);
  if (!res.ok) throw new Error(`API ${endpoint} failed`);
  return res.json();
}
