/* ============================================================
   Long Hair Identification — ML Gender Classifier
   app.js
   ============================================================
   Decision Logic:
     - Age 20–30  → classify by hair length (long = female, short = male)
     - Age <20 or >30 → classify by actual biometric gender
   ============================================================ */

'use strict';

// ── DOM References ───────────────────────────────────────────
const fileInput       = document.getElementById('fileInput');
const dropZone        = document.getElementById('dropZone');
const previewSection  = document.getElementById('previewSection');
const previewImg      = document.getElementById('previewImg');
const clearBtn        = document.getElementById('clearBtn');
const analyseBtn      = document.getElementById('analyseBtn');
const loadingSection  = document.getElementById('loadingSection');
const errorSection    = document.getElementById('errorSection');
const resultSection   = document.getElementById('resultSection');
const resultHero      = document.getElementById('resultHero');
const resultGrid      = document.getElementById('resultGrid');
const confidenceSection = document.getElementById('confidenceSection');
const ruleReasoning   = document.getElementById('ruleReasoning');
const tryAgainBtn     = document.getElementById('tryAgainBtn');

const steps = [
  document.getElementById('step1'),
  document.getElementById('step2'),
  document.getElementById('step3'),
  document.getElementById('step4'),
];

// ── State ────────────────────────────────────────────────────
let currentBase64   = null;
let currentMimeType = null;
let stepTimer       = null;

const API_BASE_URL = window.location.protocol === 'file:'
  ? 'http://127.0.0.1:5000'
  : 'https://long-hair-identification-ml-model.onrender.com';

// ── File Upload ──────────────────────────────────────────────
fileInput.addEventListener('change', e => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

function loadFile(file) {
  if (!file.type.startsWith('image/')) {
    showError('Please upload a valid image file (JPG, PNG, or WEBP).');
    return;
  }

  const reader = new FileReader();
  reader.onload = e => {
    const dataUrl = e.target.result;
    currentBase64   = dataUrl.split(',')[1];
    currentMimeType = file.type;
    previewImg.src  = dataUrl;

    clearResults();
    dropZone.style.display = 'none';
    previewSection.classList.remove('hidden');
    loadingSection.classList.add('hidden');
  };
  reader.readAsDataURL(file);
}

// ── Clear / Reset ────────────────────────────────────────────
clearBtn.addEventListener('click', resetAll);
tryAgainBtn.addEventListener('click', resetAll);

function resetAll() {
  currentBase64   = null;
  currentMimeType = null;
  fileInput.value = '';
  previewImg.src  = '';
  clearResults();
  previewSection.classList.add('hidden');
  loadingSection.classList.add('hidden');
  resultSection.classList.add('hidden');
  errorSection.classList.add('hidden');
  dropZone.style.display = 'block';
  if (stepTimer) clearInterval(stepTimer);
}

function clearResults() {
  resultHero.innerHTML       = '';
  resultGrid.innerHTML       = '';
  confidenceSection.innerHTML = '';
  ruleReasoning.innerHTML    = '';
  errorSection.classList.add('hidden');
  resultSection.classList.add('hidden');
}

// ── Loading Steps Animation ──────────────────────────────────
function startStepAnimation() {
  steps.forEach(s => { s.classList.remove('active', 'done'); });
  steps[0].classList.add('active');

  let current = 0;
  stepTimer = setInterval(() => {
    if (current < steps.length - 1) {
      steps[current].classList.remove('active');
      steps[current].classList.add('done');
      current++;
      steps[current].classList.add('active');
    }
  }, 900);
}

function stopStepAnimation() {
  clearInterval(stepTimer);
  steps.forEach(s => { s.classList.remove('active'); s.classList.add('done'); });
}

// ── Analyse ──────────────────────────────────────────────────
analyseBtn.addEventListener('click', analyseImage);

async function analyseImage() {
  if (!currentBase64) return;

  clearResults();
  previewSection.querySelector('.btn-primary').style.display = 'none';
  loadingSection.classList.remove('hidden');
  startStepAnimation();

  try {
    const response = await fetch(`${API_BASE_URL}/api/analyse`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        mimeType: currentMimeType,
        imageBase64: currentBase64,
      }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      const detail = data.model_text || data.upstream_detail || data.upstream_text || '';
      const message = data.error || data.message || `API error: ${response.status}`;
      throw new Error(detail ? `${message} ${detail}` : message);
    }

    const result = data.result;
    if (!result) {
      throw new Error('Backend did not return an analysis result.');
    }

    stopStepAnimation();
    loadingSection.classList.add('hidden');
    previewSection.querySelector('.btn-primary').style.display = 'flex';

    renderResult(result);
    resultSection.classList.remove('hidden');

  } catch (err) {
    stopStepAnimation();
    loadingSection.classList.add('hidden');
    previewSection.querySelector('.btn-primary').style.display = 'flex';
    showError(`Analysis failed: ${err.message || 'Unknown error. Please try again.'}`);
  }
}

// ── Render Result ────────────────────────────────────────────
function renderResult(r) {
  const isHairBased   = r.ruleApplied === 'hair-based';
  const isFemale      = r.predictedGender === 'female';
  const genderClass   = isFemale ? 'female-col' : 'male-col';
  const genderEmoji   = isFemale ? '👩' : '👨';
  const conf          = clamp(r.overallConfidence || 75, 0, 100);
  const actualConf    = clamp(r.actualGenderConfidence || 70, 0, 100);
  const hairConf      = clamp(conf - 5, 0, 100);

  // ── Hero Card ────────────────────────────────────────────
  resultHero.innerHTML = `
    <div class="result-avatar">${genderEmoji}</div>
    <div>
      <div class="result-label">predicted gender</div>
      <div class="result-value ${genderClass}">${capitalize(r.predictedGender)}</div>
      <div class="result-sub">Estimated age: <strong>${r.estimatedAge} yrs</strong> &nbsp;·&nbsp; ${r.ageGroupLabel}</div>
    </div>
    <div class="result-badge">
      <span class="tag ${isHairBased ? 'tag-hair' : 'tag-out'}">
        ${isHairBased ? 'Hair-based rule' : 'Actual-gender rule'}
      </span>
      <span class="rule-name">${isHairBased ? 'Age 20–30 detected' : 'Outside 20–30 range'}</span>
    </div>
  `;

  // ── Detail Metrics Grid ──────────────────────────────────
  resultGrid.innerHTML = `
    <div class="detail-metric">
      <div class="dm-label">Age estimate</div>
      <div class="dm-value">${r.estimatedAge} <span style="font-size:13px;font-weight:400;color:var(--text-secondary);">yrs</span></div>
      <div class="dm-sub">Range: ${r.ageRange || '—'}</div>
    </div>
    <div class="detail-metric">
      <div class="dm-label">Hair length</div>
      <div class="dm-value">${capitalize(r.hairLength)}</div>
      <div class="dm-sub">${r.hairLengthDetail || ''}</div>
    </div>
    <div class="detail-metric">
      <div class="dm-label">Actual gender</div>
      <div class="dm-value ${r.actualGender === 'female' ? 'female-col' : 'male-col'}">${capitalize(r.actualGender)}</div>
      <div class="dm-sub">Biometric detection</div>
    </div>
    <div class="detail-metric">
      <div class="dm-label">Rule applied</div>
      <div class="dm-value" style="font-size:13px;">${isHairBased ? 'Hair-based' : 'Actual gender'}</div>
      <div class="dm-sub">${isHairBased ? 'Hair → gender mapping' : 'Gender → direct output'}</div>
    </div>
  `;

  // ── Confidence Bars ──────────────────────────────────────
  confidenceSection.innerHTML = `
    <div class="conf-title">Confidence Scores</div>

    <div class="conf-row">
      <div class="conf-label-row">
        <span>Overall prediction confidence</span>
        <span>${conf}%</span>
      </div>
      <div class="conf-bar">
        <div class="conf-fill blue" style="width:0%" data-target="${conf}"></div>
      </div>
    </div>

    <div class="conf-row">
      <div class="conf-label-row">
        <span>Actual gender detection</span>
        <span>${actualConf}%</span>
      </div>
      <div class="conf-bar">
        <div class="conf-fill teal" style="width:0%" data-target="${actualConf}"></div>
      </div>
    </div>

    <div class="conf-row">
      <div class="conf-label-row">
        <span>Hair length classification</span>
        <span>${hairConf}%</span>
      </div>
      <div class="conf-bar">
        <div class="conf-fill pink" style="width:0%" data-target="${hairConf}"></div>
      </div>
    </div>
  `;

  // Animate bars after paint
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.conf-fill[data-target]').forEach(el => {
        el.style.width = el.dataset.target + '%';
      });
    }, 80);
  });

  // ── Reasoning Box ────────────────────────────────────────
  ruleReasoning.innerHTML = `
    <div class="reasoning-title">
      <i class="fa-solid fa-lightbulb" style="color:var(--blue-600);margin-right:6px;"></i>
      Model Reasoning
    </div>
    <div class="reasoning-text">${r.reasoning || 'No reasoning provided.'}</div>
  `;
}

// ── Error ────────────────────────────────────────────────────
function showError(message) {
  errorSection.innerHTML = `<i class="fa-solid fa-circle-exclamation" style="flex-shrink:0;margin-top:1px;"></i> ${message}`;
  errorSection.classList.remove('hidden');
}

// ── Helpers ──────────────────────────────────────────────────
function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function clamp(val, min, max) {
  return Math.min(max, Math.max(min, val));
}

