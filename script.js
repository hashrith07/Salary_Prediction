const API_BASE = (window.location.origin && window.location.origin.includes('onrender.com'))
  ? ''
  : 'https://salary-prediction-k6nv.onrender.com';

const FALLBACK_CATEGORIES = {
  "Gender": ["Female", "Male", "Other"],
  "Education_Level": ["Bachelor's", "High School", "Master's", "PhD"],
  "Job_Title": [
    "Back end Developer", "Content Marketing Manager", "Data Analyst", "Data Scientist",
    "Digital Marketing Manager", "Director of Data Science", "Director of HR", "Director of Marketing",
    "Financial Analyst", "Financial Manager", "Front End Developer", "Front end Developer",
    "Full Stack Engineer", "Human Resources Coordinator", "Human Resources Manager",
    "Junior HR Coordinator", "Junior HR Generalist", "Junior Marketing Manager",
    "Junior Sales Associate", "Junior Sales Representative", "Junior Software Developer",
    "Junior Software Engineer", "Junior Web Developer", "Marketing Analyst", "Marketing Coordinator",
    "Marketing Director", "Marketing Manager", "Operations Manager", "Others", "Product Designer",
    "Product Manager", "Receptionist", "Research Director", "Research Scientist", "Sales Associate",
    "Sales Director", "Sales Executive", "Sales Manager", "Sales Representative",
    "Senior Data Scientist", "Senior HR Generalist", "Senior Human Resources Manager",
    "Senior Product Marketing Manager", "Senior Project Engineer", "Senior Research Scientist",
    "Senior Software Engineer", "Software Developer", "Software Engineer", "Software Engineer Manager",
    "Web Developer"
  ]
};

function spawnParticles() {
  const container = document.getElementById('particles');
  if (!container) return;
  for (let i = 0; i < 18; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 6 + 2;
    p.style.cssText = `width:${size}px;height:${size}px;left:${Math.random()*100}%;animation-duration:${Math.random()*20+15}s;animation-delay:${Math.random()*-25}s;opacity:0;`;
    container.appendChild(p);
  }
}
spawnParticles();

function fmtINR(amount) {
  if (amount >= 1_00_00_000) return `₹${(amount/1_00_00_000).toFixed(2)} Cr`;
  if (amount >= 1_00_000)    return `₹${(amount/1_00_000).toFixed(2)} L`;
  return `₹${amount.toLocaleString('en-IN',{maximumFractionDigits:0})}`;
}

function populateSelect(el, options, sort=false) {
  if (!el) return;
  el.innerHTML = '';
  (sort ? [...options].sort() : options).forEach(opt => {
    const o = document.createElement('option');
    o.value = o.textContent = opt;
    el.appendChild(o);
  });
}

function showError(msg) {
  const errBox = document.getElementById('error-box');
  const errMsg = document.getElementById('error-msg');
  const resCard = document.getElementById('result-card');
  if (errMsg) errMsg.innerHTML = msg;
  if (errBox) errBox.classList.add('show');
  if (resCard) resCard.classList.remove('show');
}
function hideError() {
  const errBox = document.getElementById('error-box');
  if (errBox) errBox.classList.remove('show');
}

const expSlider = document.getElementById('experience');
const expVal    = document.getElementById('exp-val');
function updateSlider() {
  if (!expSlider || !expVal) return;
  const pct = ((expSlider.value - expSlider.min) / (expSlider.max - expSlider.min)) * 100;
  expSlider.style.background = `linear-gradient(90deg,#a855f7 ${pct}%,rgba(168,85,247,.15) ${pct}%)`;
  expVal.textContent = `${parseFloat(expSlider.value).toFixed(1)} yrs`;
}
if (expSlider) {
  expSlider.addEventListener('input', updateSlider);
  updateSlider();
}

(async () => {
  const loader = document.getElementById('init-loader');
  const form   = document.getElementById('predict-form');

  // Immediately populate with fallback so UI is instant and never blocked
  populateSelect(document.getElementById('gender'),    FALLBACK_CATEGORIES.Gender);
  populateSelect(document.getElementById('education'), FALLBACK_CATEGORIES.Education_Level);
  populateSelect(document.getElementById('job-title'), FALLBACK_CATEGORIES.Job_Title, true);

  if (loader) loader.style.display = 'none';
  if (form)   form.style.display   = 'block';

  // Fetch live categories in background if available
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 6000);
    const res = await fetch(`${API_BASE}/categories`, { signal: controller.signal });
    clearTimeout(timer);
    if (res.ok) {
      const cats = await res.json();
      if (cats.Gender) populateSelect(document.getElementById('gender'), cats.Gender);
      if (cats.Education_Level) populateSelect(document.getElementById('education'), cats.Education_Level);
      if (cats.Job_Title) populateSelect(document.getElementById('job-title'), cats.Job_Title, true);
    }
  } catch (e) {
    // Graceful fallback to embedded categories
    console.info("Using embedded categories while cloud API connects.");
  }
})();

document.getElementById('predict-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();
  const btn    = document.getElementById('predict-btn');
  const loader = document.getElementById('predict-loader');
  const card   = document.getElementById('result-card');

  btn.disabled = true;
  btn.textContent = '⚡  Calculating…';
  loader.classList.add('show');
  card.classList.remove('show');

  // Friendly feedback for free-tier Render wakeups
  const wakeUpTimer = setTimeout(() => {
    btn.textContent = '⏳ Waking up cloud server (Render free tier)…';
  }, 3500);

  const payload = {
    age:                 parseInt(document.getElementById('age').value, 10),
    gender:              document.getElementById('gender').value,
    education_level:     document.getElementById('education').value,
    job_title:           document.getElementById('job-title').value,
    years_of_experience: parseFloat(document.getElementById('experience').value),
  };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    clearTimeout(wakeUpTimer);
    loader.classList.remove('show');
    btn.disabled = false;
    btn.innerHTML = '⚡ &nbsp; Calculate My Salary';

    if (res.ok) {
      renderResult(await res.json(), payload);
    } else if (res.status === 422) {
      const err = await res.json();
      showError(Array.isArray(err.detail) ? err.detail.map(d=>`· ${d}`).join('<br>') : String(err.detail));
    } else {
      showError(`Unexpected error (${res.status}). Please try again.`);
    }
  } catch (err) {
    clearTimeout(wakeUpTimer);
    loader.classList.remove('show');
    btn.disabled = false;
    btn.innerHTML = '⚡ &nbsp; Calculate My Salary';
    showError('Connection failed. Server may be starting up — please retry in a moment.');
  }
});

function renderResult(data, payload) {
  const annual  = data.predicted_salary_india_annual_inr;
  const monthly = data.predicted_salary_india_monthly_inr;
  const [low, high] = data.salary_range_inr;

  animateCount('res-annual', annual, fmtINR, '<span class="res-annual-unit">per year</span>');
  document.getElementById('res-monthly').textContent = fmtINR(monthly);
  document.getElementById('res-low').textContent     = fmtINR(low);
  document.getElementById('res-high').textContent    = fmtINR(high);
  document.getElementById('p-role').textContent      = payload.job_title;
  document.getElementById('p-exp').textContent       = `${payload.years_of_experience.toFixed(1)} yrs`;
  document.getElementById('p-edu').textContent       = payload.education_level;
  document.getElementById('res-note').textContent    = data.note || 'Estimates based on aggregated market data';

  const card = document.getElementById('result-card');
  card.classList.add('show');
  const fill = document.getElementById('conf-fill');
  fill.style.width = '0%';
  requestAnimationFrame(() => requestAnimationFrame(() => { fill.style.width = '85%'; }));
  setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'nearest'}), 120);
}

function animateCount(elId, target, formatter, suffix='') {
  const el = document.getElementById(elId);
  if (!el) return;
  const t0 = performance.now();
  const dur = 1400;
  function tick(now) {
    const p = Math.min((now - t0) / dur, 1);
    const e = 1 - Math.pow(1 - p, 4);
    el.innerHTML = formatter(Math.round(e * target)) + suffix;
    if (p < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}
