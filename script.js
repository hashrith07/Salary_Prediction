const API_BASE = 'https://Salary_Prediction-m55r.onrender.com';

function spawnParticles() {
  const container = document.getElementById('particles');
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
  (sort ? [...options].sort() : options).forEach(opt => {
    const o = document.createElement('option');
    o.value = o.textContent = opt;
    el.appendChild(o);
  });
}

function showError(msg) {
  document.getElementById('error-msg').innerHTML = msg;
  document.getElementById('error-box').classList.add('show');
  document.getElementById('result-card').classList.remove('show');
}
function hideError() { document.getElementById('error-box').classList.remove('show'); }

const expSlider = document.getElementById('experience');
const expVal    = document.getElementById('exp-val');
function updateSlider() {
  const pct = ((expSlider.value - expSlider.min) / (expSlider.max - expSlider.min)) * 100;
  expSlider.style.background = `linear-gradient(90deg,#a855f7 ${pct}%,rgba(168,85,247,.15) ${pct}%)`;
  expVal.textContent = `${parseFloat(expSlider.value).toFixed(1)} yrs`;
}
expSlider.addEventListener('input', updateSlider);
updateSlider();

(async () => {
  let cats;
  try {
    const res = await fetch(`${API_BASE}/categories`);
    if (!res.ok) throw new Error();
    cats = await res.json();
  } catch { cats = null; }

  const loader = document.getElementById('init-loader');
  const form   = document.getElementById('predict-form');

  if (!cats) {
    loader.innerHTML = '';
    showError('⚡ Could not connect to the prediction API.<br>Please wait 30 seconds and refresh.');
    return;
  }
  populateSelect(document.getElementById('gender'),    cats.Gender         || []);
  populateSelect(document.getElementById('education'), cats.Education_Level || []);
  populateSelect(document.getElementById('job-title'), cats.Job_Title       || [], true);
  loader.style.display = 'none';
  form.style.display   = 'block';
})();

document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();
  const btn    = document.getElementById('predict-btn');
  const loader = document.getElementById('predict-loader');
  const card   = document.getElementById('result-card');

  btn.disabled = true;
  btn.textContent = '⚡  Calculating…';
  loader.classList.add('show');
  card.classList.remove('show');

  const payload = {
    age:                 parseInt(document.getElementById('age').value, 10),
    gender:              document.getElementById('gender').value,
    education_level:     document.getElementById('education').value,
    job_title:           document.getElementById('job-title').value,
    years_of_experience: parseFloat(document.getElementById('experience').value),
  };

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload),
    });
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
  } catch {
    loader.classList.remove('show');
    btn.disabled = false;
    btn.innerHTML = '⚡ &nbsp; Calculate My Salary';
    showError('Connection failed. Please retry in 30 seconds.');
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