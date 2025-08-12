// frontend/app.js

const canvas = document.getElementById('wheelCanvas');
const ctx = canvas.getContext('2d');
const predictBtn = document.getElementById('predictBtn');
const syncBtn = document.getElementById('syncBtn');
const predBox = document.getElementById('predBox');
const historyEl = document.getElementById('history');
const statusEl = document.getElementById('wheelStatus');

const size = Math.min(window.innerWidth, 720, 600);
canvas.width = size;
canvas.height = size;

const centerX = canvas.width/2;
const centerY = canvas.height/2;
const radius = Math.min(centerX, centerY) - 8;

// порядок секторов — должен совпадать с backend.sector_order
const sectors = ['one','two','five','ten','coinflip','cashhunt','pachinko','crazytime'];
const sectorColors = ['#FF8A65','#FFB74D','#FFD54F','#4FC3F7','#A1887F','#81C784','#BA68C8','#F06292'];

function drawWheel(rotationDeg=0){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    const slices = sectors.length;
    const anglePer = (Math.PI*2)/slices;
    ctx.save();
    ctx.translate(centerX, centerY);
    ctx.rotate(rotationDeg * Math.PI/180);
    // draw slices
    for(let i=0;i<slices;i++){
        const start = i * anglePer;
        ctx.beginPath();
        ctx.moveTo(0,0);
        ctx.arc(0,0, radius, start, start + anglePer);
        ctx.closePath();
        ctx.fillStyle = sectorColors[i % sectorColors.length];
        ctx.fill();
        // label
        ctx.save();
        ctx.rotate(start + anglePer/2);
        ctx.translate(radius*0.62, 0);
        ctx.rotate(Math.PI/2);
        ctx.fillStyle = "#0b1220";
        ctx.font = `${Math.max(14, radius*0.08)}px sans-serif`;
        ctx.textAlign = "center";
        ctx.fillText(sectors[i].toUpperCase(), 0, 0);
        ctx.restore();
    }
    // center circle
    ctx.beginPath();
    ctx.arc(0,0, radius*0.2, 0, Math.PI*2);
    ctx.fillStyle = "#0b1220";
    ctx.fill();
    ctx.restore();

    // pointer
    ctx.beginPath();
    ctx.moveTo(centerX, centerY - radius - 10);
    ctx.lineTo(centerX - 12, centerY - radius + 22);
    ctx.lineTo(centerX + 12, centerY - radius + 22);
    ctx.closePath();
    ctx.fillStyle = "#FFD54F";
    ctx.fill();
}

let isSpinning = false;
let currentRotation = 0; // degrees

// easing for animation (cubic)
function easeOutCubic(t){ return 1 - Math.pow(1 - t, 3); }

function spinToSector(targetSector, predictedConfidence=0.5){
    if(isSpinning) return;
    isSpinning = true;
    statusEl.textContent = `Крутим к сектору: ${targetSector} (уверенность ${Math.round(predictedConfidence*100)}%)`;
    // which sector index
    const idx = sectors.indexOf(targetSector);
    if(idx === -1){
        // fallback random
        idx = Math.floor(Math.random()*sectors.length);
    }
    // we want wheel to stop so that pointer (top) points to target sector.
    // In drawWheel rotation 0 means sector 0 at original orientation.
    // compute angle to rotate to bring that sector to top (pointer at 0 degrees)
    const slices = sectors.length;
    const anglePer = 360 / slices;
    // sector center angle in current wheel coordinates: sector i is centered at (i*anglePer + anglePer/2)
    const targetCenterAngle = idx * anglePer + anglePer/2;
    // to bring that to top (0 deg), we need rotation so that targetCenterAngle + rotation ≡ 0 (mod 360)
    // so rotation = -targetCenterAngle  (but we'll add full spins)
    const spins = 5 + Math.floor(Math.random()*3); // 5-7 full spins
    const finalRotation = spins * 360 - targetCenterAngle;
    const startRotation = currentRotation % 360;
    const delta = (finalRotation - startRotation);
    const duration = 4200 + Math.floor(Math.random()*1600); // 4.2-5.8s

    const start = performance.now();
    function frame(now){
        let t = (now - start) / duration;
        if(t >= 1) t = 1;
        const eased = easeOutCubic(t);
        const rot = startRotation + delta * eased;
        currentRotation = rot;
        drawWheel(rot);
        if(t < 1){
            requestAnimationFrame(frame);
        } else {
            isSpinning = false;
            statusEl.textContent = `Остановлено на: ${targetSector}`;
            // add to history
            addHistory({prediction: targetSector, confidence: predictedConfidence, timestamp: new Date().toISOString()});
        }
    }
    requestAnimationFrame(frame);
}

function addHistory(item){
    const div = document.createElement('div');
    div.className = 'hist-item';
    div.innerHTML = `<div>${item.prediction.toUpperCase()} — ${Math.round(item.confidence*100)}%</div><div>${new Date(item.timestamp).toLocaleString()}</div>`;
    historyEl.prepend(div);
    // keep only 50
    while(historyEl.children.length > 50) historyEl.removeChild(historyEl.lastChild);
}

async function requestPrediction(){
    predBox.textContent = "Запрос прогноза...";
    try {
        const res = await fetch('/api/signals?count=1');
        if(!res.ok) throw new Error('network');
        const data = await res.json();
        const p = data.predictions[0];
        predBox.innerHTML = `<strong>${p.prediction.toUpperCase()}</strong><div>Уверенность: ${Math.round(p.confidence*100)}%</div>`;
        spinToSector(p.prediction, p.confidence || 0.5);
    } catch(e){
        // fallback
        const choices = ['one','two','five','ten','coinflip','cashhunt','pachinko','crazytime'];
        const pick = choices[Math.floor(Math.random()*choices.length)];
        predBox.innerHTML = `<strong>${pick.toUpperCase()}</strong><div>Уверенность: 50%</div>`;
        spinToSector(pick, 0.5);
    }
}

// sync with last real spin (just visual; not predicting)
async function syncLastSpin(){
    statusEl.textContent = "Получаем последний результат...";
    try {
        const res = await fetch('/api/last-spin');
        const data = await res.json();
        if(data.last && data.last.spinResultSymbol){
            // animate quickly to real sector (1 spin)
            spinToSector(data.last.spinResultSymbol, 1.0);
        } else {
            statusEl.textContent = "Последний результат недоступен";
        }
    } catch(e){
        statusEl.textContent = "Ошибка получения последнего результата";
    }
}

predictBtn.addEventListener('click', requestPrediction);
syncBtn.addEventListener('click', syncLastSpin);

// draw initial wheel
drawWheel(0);
