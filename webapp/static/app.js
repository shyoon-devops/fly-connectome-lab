const $ = id => document.getElementById(id);
const colors = [[.18,.62,.48],[.25,.48,.82],[.65,.38,.73],[.78,.45,.34],[.90,.70,.28],[.88,.28,.38],[.38,.47,.43]];
const state = { meta:null, ws:null, running:false, game:false, physicsLive:false, physicsSeen:0, physicsTick:null, physicsProgressSeen:0, physicsResendAt:0, positions:null, regions:null, gl:null, baseColors:null, liveColors:null, previousTop:[], history:[], yaw:0, pitch:0, zoom:.82, dragging:false, autoRotate:localStorage.getItem('brainAutoRotate')!=='off', currentRegions:[], currentStimulusActivity:{}, currentTick:0, stimulusResponse:null, lastStimulusMessage:null, regionHighlight:[] };

const stimulusHelp = {
  visual:'눈과 연결된 감각 뉴런에서 신호가 시작됩니다.',
  olfactory:'더듬이의 냄새 감각 뉴런에서 신호가 시작됩니다.',
  touch:'몸과 다리에 닿은 느낌을 담당하는 뉴런에서 시작됩니다.',
  proprioception:'다리가 어디에 있고 얼마나 움직이는지 알려주는 뉴런에서 시작됩니다.',
  taste:'입과 다리의 맛 감각 뉴런에서 신호가 시작됩니다.',
  temperature:'뜨겁고 차가운 변화를 느끼는 뉴런에서 시작됩니다.',
  humidity:'공기의 습도를 느끼는 뉴런에서 시작됩니다.'
};
const regionHelp = {
  'optic lobe':'눈에서 들어온 정보를 처음 크게 처리하는 곳',
  'central brain':'여러 감각을 모아 판단과 행동을 준비하는 곳',
  'ventral nerve cord':'몸과 여섯 다리의 움직임에 가까운 곳',
  'sensory':'바깥세상의 느낌이 처음 들어오는 감각 뉴런',
  'ascending':'몸에서 생긴 신호를 뇌 쪽으로 올려 보내는 뉴런',
  'descending/motor':'뇌의 결정을 몸과 다리로 내려보내는 운동 뉴런',
  'other':'위 큰 분류에 포함되지 않은 나머지 뉴런'
};

function fmt(n){ return Number(n).toLocaleString('ko-KR'); }
async function boot(){
  state.meta = await fetch('/api/meta').then(r=>r.json());
  state.regionHighlight=state.meta.regions.map(()=>0); state.currentRegions=state.meta.regions.map(()=>0);
  $('neurons').textContent=fmt(state.meta.neurons); $('connections').textContent=fmt(state.meta.connections); $('synapses').textContent=fmt(state.meta.synapses);
  buildStimuli(); buildRegions(); await loadBrain(); connect(); bindControls(); drawLoop();
}
function buildStimuli(){
  const box=$('stimulusButtons');
  state.meta.stimuli.forEach(s=>{ const b=document.createElement('button'); b.dataset.key=s.key; b.innerHTML=`<span>${s.label}</span><small>${fmt(s.count)}</small>`; b.onclick=()=>stimulate(s.key,b); box.appendChild(b); });
}
function buildRegions(){
  const box=$('regionBars'); state.meta.regions.forEach((name,i)=>{ const row=document.createElement('div');row.className='region-row';row.innerHTML=`<div><span>${name}</span><output id="regionValue${i}">0.000</output></div><div class="bar"><i id="regionBar${i}"></i></div>`;box.appendChild(row); });
}
async function loadBrain(){
  const buffer=await fetch('/api/positions').then(r=>r.arrayBuffer()), n=state.meta.neurons, offset=n*3*4;
  state.positions=new Float32Array(buffer,0,n*3); state.regions=new Uint8Array(buffer,offset,n); initGL(); $('loadingBrain').classList.add('hidden');
}
function initGL(){
  const canvas=$('brainCanvas'), gl=canvas.getContext('webgl',{antialias:true,alpha:true}); state.gl=gl;
  const vs=`attribute vec3 p;attribute vec3 c;uniform float yaw;uniform float pitch;uniform float zoom;varying vec3 col;void main(){float cy=cos(yaw),sy=sin(yaw),cp=cos(pitch),sp=sin(pitch);vec3 q=vec3(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z);q=vec3(q.x,cp*q.y-sp*q.z,sp*q.y+cp*q.z);gl_Position=vec4(q.x*zoom,q.y*zoom,q.z*.08,1.0);gl_PointSize=2.2;col=c;}`;
  const fs=`precision mediump float;varying vec3 col;void main(){vec2 d=gl_PointCoord-.5;if(dot(d,d)>.25)discard;gl_FragColor=vec4(col,0.9);}`;
  const shader=(type,src)=>{const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);return s};const prog=gl.createProgram();gl.attachShader(prog,shader(gl.VERTEX_SHADER,vs));gl.attachShader(prog,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(prog);gl.useProgram(prog);state.program=prog;
  state.posBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,state.posBuffer);gl.bufferData(gl.ARRAY_BUFFER,state.positions,gl.STATIC_DRAW);const p=gl.getAttribLocation(prog,'p');gl.enableVertexAttribArray(p);gl.vertexAttribPointer(p,3,gl.FLOAT,false,0,0);
  state.baseColors=new Float32Array(state.meta.neurons*3);for(let i=0;i<state.meta.neurons;i++){const c=colors[state.regions[i]]||colors[6];state.baseColors.set(c.map(v=>v*.42),i*3)}state.liveColors=state.baseColors.slice();state.colorBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,state.colorBuffer);gl.bufferData(gl.ARRAY_BUFFER,state.liveColors,gl.DYNAMIC_DRAW);const c=gl.getAttribLocation(prog,'c');gl.enableVertexAttribArray(c);gl.vertexAttribPointer(c,3,gl.FLOAT,false,0,0);
  state.uYaw=gl.getUniformLocation(prog,'yaw');state.uPitch=gl.getUniformLocation(prog,'pitch');state.uZoom=gl.getUniformLocation(prog,'zoom');gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE);bindBrainMouse(canvas);
}
function bindBrainMouse(c){let x=0,y=0;c.onmousedown=e=>{state.dragging=true;x=e.clientX;y=e.clientY};window.onmouseup=()=>state.dragging=false;window.onmousemove=e=>{if(!state.dragging)return;state.yaw+=(e.clientX-x)*.008;state.pitch+=(e.clientY-y)*.008;x=e.clientX;y=e.clientY};c.onwheel=e=>{e.preventDefault();state.zoom=Math.max(.25,Math.min(1.8,state.zoom-e.deltaY*.0008))};}
function drawLoop(){
  if(state.gl){const gl=state.gl,c=$('brainCanvas'),d=devicePixelRatio||1,w=Math.floor(c.clientWidth*d),h=Math.floor(c.clientHeight*d);if(c.width!==w||c.height!==h){c.width=w;c.height=h;gl.viewport(0,0,w,h)}if(state.autoRotate&&!state.dragging)state.yaw+=.0012;gl.clearColor(.02,.05,.04,1);gl.clear(gl.COLOR_BUFFER_BIT);gl.useProgram(state.program);gl.uniform1f(state.uYaw,state.yaw);gl.uniform1f(state.uPitch,state.pitch);gl.uniform1f(state.uZoom,state.zoom);gl.bindBuffer(gl.ARRAY_BUFFER,state.posBuffer);let p=gl.getAttribLocation(state.program,'p');gl.vertexAttribPointer(p,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,state.colorBuffer);let a=gl.getAttribLocation(state.program,'c');gl.vertexAttribPointer(a,3,gl.FLOAT,false,0,0);gl.drawArrays(gl.POINTS,0,state.meta.neurons)} requestAnimationFrame(drawLoop);
}
function connect(){
  const proto=location.protocol==='https:'?'wss':'ws';state.ws=new WebSocket(`${proto}://${location.host}/ws/sim`);
  state.ws.onopen=()=>{$('connection').className='status live';$('connection').innerHTML='<i></i> 전체 뇌 연결됨'};
  state.ws.onclose=()=>{$('connection').className='status';$('connection').innerHTML='<i></i> 재연결 중';setTimeout(connect,1500)};
  state.ws.onmessage=e=>{const m=JSON.parse(e.data),now=Date.now(),physicsAdvancing=state.physicsLive&&now-state.physicsSeen<1500&&(!state.running||now-state.physicsProgressSeen<750);if(m.type==='state'&&!physicsAdvancing)update(m)};
}
function send(v){if(state.ws?.readyState===1)state.ws.send(JSON.stringify(v));}
function sendPhysics(v){const frame=$('nmfFrame');if(frame?.contentWindow)frame.contentWindow.postMessage({type:'nmf-parent-command',message:v},location.origin)}
function sendEverywhere(v){send(v);sendPhysics(v)}
function stimulate(key,b){document.querySelectorAll('.stimuli button').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.running=true;$('pauseButton').textContent='일시정지';const duration=+$('duration').value,intensity=+$('intensity').value,label=state.meta.stimuli.find(s=>s.key===key)?.label||key;const sourceBaseline=Number(state.currentStimulusActivity[key]||0);state.stimulusResponse={key,label,intensity,duration,startTick:state.currentTick,baseline:state.currentRegions.slice(),peak:state.meta.regions.map(()=>0),current:state.meta.regions.map(()=>0),sourceBaseline,sourceCurrent:0,sourcePeak:0,complete:false};renderStimulusInsight();$('brainCanvas').classList.remove('stimulus-pulse');void $('brainCanvas').offsetWidth;$('brainCanvas').classList.add('stimulus-pulse');const msg={type:'stimulus',key,intensity,duration};state.lastStimulusMessage={...msg,sentAt:Date.now()};sendEverywhere(msg);sendEverywhere({type:'run',value:true});}
function bindControls(){
  $('intensity').oninput=e=>$('intensityValue').textContent=(+e.target.value).toFixed(2);$('duration').oninput=e=>$('durationValue').textContent=e.target.value;
  $('autoRotateButton').onclick=()=>{state.autoRotate=!state.autoRotate;localStorage.setItem('brainAutoRotate',state.autoRotate?'on':'off');updateRotationButton()};
  $('resetBrainView').onclick=()=>{state.yaw=0;state.pitch=0;state.zoom=.82}; updateRotationButton();
  $('pauseButton').onclick=()=>{state.running=!state.running;sendEverywhere({type:'run',value:state.running});$('pauseButton').textContent=state.running?'일시정지':'계속 실행'};
  $('resetButton').onclick=()=>{state.stimulusResponse=null;state.lastStimulusMessage=null;state.regionHighlight.fill(0);renderStimulusInsight();document.querySelectorAll('.region-row').forEach(x=>x.classList.remove('response-hot','response-primary'));sendEverywhere({type:'reset_brain'})};
  $('gameButton').onclick=()=>{state.game=!state.game;if(state.game){state.running=true;$('pauseButton').textContent='일시정지';sendEverywhere({type:'run',value:true})}sendEverywhere({type:'game',running:state.game,learning:$('learningToggle').checked});$('gameButton').textContent=state.game?'게임 정지':'게임 시작';$('gameButton').classList.toggle('active',state.game)};
  $('learningToggle').onchange=e=>sendEverywhere({type:'game',running:state.game,learning:e.target.checked});$('resetLearning').onclick=()=>sendEverywhere({type:'reset_learning'});
  window.addEventListener('message',e=>{if(e.origin!==location.origin)return;if(e.data?.type==='nmf-run-state'){const wasRunning=state.running;state.running=!!e.data.running;if(wasRunning!==state.running)send({type:'run',value:state.running});const phase=e.data.phase;$('pauseButton').textContent=phase==='ready'?'시뮬레이션 시작':phase==='countdown'?'시작 준비 중':phase==='finished'?'다시 시작':state.running?'일시정지':'계속 실행';return}if(e.data?.type!=='nmf-brain-state')return;const m=e.data.state,now=Date.now(),first=!state.physicsLive,restarted=state.physicsTick!==null&&m.tick<state.physicsTick,advanced=state.physicsTick===null||m.tick!==state.physicsTick;state.physicsLive=true;state.physicsSeen=now;if(advanced){state.physicsTick=m.tick;state.physicsProgressSeen=now}const stalled=state.running&&now-state.physicsProgressSeen>500;if(state.running&&(first||restarted||stalled)&&now-state.physicsResendAt>500){state.physicsResendAt=now;if(state.lastStimulusMessage&&now-state.lastStimulusMessage.sentAt<15000)sendPhysics(state.lastStimulusMessage);sendPhysics({type:'run',value:true})}if(!state.running||advanced)update(m)});
}
function updateRotationButton(){const b=$('autoRotateButton');b.classList.toggle('active',state.autoRotate);b.setAttribute('aria-pressed',String(state.autoRotate));b.textContent=state.autoRotate?'↻ 자동 회전 켜짐':'Ⅱ 자동 회전 꺼짐';}
function update(m){
  $('tick').textContent=fmt(m.tick);$('activeCount').textContent=fmt(m.active_neurons);state.currentTick=m.tick;updateStimulusResponse(m);state.currentRegions=m.region_activity.slice();state.currentStimulusActivity={...(m.stimulus_activity||{})};
  const ranked=state.regionHighlight.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v);m.region_activity.forEach((v,i)=>{const row=$(`regionValue${i}`).closest('.region-row');row.classList.toggle('response-primary',ranked[0]?.i===i&&ranked[0].v>.05);row.classList.toggle('response-hot',ranked.slice(1,3).some(x=>x.i===i&&x.v>.05));$(`regionValue${i}`).textContent=v.toFixed(4);$(`regionBar${i}`).style.width=`${Math.min(100,v*180)}%`});
  state.history.push(m.region_activity);if(state.history.length>150)state.history.shift();drawActivity();updateBrain(m.top);updateTable(m.top);drawGame(m.game);updateScore(m.game);
}
function updateStimulusResponse(m){const r=state.stimulusResponse;if(!r)return;r.current=m.region_activity.map((v,i)=>Math.max(0,v-r.baseline[i]));r.current.forEach((v,i)=>r.peak[i]=Math.max(r.peak[i],v));r.sourceCurrent=Math.max(0,Number(m.stimulus_activity?.[r.key]||0)-r.sourceBaseline);r.sourcePeak=Math.max(r.sourcePeak,r.sourceCurrent);const elapsed=Math.max(0,m.tick-r.startTick);r.complete=m.stimulus_remaining===0&&elapsed>r.duration+12;const source=r.complete?r.peak:r.current,max=Math.max(...source,0.00001);state.regionHighlight=source.map(v=>v/max*(r.complete?.38:1));renderStimulusInsight();}
function renderStimulusInsight(){const panel=$('stimulusInsight'),r=state.stimulusResponse;if(!r){panel.className='stimulus-insight idle';panel.innerHTML='<div class="insight-kicker">자극 변화 안내</div><h2>왼쪽에서 자극을 눌러보세요</h2><p>누르기 전과 후를 자동으로 비교해서, 가장 많이 변한 뇌 영역을 여기에 바로 알려드립니다.</p><div class="insight-placeholder">자극 전 저장 → 신호 전파 관찰 → 가장 큰 변화 표시</div>';return}const values=r.complete?r.peak:r.current,sourceDelta=r.complete?r.sourcePeak:r.sourceCurrent,ranked=values.map((v,i)=>({i,v,base:r.baseline[i],now:r.baseline[i]+(r.complete?r.peak[i]:r.current[i])})).sort((a,b)=>b.v-a.v).slice(0,3),max=Math.max(ranked[0]?.v||0,0.00001),primary=ranked[0],hasRegionSignal=primary&&primary.v>.00005,hasDirectSignal=sourceDelta>.00005,primaryName=hasRegionSignal?state.meta.regions[primary.i]:'영역 평균 변화 작음',phase=r.complete?'관찰 완료 · 가장 컸던 변화':'실시간 비교 중 · 노란빛이 강할수록 큰 변화',heading=hasRegionSignal?`${r.label} → ${primaryName} 영역이 가장 크게 변함`:hasDirectSignal?`${r.label} 감각뉴런에 직접 신호 확인`:`${r.label} 신호가 들어가는 중…`;panel.className=`stimulus-insight ${r.complete?'complete':'live'}`;panel.innerHTML=`<div class="insight-kicker"><i></i>${r.label} 자극 · ${phase}</div><h2>${heading}</h2><p>${stimulusHelp[r.key]||'관련 감각 뉴런에서 신호가 시작됩니다.'} ${hasRegionSignal?(regionHelp[primaryName]||'연결된 뉴런이 반응하는 곳')+'이 지금 가장 크게 변했습니다.':hasDirectSignal?'큰 영역 전체로 평균내면 작아 보이지만, 자극 대상 뉴런은 분명히 반응했습니다.':'첫 전파 결과를 기다리는 중입니다.'}</p><div class="insight-source ${hasDirectSignal?'hot':''}"><b>직접 자극 감각뉴런</b><span>${r.sourceBaseline.toFixed(4)} → ${(r.sourceBaseline+sourceDelta).toFixed(4)} <em>+${sourceDelta.toFixed(4)}</em></span></div><div class="change-list">${ranked.map((x,n)=>`<div class="change-item ${n===0?'first':''}"><div><b>${n+1}. ${state.meta.regions[x.i]}</b><span>${x.base.toFixed(4)} → ${x.now.toFixed(4)} <em>+${x.v.toFixed(4)}</em></span></div><i style="width:${Math.max(3,x.v/max*100)}%"></i></div>`).join('')}</div><div class="insight-foot">자극 세기 ${r.intensity.toFixed(1)} · 직접 뉴런과 뇌 영역을 따로 측정</div>`;}
function updateBrain(top){if(!state.gl)return;for(let i=0;i<state.meta.neurons;i++){const k=i*3,boost=state.regionHighlight[state.regions[i]]||0;if(boost>.01){state.liveColors[k]=Math.min(1,state.baseColors[k]*(1-boost*.35)+1.0*boost*.72);state.liveColors[k+1]=Math.min(1,state.baseColors[k+1]*(1-boost*.35)+.72*boost*.72);state.liveColors[k+2]=Math.min(1,state.baseColors[k+2]*(1-boost*.35)+.12*boost*.72)}else{state.liveColors[k]=state.baseColors[k];state.liveColors[k+1]=state.baseColors[k+1];state.liveColors[k+2]=state.baseColors[k+2]}}top.indices.forEach((idx,j)=>{const v=top.activity[j],mag=Math.min(1,Math.abs(v));state.liveColors.set(v>=0?[1,.22+.7*mag,.12]:[.12,.45+.5*mag,1],idx*3)});state.previousTop=top.indices;state.gl.bindBuffer(state.gl.ARRAY_BUFFER,state.colorBuffer);state.gl.bufferSubData(state.gl.ARRAY_BUFFER,0,state.liveColors);}
function updateTable(top){const body=$('topNeurons');body.innerHTML='';for(let i=0;i<Math.min(14,top.indices.length);i++){const tr=document.createElement('tr');tr.innerHTML=`<td title="body ${top.body_ids[i]}">${top.types[i]}</td><td>${top.superclasses[i]}</td><td>${top.activity[i].toFixed(3)}</td>`;body.appendChild(tr)}}
function drawActivity(){const c=$('activityChart'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.clearRect(0,0,w,h);ctx.strokeStyle='#20322c';for(let y=20;y<h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}state.meta.regions.forEach((_,r)=>{ctx.strokeStyle=`rgb(${colors[r].map(v=>v*255).join(',')})`;ctx.beginPath();state.history.forEach((row,i)=>{const x=i/(149)*w,y=h-Math.min(h-2,row[r]*260);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()})}
function drawGame(g){const c=$('gameCanvas'),ctx=c.getContext('2d'),w=c.width,h=c.height,centers=[[190,105],[570,105],[190,265],[570,265]];ctx.fillStyle='#070d0b';ctx.fillRect(0,0,w,h);centers.forEach((p,i)=>{ctx.strokeStyle=i===g.action?'#fff':`rgb(${colors[i].map(v=>v*255).join(',')})`;ctx.lineWidth=i===g.action?8:3;ctx.beginPath();ctx.arc(p[0],p[1],55,0,Math.PI*2);ctx.stroke();ctx.fillStyle='#687d75';ctx.font='12px Segoe UI';ctx.fillText(`MOTOR ${i+1}`,p[0]-27,p[1]+4)});if(g.running){const p=centers[g.note_lane],radius=120-g.note_phase*65;ctx.strokeStyle='#58f2b0';ctx.lineWidth=7;ctx.beginPath();ctx.arc(p[0],p[1],radius,0,Math.PI*2);ctx.stroke()}else{ctx.fillStyle='#91a69e';ctx.font='20px Segoe UI';ctx.fillText('게임 시작을 누르면 전체 뇌 폐루프가 시작됩니다.',165,185)}}
function updateScore(g){$('score').textContent=String(g.score).padStart(7,'0');$('combo').textContent=g.combo;$('accuracy').textContent=`${(g.rolling_accuracy*100).toFixed(1)}%`;$('trials').textContent=g.trials;$('hits').textContent=g.hits;$('reward').textContent=g.reward.toFixed(2);}
boot().catch(e=>{$('connection').textContent='초기화 실패: '+e.message;console.error(e)});
