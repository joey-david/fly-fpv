/* flypv live monitor — world-space flight, connectome activity and learning. */
const ROLE = [
  ["Intrinsic", "#465671"], ["Retina", "#6cd9e7"], ["Sensory", "#8d7bf6"],
  ["Motor", "#f36a7f"], ["Descending", "#f1c46b"], ["Ascending", "#60c898"],
];
const CURVES = [
  ["ep_return", "#6cd9e7", "Return"], ["ep_gates", "#60c898", "Gates"],
  ["crash_rate", "#f36a7f", "Crash rate"], ["entropy", "#8d7bf6", "Entropy"],
  ["difficulty", "#f1c46b", "Difficulty"],
];
const S = {
  graph: null, eye: null, info: null, act: null, actHi: 1,
  eyeFrame: null, flight: null, pools: null, live: {}, metrics: {}, hist: {},
};
const el = (id) => document.getElementById(id);
const DPR = () => Math.min(devicePixelRatio || 1, 2.5);
const fmt = (v, d = 2) => v == null || Number.isNaN(v) ? "—" :
  Math.abs(v) >= 10000 ? v.toLocaleString(undefined, {maximumFractionDigits: 0}) : v.toFixed(d);
const b64 = (s) => {
  const bin = atob(s), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
};
function fitCanvas(c) {
  const r = c.getBoundingClientRect(), d = DPR();
  const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; return true; }
  return false;
}

let lastStep = -1;
async function seedHistory() {
  try {
    const r = await fetch("/api/series"), d = await r.json();
    for (const [k] of CURVES) if (d[k]) S.hist[k] = d[k].slice(-700);
    if (d.step?.length) lastStep = d.step[d.step.length - 1];
  } catch (_) {}
}
function pushHistory(m) {
  if (m.step == null || m.step === lastStep) return;
  lastStep = m.step;
  for (const [k] of CURVES) {
    if (m[k] == null) continue;
    (S.hist[k] ||= []).push(m[k]);
    if (S.hist[k].length > 700) S.hist[k].shift();
  }
}
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { el("conn").classList.add("live"); el("conn").querySelector("span").textContent = "live"; seedHistory(); };
  ws.onclose = () => {
    el("conn").classList.remove("live"); el("conn").querySelector("span").textContent = "reconnecting";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.kind === "static") {
      if (m.graph) { S.graph = m.graph; prepareGraph(); }
      if (m.eye) { S.eye = m.eye; prepareEye(); }
      if (m.info) { S.info = m.info; renderHeader(); renderKey(); }
      return;
    }
    if (m.act) { S.act = b64(m.act.data); S.actHi = m.act.hi; }
    if (m.eye_frame) S.eyeFrame = m.eye_frame;
    if (m.flight) S.flight = m.flight;
    if (m.pools) S.pools = m.pools;
    if (m.live) S.live = m.live;
    if (m.metrics) { S.metrics = m.metrics; pushHistory(m.metrics); }
  };
}

function renderHeader() {
  const i = S.info;
  const temporal = i.recurrent ? `persistent · TBPTT ${i.tbptt_steps} (${Math.round(1000 * i.tbptt_steps / i.control_hz)} ms)` : "stateless ablation";
  el("dataset").textContent = `${i.dataset} · ${i.neurons.toLocaleString()} neurons · ${i.connections.toLocaleString()} edges · ${temporal}`;
  el("graph-note").textContent = `drag to rotate · ${i.displayed.toLocaleString()} / ${i.neurons.toLocaleString()} neurons shown · activity is live`;
}
function renderCounters() {
  const m = S.metrics, l = S.live, i = S.info || {};
  const rows = [
    ["env steps", m.step ? Math.round(m.step).toLocaleString() : "—"],
    ["return", fmt(m.ep_return)], ["gates", `${fmt(m.ep_gates, 2)} / ${i.n_gates ?? "—"}`],
    ["crash", m.crash_rate == null ? "—" : `${Math.round(m.crash_rate * 100)}%`],
    ["difficulty", fmt(m.difficulty, 2)], ["airspeed", l.speed == null ? "—" : `${fmt(l.speed * 100, 1)} cm/s`],
    ["power", l.power_rel == null ? "—" : `${fmt(l.power_rel, 2)}× hover`], ["steps/s", fmt(m.fps, 0)],
  ];
  el("counters").innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}
function renderKey() {
  el("key").innerHTML = ROLE.map(([n, c], i) => {
    const count = S.graph ? S.graph.role.filter((r) => r === i).length : 0;
    return `<li><b style="background:${c}"></b>${n}<span>${count.toLocaleString()}</span></li>`;
  }).join("") + `<li><b style="background:#f0c86b"></b>exc</li><li><b style="background:#6594ff"></b>inh</li>`;
}

const add = (a,b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const sub = (a,b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const mul = (a,s) => [a[0]*s, a[1]*s, a[2]*s];
const dot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const mag = (a) => Math.hypot(a[0],a[1],a[2]);
const unit = (a) => { const m = mag(a) || 1; return mul(a, 1/m); };
const bodyToWorld = (R, v) => [
  R[0]*v[0]+R[1]*v[1]+R[2]*v[2],
  R[3]*v[0]+R[4]*v[1]+R[5]*v[2],
  R[6]*v[0]+R[7]*v[1]+R[8]*v[2],
];
function rotY(v, a) { const c=Math.cos(a), s=Math.sin(a); return [c*v[0]+s*v[2], v[1], -s*v[0]+c*v[2]]; }
function rodrigues(v, axis, a) {
  const c=Math.cos(a), s=Math.sin(a), k=unit(axis);
  return add(add(mul(v,c), mul(cross(k,v),s)), mul(k, dot(k,v)*(1-c)));
}
function perp(n) { const a = Math.abs(n[2]) < .85 ? [0,0,1] : [1,0,0]; return unit(cross(n,a)); }

function cameraFor(fl, w, h) {
  const R = fl.R, fwd = unit([R[0], R[3], R[6]]), worldUp = [0,0,1];
  const cam = add(add(fl.pos, mul(fwd, -0.036)), [0,0,0.016]);
  const target = add(fl.pos, mul(fwd, 0.075));
  const z = unit(sub(target, cam));
  let x = unit(cross(z, worldUp));
  if (mag(x) < 1e-4) x = [0,1,0];
  const y = unit(cross(x, z));
  return {p: cam, x, y, z, f: h * 0.92, cx: w*0.5, cy: h*0.53};
}
function project3(p, cam) {
  const q=sub(p,cam.p), z=dot(q,cam.z);
  if (z < .002) return null;
  return {x:cam.cx+cam.f*dot(q,cam.x)/z, y:cam.cy-cam.f*dot(q,cam.y)/z, z};
}
function line3(ctx, a, b, cam) {
  const p=project3(a,cam), q=project3(b,cam); if (!p || !q) return false;
  ctx.moveTo(p.x,p.y); ctx.lineTo(q.x,q.y); return true;
}
function drawGround(ctx, cam, fl, w, h) {
  const grad=ctx.createLinearGradient(0,0,0,h); grad.addColorStop(0,"#0c1420"); grad.addColorStop(.52,"#0b1016"); grad.addColorStop(1,"#050608");
  ctx.fillStyle=grad; ctx.fillRect(0,0,w,h);
  const gx=Math.round(fl.pos[0]/.1)*.1, gy=Math.round(fl.pos[1]/.1)*.1;
  ctx.strokeStyle="rgba(100,130,160,.12)"; ctx.lineWidth=1;
  for (let i=-10;i<=10;i++) {
    ctx.beginPath(); line3(ctx,[gx+i*.1,gy-1,0],[gx+i*.1,gy+1,0],cam); ctx.stroke();
    ctx.beginPath(); line3(ctx,[gx-1,gy+i*.1,0],[gx+1,gy+i*.1,0],cam); ctx.stroke();
  }
}
function drawGate(ctx, co, i, cam, active) {
  const c=co.center[i], n=unit(co.normal[i]), u=perp(n), v=unit(cross(n,u));
  const ro=co.r_out[i], ri=co.r_in[i], seg=56;
  const outer=[], inner=[];
  for (let k=0;k<=seg;k++) {
    const a=2*Math.PI*k/seg, d=add(mul(u,Math.cos(a)),mul(v,Math.sin(a)));
    outer.push(project3(add(c,mul(d,ro)),cam)); inner.push(project3(add(c,mul(d,ri)),cam));
  }
  if (outer.filter(Boolean).length < seg*.55) return;
  ctx.beginPath();
  let started=false; for (const p of outer) if (p) { started?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y); started=true; }
  for (let k=inner.length-1;k>=0;k--) { const p=inner[k]; if (p) ctx.lineTo(p.x,p.y); }
  ctx.closePath(); ctx.fillStyle=active?"rgba(241,196,107,.82)":"rgba(91,112,142,.45)"; ctx.fill("evenodd");
  ctx.strokeStyle=active?"#f6d78c":"#60718b"; ctx.lineWidth=active?2.2:1; ctx.stroke();
}
function wingPolygon(fl, sideIndex) {
  const m=S.info?.morphology || {wing_length:.0025, mean_chord:.00078, hinge_offset:[0,.00022,.00035], stroke_plane_angle:-.261799};
  const side=sideIndex===0?1:-1, wing=fl.wing, phi=wing.phi[sideIndex], theta=wing.theta[sideIndex], alpha=wing.alpha[sideIndex];
  const beta=m.stroke_plane_angle+(wing.stroke_tilt||0), ct=Math.cos(theta);
  const spanS=[Math.sin(phi)*ct, Math.cos(phi)*ct*side, Math.sin(theta)];
  const spanB=unit(rotY(spanS,beta)), nsp=unit(rotY([0,0,1],beta));
  const normal=unit(rodrigues(nsp,spanB,alpha)), chord=unit(cross(normal,spanB));
  const hinge=[m.hinge_offset[0], side*Math.abs(m.hinge_offset[1]), m.hinge_offset[2]];
  const rootWorld=add(fl.pos,bodyToWorld(fl.R,hinge));
  const spanWorld=bodyToWorld(fl.R,spanB), chordWorld=bodyToWorld(fl.R,chord);
  const stations=[[.10,.16],[.56,.52],[1,.08]], top=[], bot=[];
  for (const [r,cw] of stations) {
    const mid=add(rootWorld,mul(spanWorld,m.wing_length*r)), hw=m.mean_chord*cw;
    top.push(add(mid,mul(chordWorld,hw))); bot.push(add(mid,mul(chordWorld,-hw)));
  }
  return top.concat(bot.reverse());
}
function drawFly(ctx, fl, cam) {
  const m=S.info?.morphology || {body_length:.0024, body_radius:.00055};
  const R=fl.R, fwd=unit([R[0],R[3],R[6]]), right=unit([R[1],R[4],R[7]]), up=unit([R[2],R[5],R[8]]);
  for (const si of [1,0]) {
    const poly=wingPolygon(fl,si).map((p)=>project3(p,cam)); if (poly.some((p)=>!p)) continue;
    ctx.beginPath(); poly.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y)); ctx.closePath();
    ctx.fillStyle=si===0?"rgba(108,217,231,.34)":"rgba(141,123,246,.32)"; ctx.fill();
    ctx.strokeStyle=si===0?"#8be8f1":"#aa9cf8"; ctx.lineWidth=1.1*DPR(); ctx.stroke();
  }
  const front=project3(add(fl.pos,mul(fwd,m.body_length*.52)),cam), back=project3(add(fl.pos,mul(fwd,-m.body_length*.62)),cam), center=project3(fl.pos,cam);
  if (!front||!back||!center) return;
  const pxr=Math.max(2.5,cam.f*m.body_radius/center.z);
  ctx.strokeStyle="#d7b36b"; ctx.lineWidth=pxr*1.45; ctx.lineCap="round"; ctx.beginPath(); ctx.moveTo(back.x,back.y); ctx.lineTo(front.x,front.y); ctx.stroke();
  const head=project3(add(fl.pos,mul(fwd,m.body_length*.62)),cam);
  if (head) { ctx.fillStyle="#c75f69"; ctx.beginPath(); ctx.arc(head.x,head.y,pxr*.72,0,Math.PI*2); ctx.fill(); }
  const rr=project3(add(fl.pos,mul(right,m.body_radius*1.7)),cam), uu=project3(add(fl.pos,mul(up,m.body_radius*1.7)),cam);
  if (rr&&uu) { ctx.strokeStyle="rgba(255,255,255,.22)"; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(center.x,center.y); ctx.lineTo(rr.x,rr.y); ctx.moveTo(center.x,center.y); ctx.lineTo(uu.x,uu.y); ctx.stroke(); }
  ctx.lineCap="butt";
}
function drawFlight() {
  const c=el("flight"); fitCanvas(c); const ctx=c.getContext("2d"), w=c.width, h=c.height, fl=S.flight;
  ctx.fillStyle="#06080c"; ctx.fillRect(0,0,w,h); if (!fl) return;
  const cam=cameraFor(fl,w,h), co=fl.course; drawGround(ctx,cam,fl,w,h);
  ctx.strokeStyle="rgba(108,217,231,.20)"; ctx.lineWidth=1.2; ctx.beginPath();
  const all=[co.start_pos||[0,0,.3],...co.center]; for(let i=0;i<all.length-1;i++) line3(ctx,all[i],all[i+1],cam); ctx.stroke();
  if (fl.trail?.length>1) { ctx.strokeStyle="rgba(108,217,231,.72)"; ctx.lineWidth=1.5; ctx.beginPath(); for(let i=0;i<fl.trail.length-1;i++) line3(ctx,fl.trail[i],fl.trail[i+1],cam); ctx.stroke(); }
  const gateOrder=co.center.map((p,i)=>({i,p:project3(p,cam)})).filter(x=>x.p).sort((a,b)=>b.p.z-a.p.z);
  for (const g of gateOrder) drawGate(ctx,co,g.i,cam,g.i===fl.next_gate);
  drawFly(ctx,fl,cam);
  const altitude=fl.pos[2]*100, sp=fl.speed*100;
  el("flight-telemetry").innerHTML=`<span>speed <b>${fmt(sp,1)} cm/s</b></span><span>alt <b>${fmt(altitude,1)} cm</b></span><span>gate <b>${Math.min(fl.next_gate+1,co.center.length)}/${co.center.length}</b></span><span>wing <b>${fmt(fl.wing.freq,0)} Hz</b></span><span>power <b>${fmt(fl.power_rel,2)}×</b></span>`;
}

const G={yaw:-.35,pitch:.12,drag:null,sub:null,dirty:true,proj:null};
const avg=(a)=>a.reduce((s,v)=>s+v,0)/a.length;
function prepareGraph(){
  const g=S.graph,n=g.n,cx=avg(g.x),cy=avg(g.y),cz=avg(g.z);
  g.px=new Float32Array(n);g.py=new Float32Array(n);g.pz=new Float32Array(n);let ext=1;
  for(let i=0;i<n;i++){g.px[i]=g.x[i]-cx;g.py[i]=g.y[i]-cy;g.pz[i]=g.z[i]-cz;ext=Math.max(ext,Math.abs(g.px[i]),Math.abs(g.py[i]),Math.abs(g.pz[i]));}
  g.ext=ext;G.proj={sx:new Float32Array(n),sy:new Float32Array(n),depth:new Float32Array(n)};G.sub=document.createElement("canvas");G.dirty=true;
}
function projectGraph(w,h){
  const g=S.graph,p=G.proj,cy=Math.cos(G.yaw),sy=Math.sin(G.yaw),cp=Math.cos(G.pitch),sp=Math.sin(G.pitch),k=Math.min(w,h)*.47/g.ext;
  for(let i=0;i<g.n;i++){const x=g.px[i],y=g.py[i],z=g.pz[i],x1=x*cy-y*sy,y1=x*sy+y*cy,y2=y1*cp-z*sp,z2=y1*sp+z*cp;p.sx[i]=w/2+x1*k;p.sy[i]=h/2+z2*k;p.depth[i]=y2;}
}
function drawGraphEdges(w,h){
  const g=S.graph,p=G.proj,c=G.sub;c.width=w;c.height=h;const x=c.getContext("2d");x.clearRect(0,0,w,h);x.lineWidth=.55*DPR();
  for(const exc of [0,1]){x.beginPath();x.strokeStyle=exc?"rgba(240,200,107,.08)":"rgba(101,148,255,.09)";for(let e=0;e<g.edges.length/2;e++){if(g.edge_sign[e]!==exc)continue;const a=g.edges[2*e],b=g.edges[2*e+1];x.moveTo(p.sx[a],p.sy[a]);x.lineTo(p.sx[b],p.sy[b]);}x.stroke();}
}
function drawGraph(){
  const c=el("graph");if(!S.graph)return;const resized=fitCanvas(c),ctx=c.getContext("2d"),w=c.width,h=c.height;if(resized||G.dirty){projectGraph(w,h);drawGraphEdges(w,h);G.dirty=false;}
  ctx.fillStyle="#05070a";ctx.fillRect(0,0,w,h);ctx.drawImage(G.sub,0,0);const g=S.graph,p=G.proj,act=S.act,r=1.35*DPR();
  for(const pass of [0,1])for(let i=0;i<g.n;i++){const a=act?act[i]/255:0,hot=a>.28;if((pass===1)!==hot)continue;const col=ROLE[g.role[i]][1];ctx.fillStyle=col;ctx.globalAlpha=pass?Math.min(1,.45+a):((g.role[i]===0?.25:.5)+a*.35);const rr=pass?r+a*2.8*DPR():r;ctx.beginPath();ctx.arc(p.sx[i],p.sy[i],rr/2,0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;
}
(function graphInput(){
  const c=el("graph");c.addEventListener("pointerdown",e=>{G.drag={x:e.clientX,y:e.clientY,yaw:G.yaw,pitch:G.pitch};});addEventListener("pointerup",()=>G.drag=null);addEventListener("pointermove",e=>{if(!G.drag)return;G.yaw=G.drag.yaw+(e.clientX-G.drag.x)*.006;G.pitch=Math.max(-1.3,Math.min(1.3,G.drag.pitch+(e.clientY-G.drag.y)*.006));G.dirty=true;});
  c.addEventListener("pointermove",e=>{if(G.drag||!S.graph||!G.proj)return;const r=c.getBoundingClientRect(),d=DPR(),mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d,g=S.graph,p=G.proj;let best=-1,bd=80*d*d;for(let i=0;i<g.n;i++){const dx=p.sx[i]-mx,dy=p.sy[i]-my,d2=dx*dx+dy*dy;if(d2<bd){bd=d2;best=i;}}const tip=el("tip");if(best<0){tip.hidden=true;return;}const a=S.act?S.act[best]/255*S.actHi:0;tip.hidden=false;tip.style.left=`${e.clientX-r.left+10}px`;tip.style.top=`${e.clientY-r.top+10}px`;tip.innerHTML=`<b style="background:${ROLE[g.role[best]][1]}"></b>${g.types[best]||"unnamed"}<span>${ROLE[g.role[best]][0].toLowerCase()}</span><span>${a.toFixed(3)}</span>`;});
  c.addEventListener("pointerleave",()=>el("tip").hidden=true);
})();

const EY={pts:null};
function prepareEye(){
  const e=S.eye;EY.pts={R:[],L:[]};for(let i=0;i<e.n;i++)(e.side[i]?EY.pts.R:EY.pts.L).push(i);EY.xy={};EY.bounds={};
  for(const side of ["L","R"]){const ids=EY.pts[side],mx=ids.reduce((s,i)=>s+e.x[i],0)/ids.length,my=ids.reduce((s,i)=>s+e.y[i],0)/ids.length;let sxx=0,syy=0,sxy=0;for(const i of ids){const dx=e.x[i]-mx,dy=e.y[i]-my;sxx+=dx*dx;syy+=dy*dy;sxy+=dx*dy;}const th=.5*Math.atan2(2*sxy,sxx-syy),ct=Math.cos(-th),st=Math.sin(-th),X=new Float32Array(e.n),Y=new Float32Array(e.n);for(const i of ids){const dx=e.x[i]-mx,dy=e.y[i]-my;X[i]=dx*ct-dy*st;Y[i]=dx*st+dy*ct;}EY.xy[side]={X,Y};const xs=ids.map(i=>X[i]),ys=ids.map(i=>Y[i]);EY.bounds[side]={x0:Math.min(...xs),x1:Math.max(...xs),y0:Math.min(...ys),y1:Math.max(...ys)};}
}
function hexAt(ctx,x,y,r){ctx.beginPath();for(let k=0;k<6;k++){const a=Math.PI/3*k+Math.PI/6,px=x+r*Math.cos(a),py=y+r*Math.sin(a);k?ctx.lineTo(px,py):ctx.moveTo(px,py);}ctx.closePath();ctx.fill();}
function drawEye(){
  const c=el("eye");fitCanvas(c);const ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.fillStyle="#070a0f";ctx.fillRect(0,0,w,h);if(!S.eye||!EY.pts)return;
  const f=S.eyeFrame,data=f?{lum:b64(f.lum),on:b64(f.on),off:b64(f.off)}:null,rows=[{k:"lum",t:null},{k:"on",t:[108,217,231]},{k:"off",t:[243,106,127]}],rh=h/3,pad=5*DPR();
  rows.forEach((row,ri)=>{for(const [si,side] of [[0,"L"],[1,"R"]]){const ids=EY.pts[side],b=EY.bounds[side],cw=w/2,s=Math.min((cw-pad*2)/Math.max(b.x1-b.x0,1e-6),(rh-pad*2)/Math.max(b.y1-b.y0,1e-6)),ox=si*cw+(cw-(b.x1-b.x0)*s)/2,oy=ri*rh+rh-pad,rad=Math.max(.7,s*.57),xy=EY.xy[side];for(const i of ids){const v=data?data[row.k][i]/255:0;if(row.t)ctx.fillStyle=`rgba(${row.t[0]},${row.t[1]},${row.t[2]},${.05+.95*v})`;else{const g=Math.round(14+v*232);ctx.fillStyle=`rgb(${g},${g},${Math.min(255,g+10)})`;}hexAt(ctx,ox+(xy.X[i]-b.x0)*s,oy-(xy.Y[i]-b.y0)*s,rad);}}
    ctx.strokeStyle="#161c25";ctx.lineWidth=1;if(ri){ctx.beginPath();ctx.moveTo(0,ri*rh);ctx.lineTo(w,ri*rh);ctx.stroke();}ctx.beginPath();ctx.moveTo(w/2,ri*rh);ctx.lineTo(w/2,(ri+1)*rh);ctx.stroke();
  });
  ctx.fillStyle="#7c8798";ctx.font=`${9*DPR()}px ui-monospace`;ctx.fillText("luminance",7*DPR(),12*DPR());ctx.fillText("ON",7*DPR(),rh+12*DPR());ctx.fillText("OFF",7*DPR(),2*rh+12*DPR());
}

function drawWing(){
  const c=el("wing");fitCanvas(c);const ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.fillStyle="#070a0f";ctx.fillRect(0,0,w,h);const fl=S.flight;if(!fl?.scope)return;const [phiR,phiL,alR,alL]=fl.scope;
  ctx.strokeStyle="#1a202b";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();
  for(const [arr,col,rng] of [[phiR,"#f36a7f",1.6],[phiL,"#f1c46b",1.6],[alR,"#6cd9e7",1.6],[alL,"#8d7bf6",1.6]]){if(!arr?.length)continue;ctx.strokeStyle=col;ctx.lineWidth=1.3*DPR();ctx.beginPath();arr.forEach((v,i)=>{const x=i/(arr.length-1)*w,y=h/2-v/rng*h*.43;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
  el("wing-note").textContent=`${fmt(fl.wing.freq,0)} Hz · stroke R/L red/amber · pitch R/L cyan/violet`;
}
function drawMotor(){
  if(!S.pools)return;const m=S.pools.motor,keys=Object.keys(m),host=el("motor");if(host.children.length!==keys.length)host.innerHTML=keys.map(k=>`<div class="row"><span class="name">${k}</span><span class="track"><i class="fill" data-k="${k}"></i></span><span class="val" data-v="${k}">0</span></div>`).join("");const hi=Math.max(...keys.map(k=>m[k]),1e-4);for(const k of keys){host.querySelector(`[data-k="${k}"]`).style.width=`${Math.max(0,m[k]/hi*100)}%`;host.querySelector(`[data-v="${k}"]`).textContent=fmt(m[k],3);}
}

function drawCurves(){
  const c=el("curves");fitCanvas(c);const ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.fillStyle="#070a0f";ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#171d27";ctx.lineWidth=1;for(let j=1;j<4;j++){const y=j*h/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
  for(const [k,col] of CURVES){const a=S.hist[k];if(!a||a.length<2)continue;const lo=Math.min(...a),hi=Math.max(...a),r=hi-lo||1;ctx.strokeStyle=col;ctx.lineWidth=1.25*DPR();ctx.beginPath();a.forEach((v,i)=>{const x=i/(a.length-1)*w,y=h-7*DPR()-(v-lo)/r*(h-14*DPR());i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}
  if(!el("curve-key").children.length)el("curve-key").innerHTML=CURVES.map(([k,col,n])=>`<li><b style="background:${col}"></b>${n}</li>`).join("");
  const tb=S.info?.recurrent?` · ${S.info.tbptt_steps}-step TBPTT`:"";el("curve-note").textContent=S.metrics.update?`PPO update ${Math.round(S.metrics.update)}${tb} · independent y-scales`:"";
}

function frame(){drawFlight();drawGraph();drawEye();drawWing();drawMotor();drawCurves();renderCounters();requestAnimationFrame(frame);}
connect();requestAnimationFrame(frame);
