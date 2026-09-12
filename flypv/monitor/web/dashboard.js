/* flypv live monitor — rendering is deliberately decoupled from telemetry cadence. */
const ROLE = [
  ["Intrinsic", "#667085"], ["Retina", "#8aa6b8"], ["Sensory", "#9d91ad"],
  ["Motor", "#b77a72"], ["Descending", "#a99a73"], ["Ascending", "#7f9b8d"],
];
const CURVES = [
  ["ep_return", "#aeb8c4", "Return"], ["ep_gates", "#8fa89b", "Gates"],
  ["crash_rate", "#b8817b", "Crash"], ["entropy", "#948da4", "Entropy"],
  ["difficulty", "#aa9b78", "Difficulty"],
];
const S = {
  graph: null, eye: null, info: null, act: null, actHi: 1,
  eyeFrame: null, flight: null, pools: null, live: {}, metrics: {}, hist: {},
};
const DIRTY = {graph: true, eye: true, dynamics: true, curves: true, counters: true};
const el = (id) => document.getElementById(id);
const DPR = () => Math.min(devicePixelRatio || 1, 2);
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
    DIRTY.curves = true;
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
  DIRTY.curves = true;
}
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    el("conn").classList.add("live"); el("conn").querySelector("span").textContent = "live";
    seedHistory();
  };
  ws.onclose = () => {
    el("conn").classList.remove("live"); el("conn").querySelector("span").textContent = "reconnecting";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => {
    const m = JSON.parse(e.data);
    if (m.kind === "static") {
      if (m.graph) { S.graph = m.graph; prepareGraph(); DIRTY.graph = true; }
      if (m.eye) { S.eye = m.eye; prepareEye(); DIRTY.eye = true; }
      if (m.info) { S.info = m.info; renderHeader(); renderKey(); }
      return;
    }
    if (m.act) { S.act = b64(m.act.data); S.actHi = m.act.hi; DIRTY.graph = true; }
    if (m.eye_frame) { S.eyeFrame = m.eye_frame; DIRTY.eye = true; }
    if (m.flight) {
      S.flight = {...(S.flight || {}), ...m.flight, _arrival: performance.now()};
      if (m.flight.reset) resetView();
      if (m.flight.scope) DIRTY.dynamics = true;
    }
    if (m.pools) { S.pools = m.pools; DIRTY.dynamics = true; }
    if (m.live) { S.live = m.live; DIRTY.counters = true; }
    if (m.metrics) { S.metrics = m.metrics; pushHistory(m.metrics); DIRTY.counters = true; }
  };
}

function renderHeader() {
  const i = S.info;
  const temporal = i.recurrent ? `persistent · TBPTT ${i.tbptt_steps}` : "stateless";
  el("dataset").textContent = `${i.dataset} · ${i.neurons.toLocaleString()} neurons · ${temporal}`;
  el("graph-note").textContent = `${i.displayed.toLocaleString()} displayed · drag to rotate`;
  const f = i.eye_fov;
  if (f) el("eye-note").textContent = `luminance only · approx ${Math.round(f.overlap_deg)}° binocular overlap · ${Math.round(f.rear_blind_deg)}° rear blind spot`;
}
function renderCounters() {
  const m = S.metrics, l = S.live, i = S.info || {};
  const rows = [
    ["steps", m.step ? Math.round(m.step).toLocaleString() : "—"],
    ["return", fmt(m.ep_return)], ["gates", `${fmt(m.ep_gates, 2)} / ${i.n_gates ?? "—"}`],
    ["crash", m.crash_rate == null ? "—" : `${Math.round(m.crash_rate * 100)}%`],
    ["difficulty", fmt(m.difficulty, 2)], ["speed", l.speed == null ? "—" : `${fmt(l.speed * 100, 1)} cm/s`],
    ["power", l.power_rel == null ? "—" : `${fmt(l.power_rel, 2)}×`], ["train", fmt(m.fps, 0) + " env/s"],
  ];
  el("counters").innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}
function renderKey() {
  el("key").innerHTML = ROLE.map(([n, c], i) => {
    const count = S.graph ? S.graph.role.filter((r) => r === i).length : 0;
    return `<li><b style="background:${c}"></b>${n}<span>${count.toLocaleString()}</span></li>`;
  }).join("");
}

const add = (a,b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const sub = (a,b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const mul = (a,s) => [a[0]*s, a[1]*s, a[2]*s];
const lerp = (a,b,t) => a.map((v,i)=>v+(b[i]-v)*t);
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
function quatNorm(q) { const n=Math.hypot(...q)||1; return q.map(v=>v/n); }
function nlerpQuat(a,b,t) {
  let bb=b; if (a.reduce((s,v,i)=>s+v*b[i],0)<0) bb=b.map(v=>-v);
  return quatNorm(a.map((v,i)=>v+(bb[i]-v)*t));
}
function quatToMat(q0) {
  const [w,x,y,z]=quatNorm(q0);
  return [
    1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
    2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
    2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y),
  ];
}

const VIEW={pos:null,quat:null,camPos:null,camLook:null,heading:null,last:performance.now()};
function resetView(){VIEW.pos=VIEW.quat=VIEW.camPos=VIEW.camLook=VIEW.heading=null;}
function smoothFlight(now) {
  const fl=S.flight; if(!fl) return null;
  const dt=Math.min(.05,Math.max(.001,(now-VIEW.last)/1000)); VIEW.last=now;
  const age=Math.min(.08,Math.max(0,(now-(fl._arrival||now))/1000));
  const targetPos=add(fl.pos,mul(fl.vel||[0,0,0],age));
  if(!VIEW.pos||!VIEW.quat){VIEW.pos=[...targetPos];VIEW.quat=[...(fl.quat||[1,0,0,0])];}
  const a=1-Math.exp(-dt/.055);
  VIEW.pos=lerp(VIEW.pos,targetPos,a);
  VIEW.quat=nlerpQuat(VIEW.quat,fl.quat||[1,0,0,0],a);
  return {...fl,pos:VIEW.pos,R:quatToMat(VIEW.quat)};
}
function cameraFor(fl,w,h,dt) {
  const worldUp=[0,0,1], co=fl.course;
  let desired=fl.vel && mag(fl.vel)>.04 ? unit([fl.vel[0],fl.vel[1],fl.vel[2]*.28]) : null;
  if(!desired && co?.center?.length){const gi=Math.min(fl.next_gate||0,co.center.length-1);desired=unit(sub(co.center[gi],fl.pos));}
  desired=desired||[1,0,0];
  if(!VIEW.heading) VIEW.heading=[...desired];
  const ah=1-Math.exp(-dt/.30); VIEW.heading=unit(lerp(VIEW.heading,desired,ah));
  const wantedPos=add(add(fl.pos,mul(VIEW.heading,-.062)),[0,0,.025]);
  const wantedLook=add(add(fl.pos,mul(VIEW.heading,.070)),[0,0,.002]);
  if(!VIEW.camPos){VIEW.camPos=wantedPos;VIEW.camLook=wantedLook;}
  const ac=1-Math.exp(-dt/.18); VIEW.camPos=lerp(VIEW.camPos,wantedPos,ac);VIEW.camLook=lerp(VIEW.camLook,wantedLook,ac);
  const z=unit(sub(VIEW.camLook,VIEW.camPos)); let x=unit(cross(z,worldUp)); if(mag(x)<1e-4)x=[0,1,0];
  const y=unit(cross(x,z));
  return {p:VIEW.camPos,x,y,z,f:h*.90,cx:w*.5,cy:h*.54};
}
function project3(p,cam){const q=sub(p,cam.p),z=dot(q,cam.z);if(z<.0015)return null;return{x:cam.cx+cam.f*dot(q,cam.x)/z,y:cam.cy-cam.f*dot(q,cam.y)/z,z};}
function line3(ctx,a,b,cam){const p=project3(a,cam),q=project3(b,cam);if(!p||!q)return false;ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);return true;}
function drawGround(ctx,cam,fl,w,h){
  ctx.fillStyle="#0a0c0f";ctx.fillRect(0,0,w,h);
  const gx=Math.round(fl.pos[0]/.2)*.2,gy=Math.round(fl.pos[1]/.2)*.2;
  ctx.strokeStyle="rgba(180,190,200,.055)";ctx.lineWidth=1;
  for(let i=-8;i<=8;i++){ctx.beginPath();line3(ctx,[gx+i*.2,gy-1.6,0],[gx+i*.2,gy+1.6,0],cam);ctx.stroke();ctx.beginPath();line3(ctx,[gx-1.6,gy+i*.2,0],[gx+1.6,gy+i*.2,0],cam);ctx.stroke();}
}
function drawGate(ctx,co,i,cam,active){
  const c=co.center[i],n=unit(co.normal[i]),u=perp(n),v=unit(cross(n,u)),ro=co.r_out[i],ri=co.r_in[i],seg=44,outer=[],inner=[];
  for(let k=0;k<=seg;k++){const a=2*Math.PI*k/seg,d=add(mul(u,Math.cos(a)),mul(v,Math.sin(a)));outer.push(project3(add(c,mul(d,ro)),cam));inner.push(project3(add(c,mul(d,ri)),cam));}
  if(outer.filter(Boolean).length<seg*.55)return;ctx.beginPath();let started=false;for(const p of outer)if(p){started?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y);started=true;}for(let k=inner.length-1;k>=0;k--){const p=inner[k];if(p)ctx.lineTo(p.x,p.y);}ctx.closePath();
  ctx.fillStyle=active?"rgba(190,164,103,.68)":"rgba(112,119,127,.20)";ctx.fill("evenodd");ctx.strokeStyle=active?"#c8ad70":"#626970";ctx.lineWidth=active?1.7*DPR():.8*DPR();ctx.stroke();
}
function wingPolygon(fl,sideIndex){
  const m=S.info?.morphology||{wing_length:.0025,mean_chord:.00078,hinge_offset:[0,.00022,.00035],stroke_plane_angle:-.261799};
  const side=sideIndex===0?1:-1,wing=fl.wing,phi=wing.phi[sideIndex],theta=wing.theta[sideIndex],alpha=wing.alpha[sideIndex],beta=m.stroke_plane_angle+(wing.stroke_tilt||0),ct=Math.cos(theta);
  const spanS=[Math.sin(phi)*ct,Math.cos(phi)*ct*side,Math.sin(theta)],spanB=unit(rotY(spanS,beta)),nsp=unit(rotY([0,0,1],beta)),normal=unit(rodrigues(nsp,spanB,alpha)),chord=unit(cross(normal,spanB));
  const hinge=[m.hinge_offset[0],side*Math.abs(m.hinge_offset[1]),m.hinge_offset[2]],root=add(fl.pos,bodyToWorld(fl.R,hinge)),spanW=bodyToWorld(fl.R,spanB),chordW=bodyToWorld(fl.R,chord);
  const stations=[[.06,.08],[.28,.45],[.68,.58],[1,.12]],top=[],bot=[];for(const [r,cw] of stations){const mid=add(root,mul(spanW,m.wing_length*r)),hw=m.mean_chord*cw;top.push(add(mid,mul(chordW,hw)));bot.push(add(mid,mul(chordW,-hw)));}return top.concat(bot.reverse());
}
function bodyPoint(fl,x,y,z){return add(fl.pos,bodyToWorld(fl.R,[x,y,z]));}
function capsule3(ctx,a,b,r,cam,fill){const p=project3(a,cam),q=project3(b,cam);if(!p||!q)return;const depth=(p.z+q.z)/2,px=Math.max(1.4,cam.f*r/depth);ctx.strokeStyle=fill;ctx.lineWidth=px*2;ctx.lineCap="round";ctx.beginPath();ctx.moveTo(p.x,p.y);ctx.lineTo(q.x,q.y);ctx.stroke();ctx.lineCap="butt";}
function disc3(ctx,p,r,cam,fill){const q=project3(p,cam);if(!q)return;const rr=Math.max(1.5,cam.f*r/q.z);ctx.fillStyle=fill;ctx.beginPath();ctx.arc(q.x,q.y,rr,0,Math.PI*2);ctx.fill();}
function drawFly(ctx,fl,cam){
  const m=S.info?.morphology||{body_length:.0024,body_radius:.00055},L=m.body_length,R=m.body_radius;
  for(const si of [1,0]){const poly=wingPolygon(fl,si).map(p=>project3(p,cam));if(poly.some(p=>!p))continue;ctx.beginPath();poly.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fillStyle="rgba(205,210,213,.18)";ctx.fill();ctx.strokeStyle="rgba(205,210,213,.55)";ctx.lineWidth=.75*DPR();ctx.stroke();}
  ctx.strokeStyle="rgba(72,64,58,.9)";ctx.lineWidth=.7*DPR();for(const side of [-1,1])for(const x of [-.18,0,.18]){const hip=bodyPoint(fl,x*L,side*.22*R,-.05*R),knee=bodyPoint(fl,(x-.12)*L,side*1.35*R,-.75*R),foot=bodyPoint(fl,(x-.28)*L,side*2.05*R,-1.4*R);ctx.beginPath();line3(ctx,hip,knee,cam);line3(ctx,knee,foot,cam);ctx.stroke();}
  capsule3(ctx,bodyPoint(fl,-.06*L,0,0),bodyPoint(fl,.20*L,0,0),R*.72,cam,"#51463e");
  capsule3(ctx,bodyPoint(fl,-.10*L,0,0),bodyPoint(fl,-.53*L,0,-.02*L),R*.48,cam,"#6b5947");
  capsule3(ctx,bodyPoint(fl,-.48*L,0,-.02*L),bodyPoint(fl,-.72*L,0,-.025*L),R*.28,cam,"#493c31");
  const head=bodyPoint(fl,.35*L,0,.02*L);disc3(ctx,head,R*.62,cam,"#625248");
  disc3(ctx,bodyPoint(fl,.37*L,.42*R,.06*L),R*.35,cam,"#7b3f3d");disc3(ctx,bodyPoint(fl,.37*L,-.42*R,.06*L),R*.35,cam,"#7b3f3d");
  ctx.strokeStyle="#76695e";ctx.lineWidth=.65*DPR();for(const side of [-1,1]){ctx.beginPath();line3(ctx,bodyPoint(fl,.55*L,side*.16*R,.16*R),bodyPoint(fl,.72*L,side*.55*R,.34*R),cam);ctx.stroke();}
}
function drawFlight(now){
  const c=el("flight");fitCanvas(c);const ctx=c.getContext("2d"),w=c.width,h=c.height,fl=smoothFlight(now);ctx.fillStyle="#0a0c0f";ctx.fillRect(0,0,w,h);if(!fl?.course)return;
  const dt=Math.min(.05,Math.max(.001,(now-(drawFlight.last||now))/1000));drawFlight.last=now;const cam=cameraFor(fl,w,h,dt),co=fl.course;drawGround(ctx,cam,fl,w,h);
  if(fl.trail?.length>1){ctx.strokeStyle="rgba(182,190,197,.38)";ctx.lineWidth=1.1*DPR();ctx.beginPath();for(let i=0;i<fl.trail.length-1;i++)line3(ctx,fl.trail[i],fl.trail[i+1],cam);ctx.stroke();}
  const gateOrder=co.center.map((p,i)=>({i,p:project3(p,cam)})).filter(x=>x.p).sort((a,b)=>b.p.z-a.p.z);for(const g of gateOrder)drawGate(ctx,co,g.i,cam,g.i===fl.next_gate);drawFly(ctx,fl,cam);
  const altitude=fl.pos[2]*100,sp=fl.speed*100;el("flight-telemetry").innerHTML=`<span>${fmt(sp,1)} cm/s</span><span>${fmt(altitude,1)} cm</span><span>gate ${Math.min(fl.next_gate+1,co.center.length)}/${co.center.length}</span><span>${fmt(fl.wing.freq,0)} Hz</span>`;
}

const G={yaw:-.35,pitch:.12,drag:null,sub:null,dirty:true,proj:null};
const avg=(a)=>a.reduce((s,v)=>s+v,0)/a.length;
function prepareGraph(){const g=S.graph,n=g.n,cx=avg(g.x),cy=avg(g.y),cz=avg(g.z);g.px=new Float32Array(n);g.py=new Float32Array(n);g.pz=new Float32Array(n);let ext=1;for(let i=0;i<n;i++){g.px[i]=g.x[i]-cx;g.py[i]=g.y[i]-cy;g.pz[i]=g.z[i]-cz;ext=Math.max(ext,Math.abs(g.px[i]),Math.abs(g.py[i]),Math.abs(g.pz[i]));}g.ext=ext;G.proj={sx:new Float32Array(n),sy:new Float32Array(n),depth:new Float32Array(n)};G.sub=document.createElement("canvas");G.dirty=true;}
function projectGraph(w,h){const g=S.graph,p=G.proj,cy=Math.cos(G.yaw),sy=Math.sin(G.yaw),cp=Math.cos(G.pitch),sp=Math.sin(G.pitch),k=Math.min(w,h)*.47/g.ext;for(let i=0;i<g.n;i++){const x=g.px[i],y=g.py[i],z=g.pz[i],x1=x*cy-y*sy,y1=x*sy+y*cy,y2=y1*cp-z*sp,z2=y1*sp+z*cp;p.sx[i]=w/2+x1*k;p.sy[i]=h/2+z2*k;p.depth[i]=y2;}}
function drawGraphEdges(w,h){const g=S.graph,p=G.proj,c=G.sub;c.width=w;c.height=h;const x=c.getContext("2d");x.clearRect(0,0,w,h);x.lineWidth=.45*DPR();for(const exc of [0,1]){x.beginPath();x.strokeStyle=exc?"rgba(173,154,116,.055)":"rgba(118,139,160,.055)";for(let e=0;e<g.edges.length/2;e++){if(g.edge_sign[e]!==exc)continue;const a=g.edges[2*e],b=g.edges[2*e+1];x.moveTo(p.sx[a],p.sy[a]);x.lineTo(p.sx[b],p.sy[b]);}x.stroke();}}
function drawGraph(){const c=el("graph");if(!S.graph)return;const resized=fitCanvas(c),ctx=c.getContext("2d"),w=c.width,h=c.height;if(resized||G.dirty){projectGraph(w,h);drawGraphEdges(w,h);G.dirty=false;}ctx.fillStyle="#0a0c0f";ctx.fillRect(0,0,w,h);ctx.drawImage(G.sub,0,0);const g=S.graph,p=G.proj,act=S.act,r=1.25*DPR();for(const pass of [0,1])for(let i=0;i<g.n;i++){const a=act?act[i]/255:0,hot=a>.30;if((pass===1)!==hot)continue;ctx.fillStyle=ROLE[g.role[i]][1];ctx.globalAlpha=pass?Math.min(.95,.35+a):((g.role[i]===0?.16:.35)+a*.25);const rr=pass?r+a*2.4*DPR():r;ctx.beginPath();ctx.arc(p.sx[i],p.sy[i],rr/2,0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;}
(function graphInput(){const c=el("graph");c.addEventListener("pointerdown",e=>{G.drag={x:e.clientX,y:e.clientY,yaw:G.yaw,pitch:G.pitch};});addEventListener("pointerup",()=>G.drag=null);addEventListener("pointermove",e=>{if(!G.drag)return;G.yaw=G.drag.yaw+(e.clientX-G.drag.x)*.006;G.pitch=Math.max(-1.3,Math.min(1.3,G.drag.pitch+(e.clientY-G.drag.y)*.006));G.dirty=true;DIRTY.graph=true;});c.addEventListener("pointermove",e=>{if(G.drag||!S.graph||!G.proj)return;const r=c.getBoundingClientRect(),d=DPR(),mx=(e.clientX-r.left)*d,my=(e.clientY-r.top)*d,g=S.graph,p=G.proj;let best=-1,bd=80*d*d;for(let i=0;i<g.n;i++){const dx=p.sx[i]-mx,dy=p.sy[i]-my,d2=dx*dx+dy*dy;if(d2<bd){bd=d2;best=i;}}const tip=el("tip");if(best<0){tip.hidden=true;return;}const a=S.act?S.act[best]/255*S.actHi:0;tip.hidden=false;tip.style.left=`${e.clientX-r.left+10}px`;tip.style.top=`${e.clientY-r.top+10}px`;tip.innerHTML=`${g.types[best]||"unnamed"}<span>${ROLE[g.role[best]][0].toLowerCase()}</span><span>${a.toFixed(3)}</span>`;});c.addEventListener("pointerleave",()=>el("tip").hidden=true);})();

const EY={pts:null};
function prepareEye(){const e=S.eye;EY.pts={R:[],L:[]};for(let i=0;i<e.n;i++)(e.side[i]?EY.pts.R:EY.pts.L).push(i);EY.xy={};EY.bounds={};for(const side of ["L","R"]){const ids=EY.pts[side],mx=ids.reduce((s,i)=>s+e.x[i],0)/ids.length,my=ids.reduce((s,i)=>s+e.y[i],0)/ids.length;let sxx=0,syy=0,sxy=0;for(const i of ids){const dx=e.x[i]-mx,dy=e.y[i]-my;sxx+=dx*dx;syy+=dy*dy;sxy+=dx*dy;}const th=.5*Math.atan2(2*sxy,sxx-syy),ct=Math.cos(-th),st=Math.sin(-th),X=new Float32Array(e.n),Y=new Float32Array(e.n);for(const i of ids){const dx=e.x[i]-mx,dy=e.y[i]-my;X[i]=dx*ct-dy*st;Y[i]=dx*st+dy*ct;}EY.xy[side]={X,Y};const xs=ids.map(i=>X[i]),ys=ids.map(i=>Y[i]);EY.bounds[side]={x0:Math.min(...xs),x1:Math.max(...xs),y0:Math.min(...ys),y1:Math.max(...ys)};}}
function hexAt(ctx,x,y,r){ctx.beginPath();for(let k=0;k<6;k++){const a=Math.PI/3*k+Math.PI/6,px=x+r*Math.cos(a),py=y+r*Math.sin(a);k?ctx.lineTo(px,py):ctx.moveTo(px,py);}ctx.closePath();ctx.fill();}
function drawEye(){const c=el("eye");fitCanvas(c);const ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.fillStyle="#0a0c0f";ctx.fillRect(0,0,w,h);if(!S.eye||!EY.pts)return;const data=S.eyeFrame?.lum?b64(S.eyeFrame.lum):null,pad=10*DPR();for(const [si,side] of [[0,"L"],[1,"R"]]){const ids=EY.pts[side],b=EY.bounds[side],cw=w/2,s=Math.min((cw-pad*2)/Math.max(b.x1-b.x0,1e-6),(h-pad*2)/Math.max(b.y1-b.y0,1e-6)),ox=si*cw+(cw-(b.x1-b.x0)*s)/2,oy=h-pad,rad=Math.max(.7,s*.57),xy=EY.xy[side];for(const i of ids){const v=data?data[i]/255:0,g=Math.round(18+v*220);ctx.fillStyle=`rgb(${g},${g},${g})`;hexAt(ctx,ox+(xy.X[i]-b.x0)*s,oy-(xy.Y[i]-b.y0)*s,rad);}ctx.fillStyle="#737b83";ctx.font=`${9*DPR()}px ui-monospace`;ctx.fillText(side==="L"?"LEFT":"RIGHT",si*cw+7*DPR(),13*DPR());}ctx.strokeStyle="#20242a";ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.stroke();}
function drawWing(){const c=el("wing");fitCanvas(c);const ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.fillStyle="#0a0c0f";ctx.fillRect(0,0,w,h);const fl=S.flight;if(!fl?.scope)return;const [phiR,phiL,alR,alL]=fl.scope;ctx.strokeStyle="#252a30";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,h/2);ctx.lineTo(w,h/2);ctx.stroke();for(const [arr,col,rng] of [[phiR,"#aa847f",1.6],[phiL,"#a99978",1.6],[alR,"#879ca8",1.6],[alL,"#928ba0",1.6]]){if(!arr?.length)continue;ctx.strokeStyle=col;ctx.lineWidth=1.05*DPR();ctx.beginPath();arr.forEach((v,i)=>{const x=i/(arr.length-1)*w,y=h/2-v/rng*h*.43;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}el("wing-note").textContent=`${fmt(fl.wing.freq,0)} Hz · stroke and pitch`;}
function drawMotor(){if(!S.pools)return;const m=S.pools.motor,keys=Object.keys(m),host=el("motor");if(host.children.length!==keys.length)host.innerHTML=keys.map(k=>`<div class="row"><span class="name">${k}</span><span class="track"><i class="fill" data-k="${k}"></i></span><span class="val" data-v="${k}">0</span></div>`).join("");const hi=Math.max(...keys.map(k=>m[k]),1e-4);for(const k of keys){host.querySelector(`[data-k="${k}"]`).style.width=`${Math.max(0,m[k]/hi*100)}%`;host.querySelector(`[data-v="${k}"]`).textContent=fmt(m[k],3);}}
function drawCurves(){const c=el("curves");fitCanvas(c);const ctx=c.getContext("2d"),w=c.width,h=c.height;ctx.fillStyle="#0a0c0f";ctx.fillRect(0,0,w,h);ctx.strokeStyle="#20242a";ctx.lineWidth=1;for(let j=1;j<4;j++){const y=j*h/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}for(const [k,col] of CURVES){const a=S.hist[k];if(!a||a.length<2)continue;const lo=Math.min(...a),hi=Math.max(...a),r=hi-lo||1;ctx.strokeStyle=col;ctx.lineWidth=1.05*DPR();ctx.beginPath();a.forEach((v,i)=>{const x=i/(a.length-1)*w,y=h-7*DPR()-(v-lo)/r*(h-14*DPR());i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();}if(!el("curve-key").children.length)el("curve-key").innerHTML=CURVES.map(([k,col,n])=>`<li><b style="background:${col}"></b>${n}</li>`).join("");const tb=S.info?.recurrent?` · TBPTT ${S.info.tbptt_steps}`:"";el("curve-note").textContent=S.metrics.update?`PPO update ${Math.round(S.metrics.update)}${tb}`:"";}

function frame(now){
  drawFlight(now);
  if(DIRTY.graph){drawGraph();DIRTY.graph=false;}
  if(DIRTY.eye){drawEye();DIRTY.eye=false;}
  if(DIRTY.dynamics){drawWing();drawMotor();DIRTY.dynamics=false;}
  if(DIRTY.curves){drawCurves();DIRTY.curves=false;}
  if(DIRTY.counters){renderCounters();DIRTY.counters=false;}
  requestAnimationFrame(frame);
}
window.addEventListener("resize",()=>{DIRTY.graph=DIRTY.eye=DIRTY.dynamics=DIRTY.curves=true;});
connect();requestAnimationFrame(frame);
