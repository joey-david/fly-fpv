/*
 * Lightweight NeuroMechFly renderer.
 *
 * The geometry is the micro-CT-derived NeuroMechFly v2 model from FlyGym,
 * accessed through flyplotlib's pinned Apache-2.0 asset mirror.  We fetch only
 * a few simplified body meshes, reduce them to small point clouds once, and
 * render their projected silhouettes on the existing 2D canvas.  The actual
 * wing mesh is reposed every frame from flypv's simulated phi/theta/alpha.
 *
 * If the assets cannot be fetched (e.g. offline), dashboard.js's procedural
 * renderer remains as a fallback rather than breaking the monitor.
 */
(() => {
  const fallbackDrawFly = drawFly;
  const REF = "6801e03af8f6422524ff60d4c1c2b9a134f730d6";
  const BASE = `https://raw.githubusercontent.com/tkclam/flyplotlib/${REF}/src/flyplotlib/data/neuromechfly`;
  const BODY = [
    "c_thorax", "c_head", "c_rostrum", "c_haustellum",
    "c_abdomen12", "c_abdomen3", "c_abdomen4", "c_abdomen5", "c_abdomen6",
    "l_eye",
  ];
  const PARENT = {
    c_head: "c_thorax", c_rostrum: "c_head", c_haustellum: "c_rostrum",
    c_abdomen12: "c_thorax", c_abdomen3: "c_abdomen12",
    c_abdomen4: "c_abdomen3", c_abdomen5: "c_abdomen4", c_abdomen6: "c_abdomen5",
    l_eye: "c_head",
  };
  const MODEL = {ready:false, failed:false, segments:[], wing:null};

  const mm = (v, s) => [v[0]*s, v[1]*s, v[2]*s];
  const vadd = (a,b) => [a[0]+b[0],a[1]+b[1],a[2]+b[2]];
  const vsub = (a,b) => [a[0]-b[0],a[1]-b[1],a[2]-b[2]];
  const vdot = (a,b) => a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const vcross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const vnorm = (a) => { const n=Math.hypot(...a)||1; return [a[0]/n,a[1]/n,a[2]/n]; };
  const matVec = (R,v) => [
    R[0]*v[0]+R[1]*v[1]+R[2]*v[2],
    R[3]*v[0]+R[4]*v[1]+R[5]*v[2],
    R[6]*v[0]+R[7]*v[1]+R[8]*v[2],
  ];
  const matMul = (A,B) => {
    const C=new Array(9).fill(0);
    for(let r=0;r<3;r++)for(let c=0;c<3;c++)for(let k=0;k<3;k++)C[3*r+c]+=A[3*r+k]*B[3*k+c];
    return C;
  };
  function quatMat(q0){
    const n=Math.hypot(...q0)||1,[w,x,y,z]=q0.map(v=>v/n);
    return [
      1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
      2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
      2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y),
    ];
  }
  function compose(a,b){return {R:matMul(a.R,b.R),p:vadd(a.p,matVec(a.R,b.p))};}
  function applyT(T,p){return vadd(T.p,matVec(T.R,p));}

  async function readSTL(name){
    const r=await fetch(`${BASE}/meshes/${name}.stl`,{cache:"force-cache"});
    if(!r.ok)throw new Error(`mesh ${name}: HTTP ${r.status}`);
    const b=await r.arrayBuffer(),dv=new DataView(b),pts=[];
    if(b.byteLength>=84){
      const n=dv.getUint32(80,true);
      if(84+n*50===b.byteLength){
        const target=240, stride=Math.max(1,Math.floor((n*3)/target));
        let vi=0;
        for(let t=0;t<n;t++)for(let j=0;j<3;j++,vi++)if(vi%stride===0){
          const o=84+t*50+12+j*12;
          pts.push([dv.getFloat32(o,true)*1000,dv.getFloat32(o+4,true)*1000,dv.getFloat32(o+8,true)*1000]);
        }
        return pts;
      }
    }
    const text=new TextDecoder().decode(b),re=/vertex\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)/g;
    let m;while((m=re.exec(text)))pts.push([+m[1]*1000,+m[2]*1000,+m[3]*1000]);
    if(!pts.length)throw new Error(`mesh ${name}: unsupported STL`);
    const stride=Math.max(1,Math.floor(pts.length/240));
    return pts.filter((_,i)=>i%stride===0);
  }

  function hull(points){
    if(points.length<3)return points;
    const p=[...points].sort((a,b)=>a.x-b.x||a.y-b.y);
    const cross2=(o,a,b)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
    const lo=[];for(const q of p){while(lo.length>=2&&cross2(lo.at(-2),lo.at(-1),q)<=0)lo.pop();lo.push(q);}
    const hi=[];for(let i=p.length-1;i>=0;i--){const q=p[i];while(hi.length>=2&&cross2(hi.at(-2),hi.at(-1),q)<=0)hi.pop();hi.push(q);}
    lo.pop();hi.pop();return lo.concat(hi);
  }
  function projectedHull(points,fl,cam){
    const pp=[];let z=0,n=0;
    for(const p of points){const w=vadd(fl.pos,bodyToWorld(fl.R,p)),q=project3(w,cam);if(q){pp.push(q);z+=q.z;n++;}}
    return {h:hull(pp),z:n?z/n:Infinity};
  }
  function fillHull(ctx,h,color,stroke=null){
    if(h.length<3)return;ctx.beginPath();h.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.closePath();ctx.fillStyle=color;ctx.fill();
    if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=.55*DPR();ctx.stroke();}
  }

  async function loadModel(){
    try{
      const rr=await fetch(`${BASE}/rigging.json`,{cache:"force-cache"});
      if(!rr.ok)throw new Error(`rigging: HTTP ${rr.status}`);
      const rig=await rr.json(),raw={};
      await Promise.all([...BODY,"l_wing"].map(async n=>{raw[n]=await readSTL(n);}));

      const T={};
      for(const name of BODY){
        const r=rig[name],local={R:quatMat(r.quat),p:r.pos};
        T[name]=PARENT[name]?compose(T[PARENT[name]],local):local;
      }
      const bodyWorld={};
      for(const name of BODY)bodyWorld[name]=raw[name].map(p=>applyT(T[name],p));
      const thor=bodyWorld.c_thorax,origin=thor.reduce((a,p)=>vadd(a,p),[0,0,0]).map(v=>v/thor.length);
      const structural=BODY.filter(n=>!n.endsWith("eye"));
      let xmin=Infinity,xmax=-Infinity;
      for(const n of structural)for(const p of bodyWorld[n]){xmin=Math.min(xmin,p[0]);xmax=Math.max(xmax,p[0]);}
      const targetLength=S.info?.morphology?.body_length||.0024,scale=targetLength/Math.max(xmax-xmin,1e-6);
      const convert=p=>[(p[0]-origin[0])*scale,-(p[1]-origin[1])*scale,(p[2]-origin[2])*scale];
      for(const name of structural)MODEL.segments.push({name,points:bodyWorld[name].map(convert),eye:false});
      const eye=bodyWorld.l_eye.map(convert);
      MODEL.segments.push({name:"l_eye",points:eye,eye:true});
      MODEL.segments.push({name:"r_eye",points:eye.map(p=>[p[0],-p[1],p[2]]),eye:true});

      // Canonicalize the real NeuroMechFly wing mesh around its joint-local
      // origin.  The current simulated span/chord/normal basis then reposes it
      // without changing its measured planform.
      const wp=raw.l_wing;
      let far=wp[0];for(const p of wp)if(vdot(p,p)>vdot(far,far))far=p;
      const span0=vnorm(far);let chordVec=[0,1,0],best=-1;
      for(const p of wp){const q=vsub(p,mm(span0,vdot(p,span0))),d=vdot(q,q);if(d>best){best=d;chordVec=q;}}
      const chord0=vnorm(chordVec),normal0=vnorm(vcross(span0,chord0));
      const coords=wp.map(p=>[vdot(p,span0),vdot(p,chord0),vdot(p,normal0)]);
      const spanExtent=Math.max(...coords.map(p=>Math.abs(p[0])),1e-6);
      MODEL.wing={coords,unitScale:(S.info?.morphology?.wing_length||.0025)/spanExtent};
      MODEL.ready=true;
    }catch(e){
      MODEL.failed=true;
      console.warn("NeuroMechFly model unavailable; using lightweight fallback",e);
    }
  }

  function realWingPoints(fl,sideIndex){
    const m=S.info?.morphology||{wing_length:.0025,hinge_offset:[0,.00022,.00035],stroke_plane_angle:-.261799};
    const side=sideIndex===0?1:-1,w=fl.wing,phi=w.phi[sideIndex],theta=w.theta[sideIndex],alpha=w.alpha[sideIndex],beta=m.stroke_plane_angle+(w.stroke_tilt||0),ct=Math.cos(theta);
    const spanS=[Math.sin(phi)*ct,Math.cos(phi)*ct*side,Math.sin(theta)],spanB=unit(rotY(spanS,beta)),nsp=unit(rotY([0,0,1],beta)),normal=unit(rodrigues(nsp,spanB,alpha)),chord=unit(cross(normal,spanB));
    const hinge=[m.hinge_offset[0],side*Math.abs(m.hinge_offset[1]),m.hinge_offset[2]],s=MODEL.wing.unitScale;
    return MODEL.wing.coords.map(q=>vadd(hinge,vadd(mm(spanB,q[0]*s),vadd(mm(chord,q[1]*s),mm(normal,q[2]*s)))));
  }

  drawFly=function(ctx,fl,cam){
    if(!MODEL.ready){fallbackDrawFly(ctx,fl,cam);return;}
    const pieces=[];
    for(const seg of MODEL.segments){const p=projectedHull(seg.points,fl,cam);pieces.push({...p,eye:seg.eye});}
    for(const side of [1,0]){const p=projectedHull(realWingPoints(fl,side),fl,cam);pieces.push({...p,wing:true});}
    pieces.sort((a,b)=>b.z-a.z);
    for(const p of pieces){
      if(p.wing)fillHull(ctx,p.h,"rgba(205,210,213,.18)","rgba(205,210,213,.52)");
      else if(p.eye)fillHull(ctx,p.h,"#713c39");
      else fillHull(ctx,p.h,"#554a40","rgba(110,100,90,.35)");
    }
  };

  loadModel();
})();
