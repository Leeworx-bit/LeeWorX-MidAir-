/* MidAir core — the hand-tracking plumbing shared by every demo in this repo.
   Each page keeps its own rendering and its own gesture-to-action wiring;
   this module only owns the parts that were identical byte-for-byte between
   them: loading the model, reading a camera frame, and turning raw
   landmarks into stable pinch/fist state. */

import { FilesetResolver, HandLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

export const TAU = Math.PI * 2;

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

/* MediaPipe's 21-point hand skeleton, as bone pairs for line drawing. */
export const HAND_LINKS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]
];

/* ---------- one-euro filter ----------
   Smooths hard when the hand is slow, lets go when it moves fast.
   Without this the pointer is unusable. Not optional polish. */
export class OneEuro {
  constructor(minCut=1.1, beta=.012, dCut=1){
    this.mc=minCut; this.b=beta; this.dc=dCut; this.x=null; this.dx=0; this.t=null;
  }
  a(c,dt){ const tau=1/(TAU*c); return 1/(1+tau/dt); }
  f(v,t){
    if(this.x===null){ this.x=v; this.t=t; return v; }
    const dt=Math.max(1e-3,(t-this.t)/1000); this.t=t;
    const dr=(v-this.x)/dt;
    this.dx+=this.a(this.dc,dt)*(dr-this.dx);
    this.x+=this.a(this.mc+this.b*Math.abs(this.dx),dt)*(v-this.x);
    return this.x;
  }
  reset(){ this.x=null; this.dx=0; this.t=null; }
}

function handSize(lm){
  return Math.hypot(lm[0].x-lm[9].x, lm[0].y-lm[9].y) || 1;
}

/* thumb tip to index tip, normalised by hand size so it survives the
   hand moving closer to or further from the camera */
export function pinchRatio(lm){
  return Math.hypot(lm[4].x-lm[8].x, lm[4].y-lm[8].y) / handSize(lm);
}

/* mean fingertip-to-wrist distance, normalised the same way */
export function curlRatio(lm){
  let curl=0;
  for(const t of [8,12,16,20]) curl += Math.hypot(lm[t].x-lm[0].x, lm[t].y-lm[0].y);
  return (curl/4) / handSize(lm);
}

/* ---------- gesture hysteresis ----------
   Two thresholds per gesture, never one. A single threshold fires
   repeatedly while the hand hovers on the boundary. `rearmMs` keeps a
   held pinch from firing more than once. */
export class GestureTracker {
  constructor({pinchOn=.34, pinchOff=.46, fistOn=.62, fistOff=.78, rearmMs=500}={}){
    this.pinchOn=pinchOn; this.pinchOff=pinchOff;
    this.fistOn=fistOn; this.fistOff=fistOff; this.rearmMs=rearmMs;
    this.fisted=false; this.pinching=false; this.armed=true;
  }
  /* Call once per frame with the current landmarks. Returns the gesture
     state plus `justPinched`, true for exactly the frame a new pinch
     is committed (armed and past rearmMs since the last one). */
  update(lm){
    const pinchR=pinchRatio(lm), curl=curlRatio(lm);
    if(curl<this.fistOn) this.fisted=true; else if(curl>this.fistOff) this.fisted=false;

    let justPinched=false;
    if(pinchR<this.pinchOn && !this.pinching){
      this.pinching=true;
      if(this.armed){
        this.armed=false; justPinched=true;
        setTimeout(()=>{ this.armed=true; }, this.rearmMs);
      }
    }else if(pinchR>this.pinchOff && this.pinching){
      this.pinching=false;
    }

    const pinchAmt=Math.max(0,Math.min(1,(this.pinchOff-pinchR)/(this.pinchOff-this.pinchOn)));
    return {fisted:this.fisted, pinching:this.pinching, justPinched, pinchAmt};
  }
  /* Clears fisted/pinching without touching the rearm timer — call when
     the hand is lost so a stale gesture doesn't carry into the next one. */
  reset(){ this.fisted=false; this.pinching=false; }
}

/* ---------- model + camera bootstrap ----------
   Identical across every demo: load the WASM runtime, load the model,
   open the camera, in that order, so a caller can tell which step failed.
   Errors are tagged with `.stage` ('secure'|'runtime'|'model'|'camera')
   and `.cause` (the original error) so each page can render its own copy. */
function taggedError(stage, e){
  const err=new Error(e&&e.message ? e.message : String(e));
  err.stage=stage; err.cause=e;
  return err;
}

export async function initHandTracking(video){
  if(!isSecureContext) throw taggedError('secure', new Error('insecure context'));

  let vision;
  try{
    vision=await FilesetResolver.forVisionTasks(WASM_URL);
  }catch(e){ throw taggedError('runtime', e); }

  let landmarker;
  try{
    landmarker=await HandLandmarker.createFromOptions(vision,{
      baseOptions:{ modelAssetPath: MODEL_URL, delegate:'GPU' },
      runningMode:'VIDEO', numHands:1,
      minHandDetectionConfidence:.5, minTrackingConfidence:.5
    });
  }catch(e){ throw taggedError('model', e); }

  try{
    video.srcObject=await navigator.mediaDevices.getUserMedia({
      video:{facingMode:'user', width:{ideal:1280}, height:{ideal:720}}, audio:false});
    await video.play();
  }catch(e){ throw taggedError('camera', e); }

  return landmarker;
}
