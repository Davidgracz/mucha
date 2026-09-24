from __future__ import annotations

import asyncio
import json
import logging
import webbrowser
from typing import Awaitable, Callable

from aiohttp import web

log = logging.getLogger("mucha.web")

SnapshotProvider = Callable[[], Awaitable[dict]]

HTML = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha Brain Dashboard</title>
<style>
:root{
  --bg:#0b0f14;--panel:#111821;--panel2:#151e29;--text:#edf4fb;--muted:#8290a0;
  --line:#233142;--accent:#55d3c3;--accent2:#6ea8fe;--good:#54d98c;--warn:#f2c14e;--bad:#ff6b6b;
}
*{box-sizing:border-box}
body{margin:0;background:linear-gradient(180deg,#0a0e13,#0d131a 60%,#0b1016);color:var(--text);
font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1500px;margin:auto;padding:22px}
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:18px}
.brand{display:flex;align-items:center;gap:12px}.fly{font-size:32px}.title h1{font-size:23px;margin:0}.title p{margin:4px 0 0;color:var(--muted);font-size:13px}
.badges{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.badge{border:1px solid var(--line);background:#0f161f;border-radius:999px;padding:7px 10px;font-size:12px}
.grid{display:grid;grid-template-columns:1.05fr 1.35fr .9fr;gap:14px}
.card{background:rgba(17,24,33,.94);border:1px solid var(--line);border-radius:16px;padding:15px;min-width:0}
.card h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#a9b8c8;margin:0 0 12px}
.metric{display:flex;justify-content:space-between;gap:12px;padding:8px 0;border-bottom:1px solid rgba(35,49,66,.6)}
.metric:last-child{border-bottom:0}.metric span:first-child{color:var(--muted)}.metric strong{font-variant-numeric:tabular-nums}
.action{display:grid;grid-template-columns:105px 1fr 52px;gap:9px;align-items:center;margin:9px 0}
.track{height:9px;background:#0a1017;border-radius:999px;overflow:hidden;border:1px solid #1e2a37}.fill{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));width:0%;transition:width .25s ease}
.val{font-variant-numeric:tabular-nums;text-align:right}.dominant{color:var(--accent);font-weight:700}
.span2{grid-column:span 2}.span3{grid-column:span 3}
canvas{display:block;width:100%;height:220px;background:#0c1219;border-radius:12px;border:1px solid #1c2734}
.events{display:grid;grid-template-columns:1fr 1fr;gap:10px}.event{background:#0c131b;border:1px solid #1d2936;border-radius:12px;padding:11px;min-height:76px}
.event small{color:var(--muted);display:block;margin-bottom:7px}.event div{word-break:break-word}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid rgba(35,49,66,.6)}th{color:var(--muted);font-weight:600}td:last-child,th:last-child{text-align:right}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--good);margin-right:6px;box-shadow:0 0 12px rgba(84,217,140,.45)}
.footer{color:var(--muted);font-size:12px;margin-top:12px;text-align:right}
.voice-summary{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin-bottom:12px}
.voice-pill{background:#0c131b;border:1px solid #1d2936;border-radius:12px;padding:10px}
.voice-pill small{display:block;color:var(--muted);margin-bottom:5px}.voice-pill strong{font-size:14px}
.reason{padding:11px 12px;border-radius:12px;background:#0c131b;border:1px solid #1d2936;margin-bottom:12px}
.reason b{color:var(--accent)}
.ok{color:var(--good)}.no{color:var(--bad)}.warn{color:var(--warn)}
.learning-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-bottom:12px}
.kpi{background:#0c131b;border:1px solid #1d2936;border-radius:12px;padding:11px;min-width:0}
.kpi small{display:block;color:var(--muted);margin-bottom:6px}.kpi strong{font-size:16px;word-break:break-word}
.impact-row{display:grid;grid-template-columns:110px 1fr 150px;gap:8px;align-items:center;margin:7px 0}
.impact-track{height:8px;background:#0a1017;border-radius:999px;overflow:hidden;border:1px solid #1e2a37;position:relative}
.impact-zero{position:absolute;left:50%;top:0;bottom:0;width:1px;background:#536274}
.impact-fill-pos,.impact-fill-neg{position:absolute;top:0;height:100%}
.impact-fill-pos{left:50%;background:var(--good)}.impact-fill-neg{right:50%;background:var(--bad)}
.log-list{display:flex;flex-direction:column;gap:7px;max-height:300px;overflow:auto}
.log-item{display:grid;grid-template-columns:74px 88px 1fr;gap:8px;padding:8px 10px;background:#0c131b;border:1px solid #1d2936;border-radius:10px;font-size:12px}
.log-time{color:var(--muted)}.log-kind{color:var(--accent);font-weight:700;text-transform:uppercase}
.reaction-emoji{font-size:30px;line-height:1}
.legend{display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}
.legend span::before{content:"";display:inline-block;width:10px;height:3px;margin-right:5px;vertical-align:middle;border-radius:2px}
.legend .reward-line::before{background:var(--warn)}.legend .trace-line::before{background:var(--accent)}
@media(max-width:1050px){.grid{grid-template-columns:1fr 1fr}.span3{grid-column:span 2}}
@media(max-width:900px){.voice-summary,.learning-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:700px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.badges{justify-content:flex-start}.grid{grid-template-columns:1fr}.span2,.span3{grid-column:auto}.events{grid-template-columns:1fr}.voice-summary,.learning-grid{grid-template-columns:1fr}.log-item{grid-template-columns:62px 72px 1fr}}
</style>
</head>
<body>
<main>
  <div class="top">
    <div class="brand"><div class="fly">🪰</div><div class="title"><h1>Mucha Brain Dashboard</h1><p id="source">łączenie…</p></div></div>
    <div class="badges">
      <div class="badge"><span class="dot"></span><span id="live">LIVE</span></div>
      <div class="badge" id="backend">backend: —</div>
      <div class="badge" id="device">device: —</div>
      <div class="badge" id="clock">—</div>
    </div>
  </div>

  <section class="grid">
    <div class="card">
      <h2>Stan mózgu</h2>
      <div class="metric"><span>Neurony</span><strong id="neurons">—</strong></div>
      <div class="metric"><span>Połączenia</span><strong id="connections">—</strong></div>
      <div class="metric"><span>Aktywne |a| &gt; 0.1</span><strong id="active">—</strong></div>
      <div class="metric"><span>Średnia |a|</span><strong id="mean">—</strong></div>
      <div class="metric"><span>Maks. |a|</span><strong id="max">—</strong></div>
      <div class="metric"><span>Reward trace</span><strong id="reward">—</strong></div>
      <div class="metric"><span>Tick</span><strong id="ticks">—</strong></div>
    </div>

    <div class="card">
      <h2>Wyjścia connectome</h2>
      <div id="actions"></div>
    </div>

    <div class="card">
      <h2>Środowisko</h2>
      <div class="metric"><span>Język</span><strong id="language">—</strong></div>
      <div class="metric"><span>Tryb języka</span><strong id="language-mode">—</strong></div>
      <div class="metric"><span>Wiadomości</span><strong id="language-messages">—</strong></div>
      <div class="metric"><span>Przejścia znaków</span><strong id="language-transitions">—</strong></div>
      <div class="metric"><span>Bootstrap ze starej pamięci</span><strong id="language-bootstrap">—</strong></div>
      <div class="metric"><span>Gotowa pisać</span><strong id="ready">—</strong></div>
      <div class="metric"><span>Voice</span><strong id="voice">—</strong></div>
      <div class="metric"><span>Stan</span><strong id="paused">—</strong></div>
    </div>

    <div class="card span2">
      <h2>Aktywność w czasie</h2>
      <canvas id="chart" width="1000" height="220" aria-label="Wykres aktywności mózgu"></canvas>
    </div>

    <div class="card">
      <h2>Ostatnie zdarzenia</h2>
      <div class="events">
        <div class="event"><small>bodziec</small><div id="event">—</div></div>
        <div class="event"><small>akcja</small><div id="lastaction">—</div></div>
      </div>
    </div>

    <div class="card span2">
      <h2>Learning Debug</h2>
      <div class="learning-grid">
        <div class="kpi"><small>ostatni reward</small><strong id="learn-reward">—</strong></div>
        <div class="kpi"><small>target action</small><strong id="learn-action">—</strong></div>
        <div class="kpi"><small>zmienione neurony</small><strong id="learn-count">—</strong></div>
        <div class="kpi"><small>max |Δ bias|</small><strong id="learn-max">—</strong></div>
      </div>
      <div class="reason" id="learn-summary">Czekam na pierwszy reward…</div>
      <div id="learning-impact"></div>
    </div>

    <div class="card">
      <h2>Reaction Debug</h2>
      <div class="learning-grid" style="grid-template-columns:1fr 1fr">
        <div class="kpi"><small>react / próg</small><strong id="reaction-score">—</strong></div>
        <div class="kpi"><small>emoji</small><strong class="reaction-emoji" id="reaction-emoji">—</strong></div>
      </div>
      <div class="metric"><span>Decyzja</span><strong id="reaction-decision">—</strong></div>
      <div class="metric"><span>Cel</span><strong id="reaction-target">—</strong></div>
      <div class="metric"><span>Cooldown</span><strong id="reaction-cooldown">—</strong></div>
      <div class="metric"><span>Pula emoji</span><strong id="reaction-pool">—</strong></div>
      <div class="metric"><span>Ocenione teraz</span><strong id="reaction-evaluated">—</strong></div>
      <div class="footer" id="reaction-top">—</div>
    </div>

    <div class="card">
      <h2>Plasticity</h2>
      <div class="metric"><span>Średni bias</span><strong id="bias-mean">—</strong></div>
      <div class="metric"><span>Średni |bias|</span><strong id="bias-mean-abs">—</strong></div>
      <div class="metric"><span>Max |bias|</span><strong id="bias-max">—</strong></div>
      <div class="metric"><span>Dodatnie neurony</span><strong id="bias-pos">—</strong></div>
      <div class="metric"><span>Ujemne neurony</span><strong id="bias-neg">—</strong></div>
      <canvas id="bias-chart" width="520" height="150" aria-label="Histogram plastic bias" style="height:150px;margin-top:12px"></canvas>
    </div>

    <div class="card span2">
      <h2>Reward timeline</h2>
      <canvas id="reward-chart" width="1000" height="220" aria-label="Historia reward trace"></canvas>
      <div class="legend"><span class="trace-line">reward trace</span><span class="reward-line">zdarzenie reward</span></div>
    </div>

    <div class="card span2">
      <h2>Action History</h2>
      <div class="log-list" id="action-history"><div class="reason">Brak akcji.</div></div>
    </div>

    <div class="card">
      <h2>Top changed neurons</h2>
      <table><thead><tr><th>root_id</th><th>Δ bias</th><th>activation</th></tr></thead><tbody id="changed-neurons"></tbody></table>
    </div>

    <div class="card span3">
      <h2>Server Learning Context</h2>
      <table>
        <thead><tr><th>Serwer</th><th>Ostatnia nagradzalna akcja</th><th>Szczegół</th><th>Wiek</th></tr></thead>
        <tbody id="guild-learning-context"></tbody>
      </table>
    </div>

    <div class="card span3">
      <h2>Voice Debug</h2>
      <div id="voice-debug"><div class="reason">Czekam na pierwszy cykl voice…</div></div>
    </div>

    <div class="card span3">
      <h2>Najbardziej aktywne neurony</h2>
      <table><thead><tr><th>#</th><th>FlyWire root_id</th><th>activation</th><th>|a|</th></tr></thead><tbody id="top"></tbody></table>
      <div class="footer">Przy prawdziwym FAFB v783 root_id odpowiada identyfikatorowi neuronu FlyWire.</div>
    </div>
  </section>
</main>
<script>
const actionOrder=["speak","react","voice_join","voice_move","voice_leave","explore","stay"];
const history=[];
const maxHistory=180;
const $=id=>document.getElementById(id);
function fmt(n,d=3){return Number(n).toFixed(d)}
function nfmt(n){return Number(n).toLocaleString("pl-PL")}
function renderActions(scores){
  const dom=Object.entries(scores).sort((a,b)=>b[1]-a[1])[0]?.[0];
  $("actions").innerHTML=actionOrder.map(k=>{
    const v=Number(scores[k]??0);
    return '<div class="action"><div class="'+(k===dom?'dominant':'')+'">'+(k===dom?'▶ ':'')+k+'</div>'+
      '<div class="track"><div class="fill" style="width:'+Math.max(0,Math.min(100,v*100))+'%"></div></div>'+
      '<div class="val">'+v.toFixed(3)+'</div></div>';
  }).join("");
}
function esc(v){
  return String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[m]));
}
function renderVoiceDebug(items){
  const root=$("voice-debug");
  if(!Array.isArray(items)||!items.length){
    root.innerHTML='<div class="reason">Brak danych. Jeśli voice jest wyłączony w configu, pętla diagnostyczna nie wystartuje.</div>';
    return;
  }
  root.innerHTML=items.map(v=>{
    const s=v.scores||{};
    const channels=(v.channels||[]).map(ch=>{
      const aff=ch.affinity==null?"—":Number(ch.affinity).toFixed(3);
      const status=ch.eligible?"OK":ch.status;
      const cls=ch.eligible?"ok":(ch.status==="AFK"?"warn":"no");
      return '<tr>'+
        '<td>'+(ch.current?"▶ ":"")+esc(ch.name)+'</td>'+
        '<td>'+Number(ch.humans||0)+'</td>'+
        '<td class="'+(ch.view?"ok":"no")+'">'+(ch.view?"YES":"NO")+'</td>'+
        '<td class="'+(ch.connect?"ok":"no")+'">'+(ch.connect?"YES":"NO")+'</td>'+
        '<td>'+aff+'</td>'+
        '<td class="'+cls+'">'+esc(status)+'</td>'+
      '</tr>';
    }).join("");
    const overstay=Number(v.overstay_seconds||0);
    const punished=Boolean(v.overstay_punished);
    return '<div class="voice-summary">'+
      '<div class="voice-pill"><small>serwer / kanał</small><strong>'+esc(v.guild)+' / '+esc(v.current||"poza voice")+'</strong></div>'+
      '<div class="voice-pill"><small>voice_join</small><strong>'+Number(s.voice_join??0).toFixed(3)+' / '+Number(v.join_threshold??0).toFixed(3)+'</strong></div>'+
      '<div class="voice-pill"><small>voice_move</small><strong>'+Number(s.voice_move??0).toFixed(3)+' / '+Number(v.move_threshold??0).toFixed(3)+'</strong></div>'+
      '<div class="voice-pill"><small>voice_leave</small><strong>'+Number(s.voice_leave??0).toFixed(3)+' / '+Number(v.leave_threshold??0).toFixed(3)+'</strong></div>'+
      '<div class="voice-pill"><small>czas na kanale / limit</small><strong>'+Number(v.dwell_elapsed??0).toFixed(0)+' / '+Number(v.maximum_dwell_seconds??0).toFixed(0)+' s</strong></div>'+
      '<div class="voice-pill"><small>minimum dwell</small><strong>'+Number(v.dwell_remaining??0).toFixed(1)+' s</strong></div>'+
      '<div class="voice-pill"><small>overstay</small><strong class="'+(overstay>0?"no":"ok")+'">'+overstay.toFixed(0)+' s</strong></div>'+
      '<div class="voice-pill"><small>kara w tym cyklu</small><strong class="'+(punished?"no":"")+'">'+(punished?Number(v.overstay_punish_amount||0).toFixed(2):"nie")+'</strong></div>'+
      '<div class="voice-pill"><small>THREAT</small><strong class="'+(v.threat_active?"no":"ok")+'">'+(Number(v.threat_level||0)*100).toFixed(0)+'%</strong></div>'+
      '<div class="voice-pill"><small>threat magnitude</small><strong>'+Number(v.threat_magnitude||0).toFixed(3)+'</strong></div>'+
      '<div class="voice-pill"><small>effective move</small><strong>'+((v.effective_move_score==null)?"—":Number(v.effective_move_score).toFixed(3))+'</strong></div>'+
      '<div class="voice-pill"><small>effective margin</small><strong>'+((v.effective_move_margin==null)?"—":Number(v.effective_move_margin).toFixed(3))+'</strong></div>'+
      '<div class="voice-pill"><small>escape target</small><strong>'+esc(v.escape_target||"—")+'</strong></div>'+
    '</div>'+
    '<div class="reason"><b>'+esc(v.decision||"—")+'</b> — '+esc(v.reason||"—")+'</div>'+
    '<table><thead><tr><th>Kanał</th><th>Ludzie</th><th>View</th><th>Connect</th><th>Affinity</th><th>Status</th></tr></thead><tbody>'+channels+'</tbody></table>';
  }).join('<div style="height:14px"></div>');
}
function renderLearning(l){
  l=l||{};
  const amount=Number(l.amount||0);
  $("learn-reward").textContent=(amount>=0?"+":"")+amount.toFixed(2);
  $("learn-reward").className=amount>0?"ok":amount<0?"no":"";
  $("learn-action").textContent=l.action||"global / brak";
  $("learn-count").textContent=nfmt(l.changed_neurons||0);
  $("learn-max").textContent=Number(l.max_delta||0).toExponential(3);
  $("learn-summary").innerHTML='<b>'+esc(l.action||"brak targetu")+'</b> • średnie Δ bias: '+
    (Number(l.mean_delta||0)>=0?"+":"")+Number(l.mean_delta||0).toExponential(3);

  const impact=l.impact||{}, before=l.before||{}, after=l.after||{};
  const maxAbs=Math.max(.001,...Object.values(impact).map(v=>Math.abs(Number(v))));
  $("learning-impact").innerHTML=actionOrder.map(k=>{
    const v=Number(impact[k]||0), b=Number(before[k]||0), a=Number(after[k]||0);
    const pct=Math.min(50,Math.abs(v)/maxAbs*50);
    const bar=v>=0
      ?'<div class="impact-fill-pos" style="width:'+pct+'%"></div>'
      :'<div class="impact-fill-neg" style="width:'+pct+'%"></div>';
    return '<div class="impact-row"><div>'+k+'</div><div class="impact-track"><div class="impact-zero"></div>'+bar+
      '</div><div class="'+(v>0?"ok":v<0?"no":"")+'" title="'+b.toFixed(3)+' → '+a.toFixed(3)+'">'+
      b.toFixed(3)+'→'+a.toFixed(3)+' '+(v>=0?"+":"")+v.toFixed(3)+'</div></div>';
  }).join("");

  $("changed-neurons").innerHTML=(l.top_changed||[]).slice(0,10).map(n=>
    '<tr><td>'+esc(n.root_id)+'</td><td class="'+(Number(n.delta)>=0?"ok":"no")+'">'+
    (Number(n.delta)>=0?"+":"")+Number(n.delta).toExponential(3)+'</td><td>'+
    (Number(n.activation)>=0?"+":"")+Number(n.activation).toFixed(4)+'</td></tr>'
  ).join("");
}
function renderReaction(r){
  r=r||{};
  $("reaction-score").textContent=Number(r.score||0).toFixed(3)+" / "+Number(r.threshold||0).toFixed(3);
  $("reaction-emoji").textContent=r.emoji||"—";
  $("reaction-decision").textContent=r.decision||"—";
  $("reaction-target").textContent=r.target||"—";
  $("reaction-cooldown").textContent=Number(r.cooldown_remaining||0).toFixed(1)+" s";
  $("reaction-pool").textContent=nfmt(r.pool_total||0);
  $("reaction-evaluated").textContent=nfmt(r.candidates_evaluated||0);
  $("reaction-top").textContent=(r.top_candidates||[]).slice(0,6).map(x=>
    (x.emoji||"—")+" "+Number(x.score||0).toFixed(3)
  ).join("   ");
}
function renderGuildLearningContext(items){
  const root=$("guild-learning-context");
  if(!Array.isArray(items)||!items.length){
    root.innerHTML='<tr><td colspan="4">Brak zapisanych akcji per serwer.</td></tr>';
    return;
  }
  const now=Date.now()/1000;
  root.innerHTML=items.map(x=>{
    const age=Math.max(0,now-Number(x.time||0));
    return '<tr>'+
      '<td>'+esc(x.guild||x.guild_id||"—")+'</td>'+
      '<td><strong>'+esc(x.action||"—")+'</strong></td>'+
      '<td>'+esc(x.detail||"—")+'</td>'+
      '<td>'+age.toFixed(0)+' s</td>'+
    '</tr>';
  }).join("");
}
function renderActionHistory(items){
  const root=$("action-history");
  if(!Array.isArray(items)||!items.length){root.innerHTML='<div class="reason">Brak akcji.</div>';return}
  root.innerHTML=items.slice().reverse().map(x=>{
    const t=new Date(Number(x.time||0)*1000).toLocaleTimeString("pl-PL");
    const guild=x.guild?('['+esc(x.guild)+'] '):'';
    return '<div class="log-item"><div class="log-time">'+t+'</div><div class="log-kind">'+esc(x.kind||"")+
      '</div><div>'+guild+esc(x.detail||"")+'</div></div>';
  }).join("");
}
function drawBiasHistogram(hist){
  const c=$("bias-chart"),ctx=c.getContext("2d"),w=c.width,h=c.height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#0c1219";ctx.fillRect(0,0,w,h);
  const counts=(hist&&hist.counts)||[];
  if(!counts.length)return;
  const max=Math.max(1,...counts);
  const bw=w/counts.length;
  counts.forEach((v,i)=>{
    const bh=(Number(v)/max)*(h-18);
    ctx.fillStyle=i<Math.floor(counts.length/2)?"#ff6b6b":i===Math.floor(counts.length/2)?"#8290a0":"#54d98c";
    ctx.fillRect(i*bw+1,h-bh-8,Math.max(1,bw-2),bh);
  });
  ctx.strokeStyle="#536274";ctx.beginPath();ctx.moveTo(w/2,0);ctx.lineTo(w/2,h);ctx.stroke();
}
const rewardHistory=[];
function drawRewardChart(events,currentTrace){
  const c=$("reward-chart"),ctx=c.getContext("2d"),w=c.width,h=c.height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#0c1219";ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#1d2a37";ctx.lineWidth=1;
  const mid=h/2;ctx.beginPath();ctx.moveTo(0,mid);ctx.lineTo(w,mid);ctx.stroke();
  const data=(events||[]).slice(-80);
  if(data.length){
    const minT=data[0].time,maxT=Math.max(minT+1,data[data.length-1].time);
    data.forEach(e=>{
      const x=((e.time-minT)/(maxT-minT))*w;
      const amt=Number(e.amount||0);
      ctx.strokeStyle="#f2c14e";ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,mid);
      ctx.lineTo(x,mid-(amt*(h*.32)));ctx.stroke();
      ctx.fillStyle=amt>=0?"#54d98c":"#ff6b6b";ctx.beginPath();ctx.arc(x,mid-(amt*(h*.32)),3,0,Math.PI*2);ctx.fill();
    });
  }
  rewardHistory.push(Number(currentTrace||0));while(rewardHistory.length>maxHistory)rewardHistory.shift();
  if(rewardHistory.length>1){
    ctx.strokeStyle="#55d3c3";ctx.lineWidth=2;ctx.beginPath();
    rewardHistory.forEach((v,i)=>{
      const x=i/(Math.max(1,maxHistory-1))*w;
      const y=mid-(Math.max(-2,Math.min(2,v))/2)*(h*.42);
      i?ctx.lineTo(x,y):ctx.moveTo(x,y);
    });ctx.stroke();
  }
}
function draw(){
  const c=$("chart"),ctx=c.getContext("2d"),w=c.width,h=c.height;
  ctx.clearRect(0,0,w,h);ctx.fillStyle="#0c1219";ctx.fillRect(0,0,w,h);
  ctx.strokeStyle="#1d2a37";ctx.lineWidth=1;
  for(let i=1;i<5;i++){const y=h*i/5;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
  if(history.length<2)return;
  const max=Math.max(.15,...history.map(x=>x.max));
  const plot=(key,stroke)=>{
    ctx.beginPath();ctx.strokeStyle=stroke;ctx.lineWidth=2;
    history.forEach((p,i)=>{const x=(i/(Math.max(1,maxHistory-1)))*w;const y=h-(Math.min(max,p[key])/max)*(h-12)-6;i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
    ctx.stroke();
  };
  plot("mean","#55d3c3");plot("max","#6ea8fe");
}
async function update(){
  try{
    const r=await fetch("/api/state",{cache:"no-store"});if(!r.ok)throw new Error("HTTP "+r.status);
    const s=await r.json(),d=s.diag;
    $("source").textContent=s.source||"unknown";
    $("backend").textContent="backend: "+(d.backend||"cpu").toUpperCase();
    $("device").textContent="device: "+(d.device||"CPU");
    $("clock").textContent=new Date().toLocaleTimeString("pl-PL");
    $("neurons").textContent=nfmt(d.neurons);$("connections").textContent=nfmt(d.connections);
    $("active").textContent=nfmt(d.active_abs_gt_0_1);$("mean").textContent=fmt(d.mean_abs,5);
    $("max").textContent=fmt(d.max_abs,5);$("reward").textContent=(d.reward_trace>=0?"+":"")+fmt(d.reward_trace,4);
    $("ticks").textContent=nfmt(d.ticks);
    $("bias-mean").textContent=(Number(d.bias_mean||0)>=0?"+":"")+Number(d.bias_mean||0).toExponential(3);
    $("bias-mean-abs").textContent=Number(d.bias_mean_abs||0).toExponential(3);
    $("bias-max").textContent=Number(d.bias_max_abs||0).toExponential(3);
    $("bias-pos").textContent=nfmt(d.bias_positive||0);
    $("bias-neg").textContent=nfmt(d.bias_negative||0);
    drawBiasHistogram(d.bias_hist||{});
    const ld=s.language_diag||{};
    $("language").textContent=nfmt(s.language_tokens)+" znaków / "+nfmt(s.language_unique)+" unikalnych";
    $("language-mode").textContent=ld.mode||"characters";
    $("language-messages").textContent=nfmt(ld.messages||0);
    $("language-transitions").textContent=nfmt(ld.transitions||0);
    $("language-bootstrap").textContent=nfmt(ld.legacy_bootstrap_chars||0)+" znaków / "+nfmt(ld.legacy_bootstrap_items||0)+" elementów";
    $("ready").textContent=s.language_ready?"TAK":"nie";$("voice").textContent=s.voice||"poza voice";
    $("paused").textContent=s.paused?"PAUZA":"aktywny";$("event").textContent=s.last_event||"—";$("lastaction").textContent=s.last_action||"—";
    renderActions(s.scores||{});
    renderReaction(s.reaction_debug||{});
    renderLearning(s.learning_debug||{});
    renderActionHistory(s.action_history||[]);
    renderGuildLearningContext(s.guild_learning_context||[]);
    drawRewardChart(s.reward_history||[],d.reward_trace);
    renderVoiceDebug(s.voice_debug||[]);
    $("top").innerHTML=(s.top_neurons||[]).map((x,i)=>'<tr><td>'+(i+1)+'</td><td>'+x[0]+'</td><td>'+(x[1]>=0?"+":"")+Number(x[1]).toFixed(5)+'</td><td>'+Math.abs(x[1]).toFixed(5)+'</td></tr>').join("");
    history.push({mean:Number(d.mean_abs),max:Number(d.max_abs)});while(history.length>maxHistory)history.shift();draw();
    $("live").textContent="LIVE";
  }catch(e){$("live").textContent="ROZŁĄCZONO";console.error(e)}
}
setInterval(update,500);update();
window.addEventListener("resize",draw);
</script>
</body>
</html>"""


class WebDashboard:
    def __init__(
        self,
        snapshot_provider: SnapshotProvider,
        host: str,
        port: int,
        auto_open: bool = True,
        refresh_ms: int = 500,
        history_points: int = 180,
    ):
        self.snapshot_provider = snapshot_provider
        self.host = host
        self.port = int(port)
        self.auto_open = bool(auto_open)
        self.refresh_ms = max(100, int(refresh_ms))
        self.history_points = max(30, int(history_points))
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

    async def start(self) -> None:
        if self.runner is not None:
            return
        app = web.Application()
        app.router.add_get("/", self._index)
        app.router.add_get("/api/state", self._state)
        app.router.add_get("/health", self._health)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        log.info("Web UI: http://%s:%s", self.host, self.port)

        if self.auto_open and self.host in {"127.0.0.1", "localhost"}:
            url = f"http://127.0.0.1:{self.port}"
            asyncio.get_running_loop().call_later(1.0, webbrowser.open, url)

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None
            self.site = None

    async def _index(self, request: web.Request) -> web.Response:
        html = HTML.replace("const maxHistory=180;", f"const maxHistory={self.history_points};")
        html = html.replace("setInterval(update,500);", f"setInterval(update,{self.refresh_ms});")
        return web.Response(text=html, content_type="text/html")

    async def _state(self, request: web.Request) -> web.Response:
        snap = await self.snapshot_provider()
        return web.json_response(snap, dumps=lambda x: json.dumps(x, ensure_ascii=False))

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})
