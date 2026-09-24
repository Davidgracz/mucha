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
@media(max-width:1050px){.grid{grid-template-columns:1fr 1fr}.span3{grid-column:span 2}}
@media(max-width:900px){.voice-summary{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:700px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.badges{justify-content:flex-start}.grid{grid-template-columns:1fr}.span2,.span3{grid-column:auto}.events{grid-template-columns:1fr}.voice-summary{grid-template-columns:1fr}}
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
    return '<div class="voice-summary">'+
      '<div class="voice-pill"><small>serwer / kanał</small><strong>'+esc(v.guild)+' / '+esc(v.current||"poza voice")+'</strong></div>'+
      '<div class="voice-pill"><small>voice_join</small><strong>'+Number(s.voice_join??0).toFixed(3)+' / '+Number(v.join_threshold??0).toFixed(3)+'</strong></div>'+
      '<div class="voice-pill"><small>voice_move</small><strong>'+Number(s.voice_move??0).toFixed(3)+' / '+Number(v.move_threshold??0).toFixed(3)+'</strong></div>'+
      '<div class="voice-pill"><small>voice_leave</small><strong>'+Number(s.voice_leave??0).toFixed(3)+' / '+Number(v.leave_threshold??0).toFixed(3)+'</strong></div>'+
      '<div class="voice-pill"><small>dwell remaining</small><strong>'+Number(v.dwell_remaining??0).toFixed(1)+' s</strong></div>'+
    '</div>'+
    '<div class="reason"><b>'+esc(v.decision||"—")+'</b> — '+esc(v.reason||"—")+'</div>'+
    '<table><thead><tr><th>Kanał</th><th>Ludzie</th><th>View</th><th>Connect</th><th>Affinity</th><th>Status</th></tr></thead><tbody>'+channels+'</tbody></table>';
  }).join('<div style="height:14px"></div>');
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
    $("language").textContent=nfmt(s.language_tokens)+" / "+nfmt(s.language_unique);
    $("ready").textContent=s.language_ready?"TAK":"nie";$("voice").textContent=s.voice||"poza voice";
    $("paused").textContent=s.paused?"PAUZA":"aktywny";$("event").textContent=s.last_event||"—";$("lastaction").textContent=s.last_action||"—";
    renderActions(s.scores||{});
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
