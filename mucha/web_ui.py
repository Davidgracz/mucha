from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import shutil
import time
import webbrowser
from pathlib import Path
from typing import Awaitable, Callable

from aiohttp import web

log = logging.getLogger("mucha.web")

SnapshotProvider = Callable[[], Awaitable[dict]]
ConfigProvider = Callable[[], dict]
ConfigUpdater = Callable[[dict], dict]

CONFIG_HTML = r"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha — Konfiguracja</title>
<style>
:root{--bg:#080d12;--panel:#101720;--panel2:#0b1219;--line:#233143;--txt:#eef5fc;--muted:#8291a2;--a:#58dac4;--good:#55d98c;--bad:#ff7272}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#080d12,#0b1118);color:var(--txt);font-family:Inter,system-ui,"Segoe UI",sans-serif}
main{max-width:1450px;margin:auto;padding:22px}.top{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:18px}
h1{margin:0;font-size:24px}.sub{color:var(--muted);font-size:12px;margin-top:4px}.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#c6d2df;text-decoration:none;border:1px solid var(--line);background:#0e161f;padding:8px 11px;border-radius:10px;font-size:12px}.nav a.active{background:var(--a);border-color:var(--a);color:#06110e;font-weight:800}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:15px;min-width:0}.card h2{margin:0 0 12px;font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:#aebdca}.fields{display:grid;grid-template-columns:1fr 1fr;gap:9px}.field{background:var(--panel2);border:1px solid #1d2a39;border-radius:11px;padding:10px}.field label{display:block;color:var(--muted);font-size:11px;margin-bottom:6px}.field input[type=number]{width:100%;border:1px solid #263749;background:#071019;color:var(--txt);border-radius:8px;padding:8px}.toggle{display:flex;align-items:center;justify-content:space-between;gap:12px}
.channels{display:flex;flex-direction:column;gap:6px;max-height:430px;overflow:auto}.channel{display:grid;grid-template-columns:28px 1fr auto;gap:8px;align-items:center;padding:8px;background:var(--panel2);border:1px solid #1d2a39;border-radius:9px}.channel small{color:var(--muted)}button{border:0;border-radius:11px;padding:11px 16px;background:var(--a);color:#06110e;font-weight:800;cursor:pointer}.bar{position:sticky;bottom:12px;margin-top:14px;background:rgba(10,16,23,.94);border:1px solid var(--line);border-radius:14px;padding:12px;display:flex;justify-content:space-between;gap:12px;align-items:center;backdrop-filter:blur(10px)}#status{font-size:12px;color:var(--muted)}.ok{color:var(--good)!important}.bad{color:var(--bad)!important}
@media(max-width:900px){.grid{grid-template-columns:1fr}.fields{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}}
</style></head><body><main>
<div class="top"><div><h1>⚙ Konfiguracja Muchy</h1><div class="sub">Zmiany są zapisywane do config.toml i stosowane na żywo.</div></div>
<div class="nav"><a href="/">🏠 Przegląd</a><a href="/details">📋 Szczegóły</a><a href="/affinity">🤝 Affinity</a><a class="active" href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div></div>
<div class="grid">
<div class="card"><h2>Zachowanie i relacje</h2><div class="fields" id="behavior-fields"></div></div>
<div class="card"><h2>Voice / TTS</h2><div class="fields" id="voice-fields"></div></div>
<div class="card"><h2>Wykluczone kanały tekstowe</h2><div class="channels" id="text-channels"></div></div>
<div class="card"><h2>Wykluczone kanały Voice</h2><div class="channels" id="voice-channels"></div></div>
</div>
<div class="bar"><div id="status">Ładowanie konfiguracji…</div><button id="save">Zapisz konfigurację</button></div>
</main><script>
const $=id=>document.getElementById(id);
let state=null;
const fields={
 behavior:[
  ["speak_threshold","Próg mówienia","number",0.01],
  ["reaction_threshold","Próg reakcji","number",0.01],
  ["reaction_cooldown_seconds","Cooldown reakcji [s]","number",1],
  ["social_learning_enabled","Social learning","bool"],
  ["social_window_seconds","Okno uczenia społecznego [s]","number",10],
  ["word_reuse_reward","Reward za powtórzone słowo","number",0.01],
  ["phrase_reuse_reward","Reward za powtórzoną frazę","number",0.01],
  ["direct_reply_reward","Reward za reply","number",0.01],
  ["self_repeat_penalty","Kara za self-repeat","number",0.01],
  ["user_affinity_positive_step","Affinity + za reakcję","number",0.01],
  ["user_affinity_negative_step","Affinity - za reakcję","number",0.01],
  ["direct_reply_affinity_step","Affinity + za reply","number",0.001],
  ["mention_affinity_step","Affinity + za @Mucha","number",0.001],
  ["continued_conversation_affinity_step","Affinity + za kontynuację rozmowy","number",0.001],
  ["word_reuse_affinity_step","Affinity + za przejęte słowo","number",0.001],
  ["phrase_reuse_affinity_step","Affinity + za przejętą frazę","number",0.001],
  ["voice_join_affinity_step","Affinity + za wejście do VC","number",0.001],
  ["voice_stay_affinity_step","Affinity + za zostanie na VC","number",0.001],
  ["voice_stay_seconds","Po ilu sekundach liczyć wspólne VC","number",1],
  ["tts_stay_affinity_step","Affinity + za zostanie po TTS","number",0.001],
  ["tts_stay_seconds","Ile sekund po TTS obserwować","number",1],
  ["voice_leave_after_join_affinity_step","Affinity - za ucieczkę po wejściu Muchy","number",0.001],
  ["voice_leave_after_join_seconds","Okno ucieczki po wejściu Muchy [s]","number",1],
  ["tts_leave_affinity_step","Affinity - za wyjście po TTS","number",0.001],
  ["tts_leave_seconds","Okno wyjścia po TTS [s]","number",1],
  ["negative_contact_cooldown_seconds","Cooldown naturalnych minusów [s]","number",1],
  ["negative_streak_window_seconds","Okno negative streak [s]","number",10],
  ["negative_streak_multiplier_step","Wzrost mnożnika negative streak","number",0.05],
  ["negative_streak_max_multiplier","Maks. mnożnik negative streak","number",0.05],
  ["positive_contact_cooldown_seconds","Cooldown naturalnych plusów [s]","number",1],
  ["familiar_affinity_threshold","Próg znajomego użytkownika","number",0.01],
  ["user_avoid_threshold","Próg unikania użytkownika","number",0.01],
  ["ignore_disliked_users_text","Nie odpisuj nielubianym","bool"],
  ["avoid_disliked_users_on_voice","Omijaj nielubianych na VC","bool"]
 ],
 voice:[
  ["poll_seconds","Voice poll [s]","number",1],
  ["minimum_dwell_seconds","Minimum dwell [s]","number",1],
  ["maximum_dwell_seconds","Maximum dwell [s]","number",1],
  ["move_threshold","Próg move","number",0.01],
  ["join_threshold","Próg join","number",0.01],
  ["leave_threshold","Próg leave","number",0.01],
  ["include_empty_channels","Uwzględniaj puste VC","bool"],
  ["tts_enabled","TTS włączony","bool"],
  ["tts_interval_seconds","TTS interval [s]","number",1],
  ["tts_volume","Głośność TTS","number",0.05],
  ["stt_enabled","Słuchanie użytkowników (STT)","bool"],
  ["stt_model","Model Whisper","text"],
  ["stt_language","Język STT","text"],
  ["stt_device","Urządzenie STT","text"],
  ["stt_compute_type","Compute type STT","text"],
  ["stt_cpu_threads","Wątki CPU STT","number",1],
  ["stt_silence_seconds","Cisza kończąca wypowiedź [s]","number",0.1],
  ["stt_min_segment_seconds","Min. wypowiedź [s]","number",0.1],
  ["stt_max_segment_seconds","Max. segment [s]","number",0.5],
  ["stt_min_chars","Min. znaków transkrypcji","number",1],
  ["stt_beam_size","Beam size STT","number",1],
  ["random_audio_enabled","Rare audio","bool"]
 ]
};
function renderField(section,[key,label,type,step]){
 const value=state[section][key];
 if(type==="bool")return '<div class="field toggle"><label for="'+section+'-'+key+'">'+label+'</label><input id="'+section+'-'+key+'" type="checkbox" '+(value?'checked':'')+'></div>';
 if(type==="text")return '<div class="field"><label for="'+section+'-'+key+'">'+label+'</label><input id="'+section+'-'+key+'" type="text" value="'+esc(value)+'"></div>';
 return '<div class="field"><label for="'+section+'-'+key+'">'+label+'</label><input id="'+section+'-'+key+'" type="number" step="'+(step||1)+'" value="'+value+'"></div>';
}
function renderChannels(kind){
 const root=$(kind+"-channels"), items=state.channels[kind]||[];
 root.innerHTML=items.map(ch=>'<label class="channel"><input type="checkbox" data-'+kind+'="'+ch.id+'" '+(ch.blocked?'checked':'')+'><div><b>'+esc(ch.name)+'</b><br><small>'+esc(ch.guild)+'</small></div><small>'+ch.id+'</small></label>').join("")||"<small>Brak kanałów.</small>";
}
function esc(v){return String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]))}
async function load(){
 const r=await fetch("/api/config",{cache:"no-store"});if(!r.ok)throw new Error("HTTP "+r.status);state=await r.json();
 $("behavior-fields").innerHTML=fields.behavior.map(f=>renderField("behavior",f)).join("");
 $("voice-fields").innerHTML=fields.voice.map(f=>renderField("voice",f)).join("");
 renderChannels("text");renderChannels("voice");$("status").textContent="Gotowe.";
}
function collect(){
 const out={behavior:{},voice:{}};
 for(const section of ["behavior","voice"])for(const [key,,type] of fields[section]){
  const el=$(section+"-"+key);out[section][key]=type==="bool"?el.checked:(type==="text"?el.value:Number(el.value));
 }
 out.blocked_text_channel_ids=[...document.querySelectorAll("[data-text]:checked")].map(x=>Number(x.dataset.text));
 out.blocked_voice_channel_ids=[...document.querySelectorAll("[data-voice]:checked")].map(x=>Number(x.dataset.voice));
 return out;
}
$("save").onclick=async()=>{
 $("save").disabled=true;$("status").textContent="Zapisywanie…";$("status").className="";
 try{
  const r=await fetch("/api/config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(collect())});
  const d=await r.json();if(!r.ok||!d.ok)throw new Error(d.error||("HTTP "+r.status));
  state=d.config;$("status").textContent="Zapisano: "+(d.changed||[]).join(", ");$("status").className="ok";
  renderChannels("text");renderChannels("voice");
 }catch(e){$("status").textContent="Błąd: "+e.message;$("status").className="bad"}
 finally{$("save").disabled=false}
};
load().catch(e=>{$("status").textContent="Błąd ładowania: "+e.message;$("status").className="bad"});
</script></body></html>"""

AFFINITY_HTML = r"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha — Affinity</title>
<style>
:root{--bg:#080d12;--panel:#101720;--panel2:#0b1219;--line:#233143;--txt:#eef5fc;--muted:#8291a2;--a:#58dac4;--good:#55d98c;--bad:#ff7272;--warn:#f2c14e}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#080d12,#0b1118);color:var(--txt);font-family:Inter,system-ui,"Segoe UI",sans-serif}
main{max-width:1500px;margin:auto;padding:22px}.top{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:18px}
h1{margin:0;font-size:24px}.sub{color:var(--muted);font-size:12px;margin-top:4px}.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#c6d2df;text-decoration:none;border:1px solid var(--line);background:#0e161f;padding:8px 11px;border-radius:10px;font-size:12px}.nav a.active{background:var(--a);border-color:var(--a);color:#06110e;font-weight:800}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:15px;min-width:0}.span2{grid-column:span 2}.card h2{margin:0 0 12px;font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:#aebdca}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.k{background:var(--panel2);border:1px solid #1d2a39;border-radius:11px;padding:10px}.k small{display:block;color:var(--muted);font-size:11px;margin-bottom:5px}.k strong{font-size:15px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:8px;border-bottom:1px solid rgba(35,49,67,.6)}th{color:var(--muted);font-weight:600}.plus{color:var(--good);font-weight:800}.minus{color:var(--bad);font-weight:800}.warn{color:var(--warn)}
.reason{padding:10px 12px;background:var(--panel2);border:1px solid #1d2a39;border-radius:11px;color:#b9c6d3;font-size:12px;line-height:1.5}
.phrases{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.phrase{display:grid;grid-template-columns:1fr auto auto;gap:8px;align-items:center;background:var(--panel2);border:1px solid #1d2a39;border-radius:9px;padding:8px;font-size:12px}
@media(max-width:900px){.grid{grid-template-columns:1fr}.span2{grid-column:auto}.kpis{grid-template-columns:1fr 1fr}.phrases{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}}
</style></head><body><main>
<div class="top"><div><h1>🤝 Affinity / Zasady relacji</h1><div class="sub">Live podgląd tego, co zwiększa i obniża stosunek Muchy do użytkowników.</div></div>
<div class="nav"><a href="/">🏠 Przegląd</a><a href="/details">📋 Szczegóły</a><a class="active" href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div></div>

<div class="grid">
  <div class="card span2">
    <h2>Progi i zasady</h2>
    <div class="kpis">
      <div class="k"><small>ZNAJOMY od</small><strong id="thr-familiar">—</strong></div>
      <div class="k"><small>LUBI od</small><strong id="thr-liked">—</strong></div>
      <div class="k"><small>OMIJA od</small><strong id="thr-avoid">—</strong></div>
      <div class="k"><small>Negative streak</small><strong id="streak">—</strong></div>
    </div>
    <div class="reason" id="effects" style="margin-top:10px">Ładowanie…</div>
  </div>

  <div class="card">
    <h2>➕ Co zwiększa affinity</h2>
    <table><thead><tr><th>Zdarzenie</th><th>Warunek</th><th>Δ</th></tr></thead><tbody id="positive"></tbody></table>
  </div>

  <div class="card">
    <h2>➖ Co obniża affinity</h2>
    <table><thead><tr><th>Zdarzenie</th><th>Warunek</th><th>Δ</th></tr></thead><tbody id="negative"></tbody></table>
  </div>

  <div class="card span2">
    <h2>🤬 Frazy odrzucające / severity</h2>
    <div class="reason" style="margin-bottom:10px">Fraza działa tylko, gdy jest skierowana do Muchy: reply, mention/„Mucha”, albo na VC krótko po jej TTS. Silniejsze frazy dają większy minus.</div>
    <div class="phrases" id="phrases"></div>
  </div>

  <div class="card span2">
    <h2>👥 Aktualne relacje</h2>
    <table><thead><tr><th>Użytkownik</th><th>Affinity</th><th>👍 reakcje</th><th>👎 reakcje</th><th>Status</th></tr></thead><tbody id="users"></tbody></table>
  </div>

  <div class="card span2">
    <h2>Ostatni sygnał społeczny</h2>
    <div class="reason" id="last-social">—</div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const esc=v=>String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));
const num=v=>Number(v||0);
function signed(v){v=num(v);return (v>=0?"+":"")+v.toFixed(3)}
function renderRows(items,kind){
  return (items||[]).map(x=>'<tr><td>'+esc(x.event)+'</td><td>'+esc(x.condition||"—")+'</td><td class="'+(kind==="plus"?"plus":"minus")+'">'+signed(x.delta)+'</td></tr>').join("")||'<tr><td colspan="3">Brak danych.</td></tr>';
}
function render(d){
  const r=d.affinity_rules||{},t=r.thresholds||{},st=r.negative_streak||{};
  $("thr-familiar").textContent=signed(t.familiar);
  $("thr-liked").textContent=signed(t.liked);
  $("thr-avoid").textContent=signed(t.avoid);
  $("streak").textContent="+"+num(st.step).toFixed(2)+" / max ×"+num(st.max_multiplier).toFixed(2)+" / "+Math.round(num(st.window_seconds))+"s";
  $("effects").innerHTML='<b>Po progu OMIJA:</b> '+(r.avoid_effects||[]).map(esc).join(" • ")+'<br><b>Chaser:</b> '+esc(r.chaser_exception||"—")+
    '<br><b>Cooldown:</b> plusy '+Math.round(num(r.positive_cooldown_seconds))+'s • minusy '+Math.round(num(r.negative_cooldown_seconds))+'s';
  $("positive").innerHTML=renderRows(r.positive,"plus");
  $("negative").innerHTML=renderRows(r.negative,"minus");
  $("phrases").innerHTML=(r.verbal_rejections||[]).map(x=>
    '<div class="phrase"><span>'+esc(x.phrase)+'</span><span class="warn">sev '+num(x.severity).toFixed(2)+'</span><span class="minus">'+signed(x.delta)+'</span></div>'
  ).join("")||'<div class="reason">Brak fraz.</div>';

  $("users").innerHTML=(d.user_affinities||[]).map(u=>{
    const a=num(u.affinity),status=a<=num(t.avoid)?"OMIJA":a>=num(t.liked)?"LUBI":a>=num(t.familiar)?"ZNAJOMY":"NEUTRAL";
    const cls=a<0?"minus":a>0?"plus":"";
    return '<tr><td>'+esc(u.display_name||u.user_id)+'</td><td class="'+cls+'">'+signed(a)+'</td><td>'+num(u.positive_reactions)+'</td><td>'+num(u.negative_reactions)+'</td><td class="'+cls+'">'+status+'</td></tr>';
  }).join("")||'<tr><td colspan="5">Brak relacji.</td></tr>';

  const s=d.social_debug||{};
  $("last-social").innerHTML='<b>'+esc(s.event||"—")+'</b> • '+esc(s.user_name||"—")+' • '+esc(s.detail||"—")+' • Δ '+signed(s.amount||0)+' • affinity '+signed(s.affinity||0);
}
async function update(){
  try{
    const x=await fetch("/api/state",{cache:"no-store"});
    if(x.status===401){location="/login";return}
    if(!x.ok)throw new Error("HTTP "+x.status);
    render(await x.json());
  }catch(e){$("last-social").textContent="Błąd: "+e.message}
}
setInterval(update,2000);update();
</script></main></body></html>"""

LOGIN_HTML = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha Control Center — logowanie</title>
<style>
:root{color-scheme:dark;--bg:#070b10;--card:#101720;--line:#243142;--txt:#eef6ff;--muted:#8291a2;--a:#59ddc6;--b:#6ea8fe;--bad:#ff7474}
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;background:
radial-gradient(circle at 20% 10%,rgba(89,221,198,.12),transparent 35%),
radial-gradient(circle at 80% 90%,rgba(110,168,254,.14),transparent 35%),var(--bg);
font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--txt)}
.box{width:min(420px,calc(100% - 28px));background:rgba(16,23,32,.94);border:1px solid var(--line);
border-radius:22px;padding:28px;box-shadow:0 28px 80px rgba(0,0,0,.35)}
.logo{font-size:42px}.eyebrow{color:var(--a);text-transform:uppercase;letter-spacing:.14em;font-size:11px;font-weight:800}
h1{font-size:25px;margin:8px 0 5px}p{color:var(--muted);margin:0 0 22px;line-height:1.5}
label{display:block;color:#afbdcb;font-size:12px;margin:14px 0 7px}
input{width:100%;padding:12px 13px;background:#0a1017;color:var(--txt);border:1px solid #263548;border-radius:11px;outline:0}
input:focus{border-color:var(--a);box-shadow:0 0 0 3px rgba(89,221,198,.1)}
button{width:100%;margin-top:18px;padding:12px;border:0;border-radius:11px;font-weight:800;color:#06110e;
background:linear-gradient(90deg,var(--a),#79e2d1);cursor:pointer}
.err{color:var(--bad);font-size:13px;margin-top:12px}.foot{text-align:center;color:#607081;font-size:11px;margin-top:18px}
</style></head>
<body><form class="box" method="post" action="/login">
<div class="logo">🪰</div><div class="eyebrow">Mucha Control Center</div>
<h1>Prywatny dashboard</h1><p>Status VPS, Muchy, Chasera, voice i connectome w jednym miejscu.</p>
<label>Użytkownik</label><input name="username" autocomplete="username" required>
<label>Hasło</label><input type="password" name="password" autocomplete="current-password" required>
<div class="err">__ERROR__</div>
<button type="submit">Wejdź do panelu</button>
<div class="foot">Sesja jest zapisywana tylko w bezpiecznym cookie HTTP-only.</div>
</form></body></html>"""

OVERVIEW_HTML = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha Control Center</title>
<style>
:root{--bg:#070b10;--panel:#0f161f;--panel2:#0a1118;--line:#213043;--txt:#edf5fd;--muted:#8190a1;
--a:#58dac4;--blue:#6ea8fe;--good:#57db91;--warn:#f0c45b;--bad:#ff7272}
*{box-sizing:border-box}body{margin:0;background:
radial-gradient(circle at 12% 0%,rgba(88,218,196,.10),transparent 30%),
radial-gradient(circle at 88% 0%,rgba(110,168,254,.10),transparent 32%),
linear-gradient(180deg,#070b10,#0a1017 60%,#080c11);color:var(--txt);
font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1480px;margin:auto;padding:22px}.top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:18px}
.brand{display:flex;gap:13px;align-items:center}.logo{font-size:37px;filter:drop-shadow(0 0 18px rgba(88,218,196,.22))}
h1{margin:0;font-size:24px}.sub{margin-top:4px;color:var(--muted);font-size:12px}.nav{display:flex;gap:8px;flex-wrap:wrap}
.nav a{color:#b9c8d7;text-decoration:none;background:#0e1720;border:1px solid var(--line);padding:8px 11px;border-radius:10px;font-size:12px}
.nav a.active{color:#07110e;background:var(--a);border-color:var(--a);font-weight:800}
.nav a:hover{border-color:#3b566f;color:white}.nav a.active:hover{color:#07110e;border-color:var(--a)}.hero{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}
.hero-card,.card{background:rgba(15,22,31,.94);border:1px solid var(--line);border-radius:16px}
.hero-card{padding:14px}.hero-card small,.k small{display:block;color:var(--muted);font-size:11px;margin-bottom:6px}
.hero-card strong{font-size:18px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.card{padding:15px;min-width:0}
.card h2{margin:0 0 12px;font-size:12px;color:#aebdcb;text-transform:uppercase;letter-spacing:.1em}
.row{display:flex;justify-content:space-between;gap:16px;padding:8px 0;border-bottom:1px solid rgba(33,48,67,.65);font-size:13px}
.row:last-child{border-bottom:0}.row span{color:var(--muted)}.row strong{text-align:right;word-break:break-word}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px}.k{background:var(--panel2);border:1px solid #1e2b3a;border-radius:11px;padding:10px;min-width:0}
.k strong{font-size:14px;word-break:break-word}.status{display:inline-flex;align-items:center;gap:7px}.dot{width:8px;height:8px;border-radius:50%;background:var(--good);box-shadow:0 0 12px rgba(87,219,145,.55)}
.dot.bad{background:var(--bad);box-shadow:0 0 12px rgba(255,114,114,.5)}.dot.warn{background:var(--warn)}
.good{color:var(--good)}.badc{color:var(--bad)}.warnc{color:var(--warn)}.accent{color:var(--a)}
.actions{display:flex;flex-direction:column;gap:7px}.act{display:grid;grid-template-columns:95px 1fr 45px;gap:8px;align-items:center;font-size:12px}
.track{height:8px;background:#071019;border:1px solid #1d2b39;border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--a))}
.logs{background:#070d13;border:1px solid #1c2937;border-radius:11px;padding:10px;max-height:270px;overflow:auto;
font:11px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;color:#aebccc}
.logs .err{color:#ff9393}.span2{grid-column:span 2}.progress{height:8px;background:#071019;border-radius:999px;overflow:hidden;border:1px solid #1d2b39;margin-top:7px}
.progress>div{height:100%;background:linear-gradient(90deg,var(--a),var(--blue))}
.footer{text-align:right;color:#5e6e7d;font-size:11px;margin-top:12px}
@media(max-width:900px){.hero{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.span2{grid-column:auto}}
@media(max-width:560px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.hero{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}}
</style></head>
<body><main>
<div class="top">
  <div class="brand"><div class="logo">🪰</div><div><h1>Mucha Control Center</h1><div class="sub">VPS • Discord • Connectome • Chaser • Audio</div></div></div>
  <div class="nav"><a class="active" href="/">🏠 Przegląd</a><a href="/details">📋 Szczegóły</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/api/state">JSON</a><a href="/logout">Wyloguj</a></div>
</div>

<section class="hero">
 <div class="hero-card"><small>Mucha</small><strong id="hero-mucha">łączenie…</strong></div>
 <div class="hero-card"><small>Chaser</small><strong id="hero-chaser">łączenie…</strong></div>
 <div class="hero-card"><small>Voice</small><strong id="hero-voice">—</strong></div>
 <div class="hero-card"><small>Następny pościg</small><strong id="hero-next">—</strong></div>
</section>

<section class="grid">
 <div class="card">
  <h2>🪰 Mucha Service</h2>
  <div class="kpis">
   <div class="k"><small>RAM</small><strong id="mucha-ram">—</strong></div>
   <div class="k"><small>Uptime</small><strong id="mucha-up">—</strong></div>
   <div class="k"><small>PID</small><strong id="mucha-pid">—</strong></div>
  </div>
  <div class="row"><span>Stan</span><strong id="mucha-state">—</strong></div>
  <div class="row"><span>Connectome</span><strong id="connectome">—</strong></div>
  <div class="row"><span>Backend</span><strong id="backend">—</strong></div>
  <div class="row"><span>Ostatni bodziec</span><strong id="last-event">—</strong></div>
  <div class="row"><span>Ostatnia akcja</span><strong id="last-action">—</strong></div>
 </div>

 <div class="card">
  <h2>🏃 Mucha Chaser</h2>
  <div class="kpis">
   <div class="k"><small>RAM</small><strong id="chaser-ram">—</strong></div>
   <div class="k"><small>Cykl</small><strong id="chaser-cycle">—</strong></div>
   <div class="k"><small>Pościg</small><strong id="chaser-duration">—</strong></div>
  </div>
  <div class="row"><span>Stan</span><strong id="chaser-state">—</strong></div>
  <div class="row"><span>Kanał</span><strong id="chaser-channel">—</strong></div>
  <div class="row"><span>Następna runda</span><strong id="chaser-next">—</strong></div>
  <div class="row"><span>Ostatnie zdarzenie</span><strong id="chaser-event">—</strong></div>
 </div>

 <div class="card">
  <h2>🖥 VPS</h2>
  <div class="kpis">
   <div class="k"><small>Uptime</small><strong id="vps-up">—</strong></div>
   <div class="k"><small>Load</small><strong id="vps-load">—</strong></div>
   <div class="k"><small>Dysk</small><strong id="vps-disk">—</strong></div>
  </div>
  <div class="row"><span>RAM</span><strong id="vps-ram">—</strong></div>
  <div class="progress"><div id="ram-bar" style="width:0"></div></div>
  <div class="row"><span>Wolny RAM</span><strong id="vps-free">—</strong></div>
  <div class="row"><span>Aktualizacja panelu</span><strong id="updated">—</strong></div>
 </div>

 <div class="card">
  <h2>🔊 Audio / TTS</h2>
  <div class="row"><span>Status</span><strong id="audio-status">—</strong></div>
  <div class="row"><span>Etap</span><strong id="audio-stage">—</strong></div>
  <div class="row"><span>Cel</span><strong id="audio-target">—</strong></div>
  <div class="row"><span>Plik</span><strong id="audio-file">—</strong></div>
  <div class="row"><span>Tekst</span><strong id="audio-text">—</strong></div>
  <div class="row"><span>STT</span><strong id="stt-status">—</strong></div>
  <div class="row"><span>Usłyszała</span><strong id="stt-heard">—</strong></div>
 </div>

 <div class="card">
  <h2>🧠 Connectome Output</h2>
  <div class="actions" id="actions"></div>
 </div>

 <div class="card">
  <h2>📊 Brain Snapshot</h2>
  <div class="row"><span>Neurony</span><strong id="neurons">—</strong></div>
  <div class="row"><span>Połączenia</span><strong id="connections">—</strong></div>
  <div class="row"><span>Aktywne |a| &gt; .1</span><strong id="active-neurons">—</strong></div>
  <div class="row"><span>Mean |a|</span><strong id="mean-a">—</strong></div>
  <div class="row"><span>Reward trace</span><strong id="reward-trace">—</strong></div>
  <div class="row"><span>Tick</span><strong id="ticks">—</strong></div>
 </div>

 <div class="card">
  <h2>📜 Mucha — ostatnie logi</h2>
  <div class="logs" id="mucha-logs">czekam…</div>
 </div>
 <div class="card">
  <h2>📜 Chaser — ostatnie logi</h2>
  <div class="logs" id="chaser-logs">czekam…</div>
 </div>
</section>
<div class="footer">Mucha Control Center • live refresh</div>
</main>
<script>
const $=id=>document.getElementById(id);
const fmtBytes=n=>{n=Number(n||0);if(!n)return "0 B";const u=["B","KB","MB","GB","TB"];let i=0;while(n>=1024&&i<u.length-1){n/=1024;i++}return n.toFixed(i>1?2:1)+" "+u[i]};
const dur=s=>{s=Math.max(0,Number(s||0));const d=Math.floor(s/86400);s%=86400;const h=Math.floor(s/3600);s%=3600;const m=Math.floor(s/60);const x=Math.floor(s%60);return (d?d+"d ":"")+(h?h+"h ":"")+(m?m+"m ":"")+x+"s"};
const nfmt=n=>Number(n||0).toLocaleString("pl-PL");
const esc=v=>String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));
let last=null;
function serviceLabel(s){const ok=s&&s.active;return '<span class="status"><i class="dot '+(ok?'':'bad')+'"></i><span class="'+(ok?'good':'badc')+'">'+(ok?'ONLINE':'OFFLINE')+'</span></span>'}
function firstGuild(cs){const g=Object.values((cs&&cs.guilds)||{});return g.find(x=>x.active_chase)||g[0]||{}}
function nextText(ts){if(!ts)return "—";const sec=Number(ts)-Date.now()/1000;if(sec<=0)return "teraz";return "za "+dur(sec)}
function renderActions(scores){const order=["speak","react","voice_join","voice_move","voice_leave","explore","stay"];const dom=Object.entries(scores||{}).sort((a,b)=>b[1]-a[1])[0]?.[0];
 $("actions").innerHTML=order.map(k=>{const v=Number((scores||{})[k]||0);return '<div class="act"><b class="'+(k===dom?'accent':'')+'">'+(k===dom?'▶ ':'')+k+'</b><div class="track"><div class="fill" style="width:'+Math.max(0,Math.min(100,v*100))+'%"></div></div><span>'+v.toFixed(3)+'</span></div>'}).join("")}
function render(d){
 last=d;const s=d.snapshot||{},diag=s.diag||{},m=d.services?.mucha||{},ch=d.services?.chaser||{},sys=d.system||{},cs=d.chaser_status||{},cg=firstGuild(cs),a=s.audio_debug||{},stt=s.stt_debug||{};
 $("hero-mucha").innerHTML=serviceLabel(m);$("hero-chaser").innerHTML=serviceLabel(ch);
 $("hero-voice").textContent=s.voice||"poza voice";$("hero-next").textContent=nextText(cg.next_round_at);
 $("mucha-ram").textContent=fmtBytes(m.memory_bytes);$("mucha-up").textContent=dur(m.uptime_seconds);$("mucha-pid").textContent=m.pid||"—";$("mucha-state").innerHTML=serviceLabel(m);
 $("connectome").textContent=nfmt(diag.neurons)+" / "+nfmt(diag.connections);$("backend").textContent=(diag.backend||"cpu").toUpperCase()+" • "+(diag.device||"CPU");
 $("last-event").textContent=s.last_event||"—";$("last-action").textContent=s.last_action||"—";
 $("chaser-ram").textContent=fmtBytes(ch.memory_bytes);$("chaser-cycle").textContent=dur(cs.interval_seconds||0);$("chaser-duration").textContent=dur(cs.duration_seconds||0);
 $("chaser-state").innerHTML=serviceLabel(ch)+" • <span class='"+(cg.active_chase?"badc":"accent")+"'>"+esc(cg.state||"—")+"</span>";
 $("chaser-channel").textContent=cg.current_voice_channel||"poza voice";$("chaser-next").textContent=nextText(cg.next_round_at);$("chaser-event").textContent=cg.last_event||"—";
 $("vps-up").textContent=dur(sys.uptime_seconds);$("vps-load").textContent=(sys.load||[]).map(x=>Number(x).toFixed(2)).join(" / ");
 $("vps-disk").textContent=fmtBytes(sys.disk_used)+" / "+fmtBytes(sys.disk_total);$("vps-ram").textContent=fmtBytes(sys.mem_used)+" / "+fmtBytes(sys.mem_total);
 $("vps-free").textContent=fmtBytes(sys.mem_available);$("ram-bar").style.width=Math.max(0,Math.min(100,Number(sys.mem_percent||0)))+"%";$("updated").textContent=new Date().toLocaleTimeString("pl-PL");
 $("audio-status").textContent=a.status||"—";$("audio-stage").textContent=a.stage||"—";$("audio-target").textContent=(a.guild||"—")+" / "+(a.channel||"—");
 $("audio-file").textContent=a.file||"—";$("audio-text").textContent=a.text||"—";
 $("stt-status").textContent=(stt.status||"—")+" • "+(stt.model||"—");
 $("stt-heard").textContent=stt.text?((stt.user||"ktoś")+": "+stt.text):"—";
 $("neurons").textContent=nfmt(diag.neurons);$("connections").textContent=nfmt(diag.connections);$("active-neurons").textContent=nfmt(diag.active_abs_gt_0_1);
 $("mean-a").textContent=Number(diag.mean_abs||0).toFixed(5);$("reward-trace").textContent=Number(diag.reward_trace||0).toFixed(4);$("ticks").textContent=nfmt(diag.ticks);
 renderActions(s.scores||{});
 $("mucha-logs").textContent=(d.logs?.mucha||[]).join("\n")||"brak logów";$("chaser-logs").textContent=(d.logs?.chaser||[]).join("\n")||"brak logów";
}
async function update(){try{const r=await fetch("/api/overview",{cache:"no-store"});if(r.status===401){location="/login";return}if(!r.ok)throw new Error("HTTP "+r.status);render(await r.json())}catch(e){console.error(e);$("hero-mucha").innerHTML='<span class="badc">BRAK POŁĄCZENIA</span>'}}
setInterval(update,2500);setInterval(()=>{if(last){const g=firstGuild(last.chaser_status||{});$("hero-next").textContent=nextText(g.next_round_at);$("chaser-next").textContent=nextText(g.next_round_at)}},1000);update();
</script></body></html>"""

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
    <div class="brand"><div class="fly">🪰</div><div class="title"><h1>Mucha — Szczegóły</h1><p id="source">łączenie…</p></div></div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <a href="/" style="color:#c5d3e1;text-decoration:none;border:1px solid var(--line);background:#0f161f;border-radius:10px;padding:7px 10px;font-size:12px">🏠 Przegląd</a>
      <a href="/details" style="color:#07110e;text-decoration:none;border:1px solid var(--accent);background:var(--accent);border-radius:10px;padding:7px 10px;font-size:12px;font-weight:700">📋 Szczegóły</a>
      <a href="/affinity" style="color:#c5d3e1;text-decoration:none;border:1px solid var(--line);background:#0f161f;border-radius:10px;padding:7px 10px;font-size:12px">🤝 Affinity</a>
      <a href="/config" style="color:#c5d3e1;text-decoration:none;border:1px solid var(--line);background:#0f161f;border-radius:10px;padding:7px 10px;font-size:12px">⚙ Konfiguracja</a>
      <div class="badges">
      <div class="badge"><span class="dot"></span><span id="live">LIVE</span></div>
      <div class="badge" id="backend">backend: —</div>
      <div class="badge" id="device">device: —</div>
      <div class="badge" id="clock">—</div>
      </div>
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

    <div class="card span3">
      <h2>Learning Since Startup</h2>
      <div class="learning-grid" style="grid-template-columns:repeat(4,1fr)">
        <div class="kpi"><small>czas uczenia</small><strong id="session-age">—</strong></div>
        <div class="kpi"><small>nowe znaki</small><strong id="session-chars">—</strong></div>
        <div class="kpi"><small>nowe wiadomości / wypowiedzi</small><strong id="session-messages">—</strong></div>
        <div class="kpi"><small>nowe przejścia znaków</small><strong id="session-transitions">—</strong></div>
        <div class="kpi"><small>transkrypcje voice</small><strong id="session-stt">—</strong></div>
        <div class="kpi"><small>reward events</small><strong id="session-rewards">—</strong></div>
        <div class="kpi"><small>łączny +reward</small><strong class="ok" id="session-positive">—</strong></div>
        <div class="kpi"><small>łączny -reward</small><strong class="no" id="session-negative">—</strong></div>
      </div>
      <div class="events" style="margin-top:10px">
        <div class="event">
          <small>Skumulowane uczenie connectome przez reward()</small>
          <div class="metric"><span>Unikalne neurony zmienione przez reward</span><strong id="session-neurons">—</strong></div>
          <div class="metric"><span>Aktualizacje bias w rewardach</span><strong id="session-bias-updates">—</strong></div>
          <div class="metric"><span>Średnie skumulowane |Δ bias|</span><strong id="session-bias-mean">—</strong></div>
          <div class="metric"><span>Max skumulowane |Δ bias|</span><strong id="session-bias-max">—</strong></div>
        </div>
        <div class="event">
          <small>Najbardziej zmieniony neuron od startu</small>
          <div class="metric"><span>FlyWire root_id</span><strong id="session-top-root">—</strong></div>
          <div class="metric"><span>Δ bias</span><strong id="session-top-delta">—</strong></div>
          <div class="metric"><span>Aktualny bias</span><strong id="session-top-bias">—</strong></div>
          <div class="reason" id="session-summary">Czekam na pierwszą trwałą zmianę.</div>
        </div>
      </div>
    </div>

    <div class="card span2">
      <h2>Social Learning / Relacje</h2>
      <div class="learning-grid">
        <div class="kpi"><small>ostatni sygnał</small><strong id="social-event">—</strong></div>
        <div class="kpi"><small>szczegół</small><strong id="social-detail">—</strong></div>
        <div class="kpi"><small>reward</small><strong id="social-amount">—</strong></div>
        <div class="kpi"><small>próg unikania</small><strong id="social-threshold">—</strong></div>
      </div>
      <div class="reason" id="social-last">Czekam na pierwszy sygnał społeczny…</div>
      <div class="events">
        <div class="event">
          <small>Relacje z użytkownikami</small>
          <table>
            <thead><tr><th>Użytkownik</th><th>Affinity</th><th>👍</th><th>👎</th><th>Status</th></tr></thead>
            <tbody id="social-users"></tbody>
          </table>
        </div>
        <div class="event">
          <small>Najlepiej utrwalone słowa</small>
          <table>
            <thead><tr><th>Słowo</th><th>Potw.</th><th>Osoby</th><th>Reward</th></tr></thead>
            <tbody id="social-words"></tbody>
          </table>
        </div>
      </div>
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
      <h2>Audio Debug</h2>
      <div class="learning-grid" style="grid-template-columns:repeat(4,1fr)">
        <div class="kpi"><small>Status</small><strong id="audio-status">—</strong></div>
        <div class="kpi"><small>Etap</small><strong id="audio-stage">—</strong></div>
        <div class="kpi"><small>Serwer / kanał</small><strong id="audio-target">—</strong></div>
        <div class="kpi"><small>Plik</small><strong id="audio-file">—</strong></div>
      </div>
      <div class="metric"><span>FFmpeg</span><strong id="audio-ffmpeg">—</strong></div>
      <div class="metric"><span>Rozmiar pliku</span><strong id="audio-size">—</strong></div>
      <div class="metric"><span>Discord playback</span><strong id="audio-playing">—</strong></div>
      <div class="metric"><span>Voice state</span><strong id="audio-vstate">—</strong></div>
      <div class="metric"><span>Tekst TTS</span><strong id="audio-text">—</strong></div>
      <div class="reason" id="audio-error">Brak błędów audio.</div>
    </div>

    <div class="card span3">
      <h2>Voice Recognition / STT</h2>
      <div class="learning-grid" style="grid-template-columns:repeat(4,1fr)">
        <div class="kpi"><small>Status</small><strong id="stt-debug-status">—</strong></div>
        <div class="kpi"><small>Model</small><strong id="stt-debug-model">—</strong></div>
        <div class="kpi"><small>Użytkownik</small><strong id="stt-debug-user">—</strong></div>
        <div class="kpi"><small>Długość</small><strong id="stt-debug-duration">—</strong></div>
      </div>
      <div class="metric"><span>Serwer / kanał</span><strong id="stt-debug-target">—</strong></div>
      <div class="metric"><span>Język / pewność</span><strong id="stt-debug-language">—</strong></div>
      <div class="metric"><span>Kolejka</span><strong id="stt-debug-pending">—</strong></div>
      <div class="metric"><span>Ostatnia transkrypcja</span><strong id="stt-debug-text">—</strong></div>
      <div class="reason" id="stt-debug-error">Brak błędów STT.</div>
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
function renderAudioDebug(a){
  a=a||{};
  const status=String(a.status||"—");
  $("audio-status").textContent=status;
  $("audio-status").className=status==="ERROR"?"no":status==="PLAYING"?"ok":"";
  $("audio-stage").textContent=a.stage||"—";
  $("audio-target").textContent=(a.guild||"—")+" / "+(a.channel||"—");
  $("audio-file").textContent=a.file||"—";
  $("audio-ffmpeg").textContent=a.ffmpeg||"—";
  $("audio-size").textContent=nfmt(a.file_size||0)+" B";
  $("audio-playing").textContent=(a.playing==null?"—":(a.playing?"PLAYING":"STOPPED"))+
    " / "+(a.connected==null?"—":(a.connected?"CONNECTED":"DISCONNECTED"));
  $("audio-vstate").textContent=
    "mute="+Boolean(a.server_muted)+
    " deaf="+Boolean(a.server_deafened)+
    " suppress="+Boolean(a.suppressed);
  $("audio-text").textContent=a.text||"—";
  const err=a.error||"";
  $("audio-error").innerHTML=err
    ? '<b class="no">BŁĄD:</b> '+esc(err)
    : 'Brak błędów audio.';
}

function renderSttDebug(s){
  s=s||{};
  $("stt-debug-status").textContent=s.status||"—";
  $("stt-debug-model").textContent=(s.model||"—")+" / "+(s.device||"—")+" / "+(s.compute_type||"—");
  $("stt-debug-user").textContent=s.user||"—";
  $("stt-debug-duration").textContent=Number(s.duration||0).toFixed(2)+" s";
  $("stt-debug-target").textContent=(s.guild||"—")+" / "+(s.channel||"—");
  const p=s.language_probability;
  $("stt-debug-language").textContent=(s.language||"—")+" / "+(p==null?"—":(Number(p)*100).toFixed(1)+"%");
  $("stt-debug-pending").textContent=String(s.pending||0);
  $("stt-debug-text").textContent=s.text||"—";
  const err=s.error||"";
  $("stt-debug-error").innerHTML=err?'<b class="no">BŁĄD:</b> '+esc(err):'Brak błędów STT.';
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
      const explore=ch.exploration_score==null?"—":Number(ch.exploration_score).toFixed(3);
      const novelty=ch.novelty==null?"—":(Number(ch.novelty)*100).toFixed(0)+"%";
      const visitAge=ch.visit_age==null?"never":Number(ch.visit_age).toFixed(0)+"s";
      const status=ch.eligible?"OK":ch.status;
      const cls=ch.eligible?"ok":(ch.status==="AFK"?"warn":"no");
      return '<tr>'+
        '<td>'+(ch.current?"▶ ":"")+esc(ch.name)+'</td>'+
        '<td>'+Number(ch.humans||0)+'</td>'+
        '<td class="'+(ch.view?"ok":"no")+'">'+(ch.view?"YES":"NO")+'</td>'+
        '<td class="'+(ch.connect?"ok":"no")+'">'+(ch.connect?"YES":"NO")+'</td>'+
        '<td>'+aff+'</td>'+
        '<td>'+explore+'</td>'+
        '<td>'+novelty+'</td>'+
        '<td>'+visitAge+'</td>'+
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
      '<div class="voice-pill"><small>CHASE</small><strong class="'+(v.chaser_active?"no":"ok")+'">'+(v.chaser_active?"ACTIVE ":"idle ")+Number(v.chaser_remaining||0).toFixed(1)+' s</strong></div>'+
      '<div class="voice-pill"><small>chaser id</small><strong>'+esc(v.chaser_id||"—")+'</strong></div>'+
      '<div class="voice-pill"><small>threat magnitude</small><strong>'+Number(v.threat_magnitude||0).toFixed(3)+'</strong></div>'+
      '<div class="voice-pill"><small>effective move</small><strong>'+((v.effective_move_score==null)?"—":Number(v.effective_move_score).toFixed(3))+'</strong></div>'+
      '<div class="voice-pill"><small>effective margin</small><strong>'+((v.effective_move_margin==null)?"—":Number(v.effective_move_margin).toFixed(3))+'</strong></div>'+
      '<div class="voice-pill"><small>escape target</small><strong>'+esc(v.escape_target||"—")+'</strong></div>'+
    '</div>'+
    '<div class="reason"><b>'+esc(v.decision||"—")+'</b> — '+esc(v.reason||"—")+'</div>'+
    '<table><thead><tr><th>Kanał</th><th>Ludzie</th><th>View</th><th>Connect</th><th>Affinity</th><th>Explore</th><th>Novelty</th><th>Last visit</th><th>Status</th></tr></thead><tbody>'+channels+'</tbody></table>';
  }).join('<div style="height:14px"></div>');
}
function sessionDuration(seconds){
  seconds=Math.max(0,Number(seconds||0));
  const d=Math.floor(seconds/86400);seconds%=86400;
  const h=Math.floor(seconds/3600);seconds%=3600;
  const m=Math.floor(seconds/60);const s=Math.floor(seconds%60);
  return (d?d+"d ":"")+(h?h+"h ":"")+(m?m+"m ":"")+s+"s";
}
function renderLearningSinceStart(x){
  x=x||{};
  $("session-age").textContent=sessionDuration(x.uptime_seconds||0);
  $("session-chars").textContent="+"+nfmt(x.language_chars||0);
  $("session-messages").textContent="+"+nfmt(x.language_messages||0);
  $("session-transitions").textContent="+"+nfmt(x.language_transitions||0);
  $("session-stt").textContent=nfmt(x.voice_transcripts||0);
  $("session-rewards").textContent=nfmt(x.reward_events||0)+
    " ("+nfmt(x.positive_reward_events||0)+"+ / "+
    nfmt(x.negative_reward_events||0)+"-)";
  $("session-positive").textContent="+"+Number(x.positive_reward_total||0).toFixed(3);
  $("session-negative").textContent=Number(x.negative_reward_total||0).toFixed(3);
  $("session-neurons").textContent=nfmt(x.unique_neurons_changed||0);
  $("session-bias-updates").textContent=nfmt(x.bias_update_operations||0);
  $("session-bias-mean").textContent=Number(x.bias_mean_abs_delta||0).toExponential(3);
  $("session-bias-max").textContent=Number(x.bias_max_abs_delta||0).toExponential(3);

  const top=x.top_changed_neuron||null;
  $("session-top-root").textContent=top?String(top.root_id):"—";
  const delta=top?Number(top.delta||0):0;
  $("session-top-delta").textContent=top?(delta>=0?"+":"")+delta.toExponential(3):"—";
  $("session-top-delta").className=top?(delta>0?"ok":delta<0?"no":""):"";
  $("session-top-bias").textContent=top?
    (Number(top.current_bias||0)>=0?"+":"")+Number(top.current_bias||0).toExponential(3):"—";

  const changed=Number(x.unique_neurons_changed||0);
  const rewards=Number(x.reward_events||0);
  const messages=Number(x.language_messages||0);
  if(changed||messages){
    const parts=[];
    if(messages)parts.push("+"+nfmt(messages)+" nowych próbek językowych");
    if(changed)parts.push(nfmt(changed)+" neuronów zmienionych przez reward()");
    $("session-summary").innerHTML='<b class="ok">UCZENIE WIDOCZNE</b> • '+parts.join(" • ");
  }else if(rewards){
    $("session-summary").innerHTML='<b>Rewardy wystąpiły</b>, ale skumulowana zmiana bias jest poniżej progu pomiaru.';
  }else{
    $("session-summary").textContent='Czekam na nowe próbki językowe albo pierwszy reward.';
  }
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
function renderSocial(s){
  const d=s.social_debug||{}, settings=s.social_settings||{};
  const amount=Number(d.amount||0);
  $("social-event").textContent=d.event||"—";
  $("social-detail").textContent=d.detail||"—";
  $("social-amount").textContent=(amount>=0?"+":"")+amount.toFixed(2);
  $("social-amount").className=amount>0?"ok":amount<0?"no":"";
  const threshold=Number(settings.user_avoid_threshold??-0.35);
  const familiar=Number(settings.familiar_affinity_threshold??0.10);
  $("social-threshold").textContent=threshold.toFixed(2);
  $("social-last").innerHTML=d.user_name
    ? '<b>'+esc(d.user_name)+'</b> • affinity '+Number(d.affinity||0).toFixed(2)+' • '+esc(d.event||"—")
    : esc(d.event||"Czekam na pierwszy sygnał społeczny…");

  $("social-users").innerHTML=(s.user_affinities||[]).slice(0,12).map(u=>{
    const a=Number(u.affinity||0);
    const status=a<=threshold?"OMIJA":a>=0.35?"LUBI":a>=familiar?"ZNAJOMY":"NEUTRAL";
    const cls=a<=threshold?"no":a>=0.1?"ok":"";
    return '<tr><td>'+esc(u.display_name||u.user_id)+'</td>'+
      '<td class="'+cls+'">'+(a>=0?"+":"")+a.toFixed(2)+'</td>'+
      '<td>'+nfmt(u.positive_reactions||0)+'</td>'+
      '<td>'+nfmt(u.negative_reactions||0)+'</td>'+
      '<td class="'+cls+'">'+status+'</td></tr>';
  }).join("")||'<tr><td colspan="5">Brak relacji.</td></tr>';

  $("social-words").innerHTML=(s.word_feedback||[]).slice(0,12).map(w=>
    '<tr><td>'+esc(w.word)+'</td><td>'+nfmt(w.confirmations||0)+'</td>'+
    '<td>'+nfmt(w.unique_users||0)+'</td><td class="ok">'+
    Number(w.reward||0).toFixed(2)+'</td></tr>'
  ).join("")||'<tr><td colspan="4">Brak potwierdzonych słów.</td></tr>';
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
    renderLearningSinceStart(s.learning_since_start||{});
    renderSocial(s);
    renderActionHistory(s.action_history||[]);
    renderGuildLearningContext(s.guild_learning_context||[]);
    drawRewardChart(s.reward_history||[],d.reward_trace);
    renderAudioDebug(s.audio_debug||{});
    renderSttDebug(s.stt_debug||{});
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
        auth_enabled: bool = True,
        auth_username: str = "admin",
        auth_password_env: str = "MUCHA_DASHBOARD_PASSWORD",
        session_hours: int = 168,
        chaser_status_file: str = "/opt/mucha-chaser/state/chaser_status.json",
        config_provider: ConfigProvider | None = None,
        config_updater: ConfigUpdater | None = None,
    ):
        self.snapshot_provider = snapshot_provider
        self.host = host
        self.port = int(port)
        self.auto_open = bool(auto_open)
        self.refresh_ms = max(100, int(refresh_ms))
        self.history_points = max(30, int(history_points))
        self.auth_enabled = bool(auth_enabled)
        self.auth_username = str(auth_username)
        self.auth_password_env = str(auth_password_env)
        self.auth_password = os.getenv(self.auth_password_env, "")
        self.session_hours = max(1, int(session_hours))
        self.chaser_status_file = Path(chaser_status_file)
        self.config_provider = config_provider
        self.config_updater = config_updater
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None
        self._bind_host = self.host

    def _session_token(self, expires: int) -> str:
        payload = str(int(expires))
        signature = hmac.new(
            self.auth_password.encode("utf-8"),
            payload.encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload}.{signature}"

    def _is_authenticated(self, request: web.Request) -> bool:
        if not self.auth_enabled:
            return True
        if not self.auth_password:
            return False

        raw = request.cookies.get("mucha_dashboard_session", "")
        try:
            expires_text, signature = raw.split(".", 1)
            expires = int(expires_text)
        except (ValueError, TypeError):
            return False

        if expires < int(time.time()):
            return False

        expected = self._session_token(expires).split(".", 1)[1]
        return hmac.compare_digest(signature, expected)

    @web.middleware
    async def _auth_middleware(
        self,
        request: web.Request,
        handler,
    ) -> web.StreamResponse:
        if request.path in {"/login", "/health"}:
            return await handler(request)

        if self._is_authenticated(request):
            return await handler(request)

        if request.path.startswith("/api/"):
            raise web.HTTPUnauthorized(text="authentication required")
        raise web.HTTPFound("/login")

    async def start(self) -> None:
        if self.runner is not None:
            return

        if (
            self.auth_enabled
            and not self.auth_password
            and self.host not in {"127.0.0.1", "localhost", "::1"}
        ):
            self._bind_host = "127.0.0.1"
            log.error(
                "Public Web UI requested, but %s is empty. "
                "Binding to 127.0.0.1 until a dashboard password is configured.",
                self.auth_password_env,
            )
        else:
            self._bind_host = self.host

        app = web.Application(middlewares=[self._auth_middleware])
        app.router.add_get("/", self._index)
        app.router.add_get("/details", self._details)
        app.router.add_get("/affinity", self._affinity_page)
        app.router.add_get("/config", self._config_page)
        app.router.add_get("/brain", self._brain)
        app.router.add_get("/login", self._login_get)
        app.router.add_post("/login", self._login_post)
        app.router.add_get("/logout", self._logout)
        app.router.add_get("/api/state", self._state)
        app.router.add_get("/api/overview", self._overview)
        app.router.add_get("/api/config", self._config_get)
        app.router.add_post("/api/config", self._config_post)
        app.router.add_get("/health", self._health)

        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self._bind_host, self.port)
        await self.site.start()
        log.info("Web UI: http://%s:%s", self._bind_host, self.port)

        if self.auto_open and self._bind_host in {"127.0.0.1", "localhost"}:
            url = f"http://127.0.0.1:{self.port}"
            asyncio.get_running_loop().call_later(1.0, webbrowser.open, url)

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None
            self.site = None

    async def _index(self, request: web.Request) -> web.Response:
        return web.Response(text=OVERVIEW_HTML, content_type="text/html")

    async def _details(self, request: web.Request) -> web.Response:
        html = HTML.replace(
            "const maxHistory=180;",
            f"const maxHistory={self.history_points};",
        )
        html = html.replace(
            "setInterval(update,500);",
            f"setInterval(update,{self.refresh_ms});",
        )
        return web.Response(text=html, content_type="text/html")

    async def _affinity_page(self, request: web.Request) -> web.Response:
        return web.Response(text=AFFINITY_HTML, content_type="text/html")

    async def _config_page(self, request: web.Request) -> web.Response:
        return web.Response(text=CONFIG_HTML, content_type="text/html")

    async def _config_get(self, request: web.Request) -> web.Response:
        if self.config_provider is None:
            raise web.HTTPServiceUnavailable(text="config provider unavailable")
        return web.json_response(
            self.config_provider(),
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    async def _config_post(self, request: web.Request) -> web.Response:
        if self.config_updater is None:
            raise web.HTTPServiceUnavailable(text="config updater unavailable")
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("JSON musi być obiektem")
            result = self.config_updater(payload)
            return web.json_response(
                result,
                dumps=lambda x: json.dumps(x, ensure_ascii=False),
            )
        except (ValueError, TypeError, OSError) as exc:
            return web.json_response(
                {"ok": False, "error": str(exc)},
                status=400,
                dumps=lambda x: json.dumps(x, ensure_ascii=False),
            )

    async def _brain(self, request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound("/details")

    async def _login_get(self, request: web.Request) -> web.Response:
        if self._is_authenticated(request):
            raise web.HTTPFound("/")
        return web.Response(
            text=LOGIN_HTML.replace("__ERROR__", ""),
            content_type="text/html",
        )

    async def _login_post(self, request: web.Request) -> web.StreamResponse:
        if not self.auth_enabled:
            raise web.HTTPFound("/")

        form = await request.post()
        username = str(form.get("username", ""))
        password = str(form.get("password", ""))

        user_ok = hmac.compare_digest(username, self.auth_username)
        password_ok = bool(self.auth_password) and hmac.compare_digest(
            password,
            self.auth_password,
        )

        if not (user_ok and password_ok):
            return web.Response(
                text=LOGIN_HTML.replace(
                    "__ERROR__",
                    "Nieprawidłowy login lub hasło.",
                ),
                content_type="text/html",
                status=401,
            )

        expires = int(time.time() + self.session_hours * 3600)
        response = web.HTTPFound("/")
        response.set_cookie(
            "mucha_dashboard_session",
            self._session_token(expires),
            max_age=self.session_hours * 3600,
            httponly=True,
            samesite="Strict",
            secure=request.headers.get("X-Forwarded-Proto", "").lower()
            == "https",
        )
        return response

    async def _logout(self, request: web.Request) -> web.StreamResponse:
        response = web.HTTPFound("/login")
        response.del_cookie("mucha_dashboard_session")
        return response

    async def _state(self, request: web.Request) -> web.Response:
        snap = await self.snapshot_provider()
        return web.json_response(
            snap,
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    async def _service_status(self, unit: str) -> dict:
        props = (
            "ActiveState,SubState,MainPID,MemoryCurrent,CPUUsageNSec,"
            "ActiveEnterTimestampMonotonic"
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl",
                "show",
                unit,
                f"--property={props}",
                "--no-pager",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3.0,
            )
        except Exception as exc:
            return {
                "unit": unit,
                "active": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        data: dict[str, str] = {}
        for line in stdout.decode("utf-8", "replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                data[key] = value

        def as_int(value: str | None) -> int:
            try:
                return int(value or 0)
            except ValueError:
                return 0

        entered_us = as_int(data.get("ActiveEnterTimestampMonotonic"))
        uptime = (
            max(0.0, time.monotonic() - entered_us / 1_000_000.0)
            if entered_us
            else 0.0
        )
        return {
            "unit": unit,
            "active": data.get("ActiveState") == "active",
            "active_state": data.get("ActiveState", "unknown"),
            "sub_state": data.get("SubState", "unknown"),
            "pid": as_int(data.get("MainPID")),
            "memory_bytes": as_int(data.get("MemoryCurrent")),
            "cpu_seconds": as_int(data.get("CPUUsageNSec")) / 1_000_000_000.0,
            "uptime_seconds": uptime,
            "error": stderr.decode("utf-8", "replace").strip(),
        }

    def _system_status(self) -> dict:
        mem: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(
                encoding="utf-8",
            ).splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                number = value.strip().split()[0]
                mem[key] = int(number) * 1024
        except (OSError, ValueError):
            pass

        try:
            uptime = float(
                Path("/proc/uptime").read_text(
                    encoding="utf-8",
                ).split()[0]
            )
        except (OSError, ValueError, IndexError):
            uptime = 0.0

        try:
            load = list(os.getloadavg())
        except OSError:
            load = [0.0, 0.0, 0.0]

        disk = shutil.disk_usage("/")
        total = int(mem.get("MemTotal", 0))
        available = int(mem.get("MemAvailable", 0))
        used = max(0, total - available)
        percent = (used / total * 100.0) if total else 0.0

        return {
            "uptime_seconds": uptime,
            "load": load,
            "mem_total": total,
            "mem_available": available,
            "mem_used": used,
            "mem_percent": percent,
            "disk_total": disk.total,
            "disk_used": disk.used,
            "disk_free": disk.free,
        }

    def _chaser_status(self) -> dict:
        try:
            return json.loads(
                self.chaser_status_file.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return {}

    async def _journal_tail(self, unit: str, lines: int = 18) -> list[str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "journalctl",
                "-u",
                unit,
                "-n",
                str(max(1, min(50, int(lines)))),
                "--no-pager",
                "-o",
                "cat",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=3.0,
            )
            if proc.returncode != 0:
                message = stderr.decode("utf-8", "replace").strip()
                return [message] if message else []
            return stdout.decode(
                "utf-8",
                "replace",
            ).splitlines()[-lines:]
        except Exception as exc:
            return [f"{type(exc).__name__}: {exc}"]

    async def _overview(self, request: web.Request) -> web.Response:
        snapshot_task = asyncio.create_task(self.snapshot_provider())
        mucha_task = asyncio.create_task(
            self._service_status("mucha.service")
        )
        chaser_task = asyncio.create_task(
            self._service_status("mucha-chaser.service")
        )
        mucha_logs_task = asyncio.create_task(
            self._journal_tail("mucha.service")
        )
        chaser_logs_task = asyncio.create_task(
            self._journal_tail("mucha-chaser.service")
        )

        snapshot, mucha, chaser, mucha_logs, chaser_logs = await asyncio.gather(
            snapshot_task,
            mucha_task,
            chaser_task,
            mucha_logs_task,
            chaser_logs_task,
        )

        payload = {
            "now": time.time(),
            "snapshot": snapshot,
            "system": self._system_status(),
            "services": {
                "mucha": mucha,
                "chaser": chaser,
            },
            "chaser_status": self._chaser_status(),
            "logs": {
                "mucha": mucha_logs,
                "chaser": chaser_logs,
            },
        }
        return web.json_response(
            payload,
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})
