from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Awaitable, Callable

from aiohttp import web

log = logging.getLogger("mucha.web")

SnapshotProvider = Callable[[], Awaitable[dict]]
ConnectomeProvider = Callable[[bool], Awaitable[dict]]
NeuromapProvider = Callable[[str], Awaitable[dict]]
AssociationProvider = Callable[[], Awaitable[dict]]
ConfigProvider = Callable[[], dict]
ConfigUpdater = Callable[[dict], dict]

CONFIG_HTML = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha — Konfiguracja</title>
<style>
:root{--bg:#070c12;--panel:#0e1721;--panel2:#09121a;--line:#22364a;--txt:#eef7ff;--muted:#8295a8;--a:#58dac4;--blue:#70aaff;--good:#55d98c;--warn:#f2c45f;--bad:#ff7272}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0%,rgba(88,218,196,.09),transparent 28%),linear-gradient(180deg,#070c12,#091019);color:var(--txt);font-family:Inter,system-ui,"Segoe UI",sans-serif}
main{max-width:1540px;margin:auto;padding:22px}.top{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:16px}.brand{display:flex;gap:12px;align-items:center}.logo{font-size:34px}
h1{margin:0;font-size:24px}.sub{color:var(--muted);font-size:12px;margin-top:4px}.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#c6d2df;text-decoration:none;border:1px solid var(--line);background:#0e161f;padding:8px 11px;border-radius:10px;font-size:12px}.nav a.active{background:var(--a);border-color:var(--a);color:#06110e;font-weight:850}
.intro{display:grid;grid-template-columns:1fr auto;gap:14px;align-items:center;background:linear-gradient(135deg,rgba(88,218,196,.08),rgba(112,170,255,.05));border:1px solid var(--line);border-radius:16px;padding:14px 16px;margin-bottom:12px}.intro strong{font-size:13px}.intro p{margin:4px 0 0;color:var(--muted);font-size:11px;line-height:1.5}.search{width:min(360px,42vw);border:1px solid #294157;background:#071019;color:var(--txt);border-radius:10px;padding:10px 12px;outline:0}.search:focus{border-color:var(--a);box-shadow:0 0 0 3px rgba(88,218,196,.08)}
.layout{display:grid;grid-template-columns:1fr 1fr;gap:12px}.section{background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden;min-width:0}.section.wide{grid-column:span 2}.section summary{list-style:none;cursor:pointer;padding:15px 16px;display:flex;justify-content:space-between;align-items:center;gap:14px}.section summary::-webkit-details-marker{display:none}.section summary:hover{background:rgba(255,255,255,.015)}.section-title b{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.09em}.section-title small{display:block;color:var(--muted);font-size:10px;margin-top:4px;line-height:1.4}.chev{color:#6f879b;font-size:13px}.section[open] .chev{transform:rotate(90deg)}.section-body{border-top:1px solid var(--line);padding:13px}
.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.field{background:var(--panel2);border:1px solid #1c2d3e;border-radius:11px;padding:11px;min-width:0}.field.hidden{display:none}.field-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.field label{display:block;color:#d8e4ee;font-size:11px;font-weight:700;line-height:1.35}.hint{color:#72879a;font-size:9px;line-height:1.45;margin-top:4px;min-height:25px}.field input[type=number],.field input[type=text]{width:100%;margin-top:8px;border:1px solid #294057;background:#050d14;color:var(--txt);border-radius:9px;padding:9px 10px;font:inherit}.field input:focus{outline:0;border-color:var(--a);box-shadow:0 0 0 3px rgba(88,218,196,.07)}
.switch{position:relative;width:42px;height:23px;flex:0 0 auto}.switch input{opacity:0;width:0;height:0}.slider{position:absolute;inset:0;background:#172534;border:1px solid #2a4054;border-radius:999px;cursor:pointer;transition:.15s}.slider:before{content:"";position:absolute;width:17px;height:17px;left:2px;top:2px;border-radius:50%;background:#8194a6;transition:.15s}.switch input:checked+.slider{background:rgba(88,218,196,.22);border-color:rgba(88,218,196,.6)}.switch input:checked+.slider:before{transform:translateX(19px);background:var(--a);box-shadow:0 0 12px rgba(88,218,196,.45)}
.channels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px;max-height:420px;overflow:auto}.channel{display:grid;grid-template-columns:28px 1fr auto;gap:8px;align-items:center;padding:9px;background:var(--panel2);border:1px solid #1d2e3e;border-radius:9px}.channel input{accent-color:var(--a)}.channel b{font-size:11px}.channel small{color:var(--muted);font-size:9px;word-break:break-all}
.savebar{position:sticky;bottom:12px;z-index:20;margin-top:14px;background:rgba(7,13,20,.95);border:1px solid #294057;border-radius:15px;padding:12px 13px;display:flex;justify-content:space-between;gap:12px;align-items:center;backdrop-filter:blur(12px);box-shadow:0 20px 55px rgba(0,0,0,.28)}.status-wrap{min-width:0}.status{font-size:12px;color:var(--muted);line-height:1.4}.dirty{font-size:9px;color:#6f8497;margin-top:3px}.ok{color:var(--good)!important}.bad{color:var(--bad)!important}.warn{color:var(--warn)!important}
button{border:0;border-radius:11px;padding:11px 16px;background:var(--a);color:#06110e;font-weight:850;cursor:pointer;white-space:nowrap}button:disabled{opacity:.45;cursor:not-allowed}.secondary{background:#111d28;color:#b9cad8;border:1px solid #294057}
@media(max-width:980px){.layout{grid-template-columns:1fr}.section.wide{grid-column:auto}.fields,.channels{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}.intro{grid-template-columns:1fr}.search{width:100%}}@media(max-width:560px){main{padding:12px}.savebar{align-items:stretch;flex-direction:column}.savebar button{width:100%}}
</style>
</head>
<body><main>
<div class="top">
 <div class="brand"><div class="logo">⚙</div><div><h1>Konfiguracja Muchy</h1><div class="sub">Edytujesz aktywne ustawienia. Zapis trafia do config.local.toml, a Mucha automatycznie uruchamia się ponownie.</div></div></div>
 <div class="nav"><a href="/">🏠 Przegląd</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a href="/associations">🕸 Skojarzenia</a><a href="/affinity">🤝 Affinity</a><a class="active" href="/config">⚙ Konfiguracja</a><a href="/public">👁 Publiczny</a><a href="/logout">Wyloguj</a></div>
</div>

<div class="intro">
 <div><strong>Zmiany są trwałe</strong><p>Wartości są walidowane przed zapisem. Po kliknięciu zapisu panel zapisze override, zrestartuje usługę Muchy i poczeka aż dashboard ponownie odpowie.</p></div>
 <input class="search" id="search" type="search" placeholder="Szukaj ustawienia, np. TTS, reward, cooldown…">
</div>

<div class="layout" id="sections"></div>

<div class="savebar">
 <div class="status-wrap"><div class="status" id="status">Ładowanie konfiguracji…</div><div class="dirty" id="dirty">—</div></div>
 <div style="display:flex;gap:8px"><button class="secondary" id="reload" type="button">Odrzuć zmiany</button><button id="save" type="button" disabled>Zapisz i zrestartuj Muchę</button></div>
</div>
</main>

<script>
const $=id=>document.getElementById(id);
let state=null,baseline="",dirtyCount=0;

const groups=[
 {id:"language-main",title:"Język i odpowiedzi",desc:"Kiedy Mucha może mówić i jak długie odpowiedzi generuje.",section:"language",open:true,fields:[
  ["min_chars_before_speaking","Minimum danych przed mówieniem","number",50,100,1000000,"Ile poznanych znaków musi mieć model zanim zacznie odpowiadać."],
  ["min_unique_chars_before_speaking","Minimum unikalnych znaków","number",1,5,500,"Chroni przed startem na bardzo ubogim materiale."],
  ["max_generated_chars","Maksymalna długość odpowiedzi","number",5,24,700,"Twardy limit długości generowanego tekstu."],
  ["spontaneous_text","Spontaniczne pisanie","bool",0,0,0,"Pozwala Musze pisać bez bezpośredniego pytania."],
  ["reply_cooldown_seconds","Cooldown odpowiedzi","number",1,0,3600,"Minimalna przerwa między odpowiedziami na wiadomości."],
  ["spontaneous_cooldown_seconds","Cooldown spontaniczny","number",5,1,86400,"Minimalna przerwa między spontanicznymi wiadomościami."],
  ["learn_from_bots","Ucz się od botów","bool",0,0,0,"Jeśli wyłączone, wiadomości innych botów nie uczą modelu."]
 ]},
 {id:"language-model",title:"Model słów + connectome",desc:"Jak model językowy łączy statystyki słów z aktualnym stanem connectomu.",section:"language",open:true,fields:[
  ["hybrid_word_enabled","Hybrydowy generator słów","bool",0,0,0,"Łączy model słów i generator znakowy."],
  ["word_model_probability","Szansa modelu słów","number",0.01,0,1,"1.0 = prawie zawsze generator słów, 0 = generator znakowy."],
  ["word_max_tokens","Maksymalna liczba słów","number",1,3,60,"Limit słów w odpowiedzi generowanej przez model słów."],
  ["word_recent_window_seconds","Okno świeżej pamięci","number",60,60,604800,"Jak długo niedawne przejścia słów są traktowane jako świeże."],
  ["word_recent_boost","Bonus świeżych wzorców","number",0.05,1,5,"Mnożnik dla niedawno poznanych przejść."],
  ["word_frequency_exponent","Wpływ częstotliwości słów","number",0.05,0.1,2,"Wyższa wartość mocniej faworyzuje częste przejścia."],
  ["word_arousal_flatten","Losowość słów od arousal","number",0.01,0,1,"Zwiększa spłaszczenie rozkładu przy pobudzeniu."],
  ["char_frequency_exponent","Wpływ częstotliwości znaków","number",0.05,0.1,2,"Odpowiednik dla generatora znakowego."],
  ["char_arousal_flatten","Losowość znaków od arousal","number",0.01,0,1,"Losowość warstwy znakowej."],
  ["word_reward_scale","Siła rewardu słów","number",0.01,0,1,"Jak mocno reakcje wzmacniają użyte słowa/przejścia."],
  ["connectome_word_control_enabled","Connectome steruje doborem słów","bool",0,0,0,"Pozwala stanowi connectomu zmieniać szanse kandydatów słów."],
  ["connectome_word_control_min_vocab","Próg słownika dla connectome","number",50,8,100000,"Od ilu unikalnych słów włącza się sterowanie connectomu."],
  ["connectome_word_control_strength","Siła wpływu connectome","number",0.05,0,2,"0 = brak wpływu, wyższe wartości mocniej zmieniają wybór słów."],
  ["connectome_word_control_candidates","Kandydaci oceniani przez connectome","number",1,4,96,"Ile najlepszych kandydatów słów connectome ocenia na krok."]
 ]},
 {id:"behavior",title:"Zachowanie",desc:"Progi mówienia, reakcji i podstawowe parametry uczenia społecznego.",section:"behavior",open:true,fields:[
  ["speak_threshold","Próg mówienia","number",0.01,0,1,"Niżej = Mucha łatwiej decyduje się mówić."],
  ["reaction_threshold","Próg reakcji emoji","number",0.01,0,1,"Niżej = częściej reaguje emoji."],
  ["reaction_cooldown_seconds","Cooldown reakcji","number",1,0,3600,"Minimalny odstęp między reakcjami."],
  ["social_learning_enabled","Uczenie społeczne","bool",0,0,0,"Włącza reward/affinity z zachowań użytkowników."],
  ["social_window_seconds","Okno uczenia społecznego","number",10,30,86400,"Jak długo wcześniejsza akcja może dostać feedback."],
  ["word_reuse_reward","Reward za przejęte słowo","number",0.01,0,1,"Nagroda gdy użytkownik później użyje słowa Muchy."],
  ["phrase_reuse_reward","Reward za przejętą frazę","number",0.01,0,1,"Nagroda za ponowne użycie dłuższej frazy."],
  ["direct_reply_reward","Reward za odpowiedź użytkownika","number",0.01,0,1,"Nagroda gdy ktoś bezpośrednio odpowie Musze."],
  ["self_repeat_penalty","Kara za powtarzanie siebie","number",0.01,0,1,"Kara za zbyt podobne własne wypowiedzi."]
 ]},
 {id:"relations",title:"Relacje / affinity",desc:"Jak szybko Mucha zaczyna lubić, znać albo unikać użytkowników.",section:"behavior",open:false,fields:[
  ["user_affinity_positive_step","Affinity + za pozytywną reakcję","number",0.01,0,1,"Bazowy plus za pozytywne emoji."],
  ["user_affinity_negative_step","Affinity - za negatywną reakcję","number",0.01,0,1,"Bazowy minus za negatywne emoji."],
  ["direct_reply_affinity_step","Affinity + za reply","number",0.001,0,0.25,"Zmiana za bezpośrednią odpowiedź."],
  ["mention_affinity_step","Affinity + za @Mucha","number",0.001,0,0.25,"Zmiana za oznaczenie Muchy."],
  ["continued_conversation_affinity_step","Affinity + za kontynuację rozmowy","number",0.001,0,0.25,"Zmiana za dalszy ciąg rozmowy."],
  ["word_reuse_affinity_step","Affinity + za przejęte słowo","number",0.001,0,0.25,"Zmiana gdy użytkownik przejmuje słowo Muchy."],
  ["phrase_reuse_affinity_step","Affinity + za przejętą frazę","number",0.001,0,0.25,"Zmiana za przejęcie frazy."],
  ["voice_join_affinity_step","Affinity + za wejście do VC","number",0.001,0,0.25,"Plus za dołączenie do wspólnego voice."],
  ["voice_stay_affinity_step","Affinity + za pozostanie na VC","number",0.001,0,0.25,"Plus za pozostanie z Muchą."],
  ["voice_stay_seconds","Czas wymagany na VC","number",1,5,3600,"Po ilu sekundach naliczyć plus."],
  ["tts_stay_affinity_step","Affinity + za zostanie po TTS","number",0.001,0,0.25,"Plus za pozostanie po wypowiedzi głosowej."],
  ["tts_stay_seconds","Obserwacja po TTS","number",1,5,3600,"Okno oczekiwania po TTS."],
  ["voice_leave_after_join_affinity_step","Affinity - za ucieczkę z VC","number",0.001,0,0.25,"Minus gdy ktoś szybko wychodzi po wejściu Muchy."],
  ["voice_leave_after_join_seconds","Okno ucieczki z VC","number",1,1,300,"Ile sekund uznawać za szybką ucieczkę."],
  ["tts_leave_affinity_step","Affinity - za wyjście po TTS","number",0.001,0,0.25,"Minus za szybkie wyjście po TTS."],
  ["tts_leave_seconds","Okno wyjścia po TTS","number",1,1,300,"Ile sekund obserwować po TTS."],
  ["ignored_reply_affinity_step","Affinity - za ignorowanie","number",0.001,0,0.10,"Minus gdy użytkownik jest aktywny, ale ignoruje odpowiedź Muchy."],
  ["ignored_reply_seconds","Okno ignorowania","number",1,5,600,"Czas oczekiwania na odpowiedź."],
  ["negative_contact_cooldown_seconds","Cooldown naturalnych minusów","number",1,1,3600,"Ogranicza częstotliwość ujemnych zdarzeń."],
  ["negative_streak_window_seconds","Okno negative streak","number",10,30,86400,"Okno zliczania serii negatywnych reakcji."],
  ["negative_streak_multiplier_step","Wzrost mnożnika negative streak","number",0.05,0,1,"Jak szybko rośnie siła kolejnych minusów."],
  ["negative_streak_max_multiplier","Maks. mnożnik negative streak","number",0.05,1,3,"Górny limit mnożnika."],
  ["positive_contact_cooldown_seconds","Cooldown naturalnych plusów","number",1,1,3600,"Ogranicza częstotliwość dodatnich zdarzeń."],
  ["familiar_affinity_threshold","Próg znajomego","number",0.01,-1,1,"Od tej wartości użytkownik jest traktowany jako znajomy."],
  ["user_avoid_threshold","Próg unikania użytkownika","number",0.01,-1,1,"Poniżej tej wartości Mucha może unikać użytkownika."],
  ["ignore_disliked_users_text","Nie odpisuj nielubianym","bool",0,0,0,"Blokuje tekstowe odpowiedzi do mocno nielubianych."],
  ["avoid_disliked_users_on_voice","Omijaj nielubianych na VC","bool",0,0,0,"Wpływa na wybór kanałów voice."]
 ]},
 {id:"voice-main",title:"Voice",desc:"Ruch po kanałach i podstawowe zachowanie głosowe.",section:"voice",open:false,fields:[
  ["poll_seconds","Interwał decyzji voice","number",1,1,3600,"Co ile sekund Mucha ocenia sytuację na voice."],
  ["minimum_dwell_seconds","Minimalny pobyt","number",1,0,86400,"Najkrótszy normalny pobyt na kanale."],
  ["maximum_dwell_seconds","Maksymalny pobyt","number",1,1,86400,"Po tym czasie rośnie presja na zmianę kanału."],
  ["move_threshold","Próg move","number",0.01,0,1,"Próg decyzji o zmianie kanału."],
  ["join_threshold","Próg join","number",0.01,0,1,"Próg decyzji o wejściu na voice."],
  ["leave_threshold","Próg leave","number",0.01,0,1,"Próg decyzji o opuszczeniu voice."],
  ["include_empty_channels","Uwzględniaj puste kanały","bool",0,0,0,"Pozwala eksplorować puste kanały."]
 ]},
 {id:"voice-audio",title:"TTS / STT",desc:"Mówienie i rozpoznawanie głosu.",section:"voice",open:false,fields:[
  ["tts_enabled","TTS włączony","bool",0,0,0,"Pozwala Musze mówić na voice."],
  ["tts_interval_seconds","Interwał TTS","number",1,1,3600,"Minimalna przerwa między próbami TTS."],
  ["tts_volume","Głośność TTS","number",0.05,0,2,"Poziom głośności od 0 do 2."],
  ["stt_enabled","STT / słuchanie użytkowników","bool",0,0,0,"Włącza transkrypcję mowy."],
  ["stt_model","Model Whisper","text",0,0,0,"Np. tiny, base, small."],
  ["stt_language","Język STT","text",0,0,0,"Kod języka, np. pl."],
  ["stt_device","Urządzenie STT","text",0,0,0,"Np. cpu lub cuda."],
  ["stt_compute_type","Typ obliczeń STT","text",0,0,0,"Np. int8, float16."],
  ["stt_cpu_threads","Wątki CPU STT","number",1,1,64,"Liczba wątków dla STT na CPU."],
  ["stt_silence_seconds","Cisza kończąca wypowiedź","number",0.1,0.2,5,"Po jakiej ciszy zamknąć segment."],
  ["stt_min_segment_seconds","Minimalny segment","number",0.1,0.2,10,"Krótsze fragmenty są ignorowane."],
  ["stt_max_segment_seconds","Maksymalny segment","number",0.5,2,60,"Dłuższy głos zostanie pocięty."],
  ["stt_min_chars","Minimum znaków transkrypcji","number",1,1,100,"Minimalna długość zaakceptowanego tekstu."],
  ["stt_beam_size","Beam size STT","number",1,1,10,"Wyższe = wolniej, zwykle dokładniej."],
  ["random_audio_enabled","Losowe audio","bool",0,0,0,"Włącza rzadkie losowe audio."]
 ]}
];

const channelSections=[
 {id:"blocked-text",title:"Wykluczone kanały tekstowe",desc:"Zaznaczone kanały są całkowicie pomijane przez część tekstową.",kind:"text"},
 {id:"blocked-voice",title:"Wykluczone kanały voice",desc:"Mucha nie wejdzie na zaznaczone kanały voice.",kind:"voice"},
 {id:"blocked-guild",title:"Wykluczone serwery voice",desc:"Całkowity zakaz voice dla zaznaczonych serwerów.",kind:"guild"}
];

function esc(v){return String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]))}
function fieldId(section,key){return section+"-"+key}
function renderField(section,f){
 const [key,label,type,step,min,max,hint]=f,value=state[section]?.[key],id=fieldId(section,key);
 const search=(label+" "+key+" "+hint+" "+section).toLowerCase();
 if(type==="bool")return '<div class="field" data-search="'+esc(search)+'"><div class="field-head"><div><label for="'+id+'">'+esc(label)+'</label><div class="hint">'+esc(hint)+'</div></div><label class="switch"><input id="'+id+'" type="checkbox" '+(value?'checked':'')+'><span class="slider"></span></label></div></div>';
 if(type==="text")return '<div class="field" data-search="'+esc(search)+'"><label for="'+id+'">'+esc(label)+'</label><div class="hint">'+esc(hint)+'</div><input id="'+id+'" type="text" value="'+esc(value)+'"></div>';
 return '<div class="field" data-search="'+esc(search)+'"><label for="'+id+'">'+esc(label)+'</label><div class="hint">'+esc(hint)+'</div><input id="'+id+'" type="number" step="'+step+'" min="'+min+'" max="'+max+'" value="'+value+'"></div>';
}
function renderSections(){
 const root=$("sections");
 root.innerHTML=groups.map(g=>'<details class="section" '+(g.open?'open':'')+' data-group="'+g.id+'"><summary><div class="section-title"><b>'+esc(g.title)+'</b><small>'+esc(g.desc)+'</small></div><span class="chev">▶</span></summary><div class="section-body"><div class="fields">'+g.fields.map(f=>renderField(g.section,f)).join("")+'</div></div></details>').join("")
 +channelSections.map(c=>'<details class="section '+(c.kind==="guild"?'wide':'')+'" data-group="'+c.id+'"><summary><div class="section-title"><b>'+esc(c.title)+'</b><small>'+esc(c.desc)+'</small></div><span class="chev">▶</span></summary><div class="section-body"><div class="channels" id="'+c.id+'-list"></div></div></details>').join("");
 renderChannels();
 bindInputs();
}
function renderChannels(){
 const text=state.channels?.text||[],voice=state.channels?.voice||[],guilds=state.guilds||[];
 $("blocked-text-list").innerHTML=text.map(ch=>channelRow("text",ch.id,ch.name,ch.guild,ch.blocked)).join("")||'<div class="hint">Brak kanałów tekstowych.</div>';
 $("blocked-voice-list").innerHTML=voice.map(ch=>channelRow("voice",ch.id,ch.name,ch.guild,ch.blocked)).join("")||'<div class="hint">Brak kanałów voice.</div>';
 $("blocked-guild-list").innerHTML=guilds.map(g=>channelRow("voice-guild",g.id,g.name,"Całkowity zakaz VC",g.voice_blocked)).join("")||'<div class="hint">Brak serwerów.</div>';
}
function channelRow(kind,id,name,sub,checked){return '<label class="channel"><input type="checkbox" data-'+kind+'="'+id+'" '+(checked?'checked':'')+'><div><b>'+esc(name)+'</b><br><small>'+esc(sub)+'</small></div><small>'+id+'</small></label>'}
function bindInputs(){document.querySelectorAll("input").forEach(el=>{if(el.id==="search")return;el.addEventListener("input",markDirty);el.addEventListener("change",markDirty)})}
function collect(){
 const out={language:{},behavior:{},voice:{}};
 for(const g of groups)for(const [key,,type] of g.fields){
  const el=$(fieldId(g.section,key));if(!el)continue;
  out[g.section][key]=type==="bool"?el.checked:(type==="text"?el.value:Number(el.value))
 }
 out.blocked_text_channel_ids=[...document.querySelectorAll("[data-text]:checked")].map(x=>Number(x.dataset.text));
 out.blocked_voice_channel_ids=[...document.querySelectorAll("[data-voice]:checked")].map(x=>Number(x.dataset.voice));
 out.blocked_voice_guild_ids=[...document.querySelectorAll("[data-voice-guild]:checked")].map(x=>Number(x.dataset.voiceGuild));
 return out
}
function stable(v){return JSON.stringify(v,Object.keys(v).sort())}
function snapshotForm(){const o=collect();return JSON.stringify(o)}
function markDirty(){
 if(!state)return;
 const changed=snapshotForm()!==baseline;
 dirtyCount=changed?1:0;
 $("save").disabled=!changed;
 $("dirty").textContent=changed?"Masz niezapisane zmiany.":"Brak niezapisanych zmian.";
 $("dirty").className="dirty "+(changed?"warn":"")
}
function applySearch(){
 const q=$("search").value.trim().toLowerCase();
 document.querySelectorAll(".field[data-search]").forEach(el=>el.classList.toggle("hidden",!!q&&!el.dataset.search.includes(q)));
 document.querySelectorAll(".section").forEach(sec=>{
  if(!q)return;
  const fields=[...sec.querySelectorAll(".field[data-search]")];
  if(fields.some(x=>!x.classList.contains("hidden")))sec.open=true
 })
}
async function load(){
 $("status").textContent="Ładowanie konfiguracji…";$("status").className="status";
 const r=await fetch("/api/config",{cache:"no-store"});if(!r.ok)throw new Error("HTTP "+r.status);
 state=await r.json();renderSections();baseline=snapshotForm();markDirty();
 $("status").textContent="Gotowe • "+(state.config_path||"config.local.toml");$("status").className="status ok"
}
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function waitForRestart(){
 $("status").textContent="Restartuję Muchę…";$("status").className="status warn";
 await sleep(1400);
 for(let i=0;i<35;i++){
  try{
   const r=await fetch("/health?restart="+Date.now(),{cache:"no-store"});
   if(r.ok&&i>=2){
    await load();
    $("status").textContent="Zapisano i zrestartowano Muchę. Nowa konfiguracja jest aktywna.";
    $("status").className="status ok";
    return
   }
  }catch(e){}
  await sleep(900)
 }
 $("status").textContent="Konfiguracja zapisana, ale panel nie potwierdził powrotu usługi. Sprawdź status systemd.";
 $("status").className="status bad"
}
$("save").onclick=async()=>{
 $("save").disabled=true;$("reload").disabled=true;$("status").textContent="Waliduję i zapisuję config.local.toml…";$("status").className="status";
 try{
  const r=await fetch("/api/config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(collect())});
  const d=await r.json();if(!r.ok||!d.ok)throw new Error(d.error||("HTTP "+r.status));
  state=d.config;baseline=JSON.stringify(collect());
  $("dirty").textContent="Brak niezapisanych zmian.";
  if(d.changed?.length){
   $("status").textContent="Zapisano "+d.changed.length+" zmian. Restart usługi zaplanowany…";$("status").className="status ok";
   if(d.restart?.scheduled)await waitForRestart();else await load()
  }else{
   $("status").textContent="Brak faktycznych zmian do zapisania.";$("status").className="status ok";await load()
  }
 }catch(e){$("status").textContent="Błąd zapisu: "+e.message;$("status").className="status bad"}
 finally{$("reload").disabled=false;markDirty()}
};
$("reload").onclick=()=>load().catch(e=>{$("status").textContent="Błąd ładowania: "+e.message;$("status").className="status bad"});
$("search").addEventListener("input",applySearch);
load().catch(e=>{$("status").textContent="Błąd ładowania: "+e.message;$("status").className="status bad"});
</script>
</body></html>"""

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
<div class="nav"><a href="/">🏠 Przegląd</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a href="/associations">🕸 Skojarzenia</a><a class="active" href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div></div>

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
    <table><thead><tr><th>Zdarzenie</th><th>Warunek</th><th>Δ bazowe</th></tr></thead><tbody id="positive"></tbody></table>
  </div>

  <div class="card">
    <h2>➖ Co obniża affinity</h2>
    <table><thead><tr><th>Zdarzenie</th><th>Warunek</th><th>Δ bazowe</th></tr></thead><tbody id="negative"></tbody></table>
  </div>

  <div class="card span2">
    <h2>🤬 Frazy odrzucające / severity</h2>
    <div class="reason" style="margin-bottom:10px">Fraza działa tylko, gdy jest skierowana do Muchy: reply, mention/„Mucha”, albo na VC krótko po jej TTS. Silniejsze frazy dają większy minus.</div>
    <div class="phrases" id="phrases"></div>
  </div>

  <div class="card span2">
    <h2>👥 Aktualne relacje</h2>
    <table><thead><tr><th>Użytkownik</th><th>Affinity</th><th>Negative streak</th><th>👍 reakcje</th><th>👎 reakcje</th><th>Status</th></tr></thead><tbody id="users"></tbody></table>
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
    const streak=num(u.negative_streak),mult=num(u.negative_multiplier||1);
    return '<tr><td>'+esc(u.display_name||u.user_id)+'</td><td class="'+cls+'">'+signed(a)+'</td><td class="'+(streak?"minus":"")+'">'+Math.round(streak)+(streak?" ×"+mult.toFixed(2):"")+'</td><td>'+num(u.positive_reactions)+'</td><td>'+num(u.negative_reactions)+'</td><td class="'+cls+'">'+status+'</td></tr>';
  }).join("")||'<tr><td colspan="6">Brak relacji.</td></tr>';

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

ASSOCIATIONS_HTML = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha — Mapa skojarzeń</title>
<style>
:root{--bg:#050910;--panel:#0d151e;--panel2:#08111a;--line:#203247;--txt:#eef7ff;--muted:#7f92a5;--cyan:#55ead0;--blue:#6da8ff;--good:#58df98;--bad:#ff7474;--warn:#ffd166}
*{box-sizing:border-box}body{margin:0;color:var(--txt);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:radial-gradient(circle at 18% 0%,rgba(85,234,208,.10),transparent 30%),radial-gradient(circle at 84% 5%,rgba(109,168,255,.10),transparent 31%),linear-gradient(180deg,#050910,#07101a)}
main{max-width:1880px;margin:auto;padding:22px 28px 30px}.top{display:flex;justify-content:space-between;gap:18px;align-items:center;margin-bottom:16px}.brand{display:flex;gap:13px;align-items:center}.logo{font-size:38px}h1{margin:0;font-size:25px}.sub{color:var(--muted);font-size:12px;margin-top:4px;max-width:760px;line-height:1.45}.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#bacada;text-decoration:none;border:1px solid var(--line);background:#0b141d;padding:8px 11px;border-radius:10px;font-size:12px}.nav a.active{background:linear-gradient(90deg,var(--cyan),#7ce5d4);border-color:var(--cyan);color:#04120e;font-weight:850}
.hero{display:grid;grid-template-columns:repeat(5,minmax(150px,1fr));gap:12px;margin-bottom:14px}.kpi,.card{background:linear-gradient(180deg,rgba(13,21,30,.97),rgba(8,15,23,.97));border:1px solid var(--line);border-radius:17px;box-shadow:0 18px 50px rgba(0,0,0,.14)}.kpi{padding:14px 16px}.kpi small{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.11em;margin-bottom:6px}.kpi strong{display:block;font-size:18px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.kpi em{display:block;color:#8799aa;font-size:9px;font-style:normal;margin-top:5px}
.toolbar{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px;padding:11px 13px;border:1px solid var(--line);border-radius:14px;background:rgba(9,16,24,.88)}.toolgroup{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.toolgroup label{font-size:9px;color:#7f94a7;text-transform:uppercase;letter-spacing:.1em}.toolgroup output{min-width:24px;text-align:right;font-size:10px;color:#c8d7e5;font-weight:800}.toolgroup input[type=range]{width:120px;accent-color:var(--cyan)}.btn{border:1px solid #294057;background:#08121b;color:#9eb2c3;border-radius:999px;padding:7px 11px;font-size:9px;font-weight:850;letter-spacing:.05em;cursor:pointer}.btn:hover{border-color:#4b6e8a;color:white}.btn.on{color:#04120e;background:var(--cyan);border-color:var(--cyan)}
.grid{display:grid;grid-template-columns:minmax(0,1fr) 400px;gap:14px;align-items:start}.card{padding:15px;min-width:0}.head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:11px}.head h2{margin:0;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:#9eb0c1}.live{font-size:9px;color:var(--good);font-weight:850;letter-spacing:.1em}.graph{height:min(78vh,900px);min-height:680px;border:1px solid #17283a;border-radius:15px;overflow:hidden;position:relative;background:radial-gradient(circle at 50% 50%,rgba(85,234,208,.035),transparent 46%),linear-gradient(180deg,#050b12,#06101a)}.graph canvas{width:100%;height:100%;display:block}.tip{position:absolute;display:none;pointer-events:none;z-index:4;width:235px;padding:10px 11px;background:rgba(5,10,16,.97);border:1px solid #31506c;border-radius:11px;font-size:10px;line-height:1.55;box-shadow:0 16px 45px rgba(0,0,0,.35)}.graph-hint{position:absolute;left:12px;bottom:12px;padding:6px 9px;border:1px solid #1b3042;border-radius:999px;background:rgba(5,11,18,.78);color:#6f8498;font-size:9px;pointer-events:none}
.side{display:flex;flex-direction:column;gap:14px;position:sticky;top:12px}.ins{min-height:205px}.empty{color:var(--muted);font-size:11px;line-height:1.55;padding:6px 0}.word-title{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:12px}.word-title h3{margin:0;font-size:21px}.word-title span{font-size:9px;color:var(--cyan);border:1px solid rgba(85,234,208,.32);background:rgba(85,234,208,.07);padding:5px 7px;border-radius:999px}.meta{display:grid;grid-template-columns:1fr 1fr;gap:8px}.meta div{padding:10px;background:var(--panel2);border:1px solid #17283a;border-radius:10px}.meta small{display:block;color:#74899d;font-size:8px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}.meta b{font-size:12px}.assoc{display:flex;flex-direction:column;gap:7px;max-height:510px;overflow:auto;padding-right:3px}.row{display:grid;grid-template-columns:minmax(0,1fr) 54px;gap:10px;padding:10px;border:1px solid #17283a;background:var(--panel2);border-radius:10px;font-size:10px;cursor:pointer;transition:.12s ease}.row:hover{border-color:#3b617e;background:#0a1722}.row b{font-size:11px}.row small{display:block;color:var(--muted);margin-top:4px;line-height:1.45}.weight{text-align:right;font-size:12px!important;color:#dceaf5}.pos{color:var(--good)}.neg{color:var(--bad)}.note{margin-top:10px;color:#71869a;font-size:9px;line-height:1.55}.legend{display:flex;gap:14px;flex-wrap:wrap;color:#75899c;font-size:9px;margin-top:9px}.legend i{display:inline-block;width:14px;height:3px;border-radius:999px;margin-right:5px;vertical-align:middle}.lstruct{background:var(--blue)}.lpos{background:var(--good)}.lneg{background:var(--bad)}
@media(max-width:1280px){.grid{grid-template-columns:minmax(0,1fr) 350px}.graph{min-height:620px}.hero{grid-template-columns:repeat(3,1fr)}}@media(max-width:980px){main{padding:16px}.top{align-items:flex-start;flex-direction:column}.grid{grid-template-columns:1fr}.side{position:static}.graph{height:650px;min-height:0}}@media(max-width:650px){main{padding:11px}.hero{grid-template-columns:1fr 1fr}.graph{height:540px}.toolbar{align-items:flex-start}.toolgroup input[type=range]{width:95px}.meta{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<div class="top">
 <div class="brand"><div class="logo">🕸</div><div><h1>Mapa skojarzeń Muchy</h1><div class="sub">Czytelny widok relacji słów z connectomu. Domyślnie pokazuje tylko najważniejsze węzły i krawędzie; suwaki pozwalają odsłonić więcej.</div></div></div>
 <div class="nav"><a href="/">🏠 Przegląd</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a class="active" href="/associations">🕸 Skojarzenia</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div>
</div>

<section class="hero">
 <div class="kpi"><small>Węzły widoczne</small><strong id="nodes">—</strong><em id="nodes-total">z — dostępnych</em></div>
 <div class="kpi"><small>Krawędzie widoczne</small><strong id="edges">—</strong><em id="edges-total">z — dostępnych</em></div>
 <div class="kpi"><small>Najsilniejsze</small><strong id="strongest">—</strong><em id="strongest-w">—</em></div>
 <div class="kpi"><small>Reward trace</small><strong id="reward">—</strong><em>bieżący ślad nagrody</em></div>
 <div class="kpi"><small>Tick</small><strong id="tick">—</strong><em id="source">runtime</em></div>
</section>

<div class="toolbar">
 <div class="toolgroup">
  <label for="node-limit">Węzły</label><input id="node-limit" type="range" min="8" max="28" value="18" step="1"><output id="node-limit-out">18</output>
  <label for="edge-limit">Krawędzie</label><input id="edge-limit" type="range" min="6" max="50" value="16" step="1"><output id="edge-limit-out">16</output>
 </div>
 <div class="toolgroup">
  <button class="btn on" id="labels-auto">ETYKIETY AUTO</button>
  <button class="btn" id="labels-all">WSZYSTKIE ETYKIETY</button>
  <button class="btn" id="reset-layout">ROZŁÓŻ PONOWNIE</button>
  <button class="btn" id="clear-selection">WYCZYŚĆ WYBÓR</button>
 </div>
</div>

<section class="grid">
 <div class="card">
  <div class="head"><h2>Connectome word graph</h2><span class="live" id="live">LIVE</span></div>
  <div class="graph" id="wrap">
   <canvas id="canvas"></canvas>
   <div class="tip" id="tip"></div>
   <div class="graph-hint">Kliknij słowo, aby podświetlić tylko jego bezpośrednie skojarzenia.</div>
  </div>
  <div class="legend"><span><i class="lstruct"></i> struktura connectomu</span><span><i class="lpos"></i> dodatni learned bias</span><span><i class="lneg"></i> ujemny learned bias</span></div>
  <div class="note">Mapa nie ma osobnej bazy relacji. Węzły są populacjami słów w connectomie, a waga krawędzi łączy strukturę macierzy, wyuczony plastic bias oraz bieżącą aktywność pary.</div>
 </div>

 <div class="side">
  <div class="card ins">
   <div class="head"><h2>Wybrane słowo</h2><span id="event" class="live">—</span></div>
   <div id="inspector" class="empty">Kliknij słowo na mapie. Pozostałe węzły zostaną przygaszone, a jego relacje będą łatwiejsze do odczytania.</div>
  </div>
  <div class="card">
   <div class="head"><h2>Najsilniejsze połączenia</h2><span id="method" class="live">CONNECTOME</span></div>
   <div id="assoc" class="assoc"></div>
  </div>
 </div>
</section>
</main>

<script>
const $=id=>document.getElementById(id),canvas=$("canvas"),ctx=canvas.getContext("2d"),wrap=$("wrap"),tip=$("tip");
let raw={nodes:[],edges:[]},data={nodes:[],edges:[]},points=new Map(),hover=null,selected=null,mouse={x:-9999,y:-9999},dpr=1;
let nodeLimit=18,edgeLimit=16,labelMode="auto",layoutEpoch=0;
const esc=v=>String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));
const nfmt=n=>Number(n||0).toLocaleString("pl-PL");
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));

function resize(){const r=wrap.getBoundingClientRect();dpr=Math.min(2,window.devicePixelRatio||1);canvas.width=Math.max(1,Math.floor(r.width*dpr));canvas.height=Math.max(1,Math.floor(r.height*dpr));canvas.style.width=r.width+"px";canvas.style.height=r.height+"px";ctx.setTransform(dpr,0,0,dpr,0,0)}
function seed(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return (h>>>0)/4294967295}

function rebuildData(){
 const rankedEdges=(raw.edges||[]).slice().sort((a,b)=>Number(b.weight||0)-Number(a.weight||0));
 const score=new Map((raw.nodes||[]).map(n=>[n.id,2.2*Number(n.salience||0)+1.1*Number(n.brain_score||0)]));
 rankedEdges.forEach((e,i)=>{const bonus=Math.max(0,1-i/Math.max(1,rankedEdges.length));score.set(e.source,(score.get(e.source)||0)+bonus*Number(e.weight||0));score.set(e.target,(score.get(e.target)||0)+bonus*Number(e.weight||0))});
 const keep=(raw.nodes||[]).slice().sort((a,b)=>(score.get(b.id)||0)-(score.get(a.id)||0)).slice(0,nodeLimit);
 const keepIds=new Set(keep.map(n=>n.id));
 const edges=rankedEdges.filter(e=>keepIds.has(e.source)&&keepIds.has(e.target)).slice(0,edgeLimit);
 const connected=new Set(edges.flatMap(e=>[e.source,e.target]));
 const nodes=keep.slice().sort((a,b)=>Number(connected.has(b.id))-Number(connected.has(a.id))||Number(b.salience||0)-Number(a.salience||0));
 data={nodes,edges};
 syncPoints();
 renderStats();
 renderRows();
 if(selected&&!nodes.some(n=>n.id===selected.id)){selected=null;inspect(null)}
}

function resetLayout(){
 points.clear();layoutEpoch++;
 syncPoints(true);
}

function syncPoints(force=false){
 const r=wrap.getBoundingClientRect(),cx=r.width/2,cy=r.height/2,R=Math.min(r.width,r.height)*.38;
 const total=Math.max(1,data.nodes.length);
 data.nodes.forEach((n,i)=>{
  if(!force&&points.has(n.id))return;
  const ring=i<8?0.54:0.90;
  const a=2*Math.PI*(i/total+seed(n.id+"|"+layoutEpoch)*.14);
  points.set(n.id,{x:cx+Math.cos(a)*R*ring,y:cy+Math.sin(a)*R*ring,vx:0,vy:0})
 });
 for(const key of [...points.keys()])if(!data.nodes.some(n=>n.id===key))points.delete(key)
}

function neighborhood(){
 if(!selected)return null;
 const set=new Set([selected.id]);
 data.edges.forEach(e=>{if(e.source===selected.id)set.add(e.target);if(e.target===selected.id)set.add(e.source)});
 return set
}

function physics(){
 const r=wrap.getBoundingClientRect(),nodes=data.nodes,by=id=>points.get(id);
 for(let i=0;i<nodes.length;i++){
  const a=by(nodes[i].id);if(!a)continue;
  for(let j=i+1;j<nodes.length;j++){
   const b=by(nodes[j].id);if(!b)continue;
   let dx=b.x-a.x,dy=b.y-a.y,d2=Math.max(180,dx*dx+dy*dy),d=Math.sqrt(d2);
   const minDist=110;
   const repel=5200/d2+(d<minDist?(minDist-d)*.045:0);
   a.vx-=dx/d*repel;a.vy-=dy/d*repel;b.vx+=dx/d*repel;b.vy+=dy/d*repel
  }
 }
 data.edges.forEach(e=>{
  const a=by(e.source),b=by(e.target);if(!a||!b)return;
  let dx=b.x-a.x,dy=b.y-a.y,d=Math.max(1,Math.hypot(dx,dy));
  const target=150+70*(1-Number(e.weight||0));
  const f=(d-target)*(.00065+.0015*Number(e.weight||0));
  a.vx+=dx*f;a.vy+=dy*f;b.vx-=dx*f;b.vy-=dy*f
 });
 const cx=r.width/2,cy=r.height/2;
 nodes.forEach(n=>{
  const p=by(n.id);if(!p)return;
  p.vx+=(cx-p.x)*.00032;p.vy+=(cy-p.y)*.00032;
  p.vx*=.86;p.vy*=.86;
  p.x=clamp(p.x+p.vx,48,r.width-48);p.y=clamp(p.y+p.vy,48,r.height-58)
 })
}

function edgeColor(e,a){const learned=Number(e.learned||0);if(learned>.10)return "rgba(88,223,152,"+a+")";if(learned<-.10)return "rgba(255,116,116,"+a+")";return "rgba(109,168,255,"+a+")"}

function importantLabels(){
 const ids=new Set(),edgeTop=data.edges.slice(0,8);
 edgeTop.forEach(e=>{ids.add(e.source);ids.add(e.target)});
 data.nodes.slice().sort((a,b)=>Number(b.salience||0)-Number(a.salience||0)).slice(0,8).forEach(n=>ids.add(n.id));
 if(selected)ids.add(selected.id);if(hover)ids.add(hover.id);
 return ids
}

function drawLabel(n,p,alpha){
 const text=n.id,rad=11+13*Number(n.salience||0),y=p.y+rad+16;
 ctx.font="600 11px system-ui";
 const w=ctx.measureText(text).width+12;
 ctx.fillStyle="rgba(4,10,16,"+(0.76*alpha)+")";
 ctx.strokeStyle="rgba(38,62,82,"+(0.72*alpha)+")";
 ctx.lineWidth=1;
 ctx.beginPath();
 if(ctx.roundRect)ctx.roundRect(p.x-w/2,y-12,w,18,7);else ctx.rect(p.x-w/2,y-12,w,18);
 ctx.fill();ctx.stroke();
 ctx.fillStyle="rgba(234,245,255,"+alpha+")";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(text,p.x,y-3)
}

function draw(){
 physics();
 const r=wrap.getBoundingClientRect(),near=neighborhood(),labels=importantLabels();
 ctx.clearRect(0,0,r.width,r.height);

 data.edges.forEach(e=>{
  const a=points.get(e.source),b=points.get(e.target);if(!a||!b)return;
  const focus=!selected||e.source===selected.id||e.target===selected.id;
  const w=Number(e.weight||0),alpha=focus?(.16+.72*w):.035;
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=edgeColor(e,alpha);ctx.lineWidth=focus?(.8+4.2*w):.55;ctx.stroke()
 });

 hover=null;
 for(const n of data.nodes){
  const p=points.get(n.id);if(!p)continue;
  const rad=11+13*Number(n.salience||0);
  if(Math.hypot(mouse.x-p.x,mouse.y-p.y)<=rad+7)hover=n;
  const focus=!near||near.has(n.id),alpha=focus?1:.16;
  ctx.beginPath();ctx.arc(p.x,p.y,rad,0,Math.PI*2);
  ctx.fillStyle=(selected&&selected.id===n.id)?"rgba(85,234,208,.96)":"rgba(15,35,50,"+(.94*alpha)+")";
  ctx.fill();
  ctx.strokeStyle="rgba(85,234,208,"+((.23+.72*Number(n.brain_score||0))*alpha)+")";
  ctx.lineWidth=(selected&&selected.id===n.id)?3.4:1.4+2.1*Number(n.salience||0);
  ctx.stroke();
  const showLabel=labelMode==="all"||labels.has(n.id)||!!near&&near.has(n.id);
  if(showLabel)drawLabel(n,p,alpha)
 }

 if(hover){
  const p=points.get(hover.id);
  tip.style.display="block";
  tip.style.left=Math.max(8,Math.min(r.width-250,p.x+18))+"px";
  tip.style.top=Math.max(8,Math.min(r.height-132,p.y+18))+"px";
  tip.innerHTML="<b style='font-size:12px'>"+esc(hover.id)+"</b><br>brain score <b>"+Number(hover.brain_score||0).toFixed(3)+"</b><br>activation <b>"+Number(hover.activation||0).toFixed(4)+"</b><br>plastic bias <b>"+Number(hover.plastic_bias||0).toFixed(5)+"</b>"
 }else tip.style.display="none";
 requestAnimationFrame(draw)
}

function inspect(n){
 selected=n;
 if(!n){
  $("inspector").className="empty";
  $("inspector").textContent="Kliknij słowo na mapie. Pozostałe węzły zostaną przygaszone, a jego relacje będą łatwiejsze do odczytania.";
  renderRows();
  return
 }
 $("inspector").className="";
 const related=data.edges.filter(e=>e.source===n.id||e.target===n.id).sort((a,b)=>Number(b.weight||0)-Number(a.weight||0));
 $("inspector").innerHTML="<div class='word-title'><h3>"+esc(n.id)+"</h3><span>"+related.length+" relacji</span></div><div class='meta'><div><small>brain score</small><b>"+Number(n.brain_score||0).toFixed(3)+"</b></div><div><small>salience</small><b>"+Number(n.salience||0).toFixed(3)+"</b></div><div><small>activation</small><b>"+Number(n.activation||0).toFixed(5)+"</b></div><div><small>plastic bias</small><b>"+Number(n.plastic_bias||0).toFixed(6)+"</b></div><div><small>liczba wystąpień</small><b>"+nfmt(n.count)+"</b></div><div><small>language reward</small><b>"+Number(n.language_reward||0).toFixed(3)+"</b></div></div>";
 renderRows()
}

function renderStats(){
 $("nodes").textContent=nfmt(data.nodes.length);$("nodes-total").textContent="z "+nfmt(raw.node_count||0)+" dostępnych";
 $("edges").textContent=nfmt(data.edges.length);$("edges-total").textContent="z "+nfmt(raw.edge_count||0)+" dostępnych";
 $("reward").textContent=Number(raw.reward_trace||0).toFixed(3);$("tick").textContent=nfmt(raw.ticks);$("source").textContent=raw.source||"runtime";
 const s=(data.edges||[])[0];$("strongest").textContent=s?(s.source+" ↔ "+s.target):"—";$("strongest-w").textContent=s?("waga "+Number(s.weight||0).toFixed(3)):"brak krawędzi";
 $("event").textContent=raw.last_event||"—";$("method").textContent="CONNECTOME"
}

function renderRows(){
 let rows=(data.edges||[]).slice();
 if(selected)rows=rows.filter(e=>e.source===selected.id||e.target===selected.id);
 rows=rows.slice(0,14);
 $("assoc").innerHTML=rows.map(e=>"<div class='row' data-a='"+esc(e.source)+"' data-b='"+esc(e.target)+"'><div><b>"+esc(e.source)+" ↔ "+esc(e.target)+"</b><small>structure "+Number(e.structural||0).toFixed(3)+" • learned <span class='"+(Number(e.learned||0)>=0?"pos":"neg")+"'>"+(Number(e.learned||0)>=0?"+":"")+Number(e.learned||0).toFixed(3)+"</span> • activity "+Number(e.pair_activation||0).toFixed(3)+"</small></div><b class='weight'>"+Number(e.weight||0).toFixed(3)+"</b></div>").join("")||"<div class='empty'>Brak widocznych połączeń dla tego ustawienia.</div>";
 document.querySelectorAll(".row").forEach(x=>x.onclick=()=>{const preferred=selected?(x.dataset.a===selected.id?x.dataset.b:x.dataset.a):x.dataset.a;const n=data.nodes.find(n=>n.id===preferred);if(n)inspect(n)})
}

function render(d){
 raw=d||{nodes:[],edges:[]};
 rebuildData();
 if(selected){const fresh=data.nodes.find(n=>n.id===selected.id);if(fresh){selected=fresh;inspect(fresh)}else inspect(null)}
 $("live").textContent="LIVE"
}

async function update(){try{const r=await fetch("/api/associations",{cache:"no-store"});if(r.status===401){location="/login";return}if(!r.ok)throw new Error("HTTP "+r.status);render(await r.json())}catch(e){$("live").textContent="ROZŁĄCZONO";console.error(e)}}

$("node-limit").oninput=e=>{nodeLimit=Number(e.target.value);$("node-limit-out").textContent=nodeLimit;rebuildData()};
$("edge-limit").oninput=e=>{edgeLimit=Number(e.target.value);$("edge-limit-out").textContent=edgeLimit;rebuildData()};
$("labels-auto").onclick=()=>{labelMode="auto";$("labels-auto").classList.add("on");$("labels-all").classList.remove("on")};
$("labels-all").onclick=()=>{labelMode="all";$("labels-all").classList.add("on");$("labels-auto").classList.remove("on")};
$("reset-layout").onclick=resetLayout;
$("clear-selection").onclick=()=>inspect(null);
canvas.addEventListener("mousemove",e=>{const r=canvas.getBoundingClientRect();mouse.x=e.clientX-r.left;mouse.y=e.clientY-r.top});
canvas.addEventListener("mouseleave",()=>{mouse.x=-9999;mouse.y=-9999});
canvas.addEventListener("click",()=>{if(hover)inspect(hover)});
window.addEventListener("resize",()=>{resize();resetLayout()});
resize();draw();setInterval(update,1400);update();
</script>
</body></html>"""

CONNECTOME_HTML = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha — Neural Connectome</title>
<style>
:root{
 --bg:#05080d;--panel:#0b1119;--panel2:#080d14;--line:#1c2b3b;--txt:#edf7ff;--muted:#7890a5;
 --cyan:#55ead0;--blue:#6da8ff;--violet:#b58cff;--pink:#ff78b7;--good:#56e39a;--warn:#ffd166;--bad:#ff7373;
}
*{box-sizing:border-box}
body{margin:0;color:var(--txt);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:
 radial-gradient(circle at 18% -10%,rgba(85,234,208,.12),transparent 30%),
 radial-gradient(circle at 82% 0%,rgba(109,168,255,.11),transparent 30%),
 radial-gradient(circle at 52% 110%,rgba(181,140,255,.08),transparent 35%),
 linear-gradient(180deg,#05080d,#081018 52%,#05090e)}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.22;background-image:
 linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),
 linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:34px 34px}
main{max-width:1600px;margin:auto;padding:22px}
.top{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:16px}
.brand{display:flex;align-items:center;gap:13px}.logo{font-size:37px;filter:drop-shadow(0 0 18px rgba(85,234,208,.32))}
h1{margin:0;font-size:24px}.sub{color:var(--muted);font-size:12px;margin-top:4px}
.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#b9cad9;text-decoration:none;border:1px solid var(--line);background:#0b131c;padding:8px 11px;border-radius:10px;font-size:12px}
.nav a:hover{border-color:#36536e;color:white}.nav a.active{background:linear-gradient(90deg,var(--cyan),#78e6d4);color:#03110d;border-color:var(--cyan);font-weight:850}
.hero{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px}
.kpi,.card{background:linear-gradient(180deg,rgba(12,19,28,.96),rgba(8,14,21,.96));border:1px solid var(--line);border-radius:16px;box-shadow:0 16px 50px rgba(0,0,0,.18)}
.kpi{padding:13px 14px;position:relative;overflow:hidden}.kpi:after{content:"";position:absolute;inset:auto -20px -28px auto;width:86px;height:86px;border-radius:50%;background:radial-gradient(circle,rgba(85,234,208,.10),transparent 70%)}
.kpi small{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px}.kpi strong{font-size:18px}.kpi em{display:block;color:#9eb1c3;font-size:10px;font-style:normal;margin-top:4px}
.grid{display:grid;grid-template-columns:minmax(0,2.1fr) minmax(330px,.9fr);gap:12px}.card{padding:14px;min-width:0}.card h2{margin:0;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:#9eb1c3}
.card-head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}.head-tools{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.live{display:inline-flex;gap:7px;align-items:center;color:var(--good);font-size:10px;font-weight:800;letter-spacing:.1em}.live i{width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 15px var(--good);animation:pulse 1.2s infinite}
@keyframes pulse{50%{opacity:.35;transform:scale(.75)}}
.mode{font-size:9px;color:#7890a5;border:1px solid #1d3041;background:#071019;border-radius:999px;padding:6px 8px;letter-spacing:.08em}
.follow{border:1px solid #294258;background:#08121b;color:#9cb0c2;border-radius:999px;padding:6px 9px;font-size:9px;font-weight:850;letter-spacing:.07em;cursor:pointer;transition:.2s}
.follow:hover{border-color:#4d718e;color:white}.follow.on{border-color:rgba(255,209,102,.5);background:rgba(255,209,102,.1);color:var(--warn);box-shadow:0 0 18px rgba(255,209,102,.08)}
.graph-wrap{position:relative;height:650px;overflow:hidden;border-radius:14px;border:1px solid #172535;background:
 radial-gradient(circle at 50% 50%,rgba(109,168,255,.04),transparent 44%),
 linear-gradient(180deg,#050a10,#07101a)}
#net{width:100%;height:100%;display:block}.graph-label{position:absolute;top:10px;padding:5px 8px;border-radius:999px;background:rgba(5,10,16,.72);border:1px solid #1a2a39;color:#8198ac;font-size:9px;text-transform:uppercase;letter-spacing:.12em;backdrop-filter:blur(7px)}
.gl-left{left:10px}.gl-mid{left:50%;transform:translateX(-50%)}.gl-right{right:10px}
.tooltip{position:absolute;display:none;z-index:4;pointer-events:none;padding:9px 10px;border-radius:10px;background:rgba(4,9,14,.94);border:1px solid #29425a;box-shadow:0 10px 35px rgba(0,0,0,.4);font-size:11px;min-width:175px}.tooltip b{color:white}.tooltip span{color:#8ba0b3}
.flow{display:grid;grid-template-columns:1fr 46px 1.35fr 46px 1fr 46px 1.2fr;gap:7px;align-items:center;margin-bottom:12px}
.flowbox{min-height:92px;padding:11px;border-radius:14px;border:1px solid #1a2a39;background:#071019;position:relative;overflow:hidden}.flowbox:after{content:"";position:absolute;width:90px;height:90px;border-radius:50%;right:-35px;bottom:-45px;background:radial-gradient(circle,rgba(85,234,208,.09),transparent 70%)}
.flowbox small{display:block;color:#758ca1;font-size:9px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:6px}.flowbox strong{font-size:13px}.flowbox p{margin:5px 0 0;color:#8297aa;font-size:10px;line-height:1.45}
.arrow{text-align:center;color:#44647d;font-size:22px;animation:arrow 1.6s ease-in-out infinite}@keyframes arrow{50%{color:var(--cyan);text-shadow:0 0 15px rgba(85,234,208,.6)}}
.side{display:flex;flex-direction:column;gap:12px}.actions{display:flex;flex-direction:column;gap:8px}.act{display:grid;grid-template-columns:92px 1fr 42px;gap:8px;align-items:center;font-size:11px}.act label{color:#a9bac9}.track{height:8px;border-radius:999px;background:#050b11;border:1px solid #172636;overflow:hidden}.fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--blue),var(--cyan));box-shadow:0 0 14px rgba(85,234,208,.22)}
.readout{display:grid;grid-template-columns:1fr 1fr;gap:8px}.mini{padding:10px;background:#071019;border:1px solid #172635;border-radius:11px}.mini small{display:block;color:#70879b;font-size:9px;text-transform:uppercase;letter-spacing:.1em}.mini strong{display:block;margin-top:4px;font-size:13px;word-break:break-word}
.lang-status{padding:12px;border-radius:13px;border:1px solid #1a2d3e;background:linear-gradient(135deg,rgba(85,234,208,.06),rgba(109,168,255,.05))}
.lang-top{display:flex;justify-content:space-between;align-items:center;gap:10px}.badge{padding:5px 8px;border-radius:999px;font-size:9px;font-weight:850;letter-spacing:.08em}.badge.on{background:rgba(86,227,154,.13);color:var(--good);border:1px solid rgba(86,227,154,.35)}.badge.wait{background:rgba(255,209,102,.10);color:var(--warn);border:1px solid rgba(255,209,102,.28)}
.progress{height:8px;background:#050b11;border:1px solid #172636;border-radius:999px;overflow:hidden;margin:10px 0 5px}.progress div{height:100%;background:linear-gradient(90deg,var(--violet),var(--cyan));box-shadow:0 0 16px rgba(181,140,255,.25)}
.muted{color:var(--muted);font-size:10px;line-height:1.5}
table{width:100%;border-collapse:collapse;font-size:10px}th,td{padding:7px 5px;border-bottom:1px solid rgba(28,43,59,.58);text-align:left}th{color:#6f879c;font-weight:600}td{color:#b7c6d3}.plus{color:var(--cyan)}.minus{color:var(--pink)}
.event{font:10px/1.55 ui-monospace,SFMono-Regular,Consolas,monospace;color:#9eb0c0;background:#060c12;border:1px solid #162536;border-radius:11px;padding:10px;word-break:break-word}
.legend{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;color:#7990a4;font-size:9px}.lg{display:inline-flex;align-items:center;gap:5px}.lg i{width:8px;height:8px;border-radius:50%}.sens{background:var(--cyan)}.internal{background:var(--blue)}.mod{background:var(--violet)}.out{background:var(--pink)}
@media(max-width:1120px){.grid{grid-template-columns:1fr}.graph-wrap{height:560px}.hero{grid-template-columns:repeat(3,1fr)}}
@media(max-width:760px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.hero{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}.graph-wrap{height:500px}}
</style>
</head>
<body><main>
<div class="top">
 <div class="brand"><div class="logo">🧬</div><div><h1>Neural Connectome</h1><div class="sub">Live funkcjonalny widok aktywnej części mózgu Muchy — nie jest to rekonstrukcja anatomiczna.</div></div></div>
 <div class="nav"><a href="/">🏠 Przegląd</a><a href="/details">📋 Szczegóły</a><a class="active" href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a href="/associations">🕸 Skojarzenia</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div>
</div>

<section class="hero">
 <div class="kpi"><small>Neurony</small><strong id="k-neurons">—</strong><em>pełny connectome</em></div>
 <div class="kpi"><small>Połączenia</small><strong id="k-connections">—</strong><em>synapsy w macierzy</em></div>
 <div class="kpi"><small>Aktywne |a| &gt; .1</small><strong id="k-active">—</strong><em>bieżący tick</em></div>
 <div class="kpi"><small>Mean |activation|</small><strong id="k-mean">—</strong><em id="k-backend">backend —</em></div>
 <div class="kpi"><small>Reward trace</small><strong id="k-reward">—</strong><em id="k-tick">tick —</em></div>
</section>

<section class="flow">
 <div class="flowbox"><small>1 • INPUT</small><strong>Discord / Voice / słowa</strong><p>Bodźce trafiają do stabilnych populacji sensorycznych.</p></div>
 <div class="arrow">→</div>
 <div class="flowbox"><small>2 • PROPAGATION</small><strong>139k neuronów + 3.7M połączeń</strong><p>Aktywność rozchodzi się po prawdziwej topologii FlyWire i miesza z pamięcią/plastycznością.</p></div>
 <div class="arrow">→</div>
 <div class="flowbox"><small>3 • READOUT</small><strong>speak / explore / voice / react</strong><p>Populacje wyjściowe zamieniają stan mózgu na decyzje.</p></div>
 <div class="arrow">→</div>
 <div class="flowbox"><small>4 • LANGUAGE / ACTION</small><strong>generator + connectome</strong><p>Po rozbudowie słownika connectome może również zmieniać szanse konkretnych słów.</p></div>
</section>

<section class="grid">
 <div class="card">
  <div class="card-head">
   <h2>Live neural activity graph</h2>
   <div class="head-tools">
    <span class="mode" id="mode-label">STABLE WINDOW</span>
    <button class="follow" id="follow-btn" type="button">FOLLOW ACTIVITY: OFF</button>
    <div class="live"><i></i><span id="live">LIVE</span></div>
   </div>
  </div>
  <div class="graph-wrap" id="graph-wrap">
   <canvas id="net"></canvas>
   <div class="graph-label gl-left">sensory</div><div class="graph-label gl-mid">internal / modulatory</div><div class="graph-label gl-right">output</div>
   <div class="tooltip" id="tip"></div>
  </div>
  <div class="legend"><span class="lg"><i class="sens"></i> sensory</span><span class="lg"><i class="internal"></i> internal</span><span class="lg"><i class="mod"></i> modulatory</span><span class="lg"><i class="out"></i> output</span><span>• rozmiar = |aktywacja| • domyślnie węzły są trzymane ~45 s • zmiany mają fade-in / fade-out</span></div>
 </div>

 <div class="side">
  <div class="card"><div class="card-head"><h2>Action readouts</h2><span class="muted">0 → 1</span></div><div class="actions" id="actions"></div></div>
  <div class="card"><div class="card-head"><h2>Connectome → słowa</h2><span id="word-badge" class="badge wait">UCZY SŁOWNIK</span></div>
   <div class="lang-status">
    <div class="lang-top"><div><strong id="word-vocab">—</strong><div class="muted">unikalnych słów</div></div><div style="text-align:right"><strong id="word-eval">—</strong><div class="muted">ostatnio ocenionych kandydatów</div></div></div>
    <div class="progress"><div id="word-progress" style="width:0%"></div></div>
    <div class="muted" id="word-detail">—</div>
   </div>
  </div>
  <div class="card"><div class="card-head"><h2>Current brain context</h2></div>
   <div class="readout">
    <div class="mini"><small>ostatni bodziec</small><strong id="event">—</strong></div>
    <div class="mini"><small>ostatnia akcja</small><strong id="last-action">—</strong></div>
    <div class="mini"><small>wybrane neurony</small><strong id="selected">—</strong></div>
    <div class="mini"><small>krawędzie live</small><strong id="edges">—</strong></div>
    <div class="mini"><small>okno stabilne</small><strong id="stable-age">—</strong></div>
    <div class="mini"><small>ostatnia podmiana</small><strong id="replacements">—</strong></div>
   </div>
  </div>
  <div class="card"><div class="card-head"><h2>Najaktywniejsze neurony</h2></div>
   <table><thead><tr><th>root_id</th><th>rola</th><th>activation</th><th>bias</th></tr></thead><tbody id="node-table"></tbody></table>
  </div>
  <div class="card"><div class="card-head"><h2>Signal monitor</h2></div><div class="event" id="signal">czekam na dane…</div></div>
 </div>
</section>
</main>
<script>
const $=id=>document.getElementById(id);
const canvas=$("net"),ctx=canvas.getContext("2d"),wrap=$("graph-wrap"),tip=$("tip");
let stateSnap=null,visualSnap=null,layout={},hover=null,lastUpdate=0;
let followActivity=localStorage.getItem("mucha-connectome-follow")==="1";
const graphNodes=new Map(),graphEdges=new Map();
const colors={sensory:"#55ead0",internal:"#6da8ff",modulatory:"#b58cff",output:"#ff78b7"};
const nfmt=n=>Number(n||0).toLocaleString("pl-PL");
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function hash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return (h>>>0)/4294967295}
function resize(){const r=wrap.getBoundingClientRect(),dpr=Math.min(2,window.devicePixelRatio||1);canvas.width=Math.max(1,Math.floor(r.width*dpr));canvas.height=Math.max(1,Math.floor(r.height*dpr));canvas.style.width=r.width+"px";canvas.style.height=r.height+"px";ctx.setTransform(dpr,0,0,dpr,0,0)}
function roleTarget(n,w,h){
 const seed=hash(n.id),seed2=hash(n.id+"x");
 if(n.role==="sensory")return {x:w*(.07+.12*seed),y:h*(.10+.80*seed2)};
 if(n.role==="output")return {x:w*(.82+.11*seed),y:h*(.10+.80*seed2)};
 if(n.role==="modulatory")return {x:w*(.35+.30*seed),y:h*(.08+.22*seed2)};
 return {x:w*(.28+.45*seed),y:h*(.25+.64*seed2)};
}
function updateLayout(nodes){
 const r=wrap.getBoundingClientRect();
 for(const n of nodes){
  const t=roleTarget(n,r.width,r.height);
  if(!layout[n.id])layout[n.id]={x:t.x+(hash(n.id+"a")-.5)*20,y:t.y+(hash(n.id+"b")-.5)*20,tx:t.x,ty:t.y};
  layout[n.id].tx=t.x;layout[n.id].ty=t.y;
 }
}
function ingestGraph(v){
 const nowNodes=new Set((v.nodes||[]).map(n=>n.id));
 for(const entry of graphNodes.values())if(!nowNodes.has(entry.node.id))entry.targetAlpha=0;
 for(const n of (v.nodes||[])){
  const old=graphNodes.get(n.id);
  if(old){old.node=n;old.targetAlpha=1}else graphNodes.set(n.id,{node:n,alpha:0,targetAlpha:1});
 }
 const nowEdges=new Set();
 for(const e of (v.edges||[])){
  const key=e.source+">"+e.target;nowEdges.add(key);
  const old=graphEdges.get(key);
  if(old){old.edge=e;old.targetAlpha=1}else graphEdges.set(key,{edge:e,alpha:0,targetAlpha:1});
 }
 for(const [key,entry] of graphEdges)if(!nowEdges.has(key))entry.targetAlpha=0;
}
function drawGrid(w,h){
 ctx.save();ctx.strokeStyle="rgba(73,108,137,.07)";ctx.lineWidth=1;
 for(let x=24;x<w;x+=48){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}
 for(let y=24;y<h;y+=48){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
 ctx.restore();
}
function draw(){
 requestAnimationFrame(draw);
 const r=wrap.getBoundingClientRect(),w=r.width,h=r.height;ctx.clearRect(0,0,w,h);drawGrid(w,h);
 const nodeEntries=[...graphNodes.values()];
 for(const e of nodeEntries){e.alpha+=(e.targetAlpha-e.alpha)*.075}
 for(const [id,e] of graphNodes)if(e.targetAlpha===0&&e.alpha<.018){graphNodes.delete(id);delete layout[id]}
 for(const e of graphEdges.values()){e.alpha+=(e.targetAlpha-e.alpha)*.09}
 for(const [id,e] of graphEdges)if(e.targetAlpha===0&&e.alpha<.018)graphEdges.delete(id);
 const nodes=[...graphNodes.values()].map(x=>x.node);updateLayout(nodes);
 for(const p of Object.values(layout)){p.x+=(p.tx-p.x)*.045;p.y+=(p.ty-p.y)*.045}
 const byId=Object.fromEntries([...graphNodes.entries()].map(([id,e])=>[id,e]));
 let maxImp=.0001;for(const x of graphEdges.values())maxImp=Math.max(maxImp,Number(x.edge.importance||0));
 const t=performance.now()/1000;
 for(const item of graphEdges.values()){
  const e=item.edge,a=layout[e.source],b=layout[e.target];if(!a||!b)continue;
  const srcEntry=byId[e.source],dstEntry=byId[e.target];if(!srcEntry||!dstEntry)continue;
  const alpha=item.alpha*Math.min(srcEntry.alpha,dstEntry.alpha),q=clamp(Number(e.importance||0)/maxImp,0,1);
  ctx.strokeStyle="rgba(90,151,197,"+(alpha*(0.05+q*.34))+")";ctx.lineWidth=.45+q*1.35;
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();
  const source=srcEntry.node,act=Math.abs(Number(source.activation||0));
  if(act>.12&&q>.18&&alpha>.08){
   const phase=(t*(.16+.48*q)+hash(e.source+e.target))%1;
   const x=a.x+(b.x-a.x)*phase,y=a.y+(b.y-a.y)*phase;
   ctx.fillStyle=colors[source.role]||colors.internal;ctx.globalAlpha=alpha*(.35+.55*q);ctx.beginPath();ctx.arc(x,y,1.3+q*1.8,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;
  }
 }
 hover=null;
 for(const item of graphNodes.values()){
  const n=item.node,p=layout[n.id];if(!p)continue;const act=Math.abs(Number(n.activation||0)),rad=3.2+Math.min(9,act*11),c=colors[n.role]||colors.internal;
  ctx.shadowColor=c;ctx.shadowBlur=(5+act*16)*item.alpha;ctx.fillStyle=c;ctx.globalAlpha=item.alpha*(.38+Math.min(.62,act*.75));
  ctx.beginPath();ctx.arc(p.x,p.y,rad,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;ctx.shadowBlur=0;
  if(mouse.inside&&item.alpha>.45){const dx=mouse.x-p.x,dy=mouse.y-p.y;if(dx*dx+dy*dy<(rad+8)*(rad+8))hover=n}
 }
 if(hover){const p=layout[hover.id];ctx.strokeStyle="#ffffff";ctx.lineWidth=1;ctx.globalAlpha=.8;ctx.beginPath();ctx.arc(p.x,p.y,12+Math.abs(Number(hover.activation||0))*8,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1}
}
const mouse={x:0,y:0,inside:false};
canvas.addEventListener("mousemove",e=>{const r=canvas.getBoundingClientRect();mouse.x=e.clientX-r.left;mouse.y=e.clientY-r.top;mouse.inside=true;if(hover){tip.style.display="block";tip.style.left=Math.min(r.width-195,mouse.x+14)+"px";tip.style.top=Math.min(r.height-120,mouse.y+14)+"px";tip.innerHTML="<b>"+hover.id+"</b><br><span>"+hover.role+"</span><br>activation "+Number(hover.activation||0).toFixed(5)+"<br>eligibility "+Number(hover.eligibility||0).toFixed(5)+"<br>bias "+Number(hover.bias||0).toFixed(5)}else tip.style.display="none"});
canvas.addEventListener("mouseleave",()=>{mouse.inside=false;tip.style.display="none"});
function renderActions(scores){
 const order=["speak","explore","react","voice_join","voice_move","voice_leave","stay"];
 $("actions").innerHTML=order.map(k=>{const v=clamp(Number(scores[k]||0),0,1);return '<div class="act"><label>'+k+'</label><div class="track"><div class="fill" style="width:'+(v*100).toFixed(1)+'%"></div></div><b>'+v.toFixed(2)+'</b></div>'}).join("");
}
function updateModeButton(){
 const b=$("follow-btn");b.textContent="FOLLOW ACTIVITY: "+(followActivity?"ON":"OFF");b.className="follow "+(followActivity?"on":"");
 $("mode-label").textContent=followActivity?"DYNAMIC TOP ACTIVITY":"STABLE WINDOW";
}
$("follow-btn").onclick=()=>{followActivity=!followActivity;localStorage.setItem("mucha-connectome-follow",followActivity?"1":"0");updateModeButton();update()};
function render(s,v){
 stateSnap=s;visualSnap=v;ingestGraph(v);
 const d=s.diag||{},ld=s.language_diag||{},cw=ld.connectome_word_control_last||{};
 $("k-neurons").textContent=nfmt(d.neurons);$("k-connections").textContent=nfmt(d.connections);$("k-active").textContent=nfmt(d.active_abs_gt_0_1);$("k-mean").textContent=Number(d.mean_abs||0).toFixed(5);$("k-reward").textContent=Number(d.reward_trace||0).toFixed(3);$("k-backend").textContent=(d.backend||"—")+" • "+(d.device||"");$("k-tick").textContent="tick "+nfmt(d.ticks);
 $("selected").textContent=nfmt(v.selected_neurons);$("edges").textContent=nfmt(v.selected_edges);$("event").textContent=s.last_event||"—";$("last-action").textContent=s.last_action||"—";renderActions(s.scores||{});
 $("stable-age").textContent=followActivity?"FOLLOW":Math.round(Number(v.stable_age_seconds||0))+" s / "+Math.round(Number(v.stable_window_seconds||45))+" s";
 $("replacements").textContent=nfmt(v.replacements||0);
 const vocab=Number(ld.word_vocab||0),min=Number(ld.connectome_word_control_min_vocab||1),ready=!!ld.connectome_word_control_ready,pct=clamp(vocab/min*100,0,100);
 $("word-vocab").textContent=nfmt(vocab)+" / "+nfmt(min);$("word-progress").style.width=pct.toFixed(1)+"%";$("word-badge").textContent=ready?"AKTYWNY":"UCZY SŁOWNIK";$("word-badge").className="badge "+(ready?"on":"wait");
 $("word-eval").textContent=nfmt(cw.evaluated||0);$("word-detail").textContent="Siła wpływu: "+Number(ld.connectome_word_control_strength||0).toFixed(2)+" • średni ostatni score: "+Number(cw.mean_score||.5).toFixed(3)+" • generator: "+(ld.last_generator||"—");
 const nodes=(v.nodes||[]).slice().sort((a,b)=>Math.abs(Number(b.activation))-Math.abs(Number(a.activation))).slice(0,12);
 $("node-table").innerHTML=nodes.map(n=>'<tr><td>'+n.id+'</td><td>'+n.role+'</td><td class="'+(Number(n.activation)>=0?"plus":"minus")+'">'+(Number(n.activation)>=0?"+":"")+Number(n.activation).toFixed(4)+'</td><td>'+Number(n.bias||0).toFixed(5)+'</td></tr>').join("")||'<tr><td colspan="4">Brak danych.</td></tr>';
 $("signal").textContent="MODE  "+(followActivity?"FOLLOW ACTIVITY":"STABLE WINDOW")+"\nINPUT  "+(s.last_event||"—")+"\nCONNECTOME  mean |a| "+Number(d.mean_abs||0).toFixed(5)+" / max "+Number(d.max_abs||0).toFixed(5)+"\nREADOUT  speak "+Number((s.scores||{}).speak||0).toFixed(3)+" / explore "+Number((s.scores||{}).explore||0).toFixed(3)+"\nOUTPUT  "+(s.last_action||"—");
 lastUpdate=Date.now();$("live").textContent="LIVE";
}
async function update(){
 try{
  const [stateResp,visualResp]=await Promise.all([
   fetch("/api/state",{cache:"no-store"}),
   fetch("/api/connectome?follow="+(followActivity?"1":"0"),{cache:"no-store"})
  ]);
  if(stateResp.status===401||visualResp.status===401){location="/login";return}
  if(!stateResp.ok)throw new Error("state HTTP "+stateResp.status);
  if(!visualResp.ok)throw new Error("connectome HTTP "+visualResp.status);
  render(await stateResp.json(),await visualResp.json())
 }catch(e){$("live").textContent="ROZŁĄCZONO";console.error(e)}
}
window.addEventListener("resize",resize);updateModeButton();resize();draw();setInterval(update,900);update();
</script>
</body></html>"""
NEUROMAP_HTML = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha — Neuro-map</title>
<style>
:root{
 --bg:#04070c;--panel:#0a1118;--panel2:#071019;--line:#1b2b3b;--txt:#eef7ff;--muted:#758ba0;
 --cyan:#55ead0;--blue:#6ba6ff;--violet:#b68bff;--pink:#ff77b7;--good:#55df97;--warn:#ffd166;--bad:#ff7474;
}
*{box-sizing:border-box}
body{margin:0;color:var(--txt);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;background:
 radial-gradient(circle at 20% 0%,rgba(85,234,208,.12),transparent 27%),
 radial-gradient(circle at 82% 8%,rgba(107,166,255,.11),transparent 30%),
 radial-gradient(circle at 50% 100%,rgba(182,139,255,.08),transparent 36%),
 linear-gradient(180deg,#04070c,#071019 55%,#05090e)}
body:before{content:"";position:fixed;inset:0;pointer-events:none;opacity:.20;background-image:
 linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),
 linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:32px 32px}
main{max-width:1740px;margin:auto;padding:22px}
.top{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:14px}
.brand{display:flex;gap:13px;align-items:center}.logo{font-size:38px;filter:drop-shadow(0 0 22px rgba(85,234,208,.28))}
h1{margin:0;font-size:24px}.sub{color:var(--muted);font-size:12px;margin-top:4px}
.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#b9cad9;text-decoration:none;border:1px solid var(--line);background:#0a131c;padding:8px 11px;border-radius:10px;font-size:12px}
.nav a:hover{border-color:#36536e;color:white}.nav a.active{background:linear-gradient(90deg,var(--cyan),#7be7d6);color:#03110d;border-color:var(--cyan);font-weight:850}
.hero{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:12px}.kpi,.card{background:linear-gradient(180deg,rgba(11,18,27,.96),rgba(7,13,20,.96));border:1px solid var(--line);border-radius:16px;box-shadow:0 16px 50px rgba(0,0,0,.18)}
.kpi{padding:13px 14px}.kpi small{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.11em;margin-bottom:5px}.kpi strong{font-size:18px}.kpi em{display:block;color:#8da0b2;font-size:10px;font-style:normal;margin-top:4px}
.toolbar{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px;padding:10px 12px;border:1px solid var(--line);border-radius:14px;background:rgba(8,15,23,.9)}
.toolgroup{display:flex;gap:7px;align-items:center;flex-wrap:wrap}.toolgroup span{font-size:9px;color:#758ba0;text-transform:uppercase;letter-spacing:.1em;margin-right:3px}
.btn{border:1px solid #263d52;background:#08121b;color:#9db0c2;border-radius:999px;padding:7px 10px;font-size:9px;font-weight:850;letter-spacing:.06em;cursor:pointer}
.btn:hover{border-color:#527593;color:white}.btn.on{border-color:rgba(85,234,208,.55);background:rgba(85,234,208,.10);color:var(--cyan);box-shadow:0 0 18px rgba(85,234,208,.08)}
.badge{padding:6px 9px;border-radius:999px;font-size:9px;font-weight:850;letter-spacing:.08em;border:1px solid #284055;background:#071019;color:#9eb2c3}.badge.real,.badge.neuropil{color:var(--good);border-color:rgba(85,223,151,.4);background:rgba(85,223,151,.09)}.badge.hybrid{color:var(--warn);border-color:rgba(255,209,102,.35);background:rgba(255,209,102,.08)}.badge.synthetic,.badge.fallback{color:var(--bad);border-color:rgba(255,116,116,.35);background:rgba(255,116,116,.08)}
.grid{display:grid;grid-template-columns:minmax(0,2.15fr) minmax(390px,.85fr);gap:12px}.card{padding:14px;min-width:0}.card-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}.card h2{margin:0;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:#9db0c0}.live{display:inline-flex;align-items:center;gap:7px;color:var(--good);font-size:9px;font-weight:850;letter-spacing:.1em}.live i{width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 14px var(--good);animation:pulse 1.2s infinite}@keyframes pulse{50%{opacity:.35;transform:scale(.75)}}
.map-wrap{position:relative;height:790px;border-radius:15px;overflow:hidden;border:1px solid #162637;background:
 radial-gradient(circle at 50% 48%,rgba(85,234,208,.035),transparent 42%),
 linear-gradient(180deg,#050a10,#07101a)}
#brain{width:100%;height:100%;display:block}.map-title{position:absolute;left:12px;top:10px;color:#6f879a;font-size:9px;text-transform:uppercase;letter-spacing:.14em;background:rgba(4,9,14,.68);border:1px solid #172839;padding:6px 8px;border-radius:999px;backdrop-filter:blur(7px)}
.tooltip{position:absolute;z-index:5;display:none;pointer-events:none;min-width:230px;max-width:320px;padding:10px 11px;border-radius:11px;background:rgba(4,9,14,.96);border:1px solid #2b4760;box-shadow:0 16px 45px rgba(0,0,0,.45);font-size:10px;line-height:1.5}.tooltip b{font-size:11px}.tooltip .mut{color:#7890a4}.tooltip .acc{color:var(--cyan)}
.side{display:flex;flex-direction:column;gap:12px}.now{display:grid;grid-template-columns:1fr 1fr;gap:8px}.mini{padding:10px;border:1px solid #172636;background:#071019;border-radius:11px}.mini small{display:block;color:#71889c;font-size:9px;text-transform:uppercase;letter-spacing:.1em}.mini strong{display:block;margin-top:4px;font-size:12px;word-break:break-word}
.actions{display:flex;flex-direction:column;gap:7px}.act{display:grid;grid-template-columns:86px 1fr 40px;gap:8px;align-items:center;font-size:10px}.track{height:7px;background:#050b11;border:1px solid #172637;border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:999px;box-shadow:0 0 12px rgba(85,234,208,.2)}
.regions{display:flex;flex-direction:column;gap:6px;max-height:310px;overflow:auto}.region{display:grid;grid-template-columns:1fr 72px 44px;gap:8px;align-items:center;padding:8px;border:1px solid #172637;background:#071019;border-radius:10px;cursor:pointer}.region:hover,.region.on{border-color:#34536c;background:#091722}.region b{font-size:10px}.region small{color:#70879b;font-size:9px}.rtrack{height:6px;background:#050b11;border:1px solid #162536;border-radius:999px;overflow:hidden}.rfill{height:100%;background:linear-gradient(90deg,var(--violet),var(--pink));border-radius:999px}
.inspector{min-height:230px}.empty{color:#71879a;font-size:11px;line-height:1.55;padding:10px 0}.ins-title{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.ins-title strong{font-size:14px;word-break:break-all}.role{padding:4px 7px;border-radius:999px;font-size:8px;font-weight:850;letter-spacing:.08em;border:1px solid #274056;color:#a8bac9}
.meta{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:10px}.meta div{padding:8px;background:#071019;border:1px solid #172637;border-radius:9px}.meta small{display:block;color:#6f879b;font-size:8px;text-transform:uppercase;letter-spacing:.09em;margin-bottom:3px}.meta b{font-size:10px;word-break:break-word}
.effects{margin-top:9px;padding:9px;background:#061019;border:1px solid #172637;border-radius:10px}.effects small{display:block;color:#71899e;font-size:8px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px}.effect{display:grid;grid-template-columns:80px 1fr auto;gap:7px;align-items:center;font-size:9px;margin-top:5px}.effect .efill{height:6px;background:#08141f;border-radius:999px;overflow:hidden}.effect .efill i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan))}
.note{color:#71879a;font-size:9px;line-height:1.5;margin-top:8px}.legend{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;color:#748b9f;font-size:9px}.legend span{display:inline-flex;align-items:center;gap:5px}.legend i{width:8px;height:8px;border-radius:50%}.sens{background:var(--cyan)}.internal{background:var(--blue)}.mod{background:var(--violet)}.out{background:var(--pink)}
.region-inspector{min-height:300px}.spark-wrap{height:76px;border:1px solid #172637;background:#050c12;border-radius:10px;margin-top:9px;padding:5px}.spark-wrap canvas{width:100%;height:100%}.chips{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}.chip{font-size:8px;padding:4px 6px;border-radius:999px;border:1px solid #263c50;background:#071019;color:#9db0c1}.corr{display:grid;grid-template-columns:82px 1fr 42px;gap:7px;align-items:center;margin-top:6px;font-size:9px}.corrbar{height:6px;background:#07131d;border-radius:999px;overflow:hidden;position:relative}.corrbar:after{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:#28445b}.corrfill{height:100%;position:absolute;top:0}.corrfill.pos{left:50%;background:var(--cyan)}.corrfill.neg{right:50%;background:var(--pink)}.neuron-list{display:flex;flex-direction:column;gap:5px;margin-top:7px}.nrow{display:grid;grid-template-columns:1fr 62px;gap:8px;font-size:9px;padding:6px 7px;border:1px solid #172637;background:#071019;border-radius:8px}.nrow span{color:#9fb1c1}.nrow b{text-align:right}
@media(max-width:1150px){.grid{grid-template-columns:1fr}.map-wrap{height:650px}.hero{grid-template-columns:repeat(3,1fr)}}
@media(max-width:720px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.hero{grid-template-columns:1fr 1fr}.map-wrap{height:540px}.meta{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<div class="top">
 <div class="brand"><div class="logo">🧠</div><div><h1>Fly Brain Neuro-map</h1><div class="sub">Aktywność przestrzenna, nazwane neuropile, typy neuronów i runtime korelacje z zachowaniem Muchy.</div></div></div>
 <div class="nav"><a href="/">🏠 Przegląd</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a class="active" href="/neuromap">🧠 Neuro-map</a><a href="/associations">🕸 Skojarzenia</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div>
</div>

<section class="hero">
 <div class="kpi"><small>Pozycje neuronów</small><strong id="coverage">—</strong><em id="coord-count">—</em></div>
 <div class="kpi"><small>Mapa neuropili</small><strong id="neuropil-coverage">—</strong><em id="neuropil-count">—</em></div>
 <div class="kpi"><small>Aktywne |a| &gt; .1</small><strong id="active">—</strong><em>cały runtime</em></div>
 <div class="kpi"><small>Najaktywniejszy region</small><strong id="top-region">—</strong><em id="top-region-detail">—</em></div>
 <div class="kpi"><small>Dominujący readout</small><strong id="dominant">—</strong><em id="dominant-score">—</em></div>
</section>

<div class="toolbar">
 <div class="toolgroup"><span>Projekcja</span><button class="btn proj on" data-proj="xy">XY</button><button class="btn proj" data-proj="xz">XZ</button><button class="btn proj" data-proj="yz">YZ</button></div>
 <div class="toolgroup"><span>Warstwa</span><button class="btn role on" data-role="all">ALL</button><button class="btn role" data-role="sensory">SENSORY</button><button class="btn role" data-role="internal">INTERNAL</button><button class="btn role" data-role="modulatory">MODULATORY</button><button class="btn role" data-role="output">OUTPUT</button></div>
 <div class="toolgroup"><span>Widok</span><button class="btn on" id="regions-btn">REGION HEAT</button><button class="btn on" id="trail-btn">ACTIVITY TRAIL</button><span class="badge" id="region-source">REGIONS</span><span class="badge" id="coord-badge">COORDINATES</span></div>
</div>

<section class="grid">
 <div class="card">
  <div class="card-head"><h2>Brain projection / live activity</h2><div class="live"><i></i><span id="live">LIVE</span></div></div>
  <div class="map-wrap" id="map-wrap">
   <canvas id="brain"></canvas>
   <div class="map-title" id="map-title">XY PROJECTION</div>
   <div class="tooltip" id="tip"></div>
  </div>
  <div class="legend"><span><i class="sens"></i> sensory</span><span><i class="internal"></i> internal</span><span><i class="mod"></i> modulatory</span><span><i class="out"></i> output</span><span>• halo = aktywny region • kliknij region po prawej, aby go odizolować</span></div>
 </div>

 <div class="side">
  <div class="card"><div class="card-head"><h2>Co robi Mucha teraz</h2><span class="badge" id="source">runtime</span></div>
   <div class="now"><div class="mini"><small>bodziec</small><strong id="event">—</strong></div><div class="mini"><small>akcja</small><strong id="last-action">—</strong></div></div>
   <div class="actions" id="actions" style="margin-top:10px"></div>
  </div>

  <div class="card"><div class="card-head"><h2>Najaktywniejsze neuropile / rejony</h2><span class="badge" id="region-filter">ALL</span></div><div class="regions" id="regions"></div><div class="note" id="region-note">—</div></div>

  <div class="card region-inspector"><div class="card-head"><h2>Region inspector</h2><span class="badge" id="region-picked">kliknij region</span></div><div id="region-inspector" class="empty">Wybierz region z listy. Zobaczysz historię aktywności od otwarcia Neuro-map, najaktywniejsze neurony, dominujące typy komórek oraz korelacje z readoutami Muchy.</div></div>

  <div class="card inspector"><div class="card-head"><h2>Neuron inspector</h2><span class="badge" id="picked">kliknij neuron</span></div><div id="inspector" class="empty">Kliknij świecący neuron na mapie, aby zobaczyć jego adnotacje biologiczne, top neuropile, aktywację i bezpośrednie połączenia do sztucznych readoutów Muchy.</div></div>
 </div>
</section>
</main>
<script>
const $=id=>document.getElementById(id);
const canvas=$("brain"),ctx=canvas.getContext("2d"),wrap=$("map-wrap"),tip=$("tip");
const colors={sensory:"#55ead0",internal:"#6ba6ff",modulatory:"#b68bff",output:"#ff77b7"};
let projection="xy",roleFilter="all",showRegions=true,showTrail=true,data=null,hover=null,selected=null,selectedRegion="",lastFetch=0;
const trail=new Map(),mouse={x:0,y:0,inside:false};
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v)),nfmt=n=>Number(n||0).toLocaleString("pl-PL");
const esc=v=>String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));
function axes(p){return p==="xy"?["x","y"]:p==="xz"?["x","z"]:["y","z"]}
function point(o,w,h,pad=30){const [a,b]=axes(projection);return {x:pad+clamp(Number(o[a]||0),0,1)*(w-pad*2),y:pad+(1-clamp(Number(o[b]||0),0,1))*(h-pad*2)}}
function resize(){const r=wrap.getBoundingClientRect(),dpr=Math.min(2,window.devicePixelRatio||1);canvas.width=Math.max(1,Math.floor(r.width*dpr));canvas.height=Math.max(1,Math.floor(r.height*dpr));canvas.style.width=r.width+"px";canvas.style.height=r.height+"px";ctx.setTransform(dpr,0,0,dpr,0,0)}
function roleVisible(n){return roleFilter==="all"||n.role===roleFilter}
function regionOf(n){if(!data)return "";return data.region_source==="neuropil"?(n.primary_neuropil||"").trim():(n.cell_class||n.super_class||"").trim()}
function drawReference(w,h){
 if(!data)return;ctx.save();ctx.globalCompositeOperation="lighter";
 for(const p0 of (data.reference||[])){const p=point(p0,w,h,24);ctx.fillStyle=p0.real?"rgba(100,145,178,.075)":"rgba(100,145,178,.025)";ctx.beginPath();ctx.arc(p.x,p.y,p0.real?1.05:.75,0,Math.PI*2);ctx.fill()}
 ctx.restore();
}
function drawRegions(w,h){
 if(!data||!showRegions)return;const regs=(data.regions||[]),max=Math.max(.0001,...regs.map(r=>Number(r.score||0)));
 ctx.save();ctx.globalCompositeOperation="lighter";
 for(const r of regs){if(selectedRegion&&r.name!==selectedRegion)continue;const p=point(r,w,h,36),q=clamp(Number(r.score||0)/max,0,1),rad=18+q*62;
  const g=ctx.createRadialGradient(p.x,p.y,0,p.x,p.y,rad);g.addColorStop(0,"rgba(182,139,255,"+(0.08+q*.21)+")");g.addColorStop(.45,"rgba(107,166,255,"+(0.04+q*.11)+")");g.addColorStop(1,"rgba(0,0,0,0)");ctx.fillStyle=g;ctx.beginPath();ctx.arc(p.x,p.y,rad,0,Math.PI*2);ctx.fill();
  if(q>.25||selectedRegion===r.name){ctx.globalCompositeOperation="source-over";ctx.fillStyle="rgba(178,199,216,"+(0.42+q*.4)+")";ctx.font="9px Inter,system-ui";ctx.fillText(r.name,p.x+7,p.y-7);ctx.globalCompositeOperation="lighter"}
 }
 ctx.restore();
}
function updateTrail(){
 if(!data)return;const seen=new Set();
 for(const n of (data.nodes||[])){if(!roleVisible(n))continue;if(selectedRegion&&regionOf(n)!==selectedRegion)continue;seen.add(n.id);const a=Math.abs(Number(n.activation||0)),old=trail.get(n.id)||0;trail.set(n.id,Math.max(a,old*.94))}
 for(const [id,v] of trail){if(!seen.has(id)){const nv=v*.90;if(nv<.015)trail.delete(id);else trail.set(id,nv)}}
}
function drawNodes(w,h){
 if(!data)return;hover=null;const nodes=(data.nodes||[]);let maxA=.0001;for(const n of nodes)maxA=Math.max(maxA,Math.abs(Number(n.activation||0)));
 for(const n of nodes){if(!roleVisible(n))continue;if(selectedRegion&&regionOf(n)!==selectedRegion)continue;const p=point(n,w,h,26),a=Math.abs(Number(n.activation||0)),q=clamp(a/maxA,0,1),hist=showTrail?(trail.get(n.id)||a):a,c=colors[n.role]||colors.internal;
  if(showTrail&&hist>a+.015){ctx.strokeStyle=c;ctx.globalAlpha=clamp(hist*.34,0,.28);ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.x,p.y,6+hist*24,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1}
  ctx.shadowColor=c;ctx.shadowBlur=4+q*22;ctx.fillStyle=c;ctx.globalAlpha=.25+q*.75;ctx.beginPath();ctx.arc(p.x,p.y,2.1+q*5.4,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;ctx.shadowBlur=0;
  if(!n.real_position){ctx.strokeStyle="rgba(255,209,102,.45)";ctx.lineWidth=.6;ctx.beginPath();ctx.arc(p.x,p.y,4+q*5.6,0,Math.PI*2);ctx.stroke()}
  if(mouse.inside){const dx=mouse.x-p.x,dy=mouse.y-p.y;if(dx*dx+dy*dy<120)hover=n}
  if(selected&&selected.id===n.id){ctx.strokeStyle="#fff";ctx.lineWidth=1.2;ctx.globalAlpha=.9;ctx.beginPath();ctx.arc(p.x,p.y,11+q*8,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1}
 }
}
function draw(){
 requestAnimationFrame(draw);const r=wrap.getBoundingClientRect(),w=r.width,h=r.height;ctx.clearRect(0,0,w,h);
 const grad=ctx.createRadialGradient(w*.5,h*.5,20,w*.5,h*.5,Math.max(w,h)*.55);grad.addColorStop(0,"rgba(34,67,91,.07)");grad.addColorStop(1,"rgba(0,0,0,0)");ctx.fillStyle=grad;ctx.fillRect(0,0,w,h);
 drawReference(w,h);drawRegions(w,h);drawNodes(w,h)
}
function renderActions(scores){
 const order=["speak","explore","react","voice_join","voice_move","voice_leave","stay"];
 $("actions").innerHTML=order.map(k=>{const v=clamp(Number(scores[k]||0),0,1);return '<div class="act"><span>'+k+'</span><div class="track"><div class="fill" style="width:'+(v*100).toFixed(1)+'%"></div></div><b>'+v.toFixed(2)+'</b></div>'}).join("");
 const best=order.map(k=>[k,Number(scores[k]||0)]).sort((a,b)=>b[1]-a[1])[0]||["—",0];$("dominant").textContent=best[0];$("dominant-score").textContent="score "+best[1].toFixed(3)
}
function sparkline(canvasEl,values){
 const r=canvasEl.getBoundingClientRect(),dpr=Math.min(2,window.devicePixelRatio||1);canvasEl.width=Math.max(1,Math.floor(r.width*dpr));canvasEl.height=Math.max(1,Math.floor(r.height*dpr));const c=canvasEl.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);const w=r.width,h=r.height;c.clearRect(0,0,w,h);
 if(!values||values.length<2)return;const min=Math.min(...values),max=Math.max(...values),span=Math.max(.000001,max-min);
 c.strokeStyle="rgba(85,234,208,.92)";c.lineWidth=1.4;c.shadowColor="rgba(85,234,208,.45)";c.shadowBlur=7;c.beginPath();
 values.forEach((v,i)=>{const x=i/(values.length-1)*w,y=h-5-((v-min)/span)*(h-10);if(i===0)c.moveTo(x,y);else c.lineTo(x,y)});c.stroke();c.shadowBlur=0
}
function corrRows(rows){
 if(!rows||!rows.length)return '<div class="note">Za mało próbek albo brak zmienności. Korelacje pojawią się po kilku sekundach działania Neuro-map.</div>';
 return rows.map(x=>{const c=clamp(Number(x.correlation||0),-1,1),width=Math.abs(c)*50;return '<div class="corr"><span>'+esc(x.action)+'</span><div class="corrbar"><i class="corrfill '+(c>=0?"pos":"neg")+'" style="width:'+width.toFixed(1)+'%"></i></div><b>'+(c>=0?"+":"")+c.toFixed(2)+'</b></div>'}).join("")
}
function regionInspector(){
 const root=$("region-inspector");if(!data||!selectedRegion){$("region-picked").textContent="kliknij region";root.className="empty";root.innerHTML="Wybierz region z listy. Zobaczysz historię aktywności od otwarcia Neuro-map, najaktywniejsze neurony, dominujące typy komórek oraz korelacje z readoutami Muchy.";return}
 const r=(data.regions||[]).find(x=>x.name===selectedRegion);if(!r){root.className="empty";root.textContent="Wybrany region nie jest teraz w TOP aktywnych regionów.";return}
 $("region-picked").textContent=data.region_source==="neuropil"?"NEUROPIL":"CLASS GROUP";root.className="";
 const types=(r.dominant_types||[]).map(x=>'<span class="chip">'+esc(x.name)+' ×'+nfmt(x.count)+'</span>').join("")||'<span class="chip">brak typów</span>';
 const neurons=(r.top_neurons||[]).map(n=>'<div class="nrow"><span>'+esc(n.id)+' • '+esc(n.primary_type||"—")+' • '+esc(n.nt_type||"—")+'</span><b class="'+(Number(n.activation)>=0?"plus":"minus")+'">'+(Number(n.activation)>=0?"+":"")+Number(n.activation||0).toFixed(4)+'</b></div>').join("");
 root.innerHTML='<div class="ins-title"><div><strong>'+esc(r.name)+'</strong><div class="note">'+nfmt(r.active_count)+' / '+nfmt(r.total)+' neuronów ma |a| &gt; 0.1</div></div><span class="role">mean '+Number(r.mean_abs||0).toFixed(4)+'</span></div>'+
 '<div class="meta"><div><small>max activation</small><b>'+Number(r.max_abs||0).toFixed(4)+'</b></div><div><small>history samples</small><b>'+nfmt(r.correlation_samples||0)+'</b></div></div>'+
 '<div class="spark-wrap"><canvas id="region-spark"></canvas></div>'+
 '<div class="effects"><small>Dominujące typy neuronów</small><div class="chips">'+types+'</div></div>'+
 '<div class="effects"><small>Runtime correlation z readoutami</small>'+corrRows(r.readout_correlations||[])+'<div class="note">To korelacja czasowa aktywności regionu z readoutem Muchy, nie dowód biologicznej funkcji ani przyczynowości.</div></div>'+
 '<div class="effects"><small>Najaktywniejsze neurony w regionie</small><div class="neuron-list">'+neurons+'</div></div>';
 requestAnimationFrame(()=>{const c=$("region-spark");if(c)sparkline(c,r.history||[])})
}
function renderRegions(){
 const regs=data.regions||[],max=Math.max(.0001,...regs.map(r=>Number(r.score||0)));
 $("regions").innerHTML=regs.slice(0,20).map(r=>{const q=clamp(Number(r.score||0)/max,0,1);return '<div class="region '+(selectedRegion===r.name?"on":"")+'" data-region="'+esc(r.name)+'"><div><b>'+esc(r.name)+'</b><br><small>'+nfmt(r.active_count)+' / '+nfmt(r.total)+' active</small></div><div class="rtrack"><div class="rfill" style="width:'+(q*100).toFixed(1)+'%"></div></div><small>'+Number(r.mean_abs||0).toFixed(3)+'</small></div>'}).join("")||'<div class="empty">Brak nazwanych regionów w aktualnym cache.</div>';
 document.querySelectorAll("[data-region]").forEach(el=>el.onclick=()=>{const name=el.dataset.region;selectedRegion=selectedRegion===name?"":name;$("region-filter").textContent=selectedRegion||"ALL";renderRegions();regionInspector();updateTrail()})
}
function effectRows(n){
 const xs=n.system_actions||[];if(!xs.length)return '<div class="note">Brak bezpośredniego połączenia tego neuronu do sztucznych populacji action-readout w pokazanym kierunku.</div>';
 const max=Math.max(...xs.map(x=>Number(x.strength||0)),.0001);return xs.map(x=>'<div class="effect"><span>'+esc(x.name)+'</span><div class="efill"><i style="width:'+(Number(x.strength||0)/max*100).toFixed(1)+'%"></i></div><b>'+Number(x.strength||0).toFixed(4)+'</b></div>').join("")
}
function neuropilRows(n){
 const xs=n.neuropils||[];if(!xs.length)return '<div class="note">Brak summary neuropili dla tego neuronu. Przebuduj neuron_meta.npz z plikiem connections_princeton.csv.gz.</div>';
 return xs.map(x=>'<div class="effect"><span>'+esc(x.name)+'</span><div class="efill"><i style="width:'+(clamp(Number(x.share||0),0,1)*100).toFixed(1)+'%"></i></div><b>'+(Number(x.share||0)*100).toFixed(0)+'%</b></div>').join("")
}
function inspect(n){
 selected=n||selected;if(!selected)return;const n0=selected;$("picked").textContent=n0.real_position?"REAL POSITION":"FALLBACK POSITION";
 $("inspector").className="";$("inspector").innerHTML='<div class="ins-title"><div><strong>'+esc(n0.id)+'</strong><div class="note">'+esc(n0.primary_type||n0.sub_class||n0.cell_class||n0.super_class||"brak typu")+'</div></div><span class="role">'+esc(n0.role)+'</span></div>'+
 '<div class="meta"><div><small>activation</small><b>'+(Number(n0.activation)>=0?"+":"")+Number(n0.activation||0).toFixed(5)+'</b></div><div><small>eligibility</small><b>'+Number(n0.eligibility||0).toFixed(5)+'</b></div>'+
 '<div><small>class</small><b>'+esc(n0.cell_class||"—")+'</b></div><div><small>sub_class</small><b>'+esc(n0.sub_class||"—")+'</b></div>'+
 '<div><small>super_class</small><b>'+esc(n0.super_class||"—")+'</b></div><div><small>side / flow</small><b>'+esc((n0.side||"—")+" / "+(n0.flow||"—"))+'</b></div>'+
 '<div><small>neurotransmitter</small><b>'+esc(n0.nt_type||"—")+'</b></div><div><small>primary neuropil</small><b>'+esc(n0.primary_neuropil||"—")+'</b></div></div>'+
 '<div class="effects"><small>Top neuropile wg incident synapse mass</small>'+neuropilRows(n0)+'</div>'+
 '<div class="effects"><small>Wpływ na systemowe readouty Muchy</small>'+effectRows(n0)+'</div>'+
 '<div class="note">Neuropil jest skrótem opartym o sumę syn_count połączeń neuronu w regionach FlyWire. Readout opisuje sztuczny interfejs Muchy, a nie biologiczną funkcję neuronu.</div>'
}
canvas.addEventListener("mousemove",e=>{const r=canvas.getBoundingClientRect();mouse.x=e.clientX-r.left;mouse.y=e.clientY-r.top;mouse.inside=true;if(hover){tip.style.display="block";tip.style.left=Math.min(r.width-255,mouse.x+13)+"px";tip.style.top=Math.min(r.height-155,mouse.y+13)+"px";tip.innerHTML='<b>'+esc(hover.id)+'</b><br><span class="mut">'+esc(hover.primary_type||hover.cell_class||hover.super_class||hover.role)+'</span><br><span class="acc">activation '+Number(hover.activation||0).toFixed(5)+'</span><br>neuropil '+esc(hover.primary_neuropil||"—")+'<br>'+esc(hover.side||"")+' '+esc(hover.nt_type||"")}else tip.style.display="none"});
canvas.addEventListener("mouseleave",()=>{mouse.inside=false;tip.style.display="none"});
canvas.addEventListener("click",()=>{if(hover){selected=hover;inspect(selected)}});

document.querySelectorAll(".proj").forEach(b=>b.onclick=()=>{projection=b.dataset.proj;document.querySelectorAll(".proj").forEach(x=>x.classList.toggle("on",x===b));$("map-title").textContent=projection.toUpperCase()+" PROJECTION";update()});
document.querySelectorAll(".role").forEach(b=>b.onclick=()=>{roleFilter=b.dataset.role;document.querySelectorAll(".role").forEach(x=>x.classList.toggle("on",x===b));selectedRegion="";$("region-filter").textContent=roleFilter.toUpperCase();regionInspector();updateTrail()});
$("regions-btn").onclick=()=>{showRegions=!showRegions;$("regions-btn").classList.toggle("on",showRegions)};
$("trail-btn").onclick=()=>{showTrail=!showTrail;$("trail-btn").classList.toggle("on",showTrail);if(!showTrail)trail.clear()};

function render(payload){
 const m=payload.brain_map||{};data=m;updateTrail();const scores=payload.scores||{};
 const coverage=Number(m.coordinate_coverage||0);$("coverage").textContent=(coverage*100).toFixed(1)+"%";$("coord-count").textContent=nfmt(m.coordinate_neurons)+" / "+nfmt(m.total_neurons)+" neurons";
 $("neuropil-coverage").textContent=(Number(m.neuropil_coverage||0)*100).toFixed(1)+"%";$("neuropil-count").textContent=nfmt(m.neuropil_labels)+" nazwanych neuropili";
 $("active").textContent=nfmt(m.active_abs_gt_0_1);
 const top=(m.regions||[])[0];$("top-region").textContent=top?top.name:"—";$("top-region-detail").textContent=top?(nfmt(top.active_count)+" active • mean "+Number(top.mean_abs||0).toFixed(3)):"brak adnotacji";
 const badge=$("coord-badge"),mode=m.coordinate_mode||"synthetic";badge.textContent=mode==="real"?"REAL FAFB COORDS":mode==="hybrid"?"HYBRID COORDS":"FALLBACK LAYOUT";badge.className="badge "+mode;
 const rs=$("region-source");rs.textContent=m.region_source==="neuropil"?"NAMED NEUROPILS":"CLASS FALLBACK";rs.className="badge "+(m.region_source==="neuropil"?"neuropil":"fallback");$("region-note").textContent=m.region_source_detail||"—";
 $("event").textContent=payload.last_event||"—";$("last-action").textContent=payload.last_action||"—";$("source").textContent=(payload.source||"runtime").includes("FlyWire")?"FAFB v783":"runtime";renderActions(scores);renderRegions();regionInspector();
 if(selected){const fresh=(m.nodes||[]).find(n=>n.id===selected.id);if(fresh){selected=fresh;inspect(fresh)}}
 $("live").textContent="LIVE";lastFetch=Date.now()
}
async function update(){
 try{const r=await fetch("/api/neuromap?projection="+projection,{cache:"no-store"});if(r.status===401){location="/login";return}if(!r.ok)throw new Error("HTTP "+r.status);render(await r.json())}
 catch(e){$("live").textContent="ROZŁĄCZONO";console.error(e)}
}
window.addEventListener("resize",resize);resize();draw();setInterval(update,1100);update();
</script>
</body></html>"""
PUBLIC_OVERVIEW_HTML = r"""<!doctype html>
<html lang="pl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha — publiczny podgląd</title>
<style>
:root{--bg:#070c12;--panel:#0e1721;--line:#22364a;--txt:#eef7ff;--muted:#8295a8;--a:#58dac4;--blue:#70aaff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0%,rgba(88,218,196,.10),transparent 30%),linear-gradient(180deg,#070c12,#091019);color:var(--txt);font-family:Inter,system-ui,"Segoe UI",sans-serif}main{max-width:1320px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;gap:15px;align-items:center;margin-bottom:18px}.brand{display:flex;gap:12px;align-items:center}.logo{font-size:38px}h1{margin:0;font-size:25px}.sub{color:var(--muted);font-size:12px;margin-top:4px}.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#c6d2df;text-decoration:none;border:1px solid var(--line);background:#0e161f;padding:8px 11px;border-radius:10px;font-size:12px}.nav a.active{background:var(--a);border-color:var(--a);color:#06110e;font-weight:850}
.hero{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px}.card small{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px}.card strong{font-size:20px}.section{margin-top:12px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:16px}.section h2{margin:0 0 13px;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:#aabccc}.actions{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}.act{display:grid;grid-template-columns:90px 1fr 48px;gap:8px;align-items:center;font-size:11px}.track{height:8px;background:#071019;border:1px solid #1d2b39;border-radius:999px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--a));border-radius:999px}.foot{margin-top:12px;color:#687c8f;font-size:10px}
@media(max-width:760px){main{padding:13px}.top{align-items:flex-start;flex-direction:column}.hero{grid-template-columns:1fr 1fr}.actions{grid-template-columns:1fr}}
</style></head><body><main>
<div class="top"><div class="brand"><div class="logo">🪰</div><div><h1>Mucha — publiczny podgląd</h1><div class="sub">Tryb tylko do odczytu. Nie daje dostępu do konfiguracji, logów ani prywatnych danych użytkowników.</div></div></div>
<div class="nav"><a class="active" href="/public">🏠 Podgląd</a><a href="/public/connectome">🧬 Connectome</a><a href="/public/neuromap">🧠 Neuro-map</a><a href="/public/associations">🕸 Skojarzenia</a><a href="/login">🔒 Admin</a></div></div>
<section class="hero">
 <div class="card"><small>Neurony</small><strong id="neurons">—</strong></div>
 <div class="card"><small>Połączenia</small><strong id="connections">—</strong></div>
 <div class="card"><small>Aktywne neurony</small><strong id="active">—</strong></div>
 <div class="card"><small>Tick</small><strong id="tick">—</strong></div>
</section>
<div class="section"><h2>Aktualne readouty zachowania</h2><div class="actions" id="actions"></div></div>
<div class="section"><h2>Język</h2><div class="actions"><div class="act"><b>Słownik</b><div></div><span id="vocab">—</span></div><div class="act"><b>Generator</b><div></div><span id="generator">—</span></div><div class="act"><b>Reward</b><div></div><span id="reward">—</span></div><div class="act"><b>Status</b><div></div><span id="ready">—</span></div></div></div>
<div class="foot">Publiczny endpoint pokazuje tylko dane techniczne i wizualizacje. Panel administratora pozostaje chroniony logowaniem.</div>
<script>
const $=id=>document.getElementById(id),nfmt=n=>Number(n||0).toLocaleString("pl-PL");
function renderActions(scores){const order=["speak","react","voice_join","voice_move","voice_leave","explore","stay"];$("actions").innerHTML=order.map(k=>{const v=Number((scores||{})[k]||0);return '<div class="act"><b>'+k+'</b><div class="track"><div class="fill" style="width:'+Math.max(0,Math.min(100,v*100))+'%"></div></div><span>'+v.toFixed(3)+'</span></div>'}).join("")}
async function update(){try{const r=await fetch("/api/public/state",{cache:"no-store"});if(!r.ok)throw new Error("HTTP "+r.status);const s=await r.json(),d=s.diag||{},l=s.language_diag||{};$("neurons").textContent=nfmt(d.neurons);$("connections").textContent=nfmt(d.connections);$("active").textContent=nfmt(d.active_abs_gt_0_1);$("tick").textContent=nfmt(d.ticks);$("vocab").textContent=nfmt(l.word_vocab||0);$("generator").textContent=l.last_generator||"—";$("reward").textContent=Number(d.reward_trace||0).toFixed(3);$("ready").textContent=s.language_ready?"GOTOWA":"UCZY SIĘ";renderActions(s.scores||{})}catch(e){console.error(e)}}setInterval(update,1200);update();
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
  <div class="nav"><a class="active" href="/">🏠 Przegląd</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a href="/associations">🕸 Skojarzenia</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/api/state">JSON</a><a href="/logout">Wyloguj</a></div>
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
      <div class="metric"><span>Słownik słów</span><strong id="language-word-vocab">—</strong></div>
      <div class="metric"><span>Bigramy słów</span><strong id="language-word-bigrams">—</strong></div>
      <div class="metric"><span>Trigramy słów</span><strong id="language-word-trigrams">—</strong></div>
      <div class="metric"><span>Ostatni generator</span><strong id="language-generator">—</strong></div>
      <div class="metric"><span>Recent boost</span><strong id="language-recent">—</strong></div>
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
        <div class="kpi"><small>nowe tokeny słów</small><strong id="session-word-tokens">—</strong></div>
        <div class="kpi"><small>nowe słowa w słowniku</small><strong id="session-word-vocab">—</strong></div>
        <div class="kpi"><small>nowe trigramy słów</small><strong id="session-word-trigrams">—</strong></div>
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
  $("session-word-tokens").textContent="+"+nfmt(x.language_word_tokens||0);
  $("session-word-vocab").textContent="+"+nfmt(x.language_word_vocab||0);
  $("session-word-trigrams").textContent="+"+nfmt(x.language_word_trigrams||0);
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
    $("language-word-vocab").textContent=nfmt(ld.word_vocab||0);
    $("language-word-bigrams").textContent=nfmt(ld.word_bigrams||0);
    $("language-word-trigrams").textContent=nfmt(ld.word_trigrams||0);
    $("language-generator").textContent=ld.last_generator||"—";
    $("language-recent").textContent="×"+Number(ld.word_recent_boost||1).toFixed(2)+" / p="+Number(ld.word_model_probability||0).toFixed(2);
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
        connectome_provider: ConnectomeProvider | None = None,
        neuromap_provider: NeuromapProvider | None = None,
        association_provider: AssociationProvider | None = None,
        public_readonly_enabled: bool = False,
        service_unit: str = "mucha.service",
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
        self.connectome_provider = connectome_provider
        self.neuromap_provider = neuromap_provider
        self.association_provider = association_provider
        self.public_readonly_enabled = bool(public_readonly_enabled)
        self.service_unit = str(service_unit or "mucha.service").strip()
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

        if self.public_readonly_enabled and (
            request.path == "/public"
            or request.path.startswith("/public/")
            or request.path.startswith("/api/public/")
        ):
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
        app.router.add_get("/connectome", self._connectome_page)
        app.router.add_get("/neuromap", self._neuromap_page)
        app.router.add_get("/associations", self._associations_page)
        app.router.add_get("/config", self._config_page)
        app.router.add_get("/public", self._public_index)
        app.router.add_get("/public/connectome", self._public_connectome_page)
        app.router.add_get("/public/neuromap", self._public_neuromap_page)
        app.router.add_get("/public/associations", self._public_associations_page)
        app.router.add_get("/brain", self._brain)
        app.router.add_get("/login", self._login_get)
        app.router.add_post("/login", self._login_post)
        app.router.add_get("/logout", self._logout)
        app.router.add_get("/api/state", self._state)
        app.router.add_get("/api/connectome", self._connectome_state)
        app.router.add_get("/api/neuromap", self._neuromap_state)
        app.router.add_get("/api/associations", self._associations_state)
        app.router.add_get("/api/public/state", self._public_state)
        app.router.add_get("/api/public/connectome", self._public_connectome_state)
        app.router.add_get("/api/public/neuromap", self._public_neuromap_state)
        app.router.add_get("/api/public/associations", self._public_associations_state)
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

    async def _connectome_page(self, request: web.Request) -> web.Response:
        return web.Response(text=CONNECTOME_HTML, content_type="text/html")

    async def _neuromap_page(self, request: web.Request) -> web.Response:
        return web.Response(text=NEUROMAP_HTML, content_type="text/html")

    async def _associations_page(self, request: web.Request) -> web.Response:
        return web.Response(text=ASSOCIATIONS_HTML, content_type="text/html")

    def _ensure_public_enabled(self) -> None:
        if not self.public_readonly_enabled:
            raise web.HTTPNotFound(text="public dashboard disabled")

    @staticmethod
    def _publicize_html(html: str) -> str:
        replacements = (
            ('href="/"', 'href="/public"'),
            ('href="/connectome"', 'href="/public/connectome"'),
            ('href="/neuromap"', 'href="/public/neuromap"'),
            ('href="/associations"', 'href="/public/associations"'),
            ('fetch("/api/state"', 'fetch("/api/public/state"'),
            ('fetch("/api/connectome', 'fetch("/api/public/connectome'),
            ('fetch("/api/neuromap', 'fetch("/api/public/neuromap'),
            ('fetch("/api/associations"', 'fetch("/api/public/associations"'),
            ('<a href="/logout">Wyloguj</a>', '<a href="/login">🔒 Admin</a>'),
        )
        for old, new in replacements:
            html = html.replace(old, new)
        for link in (
            '<a href="/details">📋 Szczegóły</a>',
            '<a href="/affinity">🤝 Affinity</a>',
            '<a href="/config">⚙ Konfiguracja</a>',
        ):
            html = html.replace(link, "")
        return html

    async def _public_index(self, request: web.Request) -> web.Response:
        self._ensure_public_enabled()
        return web.Response(text=PUBLIC_OVERVIEW_HTML, content_type="text/html")

    async def _public_connectome_page(self, request: web.Request) -> web.Response:
        self._ensure_public_enabled()
        return web.Response(
            text=self._publicize_html(CONNECTOME_HTML),
            content_type="text/html",
        )

    async def _public_neuromap_page(self, request: web.Request) -> web.Response:
        self._ensure_public_enabled()
        return web.Response(
            text=self._publicize_html(NEUROMAP_HTML),
            content_type="text/html",
        )

    async def _public_associations_page(self, request: web.Request) -> web.Response:
        self._ensure_public_enabled()
        return web.Response(
            text=self._publicize_html(ASSOCIATIONS_HTML),
            content_type="text/html",
        )

    @staticmethod
    def _safe_public_state(snap: dict) -> dict:
        diag = dict(snap.get("diag") or {})
        language_diag = dict(snap.get("language_diag") or {})
        return {
            "diag": diag,
            "scores": dict(snap.get("scores") or {}),
            "language_diag": language_diag,
            "language_ready": bool(snap.get("language_ready")),
            "paused": bool(snap.get("paused")),
            "source": snap.get("source", "unknown"),
            "last_event": "ukryte w trybie publicznym",
            "last_action": "ukryte w trybie publicznym",
        }

    async def _public_state(self, request: web.Request) -> web.Response:
        self._ensure_public_enabled()
        snap = await self.snapshot_provider()
        return web.json_response(
            self._safe_public_state(snap),
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    async def _public_connectome_state(self, request: web.Request) -> web.Response:
        self._ensure_public_enabled()
        if self.connectome_provider is None:
            raise web.HTTPServiceUnavailable(text="connectome provider unavailable")
        raw = str(request.query.get("follow", "")).strip().lower()
        snap = await self.connectome_provider(raw in {"1", "true", "yes", "on"})
        return web.json_response(
            snap,
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    async def _public_neuromap_state(self, request: web.Request) -> web.Response:
        self._ensure_public_enabled()
        if self.neuromap_provider is None:
            raise web.HTTPServiceUnavailable(text="neuromap provider unavailable")
        projection = str(request.query.get("projection", "xy")).strip().lower()
        if projection not in {"xy", "xz", "yz"}:
            projection = "xy"
        snap = dict(await self.neuromap_provider(projection))
        snap["last_event"] = "ukryte w trybie publicznym"
        snap["last_action"] = "ukryte w trybie publicznym"
        return web.json_response(
            snap,
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    async def _public_associations_state(self, request: web.Request) -> web.Response:
        self._ensure_public_enabled()
        if self.association_provider is None:
            raise web.HTTPServiceUnavailable(text="association provider unavailable")
        snap = dict(await self.association_provider())
        snap["last_event"] = "ukryte w trybie publicznym"
        snap["last_action"] = "ukryte w trybie publicznym"
        return web.json_response(
            snap,
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    async def _config_get(self, request: web.Request) -> web.Response:
        if self.config_provider is None:
            raise web.HTTPServiceUnavailable(text="config provider unavailable")
        payload = self.config_provider()
        payload["config_path"] = str(
            Path(__file__).resolve().parents[1] / "config.local.toml"
        )
        return web.json_response(
            payload,
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
            changed = list(result.get("changed") or [])
            if result.get("ok") and changed:
                result["restart"] = {
                    "scheduled": True,
                    "unit": self.service_unit,
                    "delay_seconds": 1.2,
                }
                asyncio.create_task(self._restart_service_after_config_save())
            else:
                result["restart"] = {
                    "scheduled": False,
                    "unit": self.service_unit,
                }
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

    async def _restart_service_after_config_save(self) -> None:
        await asyncio.sleep(1.2)
        command = ["systemctl", "restart", self.service_unit]
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            command = ["sudo", "-n", *command]
        try:
            subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            log.exception(
                "Nie udało się uruchomić restartu usługi %s",
                self.service_unit,
            )

    async def _brain(self, request: web.Request) -> web.StreamResponse:
        raise web.HTTPFound("/connectome")

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

    async def _connectome_state(
        self,
        request: web.Request,
    ) -> web.Response:
        if self.connectome_provider is None:
            raise web.HTTPServiceUnavailable(
                text="connectome provider unavailable"
            )
        raw = str(request.query.get("follow", "")).strip().lower()
        follow_activity = raw in {"1", "true", "yes", "on"}
        snap = await self.connectome_provider(follow_activity)
        return web.json_response(
            snap,
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    async def _neuromap_state(
        self,
        request: web.Request,
    ) -> web.Response:
        if self.neuromap_provider is None:
            raise web.HTTPServiceUnavailable(
                text="neuromap provider unavailable"
            )
        projection = str(
            request.query.get("projection", "xy")
        ).strip().lower()
        if projection not in {"xy", "xz", "yz"}:
            projection = "xy"
        snap = await self.neuromap_provider(projection)
        return web.json_response(
            snap,
            dumps=lambda x: json.dumps(x, ensure_ascii=False),
        )

    async def _associations_state(
        self,
        request: web.Request,
    ) -> web.Response:
        if self.association_provider is None:
            raise web.HTTPServiceUnavailable(
                text="association provider unavailable"
            )
        snap = await self.association_provider()
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
