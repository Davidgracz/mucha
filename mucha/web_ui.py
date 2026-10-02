from __future__ import annotations

import asyncio
import ctypes
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

try:
    import psutil
except ImportError:
    psutil = None

log = logging.getLogger("mucha.web")

SnapshotProvider = Callable[[], Awaitable[dict]]
ConnectomeProvider = Callable[[bool, str | None], Awaitable[dict]]
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
.fields{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.field{background:var(--panel2);border:1px solid #1c2d3e;border-radius:11px;padding:11px;min-width:0}.field.hidden{display:none}.field-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.field label{display:block;color:#d8e4ee;font-size:11px;font-weight:700;line-height:1.35}.config-key{display:block;color:#eef7ff;font:700 11px/1.35 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.friendly-name{display:block;color:#8fa5b8;font-size:9px;font-weight:600;margin-top:3px}.hint{color:#72879a;font-size:9px;line-height:1.45;margin-top:4px;min-height:25px}.field input[type=number],.field input[type=text]{width:100%;margin-top:8px;border:1px solid #294057;background:#050d14;color:var(--txt);border-radius:9px;padding:9px 10px;font:inherit}.field input:focus{outline:0;border-color:var(--a);box-shadow:0 0 0 3px rgba(88,218,196,.07)}
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
 <div class="nav"><a href="/">🏠 Przegląd</a><a href="/autonomy">🧭 Autonomia</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a href="/associations">🗣 Mowa</a><a href="/affinity">🤝 Affinity</a><a class="active" href="/config">⚙ Konfiguracja</a><a href="/public">👁 Publiczny</a><a href="/logout">Wyloguj</a></div>
</div>

<div class="intro">
 <div><strong>Zmiany są trwałe</strong><p>Nazwa każdego pola pokazuje dokładną ścieżkę z configu, np. <code>voice.maximum_dwell_seconds</code>. Polska nazwa pod spodem wyjaśnia znaczenie. Wartości są walidowane przed zapisem do config.local.toml.</p></div>
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
 {id:"synapses",title:"Plastyczność synaptyczna",desc:"Reward i punish mogą zmieniać siłę rzeczywistych połączeń FAFB uczestniczących w ostatnim śladzie aktywności.",section:"brain",open:true,fields:[
  ["synaptic_plasticity_enabled","Uczenie synaps","bool",0,0,0,"Włącza rzadką nakładkę uczonych wag na istniejących połączeniach connectomu."],
  ["synaptic_plasticity_lr","Tempo uczenia synaps","number",0.0001,0,0.05,"Jak mocno pojedynczy reward/punish zmienia aktywne połączenia."],
  ["synaptic_plasticity_max_delta","Maks. zmiana synapsy","number",0.005,0.001,0.5,"Limit odchylenia uczonej wagi od bazowej wagi FAFB."],
  ["synaptic_plasticity_trace_neurons","Neurony śladu dla synaps","number",16,32,1024,"Ile najsilniejszych neuronów eligibility analizować przy reward/punish."],
  ["synaptic_plasticity_max_edges","Limit uczonych synaps","number",1000,1000,250000,"Maksymalna liczba zapamiętanych zmian połączeń; najsilniejsze są zachowywane."],
  ["consolidation_enabled","Konsolidacja + zapominanie","bool",0,0,0,"Włącza czasowy decay wyłącznie dla nauczonych biasów i synaptic delta. Bazowy connectome FAFB nie jest zmieniany."],
  ["consolidation_interval_seconds","Interwał konsolidacji","number",10,10,86400,"Jak często wykonywać czasowe zapominanie i porządkowanie śladów."],
  ["bias_forgetting_half_life_hours","Półokres plastic bias","number",1,1,8760,"Po ilu godzinach nieutrwalony plastic bias spada o połowę wskutek czasowego zapominania."],
  ["synaptic_forgetting_half_life_hours","Półokres learned synapse","number",1,1,8760,"Bazowy półokres zaniku synaptic delta przed ochroną wynikającą z konsolidacji."],
  ["synaptic_consolidation_gain","Tempo konsolidacji synaps","number",0.01,0,1,"Jak szybko spójne kolejne rewardy zwiększają consolidation strength synapsy."],
  ["synaptic_consolidation_decay_half_life_days","Półokres consolidation strength","number",0.25,0.25,3650,"Jak wolno zanika sama odporność utrwalonego śladu."],
  ["synaptic_consolidation_protection","Ochrona utrwalonych synaps","number",0.1,0,50,"Mnożnik wydłużający półokres learned synapse wraz ze wzrostem consolidation strength."],
  ["synaptic_prune_threshold","Próg usuwania śladu","number",0.00001,0,0.05,"Bardzo słabe i nieutrwalone synaptic delta poniżej tego progu są usuwane."],
  ["synaptic_consolidated_threshold","Próg CONSOLIDATED","number",0.01,0,1,"Od jakiej siły pamięci learned synapse jest oznaczana jako utrwalona."]
 ]},
 {id:"neuromod",title:"Neuromodulatory v2",desc:"Dopamina, serotonina i octopamina sterują globalną dynamiką sieci zamiast działać wyłącznie jak zwykłe dodatnie synapsy.",section:"brain",open:true,fields:[
  ["neuromodulation_enabled","Neuromodulacja globalna","bool",0,0,0,"Włącza osobne stany dopaminy, serotoniny i octopaminy."],
  ["neuromodulatory_direct_residual","Resztkowy wpływ bezpośredni","number",0.01,0,1,"Jaka część starego bezpośredniego wpływu synaptycznego neuronów modulacyjnych zostaje w macierzy."],
  ["dopamine_plasticity_gain","Dopamina → plastyczność","number",0.05,0,4,"Jak mocno aktywność dopaminowa zwiększa tempo uczenia bias i synaps."],
  ["serotonin_stability_gain","Serotonina → stabilność","number",0.01,0,0.3,"Zwiększa pamięć stanu i lekko tłumi szum przy aktywności serotoninowej."],
  ["octopamine_arousal_gain","Octopamina → pobudzenie","number",0.05,0,1.5,"Zwiększa propagation gain i szum przy aktywności octopaminowej."],
  ["neuromodulator_smoothing","Bezwładność neuromodulatorów","number",0.01,0,0.999,"Wyżej = poziomy zmieniają się wolniej i utrzymują się dłużej."]
 ]},
 {id:"internal-states",title:"Internal states / attractors",desc:"SOCIAL NEED, CURIOSITY, STRESS, SATIETY i AROUSAL są odczytywane z aktywności zespołów neuronów. Bodziec wchodzi przez sensory, a podtrzymanie używa wyłącznie istniejących rekurencyjnych krawędzi FAFB.",section:"brain",open:true,fields:[
  ["internal_states_enabled","Neural internal states","bool",0,0,0,"Włącza wewnętrzne attractory. Nie dodaje bezpośrednich bonusów do action score."],
  ["internal_state_pool_size","Neurony na attractor","number",8,24,1024,"Rozmiar zespołu neuronów wewnętrznych dla każdego stanu."],
  ["internal_state_entry_width","Sensory entry neurons","number",8,16,1024,"Ile neuronów sensorycznych może pobudzać każdy attractor na podstawie realnego reachability."],
  ["internal_state_recurrent_gain","Recurrent FAFB gain","number",0.01,0,2,"Dodatkowe wzmocnienie tylko istniejących krawędzi rekurencyjnych wewnątrz attractoru."],
  ["internal_state_level_gain","Czułość odczytu stanu","number",0.1,0.1,20,"Przelicza dodatnią aktywność neuronalną attractoru na poziom 0–1."],
  ["internal_state_arousal_gain","AROUSAL → propagation","number",0.01,0,1.5,"Jak mocno neuronalny AROUSAL zwiększa propagation gain."],
  ["internal_state_stress_gain","STRESS → propagation","number",0.01,0,1.5,"Jak mocno neuronalny STRESS zwiększa propagation gain i pobudzenie."],
  ["internal_state_satiety_stability_gain","SATIETY → stabilność","number",0.01,0,0.5,"Jak mocno neuronalny SATIETY zwiększa leak/stabilność stanu sieci."]
 ]},
 {id:"motivation",title:"Stage 30 — Natural motivation",desc:"SOCIAL, NOVELTY, SAFETY i REST powstają z homeostatic drives oraz affectu. Frustracja rośnie przy długo niezaspokojonej potrzebie, a satiation chwilowo tłumi motyw po sukcesie. Urgency skaluje wyłącznie wejście drive → attractor → FAFB.",section:"brain",open:true,fields:[
  ["motivation_enabled","Natural motivation","bool",0,0,0,"Włącza wspólny układ motywacyjny Stage 30."],
  ["motivation_frustration_threshold","Próg frustracji","number",0.01,0,0.95,"Powyżej jakiego pressure niezaspokojony motyw zaczyna gromadzić frustrację."],
  ["motivation_frustration_per_minute","Narastanie frustracji / min","number",0.005,0,1,"Tempo wzrostu frustracji przy utrzymującej się potrzebie."],
  ["motivation_frustration_decay_per_minute","Spadek frustracji / min","number",0.005,0,1,"Tempo wygaszania frustracji, gdy pressure spadnie."],
  ["motivation_satiation_decay_per_minute","Spadek satiation / min","number",0.005,0,1,"Jak szybko po zaspokojeniu potrzeba odzyskuje zdolność do ponownego motywowania."],
  ["motivation_frustration_gain","Frustration → urgency","number",0.05,0,3,"Jak mocno skumulowana frustracja wzmacnia urgency."],
  ["motivation_satiation_gain","Satiation → inhibition","number",0.05,0,1,"Jak mocno chwilowe nasycenie osłabia urgency."],
  ["motivation_neural_gain","Urgency → neural input","number",0.05,0,1.5,"Jak mocno urgency skaluje istniejące wejścia homeostatic drive do neuronalnych attractorów."]
 ]},
 {id:"foresight",title:"Stage 31 — Internal foresight",desc:"Przed finalną konkurencją Mucha symuluje na kopii swojego stanu, które potrzeby każdy kandydat może rozładować. Symulacja nie zmienia live state i nie wybiera akcji bezpośrednio; jej wynik wraca jako sensory evidence do FAFB.",section:"brain",open:true,fields:[
  ["foresight_enabled","Internal foresight","bool",0,0,0,"Włącza kontrfaktyczną symulację kandydatów Stage 31."],
  ["foresight_drive_relief_scale","Skala przewidywanego reliefu","number",0.02,0,1,"Jak duży hipotetyczny spadek zgodnego drive może zasymulować jedna akcja."],
  ["foresight_state_signal_gain","Relief → sensory signal","number",0.05,0,2,"Jak mocno przewidywana poprawa stanu wraca jako sensory evidence dla kandydata."],
  ["foresight_base_confidence","Bazowa pewność modelu","number",0.05,0,1,"Pewność strukturalnej części symulacji przed uwzględnieniem historii rewardu."],
  ["foresight_uncertainty_weight","Waga niepewności","number",0.05,0,1,"Ile niepewność historii rewardu wnosi do diagnostycznego risk." ]
 ]},
 {id:"intention",title:"Stage 32 — Persistent intent",desc:"Finalny neuronalny winner autonomii może utworzyć krótkotrwałą intencję. Intencja nie wymusza akcji: przy następnym ticku wraca jako action-guided sensory cue, propaguje się przez FAFB i może zostać przegłosowana przez świeżą konkurencję.",section:"brain",open:true,fields:[
  ["intention_enabled","Persistent intent","bool",0,0,0,"Włącza pamięć krótkotrwałego zamiaru Stage 32."],
  ["intention_half_life_seconds","Half-life intencji [s]","number",1,1,86400,"Co ile sekund siła niepodtrzymywanej intencji spada o połowę."],
  ["intention_max_age_seconds","Maks. wiek intencji [s]","number",5,1,604800,"Po tym czasie intencja jest kasowana niezależnie od siły."],
  ["intention_signal_gain","Intencja → sensory signal","number",0.05,0,2,"Jak mocno zapamiętany zamiar wraca do sensorycznych ścieżek kandydata."],
  ["intention_reinforcement_gain","Wzmocnienie po ponownym winnerze","number",0.05,0,1,"Jak mocno kolejny zgodny winner utrwala istniejącą intencję."],
  ["intention_switch_margin","Próg zmiany zamiaru","number",0.01,0,1,"Jak dużo silniejszy musi być nowy evidence, aby zastąpić aktywną intencję."],
  ["intention_min_evidence","Minimum evidence","number",0.01,0,1,"Minimalna siła świeżego neuronalnego zwycięstwa potrzebna do utworzenia intencji."],
  ["intention_outcome_gain","Wpływ reward/punish","number",0.05,0,1,"Jak mocno realny outcome wzmacnia lub osłabia intencję tej samej akcji."]
 ]},
 {id:"goal",title:"Stage 33 — Multi-step motivational goals",desc:"Mucha może utrzymywać cel typu SOCIAL / NOVELTY / SAFETY / REST i realizować go przez serię kolejnych neuronalnie wybranych działań. Goal nie wybiera akcji bezpośrednio — tylko wzmacnia sensorycznie kandydatów, dla których Stage 31 przewiduje relief docelowej potrzeby.",section:"brain",open:true,fields:[
  ["goal_enabled","Multi-step goals","bool",0,0,0,"Włącza cele wieloetapowe Stage 33."],
  ["goal_signal_gain","Goal → sensory signal","number",0.05,0,2,"Jak mocno przewidywany relief docelowej potrzeby wraca jako sensory cue dla kandydata."],
  ["goal_min_relief","Minimum relief do utworzenia celu","number",0.005,0,1,"Minimalny Stage-31 relief wymagany, aby winner mógł uruchomić nowy cel."],
  ["goal_min_start_urgency","Minimum urgency celu","number",0.05,0,1.5,"Minimalna bieżąca urgency potrzeby, aby powstał cel wieloetapowy."],
  ["goal_success_progress","Próg ukończenia","number",0.05,0.05,1,"Jaka część początkowej urgency musi zostać realnie zredukowana, aby cel został uznany za ukończony."],
  ["goal_max_age_seconds","Maks. wiek celu [s]","number",10,1,604800,"Po jakim czasie nierozwiązany cel zostaje porzucony."],
  ["goal_max_steps","Maks. liczba kroków","number",1,1,100,"Ile wykonanych prób może należeć do jednego celu."],
  ["goal_max_failed_steps","Maks. nieudanych kroków","number",1,1,50,"Po ilu nieudanych wykonaniach cel zostaje porzucony."]
 ]},
 {id:"personality",title:"Stage 35 — Emergent personality",desc:"Trwały temperament uczy się powoli z faktycznych zachowań, reward/punish i sukcesów lub porzuceń celów. Cechy nie dodają punktów do action score — powyżej neutralnego poziomu wracają tylko przez sensoryczne wejścia istniejących attractorów FAFB.",section:"brain",open:true,fields:[
  ["personality_enabled","Emergent personality","bool",0,0,0,"Włącza trwałe cechy Stage 35."],
  ["personality_learning_rate","Personality learning rate","number",0.005,0.0001,0.5,"Jak wolno nowe doświadczenia przesuwają trwały temperament. Ma być znacznie wolniejsze niż zwykły reward learning."],
  ["personality_signal_gain","Personality → attractor signal","number",0.02,0,1.5,"Maksymalny wpływ wyuczonej cechy na istniejące internal-state sensory entries."],
  ["personality_min_observations","Obserwacje do confidence","number",1,1,1000,"Skala liczby doświadczeń potrzebna, aby wyuczona cecha zaczęła być mocno wyrażana neuronalnie."]
 ]},
 {id:"action-policy",title:"Learned Action Policy",desc:"Reward i punish uczą osobny bias każdej akcji. Connectome nadal daje surowy readout, a policy tylko przesuwa jego skuteczną wartość w ograniczonym zakresie.",section:"brain",open:true,fields:[
  ["action_policy_enabled","Learned action policy","bool",0,0,0,"Włącza trwałe uczenie preferencji akcji na podstawie reward/punish."],
  ["action_policy_lr","Policy learning rate","number",0.005,0,1,"Jak szybko reward zmienia bias wybranej akcji."],
  ["action_policy_max_bias","Maks. |policy bias|","number",0.01,0,3,"Ogranicza jak mocno policy może przesunąć decyzję connectomu."],
  ["action_policy_decay","Policy decay","number",0.0001,0.9,1,"Przy każdym rewardzie stare biasy są lekko wygaszane. 1.0 = bez wygaszania."]
 ]},
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
  ["connectome_word_control_candidates","Kandydaci oceniani przez connectome","number",1,4,96,"Ile najlepszych kandydatów słów connectome ocenia na krok."],
  ["connectome_word_feedback_enabled","Rekurencyjny feedback słów","bool",0,0,0,"Po wyborze każdego słowa przepuszcza jego ślad z powrotem przez connectome przed wyborem następnego."],
  ["connectome_word_feedback_steps","Ticki mózgu na słowo","number",1,1,8,"Ile kroków connectomu wykonuje Mucha po każdym wybranym słowie."],
  ["connectome_word_feedback_magnitude","Siła feedbacku słowa","number",0.01,0,1,"Jak mocno wybrane słowo zmienia stan connectomu przed wyborem kolejnego."]
 ]},
 {id:"attention",title:"Attention / Working Memory",desc:"Krótkotrwała uwaga utrzymuje osoby, kanały i tematy jako zanikający kontekst, który jest ponownie podawany do connectomu.",section:"behavior",open:true,fields:[
  ["attention_enabled","Attention + working memory","bool",0,0,0,"Włącza krótkotrwały stan uwagi i jego neuronalną reiniekcję."],
  ["attention_half_life_seconds","Półokres uwagi","number",5,5,3600,"Po ilu sekundach siła nieodświeżanego elementu uwagi spada o połowę."],
  ["working_memory_seconds","Okno working memory","number",10,10,3600,"Jak długo ostatnie sceny tekst/voice pozostają dostępne jako aktywny kontekst."],
  ["attention_max_items","Maks. elementów uwagi","number",1,1,32,"Ile najsilniejszych osób, kanałów i tematów może być jednocześnie reiniektowanych."],
  ["attention_reinject_magnitude","Siła reiniekcji uwagi","number",0.01,0,1.5,"Jak mocno aktywny kontekst wraca do sensorycznych populacji connectomu przy idle tick."],
  ["attention_topic_words","Słów tematu na scenę","number",1,1,12,"Ile istotnych słów z wiadomości może utworzyć krótkotrwałe ślady topic."],
  ["attention_mention_boost","Boost uwagi po @Mucha","number",0.05,0,1,"Dodatkowa siła focusu osoby, która bezpośrednio zwraca się do Muchy."]
 ]},
 {id:"behavior",title:"Zachowanie",desc:"Progi mówienia, reakcji i podstawowe parametry uczenia społecznego.",section:"behavior",open:true,fields:[
  ["speak_threshold","LEGACY: próg mówienia","number",0.01,0,1,"Używany tylko gdy Connectome behavior competition jest wyłączone. W trybie neural competition SPEAK konkuruje bezpośrednio ze STAY."],
  ["reaction_threshold","LEGACY: próg reakcji emoji","number",0.01,0,1,"Używany tylko w trybie legacy. Przy neural competition REACT konkuruje bezpośrednio ze STAY."],
  ["connectome_behavior_competition_enabled","Connectome behavior competition","bool",0,0,0,"Domyślny tryb: tekstowe SPEAK, REACT i spontaneous SPEAK wybiera konkurencja readoutów zamiast ręcznych progów."],
  ["reaction_cooldown_seconds","Cooldown reakcji","number",1,0,3600,"Techniczny limit częstotliwości reakcji; nie jest preferencją neuronalną."],
  ["social_learning_enabled","Uczenie społeczne","bool",0,0,0,"Włącza reward/affinity z zachowań użytkowników."],
  ["neural_social_memory_enabled","Neuralna pamięć użytkowników","bool",0,0,0,"Każdy użytkownik dostaje stabilną reprezentację neuronalną i trwałe zmiany synaps w connectomie."],
  ["neural_affinity_weight","Udział neural affinity","number",0.05,0,1,"Maksymalny udział pamięci connectomu w relacji. Reszta pochodzi z legacy affinity podczas migracji."],
  ["neural_social_learning_scale","Siła neural social learning","number",0.05,0,3,"Mnożnik zapisu pozytywnych i negatywnych zdarzeń do neuronalnej pamięci użytkownika."],
  ["person_model_enabled","Długoterminowe modele ludzi","bool",0,0,0,"Buduje trwały profil doświadczeń konkretnej osoby z pamięci semantycznej i reaktywuje go jako sensoryczny kontekst."],
  ["person_model_min_observations","Model osoby: min. obserwacji","number",1,1,100,"Ile zapisanych doświadczeń potrzeba, zanim profil osoby zacznie być podawany do connectomu."],
  ["person_model_sensory_magnitude","Model osoby: siła sensoryczna","number",0.05,0,1.5,"Siła reiniekcji profilu osoby. Nie jest bonusem do akcji; sygnał przechodzi przez zwykłe neurony sensoryczne."],
  ["channel_model_enabled","Długoterminowe modele kanałów","bool",0,0,0,"Buduje trwały profil miejsc voice z wizyt, epizodów, ludzi i dynamiki rozmowy."],
  ["channel_model_min_observations","Model kanału: min. obserwacji","number",1,1,1000,"Ile trwałych obserwacji potrzeba, zanim profil miejsca zacznie być podawany do connectomu."],
  ["channel_model_sensory_magnitude","Model kanału: siła sensoryczna","number",0.05,0,1.5,"Siła sensorycznej reiniekcji pamięci miejsca. Nie daje bezpośredniego bonusu JOIN/MOVE/STAY."],
  ["social_scene_model_enabled","Długoterminowe sytuacje społeczne","bool",0,0,0,"Łączy ludzi, kanał, dynamikę rozmowy i stan Muchy w trwałe powtarzalne sceny."],
  ["social_scene_min_observations","Sytuacja społeczna: min. obserwacji","number",1,1,1000,"Ile obserwacji sytuacji potrzeba, zanim jej profil zacznie wracać do connectomu."],
  ["social_scene_sensory_magnitude","Sytuacja społeczna: siła sensoryczna","number",0.05,0,1.5,"Siła reiniekcji znanej sytuacji jako zwykłego sensorycznego kontekstu connectomu."],
  ["voice_dynamics_learning_enabled","Uczenie dynamiki rozmowy z rewardu","bool",0,0,0,"Uczy wyniki SPEAK/STAY/JOIN/MOVE/LEAVE dla powtarzalnych wzorców rozmowy niezależnie od konkretnej osoby i kanału."],
  ["voice_dynamics_min_observations","Dynamika rozmowy: min. obserwacji","number",1,1,1000,"Od ilu obserwacji znany wzorzec dynamiki wraca do connectomu."],
  ["voice_dynamics_sensory_magnitude","Dynamika rozmowy: siła sensoryczna","number",0.05,0,1.5,"Siła reiniekcji wyuczonej historii dynamiki rozmowy. Nie modyfikuje action score bezpośrednio."],
  ["voice_dynamics_seen_cooldown_seconds","Dynamika rozmowy: cooldown familiarity","number",5,5,3600,"Minimalny odstęp między neutralnymi obserwacjami tego samego wzorca, aby polling nie pompował familiarity."],
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
  ["connectome_voice_control_enabled","Voice sterowany connectomem","bool",0,0,0,"Join / stay / move / leave wybiera konkurencja readoutów. Progi voice zostają tylko jako tryb legacy po wyłączeniu tej opcji."],
  ["voice_sensory_enabled","Voice Sensory Bus","bool",0,0,0,"Live sensory z PCM i sceny Discorda: kto mówi, cisza, overlap, tempo rozmowy, affinity i odpowiedź po TTS. Dane trafiają do sensory neurons bez bezpośredniego bonusu do akcji."],
  ["voice_sensory_interval_seconds","Sensory Bus: interwał","number",0.05,0.25,10,"Co ile sekund aktualna scena voice jest ponownie podawana do connectomu."],
  ["voice_sensory_speaker_timeout_seconds","Sensory Bus: timeout mówcy","number",0.05,0.15,3,"Po jakiej przerwie w pakietach PCM uznać, że użytkownik przestał mówić."],
  ["voice_sensory_reply_window_seconds","Sensory Bus: reply po TTS","number",1,1,120,"Okno czasu, w którym wypowiedź użytkownika po TTS Muchy jest oznaczana jako odpowiedź na jej głos."],
  ["voice_sensory_base_magnitude","Sensory Bus: siła bazowa","number",0.05,0.05,2,"Bazowa amplituda surowych cue voice. Nie jest action score ani JOIN boostem."],
  ["voice_sensory_steps","Sensory Bus: ticki","number",1,1,8,"Liczba ticków propagacji po każdej aktualizacji surowej sceny voice."],
  ["social_drive_enabled","Neuralny social drive","bool",0,0,0,"Długi pobyt poza voice przy dostępnych ludziach daje narastający bodziec sensoryczny. Nie wymusza JOIN — decyzję nadal podejmuje connectome."],
  ["social_drive_start_seconds","Social drive: start","number",5,0,86400,"Po ilu sekundach poza voice zaczyna rosnąć bodziec społeczny, jeśli są dostępni ludzie."],
  ["social_drive_ramp_seconds","Social drive: ramp","number",5,1,86400,"Ile sekund trwa wzrost od 0 do pełnej siły bodźca."],
  ["social_drive_max_magnitude","Social drive: siła sensoryczna","number",0.05,0,4,"Maksymalna amplituda bodźca wpuszczanego do sensory neurons. Nie jest bonusem do score JOIN."],
  ["social_drive_stay_punish","Social drive: nauka przeciw STAY","number",0.005,0,0.5,"Delikatny punish ścieżki STAY, gdy przy silnym social drive connectome nadal wybiera pozostanie poza voice."],
  ["social_drive_learning_interval_seconds","Social drive: interwał nauki","number",5,5,3600,"Minimalny odstęp między kolejnymi punishami STAY poza voice."],
  ["social_join_reward","Reward za neuralny JOIN","number",0.01,0,1,"Reward dla ścieżki voice_join po udanym wejściu wybranym przez connectome."],
  ["reward_opportunity_enabled","Losowa możliwość nagrody na VC","bool",0,0,0,"Jeden dostępny kanał dostaje sensoryczny sygnał potencjalnej nagrody. Nie dodaje punktów bezpośrednio do voice_join."],
  ["reward_opportunity_ttl_seconds","Możliwość nagrody: czas życia","number",5,10,3600,"Jak długo wylosowany kanał pozostaje aktualną możliwością nagrody."],
  ["reward_opportunity_min_strength","Możliwość nagrody: min sygnału","number",0.05,0,4,"Dolna amplituda sensorycznego reward-opportunity cue."],
  ["reward_opportunity_max_strength","Możliwość nagrody: max sygnału","number",0.05,0,4,"Górna amplituda sensorycznego reward-opportunity cue."],
  ["reward_opportunity_success_chance","Szansa realnej nagrody","number",0.01,0,1,"Po sprawdzeniu wskazanego kanału nagroda nie jest pewna; ta wartość określa prawdopodobieństwo."],
  ["reward_opportunity_reward","Wartość znalezionej nagrody","number",0.01,0,1,"Reward dla ścieżki voice_join, gdy potencjalna nagroda okaże się prawdziwa."],
  ["reward_opportunity_stay_punish","Kara za zignorowaną możliwość","number",0.005,0,0.5,"Gdy connectome widzi reward opportunity, ale wybiera STAY, kara trafia do śladu STAY zamiast sztucznie podbijać JOIN."],
  ["reward_opportunity_stay_punish_interval_seconds","Interwał kary za ignorowanie","number",5,5,3600,"Minimalny odstęp między kolejnymi karami STAY przy aktywnej możliwości nagrody."],
  ["motivation_propagation_steps","Ticki propagacji motywacji","number",1,2,12,"Ile kroków connectomu dostają wolne stany motywacyjne zanim zostaną odczytane przez readouty."],
  ["homeostasis_enabled","Homeostaza voice","bool",0,0,0,"Włącza wolne stany wewnętrzne: zmęczenie społeczne, habituację i potrzebę eksploracji. Stany dają bodźce connectomowi, ale nie wymuszają akcji."],
  ["social_fatigue_start_seconds","Zmęczenie społeczne: start","number",5,0,86400,"Po ilu sekundach ciągłego pobytu z ludźmi zaczyna narastać social fatigue."],
  ["social_fatigue_ramp_seconds","Zmęczenie społeczne: ramp","number",5,1,86400,"Czas wzrostu social fatigue od 0 do pełnego poziomu."],
  ["social_fatigue_max_magnitude","Zmęczenie społeczne: siła","number",0.05,0,4,"Maksymalna siła sensoryczna kierowana przez connectome w stronę MOVE/LEAVE."],
  ["habituation_enabled","Habituacja sceny VC","bool",0,0,0,"Powtarzający się kanał i ten sam skład osób stopniowo wywołują słabszy bodziec sensoryczny."],
  ["habituation_half_life_seconds","Habituacja: półczas","number",5,1,86400,"Po tym czasie niezmieniona scena osiąga około 50% poziomu habituacji."],
  ["habituation_max_suppression","Habituacja: maks. wygaszenie","number",0.01,0,0.95,"Maksymalna część bodźca aktualnego kanału i pamięci osób, która może zostać wygaszona."],
  ["habituation_change_magnitude","Habituacja: potrzeba zmiany","number",0.05,0,4,"Siła dodatkowego bodźca zmiany środowiska przepuszczanego przez connectome."],
  ["exploration_drive_enabled","Głód eksploracji VC","bool",0,0,0,"Włącza narastającą potrzebę sprawdzenia innego dostępnego kanału."],
  ["exploration_drive_start_seconds","Eksploracja: start","number",5,0,86400,"Po ilu sekundach niezmienionej sceny zaczyna rosnąć exploration drive."],
  ["exploration_drive_ramp_seconds","Eksploracja: ramp","number",5,1,86400,"Czas wzrostu exploration drive od 0 do 100%."],
  ["exploration_drive_max_magnitude","Eksploracja: siła","number",0.05,0,4,"Maksymalna siła bodźca kierowanego przez realne połączenia do voice_move."],
  ["episodic_prediction_enabled","Pamięć epizodyczna / prediction error","bool",0,0,0,"Zapamiętuje wyniki decyzji voice w podobnych kontekstach i oblicza błąd przewidywania nagrody."],
  ["episodic_database","Baza pamięci epizodycznej","text",0,0,0,"Plik SQLite z trwałą pamięcią epizodów i predykcji. Domyślnie state/voice_episodes.sqlite3."],
  ["episodic_memory_size","Cache ostatnich epizodów","number",16,16,5000,"Ile ostatnich epizodów trzymać dodatkowo w RAM dla szybkiego podglądu."],
  ["episodic_max_persisted_events","Maks. trwałych epizodów","number",100,16,1000000,"Limit rekordów w SQLite. Najstarsze epizody są usuwane dopiero po przekroczeniu tego limitu."],
  ["episodic_recall_magnitude","Siła przypomnienia epizodu","number",0.05,0,4,"Dodatnie oczekiwane rewardy z podobnej sceny wracają jako sensoryczny cue przez connectome do właściwego readoutu."],
  ["prediction_learning_rate","Tempo uczenia predykcji","number",0.01,0.001,1,"Jak szybko przewidywana nagroda context+action zbliża się do rzeczywistych wyników."],
  ["prediction_error_scale","Wpływ prediction error na connectome","number",0.01,0,1,"Jaka część błędu przewidywania trafia jako dodatkowa korekta reward/punish do zapisanego śladu neuronalnego."],
  ["prediction_error_max_correction","Limit korekty prediction error","number",0.01,0,0.5,"Maksymalna dodatkowa korekta pojedynczego epizodu."],
  ["prediction_max_age_seconds","Maks. wiek oczekiwanego wyniku","number",5,5,3600,"Jak długo po decyzji reward/punish może zostać przypisany do jej przewidywanego wyniku."],
  ["prediction_credit_queue_size","Kolejka temporal credit","number",1,1,64,"Ile ostatnich decyzji voice może współdzielić późniejszy reward/punish."],
  ["prediction_credit_decay_seconds","Zanik temporal credit","number",1,1,3600,"Stała czasowa zaniku kredytu. Nowsze decyzje dostają większy udział."],
  ["memory_replay_enabled","MEMORY REPLAY","bool",0,0,0,"Po okresie ciszy odtwarza ważne epizody jako słabe bodźce i ponownie przepuszcza je przez connectome."],
  ["memory_replay_idle_seconds","Replay: cisza przed startem","number",10,10,86400,"Minimalny czas bez nowych wiadomości i zmian voice przed konsolidacją pamięci."],
  ["memory_replay_interval_seconds","Replay: interwał","number",10,30,86400,"Minimalny odstęp pomiędzy kolejnymi sesjami replay."],
  ["memory_replay_batch_size","Replay: epizody na sesję","number",1,1,8,"Liczba wspomnień reaktywowanych w jednej sesji."],
  ["memory_replay_magnitude","Replay: siła bodźca","number",0.01,0,1.5,"Siła reaktywacji sceny i action-guided sensory cue."],
  ["memory_replay_reward_scale","Replay: siła uczenia","number",0.01,0,0.5,"Mała część oryginalnego reward/prediction error używana do ponownej plastyczności."],
  ["memory_replay_steps","Replay: ticki connectomu","number",1,1,24,"Liczba kroków propagacji realnego connectome przed ponownym rewardem."],
  ["memory_replay_max_age_days","Replay: maks. wiek wspomnienia","number",1,1,365,"Jak stare epizody mogą wracać podczas konsolidacji."],
  ["sleep_enabled","SLEEP / offline consolidation","bool",0,0,0,"Po dłuższej ciszy Mucha wchodzi w osobny stan snu i wykonuje serię mocniejszych replay bez normalnych decyzji voice/text."],
  ["sleep_idle_seconds","Sleep: zwykła cisza przed snem","number",60,60,604800,"Ile sekund realnej ciszy Discord musi minąć przed zwykłym snem, gdy Mucha nie jest jeszcze TIRED."],
  ["sleep_tired_idle_seconds","Sleep: cisza gdy TIRED","number",5,5,3600,"Gdy fatigue przekroczy circadian tired threshold, po wyjściu z VC wystarczy ten krótszy czas ciszy przed snem."],
  ["sleep_leave_signal_gain","TIRED → VOICE_LEAVE signal","number",0.05,0,4,"Siła neuronalnego sleep-pressure cue prowadzonego do VOICE_LEAVE po przekroczeniu progu TIRED."],
  ["sleep_force_disconnect_fatigue","Force leave przy fatigue","number",0.01,0.65,1,"Przy tak wysokim fatigue rozłączenie z VC staje się technicznym warunkiem przygotowania do snu i omija minimum dwell."],
  ["sleep_cycle_interval_seconds","Sleep: odstęp cykli","number",5,5,3600,"Minimalny odstęp pomiędzy kolejnymi cyklami offline replay podczas snu."],
  ["sleep_max_cycles","Sleep: cykle na sesję","number",1,1,64,"Po ilu cyklach kończy się jedna sesja snu. Jeśli dalej trwa cisza, Mucha automatycznie uzbroi kolejną sesję po następnym sleep_idle_seconds; aktywność wybudza ją i zeruje licznik."],
  ["sleep_replay_batch_size","Sleep: epizody na cykl","number",1,1,16,"Ile istotnych wspomnień może zostać reaktywowanych w jednym cyklu snu."],
  ["sleep_replay_magnitude_multiplier","Sleep: × siła reaktywacji","number",0.05,0.25,4,"Mnożnik siły sensorycznego replay względem zwykłego MEMORY REPLAY."],
  ["sleep_reward_scale_multiplier","Sleep: × plastyczność replay","number",0.05,0.25,4,"Mnożnik niewielkiego reward/punish używanego do ponownej plastyczności podczas snu."],
  ["sleep_steps_multiplier","Sleep: × ticki connectomu","number",0.25,0.5,4,"Mnożnik liczby kroków propagacji connectomu podczas każdego replay w stanie snu."],
  ["episodic_consolidation_gain","Tempo utrwalania wspomnień","number",0.01,0,1,"Jak szybko powtarzane doświadczenia i MEMORY REPLAY zwiększają siłę pamięci sceny."],
  ["episodic_forgetting_half_life_days","Półokres pamięci epizodycznej","number",0.25,0.25,3650,"Jak szybko bez ponownego wzmacniania zanikają expected reward i consolidation strength scen."],
  ["episodic_forgetting_interval_seconds","Interwał zapominania pamięci","number",30,30,86400,"Jak często stosować czasowy decay pamięci epizodycznej."],
  ["episodic_consolidated_threshold","Próg utrwalonej sceny","number",0.01,0,1,"Od jakiej siły scena jest traktowana jako skonsolidowane wspomnienie."],
  ["semantic_memory_enabled","Pamięć semantyczna z epizodów","bool",0,0,0,"Uogólnia powtarzające się doświadczenia po użytkowniku, kanale i stanie homeostatycznym."],
  ["semantic_recall_min_observations","Minimum doświadczeń semantycznych","number",1,1,1000,"Ile podobnych zdarzeń musi istnieć, zanim uogólnienie zacznie być przypominane."],
  ["semantic_recall_magnitude","Siła semantic recall","number",0.05,0,4,"Skaluje dodatni lub ujemny bodziec pamięci semantycznej podawany do connectomu."],
  ["semantic_recall_steps","Ticki semantic recall","number",1,1,12,"Minimalna liczba kroków propagacji, gdy aktywne jest przypomnienie semantyczne."],
  ["uncertainty_exploration_enabled","Curiosity / uncertainty exploration","bool",0,0,0,"Nieznane kanały i sytuacje pobudzają attractor CURIOSITY zamiast bezpośrednio zmieniać action score."],
  ["uncertainty_curiosity_magnitude","Siła uncertainty → CURIOSITY","number",0.05,0,4,"Maksymalna amplituda bodźca curiosity przy pełnej niepewności semantycznej."],
  ["uncertainty_curiosity_steps","Ticki propagacji uncertainty","number",1,1,12,"Minimalna liczba kroków connectomu, gdy aktywny jest cue z niepewności."],
  ["uncertainty_target_weight","Waga uncertainty przy wyborze celu","number",0.01,0,2,"Po decyzji JOIN/MOVE zwiększa szansę wybrania mniej znanego kanału; nie zmienia samego readoutu akcji."],
  ["information_gain_reward_scale","Reward za information gain","number",0.01,0,1,"Skaluje wewnętrzną nagrodę, gdy nowe doświadczenie realnie zmniejsza niepewność."],
  ["information_gain_reward_max","Maks. reward information gain","number",0.01,0,0.5,"Górny limit pojedynczej nagrody za zdobycie informacji."],
  ["information_gain_min_delta","Minimum information gain","number",0.005,0,1,"Minimalny spadek uncertainty wymagany do przyznania wewnętrznego rewardu."],
  ["minimum_dwell_seconds","Motor refractory po wejściu","number",1,0,86400,"W tym czasie move/leave są fizycznie niedostępne; connectome nadal widzi bodziec early-dwell."],
  ["maximum_dwell_seconds","Maksymalny pobyt","number",1,1,86400,"Po tym czasie uruchamia się mechanizm overstay/threat."],
  ["overstay_punish_amount","Kara STAY za zbyt długi pobyt","number",0.05,0,1,"Kara ucząca ścieżkę STAY, gdy Mucha po maximum_dwell_seconds nadal zostaje na kanale."],
  ["overstay_punish_interval_seconds","Interwał kary overstay","number",1,1,3600,"Minimalny odstęp między kolejnymi karami STAY za zbyt długi pobyt."],
  ["threat_ramp_seconds","Narastanie presji overstay","number",1,1,86400,"Czas narastania sygnału threat od początku overstay do pełnej siły."],
  ["threat_magnitude","Siła sygnału threat","number",0.05,0,4,"Maksymalna amplituda sygnału threat używanego po przekroczeniu maksymalnego pobytu."],
  ["threat_move_boost","Legacy boost MOVE","number",0.01,0,1,"Dodatkowe wzmocnienie MOVE w trybie legacy; przy sterowaniu connectomem nie zastępuje decyzji sieci."],
  ["threat_affinity_relaxation","Rozluźnienie affinity przy threat","number",0.01,0,1,"Zmniejsza wpływ przywiązania do bieżącego kanału podczas presji overstay."],
  ["threat_escape_reward","Reward za ucieczkę z threat","number",0.01,0,1,"Nagroda za skuteczne opuszczenie/przeniesienie się z kanału po presji threat."],
  ["threat_steps","Ticki propagacji threat","number",1,1,12,"Liczba kroków connectomu używana do propagacji sygnału threat."],
  ["move_threshold","Próg move","number",0.01,0,1,"Próg decyzji o zmianie kanału w trybie legacy."],
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
 {id:"blocked-text",title:"discord.blocked_text_channel_ids",desc:"Wykluczone kanały tekstowe — zaznaczone kanały są całkowicie pomijane przez część tekstową.",kind:"text"},
 {id:"blocked-voice",title:"voice.blocked_voice_channel_ids",desc:"Wykluczone kanały voice — Mucha nie wejdzie na zaznaczone kanały.",kind:"voice"},
 {id:"blocked-guild",title:"voice.blocked_voice_guild_ids",desc:"Wykluczone serwery voice — całkowity zakaz VC dla zaznaczonych serwerów.",kind:"guild"}
];

function esc(v){return String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]))}
function fieldId(section,key){return section+"-"+key}
function renderField(section,f){
 const [key,label,type,step,min,max,hint]=f,value=state[section]?.[key],id=fieldId(section,key),configKey=section+"."+key;
 const search=(label+" "+configKey+" "+hint+" "+section).toLowerCase();
 const name='<code class="config-key">'+esc(configKey)+'</code><span class="friendly-name">'+esc(label)+'</span>';
 if(type==="bool")return '<div class="field" data-search="'+esc(search)+'"><div class="field-head"><div><label for="'+id+'">'+name+'</label><div class="hint">'+esc(hint)+'</div></div><label class="switch"><input id="'+id+'" type="checkbox" '+(value?'checked':'')+'><span class="slider"></span></label></div></div>';
 if(type==="text")return '<div class="field" data-search="'+esc(search)+'"><label for="'+id+'">'+name+'</label><div class="hint">'+esc(hint)+'</div><input id="'+id+'" type="text" value="'+esc(value)+'"></div>';
 return '<div class="field" data-search="'+esc(search)+'"><label for="'+id+'">'+name+'</label><div class="hint">'+esc(hint)+'</div><input id="'+id+'" type="number" step="'+step+'" min="'+min+'" max="'+max+'" value="'+value+'"></div>';
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
 $("blocked-text-list").innerHTML=text.map(ch=>channelRow("text",ch.id,ch.name,ch.guild,ch.blocked,ch.hard_blocked)).join("")||'<div class="hint">Brak kanałów tekstowych.</div>';
 $("blocked-voice-list").innerHTML=voice.map(ch=>channelRow("voice",ch.id,ch.name,ch.guild,ch.blocked)).join("")||'<div class="hint">Brak kanałów voice.</div>';
 $("blocked-guild-list").innerHTML=guilds.map(g=>channelRow("voice-guild",g.id,g.name,"Całkowity zakaz VC",g.voice_blocked)).join("")||'<div class="hint">Brak serwerów.</div>';
}
function channelRow(kind,id,name,sub,checked,hardBlocked=false){const locked=kind==="text"&&hardBlocked;return '<label class="channel"><input type="checkbox" data-'+kind+'="'+id+'" '+(checked?'checked':'')+' '+(locked?'disabled':'')+'><div><b>'+esc(name)+(locked?' <span style="color:#ffb36b">• NA SZTYWNO</span>':'')+'</b><br><small>'+esc(sub)+'</small></div><small>'+id+'</small></label>'}
function bindInputs(){document.querySelectorAll("input").forEach(el=>{if(el.id==="search")return;el.addEventListener("input",markDirty);el.addEventListener("change",markDirty)})}
function collect(){
 const out={brain:{},language:{},behavior:{},voice:{}};
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
.memory-note{margin-bottom:10px}.memory-bar{height:7px;background:#071019;border:1px solid #1d2a39;border-radius:999px;overflow:hidden;min-width:86px}.memory-bar>i{display:block;height:100%;background:linear-gradient(90deg,#6da8ff,var(--a))}.path{font:10px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;color:#8298aa;max-width:360px;word-break:break-all}.neural{color:#6da8ff;font-weight:800}.effective{color:var(--a);font-weight:850}
.person-list{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.person-card{background:linear-gradient(145deg,rgba(88,218,196,.055),rgba(11,18,25,.98));border:1px solid #22384a;border-radius:14px;padding:12px;min-width:0}.person-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.person-name{font-size:15px;font-weight:850}.person-id{display:block;color:var(--muted);font:9px/1.4 ui-monospace,Consolas,monospace;margin-top:3px}.person-state{border:1px solid #2d475c;border-radius:999px;padding:4px 7px;font-size:9px;font-weight:850;text-transform:uppercase}.person-state.positive{color:var(--good);border-color:#2e654c}.person-state.negative{color:var(--bad);border-color:#66373d}.person-state.mixed{color:var(--warn);border-color:#66572e}.person-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin-top:10px}.person-k{background:#081018;border:1px solid #192a38;border-radius:9px;padding:8px;min-width:0}.person-k small{display:block;color:var(--muted);font-size:8px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px}.person-k b{font-size:11px;word-break:break-word}.person-bars{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:8px}.person-sub{margin-top:8px;padding-top:8px;border-top:1px solid #1a2a37;color:#96a8b7;font-size:10px;line-height:1.5}.person-tags{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}.person-tag{border:1px solid #274056;background:#08121a;border-radius:999px;padding:4px 6px;font-size:8px;color:#a7b8c7}.person-tag.good{border-color:#2e614b;color:#79dba4}.person-tag.bad{border-color:#60363b;color:#ec9198}.person-empty{padding:12px;border:1px dashed #263a4a;border-radius:11px;color:var(--muted);font-size:11px}
@media(max-width:1100px){.person-list{grid-template-columns:1fr}}
@media(max-width:900px){.grid{grid-template-columns:1fr}.span2{grid-column:auto}.kpis{grid-template-columns:1fr 1fr}.phrases{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}}
</style></head><body><main>
<div class="top"><div><h1>🤝 Affinity / Zasady relacji</h1><div class="sub">Live podgląd tego, co zwiększa i obniża stosunek Muchy do użytkowników.</div></div>
<div class="nav"><a href="/">🏠 Przegląd</a><a href="/autonomy">🧭 Autonomia</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a href="/associations">🗣 Mowa</a><a class="active" href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div></div>

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
    <table><thead><tr><th>Użytkownik</th><th>Affinity używane</th><th>Negative streak</th><th>👍 reakcje</th><th>👎 reakcje</th><th>Status</th></tr></thead><tbody id="users"></tbody></table>
  </div>

  <div class="card span2">
    <h2>🧠 Social Neural Memory</h2>
    <div class="reason memory-note" id="neural-memory-note">Ładowanie pamięci connectomu…</div>
    <table><thead><tr><th>Użytkownik</th><th>Neural</th><th>Legacy</th><th>Używane</th><th>Dojrzałość</th><th>Assembly</th><th>Uczone synapsy</th><th>Aktywność</th><th>Najsilniejsza ścieżka Δ</th></tr></thead><tbody id="neural-users"></tbody></table>
  </div>

  <div class="card span2">
    <h2>🧬 Long-term Person Models</h2>
    <div class="reason memory-note" id="person-model-note">Ładowanie profili ludzi…</div>
    <div class="person-list" id="person-models"></div>
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

  const users=d.user_affinities||[];
  $("users").innerHTML=users.map(u=>{
    const a=num(u.affinity),status=a<=num(t.avoid)?"OMIJA":a>=num(t.liked)?"LUBI":a>=num(t.familiar)?"ZNAJOMY":"NEUTRAL";
    const cls=a<0?"minus":a>0?"plus":"";
    const streak=num(u.negative_streak),mult=num(u.negative_multiplier||1);
    return '<tr><td>'+esc(u.display_name||u.user_id)+'</td><td class="effective">'+signed(a)+'</td><td class="'+(streak?"minus":"")+'">'+Math.round(streak)+(streak?" ×"+mult.toFixed(2):"")+'</td><td>'+num(u.positive_reactions)+'</td><td>'+num(u.negative_reactions)+'</td><td class="'+cls+'">'+status+'</td></tr>';
  }).join("")||'<tr><td colspan="6">Brak relacji.</td></tr>';

  const settings=d.social_settings||{},maxWeight=num(settings.neural_affinity_weight);
  $("neural-memory-note").innerHTML=settings.neural_social_memory_enabled
    ? '<b>AKTYWNA.</b> Używane affinity = legacy + neural memory. Maksymalny udział connectomu: <b>'+Math.round(maxWeight*100)+'%</b>; udział rośnie wraz z dojrzałością assembly. Reward/punish zapisuje bias neuronów i prawdziwe learned synapses.'
    : '<b>WYŁĄCZONA.</b> Zachowanie korzysta wyłącznie z legacy affinity.';
  $("neural-users").innerHTML=users.filter(u=>u.neural_memory).slice(0,20).map(u=>{
    const m=u.neural_memory||{},neural=num(u.neural_affinity),legacy=num(u.legacy_affinity),effective=num(u.effective_affinity??u.affinity),maturity=Math.max(0,Math.min(1,num(u.neural_maturity)));
    const edge=(m.top_edges||[])[0];
    const path=edge
      ? '#'+esc(edge.source)+' → #'+esc(edge.target)+' '+signed(edge.delta)
      : 'jeszcze brak learned edge';
    const age=m.last_activation_age==null?'—':(num(m.last_activation_age)<60?num(m.last_activation_age).toFixed(0)+' s':(num(m.last_activation_age)/60).toFixed(1)+' min');
    return '<tr>'+
      '<td><b>'+esc(u.display_name||u.user_id)+'</b><br><small>'+esc(u.user_id)+'</small></td>'+
      '<td class="neural">'+signed(neural)+'</td>'+
      '<td>'+signed(legacy)+'</td>'+
      '<td class="effective">'+signed(effective)+'</td>'+
      '<td><div class="memory-bar"><i style="width:'+(maturity*100).toFixed(1)+'%"></i></div><small>'+(maturity*100).toFixed(1)+'% • w '+(num(u.neural_weight)*100).toFixed(1)+'%</small></td>'+
      '<td>'+num(m.identity_neurons)+' ID → '+num(m.memory_neurons)+' mem</td>'+
      '<td>'+num(m.learned_synapses)+'<br><small>mean |Δ| '+num(m.mean_abs_synaptic_delta).toExponential(2)+'</small></td>'+
      '<td>'+num(m.memory_activity).toFixed(4)+'<br><small>ostatnio '+age+'</small></td>'+
      '<td class="path">'+path+'</td>'+
    '</tr>';
  }).join("")||'<tr><td colspan="9">Pamięć neuronalna nie ma jeszcze użytkowników do pokazania.</td></tr>';

  const profiles=d.person_profiles||[];
  $("person-model-note").innerHTML=settings.person_model_enabled
    ? '<b>AKTYWNE.</b> Profil osoby powstaje z trwałej pamięci semantycznej: osoba + akcja, osoba + kanał i osoba + stan sytuacji. Od <b>'+Math.round(num(settings.person_model_min_observations||2))+' obserwacji</b> wraca do connectomu jako raw sensory context o bazowej sile <b>'+num(settings.person_model_sensory_magnitude||0.35).toFixed(2)+'</b>. Nie dodaje punktów bezpośrednio do JOIN/MOVE/STAY/SPEAK.'
    : '<b>WYŁĄCZONE.</b> Profile są nadal zapisane w pamięci, ale nie są reiniektowane do connectomu.';
  $("person-models").innerHTML=profiles.slice(0,20).map(p=>{
    const fam=Math.max(0,Math.min(1,num(p.familiarity))),conf=Math.max(0,Math.min(1,num(p.confidence))),val=num(p.valence);
    const state=String(p.valence_label||"neutral"),preferred=p.preferred_action||null,avoided=p.avoided_action||null;
    const channels=(p.channels||[]).slice(0,3),contexts=(p.contexts||[]).slice(0,2),inj=p.last_injection||{},brain=inj.brain||{};
    const last=p.last_seen?new Date(Number(p.last_seen)*1000).toLocaleString("pl-PL"):"—";
    const channelTags=channels.map(x=>'<span class="person-tag">'+esc(x.channel_name||x.channel_id)+' • n='+num(x.observations)+' • '+signed(x.expected_reward)+'</span>').join("");
    const contextTags=contexts.map(x=>'<span class="person-tag">'+esc(String(x.context||"").slice(0,72))+' • n='+num(x.observations)+'</span>').join("");
    const actionTags=(p.actions||[]).slice(0,4).map(x=>'<span class="person-tag '+(num(x.signal)>0.03?"good":num(x.signal)<-0.03?"bad":"")+'">'+esc(x.action)+' '+signed(x.signal)+'</span>').join("");
    return '<div class="person-card">'+
      '<div class="person-head"><div><div class="person-name">'+esc(p.display_name||p.user_id)+'</div><span class="person-id">'+esc(p.user_id)+' • ostatnio '+esc(last)+'</span></div><span class="person-state '+esc(state)+'">'+esc(state)+'</span></div>'+
      '<div class="person-grid">'+
        '<div class="person-k"><small>obserwacje</small><b>'+num(p.observations)+'</b></div>'+
        '<div class="person-k"><small>familiarity</small><b>'+(fam*100).toFixed(0)+'%</b></div>'+
        '<div class="person-k"><small>confidence</small><b>'+(conf*100).toFixed(0)+'%</b></div>'+
        '<div class="person-k"><small>history valence</small><b class="'+(val>0.05?"plus":val<-0.05?"minus":"")+'">'+signed(val)+'</b></div>'+
        '<div class="person-k"><small>reward + / -</small><b>'+num(p.positive_count)+' / '+num(p.negative_count)+'</b></div>'+
        '<div class="person-k"><small>preferred</small><b>'+(preferred?esc(preferred.action)+' '+signed(preferred.signal):'—')+'</b></div>'+
        '<div class="person-k"><small>avoided</small><b>'+(avoided?esc(avoided.action)+' '+signed(avoided.signal):'—')+'</b></div>'+
        '<div class="person-k"><small>ostatni sensory cue</small><b>'+num(brain.cue_count)+' cue</b></div>'+
      '</div>'+
      '<div class="person-bars"><div><small>familiarity</small><div class="memory-bar"><i style="width:'+(fam*100).toFixed(1)+'%"></i></div></div><div><small>confidence</small><div class="memory-bar"><i style="width:'+(conf*100).toFixed(1)+'%"></i></div></div></div>'+
      '<div class="person-sub"><b>Akcje:</b><div class="person-tags">'+(actionTags||'<span class="person-tag">brak dojrzałych skojarzeń</span>')+'</div></div>'+
      '<div class="person-sub"><b>Kanały:</b><div class="person-tags">'+(channelTags||'<span class="person-tag">brak</span>')+'</div></div>'+
      '<div class="person-sub"><b>Konteksty:</b><div class="person-tags">'+(contextTags||'<span class="person-tag">brak</span>')+'</div></div>'+
    '</div>';
  }).join("")||'<div class="person-empty">Brak dojrzałych profili. Powstaną z kolejnych zapisanych doświadczeń voice.</div>';

  const s=d.social_debug||{};
  $("last-social").innerHTML='<b>'+esc(s.event||"—")+'</b> • '+esc(s.user_name||"—")+' • '+esc(s.detail||"—")+' • Δ '+signed(s.amount||0)+' • używane '+signed(s.affinity||0)+' • neural '+signed(s.neural_affinity||0)+' • legacy '+signed(s.legacy_affinity||0)+' • maturity '+(num(s.neural_maturity)*100).toFixed(1)+'%';
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
<title>Mucha — Language Brain</title>
<style>
:root{
 --bg:#050910;--panel:#0c141d;--panel2:#08111a;--panel3:#0a1722;--line:#203247;
 --txt:#eef7ff;--muted:#7f92a5;--cyan:#55ead0;--blue:#6da8ff;--violet:#b58cff;
 --good:#58df98;--bad:#ff7474;--warn:#ffd166;--orange:#ffad66
}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;color:var(--txt);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
 background:radial-gradient(circle at 14% 0%,rgba(85,234,208,.10),transparent 27%),
 radial-gradient(circle at 88% 0%,rgba(109,168,255,.10),transparent 30%),
 linear-gradient(180deg,#050910,#07101a 54%,#050a10)}
main{max-width:1880px;margin:auto;padding:22px 28px 36px}
.top{display:flex;justify-content:space-between;gap:18px;align-items:center;margin-bottom:16px}
.brand{display:flex;gap:13px;align-items:center}.logo{font-size:38px}h1{margin:0;font-size:25px}
.sub{color:var(--muted);font-size:12px;margin-top:4px;max-width:900px;line-height:1.5}
.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#bacada;text-decoration:none;border:1px solid var(--line);
 background:#0b141d;padding:8px 11px;border-radius:10px;font-size:12px}
.nav a.active{background:linear-gradient(90deg,var(--cyan),#7ce5d4);border-color:var(--cyan);color:#04120e;font-weight:850}
.notice{display:flex;gap:10px;align-items:flex-start;padding:11px 13px;margin-bottom:13px;border:1px solid #244057;
 border-radius:13px;background:rgba(8,17,26,.86);font-size:10px;line-height:1.55;color:#9db0c1}
.notice b{color:#dceaf5}.help{display:inline-grid;place-items:center;width:16px;height:16px;border-radius:50%;
 border:1px solid #38566e;color:#8feadd;background:#0a151e;font:800 9px/1 ui-monospace,Consolas,monospace;cursor:help;vertical-align:middle}
.help:hover,.help:focus{border-color:var(--cyan);color:white}
.tooltip{position:fixed;z-index:99;display:none;width:min(380px,calc(100vw - 24px));padding:12px 13px;
 border:1px solid #35536c;border-radius:13px;background:#061019;box-shadow:0 18px 55px rgba(0,0,0,.55);pointer-events:none}
.tooltip.show{display:block}.tooltip b{display:block;font-size:12px;margin-bottom:5px}.tooltip p{margin:0;color:#a8bac8;font-size:10px;line-height:1.55}
.tooltip em{display:block;margin-top:7px;padding-top:7px;border-top:1px solid #1a2b39;color:#72dccc;font-size:10px;font-style:normal;line-height:1.5}

.hero{display:grid;grid-template-columns:repeat(6,minmax(135px,1fr));gap:10px;margin-bottom:13px}
.kpi,.card,.stage,.step-card{background:linear-gradient(180deg,rgba(13,21,30,.97),rgba(8,15,23,.97));
 border:1px solid var(--line);border-radius:16px;box-shadow:0 16px 44px rgba(0,0,0,.12)}
.kpi{padding:12px 14px;min-width:0}.kpi small{display:flex;gap:5px;align-items:center;color:var(--muted);font-size:8px;
 text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px}.kpi strong{display:block;font-size:17px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.kpi em{display:block;color:#8498aa;font-size:9px;font-style:normal;margin-top:4px;line-height:1.35}
.good{color:var(--good)!important}.bad{color:var(--bad)!important}.warn{color:var(--warn)!important}.cyan{color:var(--cyan)!important}

.process{display:grid;grid-template-columns:repeat(7,minmax(120px,1fr));gap:7px;margin-bottom:14px}
.stage{padding:11px;min-width:0;position:relative}.stage:after{content:"→";position:absolute;right:-11px;top:50%;transform:translateY(-50%);z-index:2;color:#3d5a70;font-size:17px}
.stage:last-child:after{display:none}.stage small{display:flex;align-items:center;gap:5px;color:#74899d;font-size:8px;text-transform:uppercase;letter-spacing:.11em;margin-bottom:5px}
.stage strong{display:block;font-size:12px;line-height:1.35;word-break:break-word}.stage p{margin:5px 0 0;color:#73889b;font-size:8px;line-height:1.4}
.stage.active{border-color:#376a70;background:linear-gradient(180deg,rgba(22,49,52,.72),rgba(8,17,26,.97))}
.stage.blue{border-color:#304d70}.stage.violet{border-color:#4b3d68}.stage.goodstage{border-color:#315b4b}

.grid{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(330px,.55fr);gap:13px;align-items:start}
.card{padding:14px;min-width:0}.card.full{grid-column:1/-1}.head{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:10px}
.head h2{display:flex;align-items:center;gap:6px;margin:0;font-size:11px;text-transform:uppercase;letter-spacing:.11em;color:#a6b8c8}
.live{font-size:9px;color:var(--good);font-weight:850;letter-spacing:.1em}
.output{font-size:22px;font-weight:800;line-height:1.45;padding:15px;border:1px solid #264254;background:
 linear-gradient(135deg,rgba(85,234,208,.07),rgba(109,168,255,.035));border-radius:13px;min-height:66px;word-break:break-word}
.context{margin-top:8px;padding:10px 11px;background:#071019;border:1px solid #172a3b;border-radius:11px;color:#a9bac8;font-size:10px;line-height:1.5;max-height:120px;overflow:auto}
.context b{color:#d8e8f3}

.state-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.statebox{padding:10px;border:1px solid #17283a;background:var(--panel2);border-radius:10px}
.statebox small{display:flex;align-items:center;gap:5px;color:#74899c;font-size:8px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px}
.statebox b{font-size:12px}.bar{height:7px;background:#071019;border:1px solid #172a38;border-radius:999px;overflow:hidden;margin-top:6px}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan));border-radius:999px}

.formulas{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.formula{padding:11px;border:1px solid #1b3042;border-radius:11px;background:#071019;min-width:0}
.formula small{display:flex;gap:5px;align-items:center;color:#8196a9;font-size:8px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.formula code{display:block;color:#d8e7f2;font:10px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:normal;word-break:break-word}
.formula p{margin:6px 0 0;color:#74899b;font-size:8px;line-height:1.45}

.attempt-tabs{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}.tab{border:1px solid #294259;background:#08121b;color:#9db1c2;border-radius:999px;padding:7px 10px;font-size:9px;font-weight:800;cursor:pointer}
.tab.on{background:var(--cyan);border-color:var(--cyan);color:#04130f}.tab.reject{border-color:#5b363b;color:#e7a1a7}
.trace-summary{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;color:#8fa2b3;font-size:9px}.chip{display:inline-flex;align-items:center;gap:4px;padding:4px 7px;border-radius:999px;border:1px solid #263c50;background:#08121b;color:#9fb3c4;font-size:8px}
.chip.tri{border-color:#425a80}.chip.bi{border-color:#3a665e}.chip.uni{border-color:#5d4b72}.chip.feedback{border-color:#34614c;color:#8de2ad}.chip.selected{border-color:var(--cyan);color:var(--cyan)}
.steps{display:flex;flex-direction:column;gap:9px}.step-card{overflow:hidden}.step-head{display:grid;grid-template-columns:56px minmax(150px,1fr) auto auto;gap:9px;align-items:center;padding:10px 12px;cursor:pointer;background:#0a141e}
.step-head:hover{background:#0d1924}.step-no{font:800 9px/1 ui-monospace,Consolas,monospace;color:#70879a}.chosen{font-size:14px;font-weight:850;color:#dcebf4;word-break:break-word}
.step-meta{color:#8499aa;font-size:8px;text-align:right}.roll{font:800 9px/1 ui-monospace,Consolas,monospace;color:#8ee7da}
.step-body{display:none;border-top:1px solid #1b2d3d;padding:11px}.step-card.open .step-body{display:block}.step-card.open .step-head{background:#0d1a24}
.source-line{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}.explain{padding:8px 9px;border:1px dashed #2a4153;border-radius:9px;color:#8fa2b1;font-size:9px;line-height:1.5;margin-bottom:9px}

.tablewrap{overflow:auto;border:1px solid #17283a;border-radius:10px}table{width:100%;border-collapse:collapse;font-size:9px;min-width:840px}
th,td{padding:7px 8px;border-bottom:1px solid #152636;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#0a141d;color:#7890a3;font-size:8px;text-transform:uppercase;letter-spacing:.07em;z-index:1}
th:first-child,td:first-child{text-align:left}.selected-row{background:rgba(85,234,208,.075)}.selected-row td:first-child{color:var(--cyan);font-weight:850}
.brainhigh{color:var(--good)}.brainlow{color:#f1a3a3}.interval{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#8398aa}
.sources{display:flex;gap:3px;justify-content:flex-end}.src{font-size:7px;padding:2px 4px;border:1px solid #274056;border-radius:4px;color:#9db0c1}

.mapgrid{display:grid;grid-template-columns:1fr 1fr;gap:9px}.list{display:flex;flex-direction:column;gap:6px;max-height:430px;overflow:auto}.assoc-row{display:grid;grid-template-columns:minmax(0,1fr) 65px 65px;gap:8px;padding:8px 9px;border:1px solid #17283a;background:#071019;border-radius:9px;font-size:9px;align-items:center}
.assoc-row b{font-size:10px}.assoc-row small{display:block;color:#74899c;margin-top:3px}.assoc-row span{text-align:right;font-variant-numeric:tabular-nums}
.empty{padding:10px;color:#7b8fa1;font-size:10px;line-height:1.5}.footer{text-align:right;color:#607588;font-size:9px;margin-top:12px}

@media(max-width:1450px){.hero{grid-template-columns:repeat(3,1fr)}.process{grid-template-columns:repeat(4,1fr)}.stage:nth-child(4):after{display:none}}
@media(max-width:1080px){main{padding:16px}.top{align-items:flex-start;flex-direction:column}.grid{grid-template-columns:1fr}.formulas{grid-template-columns:1fr 1fr}.process{grid-template-columns:repeat(2,1fr)}.stage:nth-child(even):after{display:none}.mapgrid{grid-template-columns:1fr}}
@media(max-width:650px){main{padding:11px}.hero{grid-template-columns:1fr 1fr}.process{grid-template-columns:1fr}.stage:after{display:none}.formulas,.state-grid{grid-template-columns:1fr}.step-head{grid-template-columns:44px 1fr}.step-meta,.roll{display:none}}
</style>
</head>
<body><main>
<div class="top">
 <div class="brand"><div class="logo">🗣</div><div><h1>Mowa / Language Brain</h1>
 <div class="sub">Live podgląd tego, jak Mucha składa wypowiedź: pamięć słów → kandydaci → score connectomu → losowanie → feedback wybranego słowa z powrotem do sieci.</div></div></div>
 <div class="nav"><a href="/">🏠 Przegląd</a><a href="/autonomy">🧭 Autonomia</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a class="active" href="/associations">🗣 Mowa</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div>
</div>

<div class="notice"><span>ℹ️</span><div><b>To jest trace algorytmu generacji, nie ukryty monolog ani „świadomość”.</b> Pokazuje rzeczywiste dane użyte przez kod: kontekst, wagi Markova, brain score, probabilistyczny wybór i recurrent feedback. Najedź na <span class="help" data-help="trace">?</span>, jeśli chcesz wiedzieć dokładnie, jak czytać tę stronę.</div></div>

<section class="hero">
 <div class="kpi"><small>Generator <span class="help" data-help="generator">?</span></small><strong id="gen">—</strong><em id="gen-sub">czekam na wypowiedź</em></div>
 <div class="kpi"><small>Arousal <span class="help" data-help="arousal">?</span></small><strong id="arousal">—</strong><em>wpływa na backoff i losowość</em></div>
 <div class="kpi"><small>Słownik <span class="help" data-help="vocab">?</span></small><strong id="vocab">—</strong><em id="vocab-sub">—</em></div>
 <div class="kpi"><small>Connectome word control <span class="help" data-help="brain-control">?</span></small><strong id="brain-control">—</strong><em id="brain-control-sub">—</em></div>
 <div class="kpi"><small>Ocenione brain score <span class="help" data-help="brain-score">?</span></small><strong id="brain-eval">—</strong><em id="brain-mean">—</em></div>
 <div class="kpi"><small>Recurrent feedback <span class="help" data-help="feedback">?</span></small><strong id="feedback">—</strong><em id="feedback-sub">—</em></div>
</section>

<section class="process">
 <div class="stage active"><small>1 • KONTEKST <span class="help" data-help="context">?</span></small><strong id="p-context">—</strong><p>Working memory / ostatnia rozmowa.</p></div>
 <div class="stage"><small>2 • PAMIĘĆ JĘZYKA <span class="help" data-help="markov">?</span></small><strong>trigram + bigram + unigram</strong><p>Model wyuczony wyłącznie z rozmów.</p></div>
 <div class="stage blue"><small>3 • MIX WAG <span class="help" data-help="mix">?</span></small><strong id="p-mix">—</strong><p>Historia kontra eksploracja.</p></div>
 <div class="stage violet"><small>4 • CONNECTOME <span class="help" data-help="brain-score">?</span></small><strong id="p-brain">—</strong><p>Każdy top kandydat dostaje brain score.</p></div>
 <div class="stage"><small>5 • LOSOWANIE <span class="help" data-help="random">?</span></small><strong id="p-roll">—</strong><p>Wybór ważony, nie zawsze argmax.</p></div>
 <div class="stage goodstage"><small>6 • FEEDBACK <span class="help" data-help="feedback">?</span></small><strong id="p-feedback">—</strong><p>Wybrane słowo zmienia stan mózgu.</p></div>
 <div class="stage goodstage"><small>7 • WYPOWIEDŹ</small><strong id="p-output">—</strong><p>Kolejne słowo liczy się już z nowego stanu.</p></div>
</section>

<section class="grid">
 <div class="card">
  <div class="head"><h2>Ostatnia wygenerowana wypowiedź <span class="help" data-help="output">?</span></h2><span class="live" id="live">ŁĄCZENIE…</span></div>
  <div class="output" id="output">Czekam, aż Mucha coś wygeneruje…</div>
  <div class="context"><b>Kontekst wejściowy:</b> <span id="context">—</span></div>
 </div>

 <div class="card">
  <div class="head"><h2>Stan przed / wokół generacji <span class="help" data-help="state">?</span></h2></div>
  <div class="state-grid">
   <div class="statebox"><small>Focus attention</small><b id="focus">—</b><div class="bar"><i id="focusbar" style="width:0"></i></div></div>
   <div class="statebox"><small>Dominujący attractor</small><b id="internal">—</b><div class="bar"><i id="internalbar" style="width:0"></i></div></div>
   <div class="statebox"><small>speak readout</small><b id="speak">—</b><div class="bar"><i id="speakbar" style="width:0"></i></div></div>
   <div class="statebox"><small>Reward trace</small><b id="reward">—</b><div class="bar"><i id="rewardbar" style="width:0"></i></div></div>
  </div>
 </div>

 <div class="card full">
  <div class="head"><h2>Dokładny algorytm doboru słowa <span class="help" data-help="formula">?</span></h2></div>
  <div class="formulas">
   <div class="formula"><small>A • Waga z pamięci</small><code>raw = n^exponent × exp(reward) × recent × repeat_penalty</code><p>n = ile razy przejście wystąpiło; reward i świeżość zmieniają jego atrakcyjność.</p></div>
   <div class="formula"><small>B • Interpolacja</small><code>LM = tri·P3 + bi·P2 + uni·P1</code><p>Każde źródło jest najpierw normalizowane osobno. Arousal i długość wypowiedzi przesuwają wagę z trigramów w stronę krótszej pamięci.</p></div>
   <div class="formula"><small>C • Connectome</small><code>brain× = exp(strength × 2 × (brain_score − 0.5))</code><p>0.5 = neutralnie. Powyżej 0.5 connectome podbija kandydata, poniżej osłabia.</p></div>
   <div class="formula"><small>D • Finalna waga</small><code>final = LM × brain×</code><p>Connectome nie wymyśla słowa spoza modelu. Zmienia szanse kandydatów wyuczonych przez model języka.</p></div>
   <div class="formula"><small>E • Losowanie</small><code>P(word) = final / Σ final</code><p>Los 0–1 wpada w przedział jednego kandydata. Dlatego czasem wygrywa słowo inne niż top 1.</p></div>
   <div class="formula"><small>F • Recurrent feedback</small><code>wybrane słowo → output/sensory → brain.step() → następny wybór</code><p>Po wyborze słowo wraca do connectomu i zmienia score kandydatów następnego kroku.</p></div>
  </div>
 </div>

 <div class="card full">
  <div class="head"><h2>Live trace ostatniej generacji <span class="help" data-help="steps">?</span></h2><div id="attempt-tabs" class="attempt-tabs"></div></div>
  <div id="trace-summary" class="trace-summary"></div>
  <div id="steps" class="steps"><div class="empty">Brak trace — poczekaj na pierwszą wypowiedź.</div></div>
 </div>

 <div class="card full">
  <div class="head"><h2>Pamięć skojarzeń słów <span class="help" data-help="associations">?</span></h2><span style="color:var(--muted);font-size:9px">sekcja pomocnicza — nie jest generatorem sama w sobie</span></div>
  <div class="mapgrid">
   <div><div style="font-size:9px;color:#8195a7;margin-bottom:7px;text-transform:uppercase;letter-spacing:.08em">Najsilniejsze aktywne słowa</div><div class="list" id="word-list"></div></div>
   <div><div style="font-size:9px;color:#8195a7;margin-bottom:7px;text-transform:uppercase;letter-spacing:.08em">Najsilniejsze relacje</div><div class="list" id="edge-list"></div></div>
  </div>
 </div>
</section>

<div class="footer">Mucha • live language telemetry • /associations pozostaje URL-em kompatybilności</div>
<div class="tooltip" id="tooltip"></div>
</main>

<script>
const $=id=>document.getElementById(id);
const HELP={
 trace:["Co dokładnie oglądasz?","To zapis danych z algorytmu generacji: nie prywatne rozumowanie. Każdy krok pochodzi z wag modelu, score connectomu i faktycznego losowania.","Najważniejsze są: LM base → brain score → final share → przedział losowania → selected."],
 generator:["Generator","Mucha ma generator hybrydowy. Zwykle wybiera model słów, a czasem fallback znakowy.","WORDS daje pełny trace kandydatów. CHARACTERS pokaże powód fallbacku i liczbę kroków znakowych."],
 arousal:["Arousal","Efektywny poziom pobudzenia używany podczas generacji.","Wyższy arousal spłaszcza rozkład i zwiększa backoff/eksplorację, więc wypowiedź jest mniej zachowawcza."],
 vocab:["Słownik","Liczba unikalnych tokenów słownych wyuczonych online.","Connectome word control włącza się dopiero po osiągnięciu skonfigurowanego minimum słownika."],
 "brain-control":["Connectome word control","Dla top kandydatów model pyta live connectome o language_word_score(token).","Connectome nie tworzy słów od zera. Przeważa kandydatów, które już podał model językowy."],
 "brain-score":["Brain score 0–1","Odczyt bieżącej preferencji connectomu dla konkretnego słowa. Łączy aktywność sensory/output i mniejszy składnik plastic bias.","0.5 jest neutralne; >0.5 zwiększa wagę; <0.5 ją zmniejsza."],
 feedback:["Recurrent feedback","Po wybraniu słowa jego populacja output i sensory zostaje pobudzona, a connectome wykonuje kolejne ticki.","Następne słowo jest więc wybierane z innego stanu mózgu niż poprzednie."],
 context:["Kontekst","Tekst wejściowy podany do generatora. Może pochodzić z wiadomości, STT albo Working Memory przy wypowiedzi spontanicznej.","Model wykorzystuje końcówkę kontekstu jako podpowiedź, ale celowo karze zbyt dokładne kopiowanie promptu."],
 markov:["Pamięć języka","Online word unigram/bigram/trigram w SQLite. Wszystkie przejścia są nauczone z Discorda/STT.","Trigram pamięta dwie poprzednie pozycje, bigram jedną, unigram daje szeroki fallback."],
 mix:["Mix trigram / bigram / unigram","Trzy osobno znormalizowane źródła są interpolowane. Wagi zmieniają się wraz z arousal i długością generacji.","Mniejsza dominacja trigramu = większa możliwość tworzenia nowych kombinacji."],
 random:["Losowanie ważone","Po korekcie connectomu wszystkie finalne wagi są normalizowane do prawdopodobieństw, a kod losuje punkt 0–1.","Tabela pokazuje przedział każdego kandydata i dokładny roll dla danego kroku."],
 output:["Wynik","Finalny tekst wygenerowany w ostatnim wywołaniu generatora.","Jeśli model odrzucił wcześniejszą próbę jako zbyt podobną do kontekstu, zobaczysz kilka attemptów."],
 state:["Stan runtime'u","Bieżący Attention, dominujący internal attractor, speak readout i reward trace.","To kontekst diagnostyczny. Trace słów zapisuje dokładne score kandydatów z chwili generacji."],
 formula:["Wzory","To skrócona wersja dokładnych obliczeń użytych przez generator słów.","W tabeli możesz przejść od bazowej wagi modelu do korekty connectomu i finalnej szansy."],
 steps:["Kroki słowo po słowie","Każda karta odpowiada jednemu wyborowi w modelu słów. Kandydaci są sortowani wg finalnej wagi, ale selected może pochodzić z niższej pozycji przez losowanie.","Kliknij krok, aby rozwinąć pełną tabelę kandydatów."],
 associations:["Pamięć skojarzeń","To pomocniczy widok strukturalnych/uczonych relacji pomiędzy słowami w connectomie.","Nie jest osobną bazą 'myśli'. Pokazuje jak reprezentacje słów są powiązane w bieżącym stanie i plastyczności."]
};
function esc(v){return String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]))}
function num(v,d=3){return Number(v||0).toFixed(d)}
function pct(v){return (Math.max(0,Math.min(1,Number(v||0)))*100).toFixed(1)+"%"}
function nfmt(v){return Number(v||0).toLocaleString("pl-PL")}
function age(ts){if(!ts)return "—";const s=Math.max(0,Date.now()/1000-Number(ts));return s<60?s.toFixed(0)+" s":(s/60).toFixed(1)+" min"}
function bindHelp(){
 document.querySelectorAll("[data-help]").forEach(x=>{if(x.dataset.bound)return;x.dataset.bound="1";
  const show=()=>{const h=HELP[x.dataset.help],t=$("tooltip");if(!h||!t)return;t.innerHTML="<b>"+esc(h[0])+"</b><p>"+esc(h[1])+"</p><em>"+esc(h[2])+"</em>";t.classList.add("show");
   const r=x.getBoundingClientRect(),w=Math.min(380,innerWidth-24);t.style.width=w+"px";t.style.left=Math.max(10,Math.min(innerWidth-w-10,r.left+r.width/2-w/2))+"px";t.style.top=(r.bottom+8)+"px";
   const tr=t.getBoundingClientRect();if(tr.bottom>innerHeight-10)t.style.top=Math.max(10,r.top-tr.height-8)+"px"};
  x.onmouseenter=show;x.onfocus=show;x.onmouseleave=()=>$("tooltip").classList.remove("show");x.onblur=()=>$("tooltip").classList.remove("show");
 })
}
let raw=null,attemptIndex=0,lastTraceStarted=null;
function acceptedAttempt(trace){
 const a=trace.attempts||[];return a.findIndex(x=>x.accepted)>=0?a.findIndex(x=>x.accepted):Math.max(0,a.length-1)
}
function renderHero(d){
 const tr=d.generation_trace||{},ld=d.language_diag||{},attempts=tr.attempts||[];
 const a=attempts[attemptIndex]||attempts[acceptedAttempt(tr)]||{};
 $("gen").textContent=String(tr.generator||ld.last_generator||"—").toUpperCase();
 const roll=tr.word_model_roll;
 $("gen-sub").textContent=roll==null?"brak losu modelu":("word roll "+num(roll,3)+" / p "+num(tr.word_model_probability,2));
 $("arousal").textContent=pct(tr.arousal||0);
 $("vocab").textContent=nfmt(ld.word_vocab||a.vocab||0);
 $("vocab-sub").textContent="minimum connectome "+nfmt(ld.connectome_word_control_min_vocab||a.min_vocab||0);
 const bc=ld.connectome_word_control_last||{};
 $("brain-control").textContent=(a.brain_active??bc.active)?"ACTIVE":"inactive";
 $("brain-control").className=(a.brain_active??bc.active)?"good":"";
 $("brain-control-sub").textContent="strength "+num(tr.brain_control_strength??a.control_strength??0,2)+" • top "+nfmt(tr.brain_candidate_limit??a.candidate_limit??0);
 $("brain-eval").textContent=nfmt(a.brain_scores_evaluated??bc.evaluated??0);
 $("brain-mean").textContent="mean "+num(a.brain_mean_score??bc.mean_score??0.5,3);
 $("feedback").textContent=(a.brain_feedback_active??bc.recurrent_feedback)?"ON":"OFF";
 $("feedback").className=(a.brain_feedback_active??bc.recurrent_feedback)?"good":"";
 const cfg=d.language_config||{};
 $("feedback-sub").textContent=nfmt(a.feedback_words??bc.feedback_words??0)+" słów • "+num(cfg.connectome_word_feedback_magnitude||0,2)+" × "+nfmt(cfg.connectome_word_feedback_steps||0)+" tick";
}
function renderProcess(d){
 const tr=d.generation_trace||{},a=(tr.attempts||[])[attemptIndex]||{},steps=a.steps||[],last=steps[steps.length-1]||{};
 $("p-context").textContent=tr.context?tr.context.slice(-70):"brak kontekstu";
 const mix=last.source_mix||{};
 $("p-mix").textContent=Object.entries(mix).map(([k,v])=>k+" "+pct(v)).join(" • ")||"—";
 $("p-brain").textContent=a.brain_active?("mean "+num(a.brain_mean_score||0.5,3)):"neutral / OFF";
 $("p-roll").textContent=last.selection_roll==null?"—":num(last.selection_roll,4);
 $("p-feedback").textContent=last.feedback_applied?"applied":"—";
 $("p-output").textContent=tr.result?tr.result.slice(0,75):"—";
}
function renderState(d){
 const att=d.attention||{},g=(att.guilds||[]).find(x=>x.focus)||(att.guilds||[])[0]||{},f=g.focus||null;
 $("focus").textContent=f?String(f.label||f.key)+" • "+num(f.score,2):"—";$("focusbar").style.width=pct(f?.score||0);
 const ins=d.internal_states||{},name=ins.dominant||"—",lvl=Number(ins.dominant_level||0);
 $("internal").textContent=name+" • "+pct(lvl);$("internalbar").style.width=pct(lvl);
 const speak=Number((d.scores||{}).speak||0);$("speak").textContent=num(speak,3);$("speakbar").style.width=pct(speak);
 const rt=Number((d.brain_diag||{}).reward_trace||0);$("reward").textContent=(rt>=0?"+":"")+num(rt,3);$("rewardbar").style.width=pct(Math.min(1,Math.abs(rt)));
}
function sourceHTML(c){
 const src=c.sources||{};
 return Object.entries(src).map(([k,v])=>{
  const title="n="+String(v.n??"—")+" reward="+num(v.reward||0,3)+" recent×="+num(v.recent_multiplier||1,2)+" repeat×="+num(v.repeat_penalty||1,2);
  return "<span class='src' title='"+esc(title)+"'>"+esc(k)+" "+pct(v.normalized_contribution||0)+"</span>"
 }).join("")
}
function renderSteps(d){
 const tr=d.generation_trace||{},attempts=tr.attempts||[];
 const tabs=$("attempt-tabs");
 if(!attempts.length){
  tabs.innerHTML="";$("trace-summary").innerHTML="";$("steps").innerHTML="<div class='empty'>Brak trace. Poczekaj, aż Mucha wygeneruje tekst.</div>";return
 }
 if(attemptIndex>=attempts.length)attemptIndex=acceptedAttempt(tr);
 tabs.innerHTML=attempts.map((a,i)=>"<button class='tab "+(i===attemptIndex?"on ":"")+(a.rejected_reason?"reject":"")+"' data-attempt='"+i+"'>PRÓBA "+(i+1)+(a.accepted?" ✓":a.rejected_reason?" ✕":"")+"</button>").join("");
 tabs.querySelectorAll("[data-attempt]").forEach(b=>b.onclick=()=>{attemptIndex=Number(b.dataset.attempt);renderAll(raw)});
 const a=attempts[attemptIndex]||{},steps=a.steps||[];
 $("trace-summary").innerHTML=[
  "<span class='chip'>mode "+esc(a.mode||tr.generator||"—")+"</span>",
  "<span class='chip'>vocab "+nfmt(a.vocab||0)+"</span>",
  "<span class='chip'>brain "+(a.brain_active?"ACTIVE":"OFF")+"</span>",
  "<span class='chip'>evaluated "+nfmt(a.brain_scores_evaluated||0)+"</span>",
  "<span class='chip feedback'>feedback "+nfmt(a.feedback_words||0)+"</span>",
  a.rejected_reason?"<span class='chip' style='border-color:#653f44;color:#ff9ca4'>odrzucona: "+esc(a.rejected_reason)+"</span>":""
 ].join("");
 if(!steps.length){
  $("steps").innerHTML=tr.generator==="characters"
   ?"<div class='empty'><b>Generator znakowy.</b><br>Word model nie został wybrany lub nie wyprodukował użytecznej wypowiedzi. Powód: "+esc(tr.char_fallback_reason||"—")+" • kroki znakowe: "+nfmt(tr.char_steps||0)+".</div>"
   :"<div class='empty'>Ta próba nie doszła do wyboru słów.</div>";
  return
 }
 $("steps").innerHTML=steps.map((s,idx)=>{
  const cand=s.candidates||[],mix=s.source_mix||{},roll=s.selection_roll;
  const rows=cand.map((x,rank)=>{
   const selected=String(x.token)===String(s.selected),bs=x.brain_score;
   const brainCls=bs==null?"":Number(bs)>.53?"brainhigh":Number(bs)<.47?"brainlow":"";
   const interval=x.selection_from==null?"—":num(x.selection_from,3)+"–"+num(x.selection_to,3);
   return "<tr class='"+(selected?"selected-row":"")+"'><td>"+(selected?"▶ ":"")+(rank+1)+". "+esc(x.token)+"</td>"+
    "<td>"+pct(x.base_weight||0)+"</td><td class='"+brainCls+"'>"+(bs==null?"—":num(bs,3))+"</td>"+
    "<td>"+num(x.brain_multiplier||1,3)+"×</td><td>"+pct(x.choice_share||0)+"</td><td class='interval'>"+interval+"</td><td><div class='sources'>"+sourceHTML(x)+"</div></td></tr>"
  }).join("");
  const mixHtml=Object.entries(mix).map(([k,v])=>"<span class='chip "+(k==="trigram"?"tri":k==="bigram"?"bi":k==="unigram"?"uni":"")+"'>"+esc(k)+" "+pct(v)+"</span>").join("");
  return "<div class='step-card "+(idx===0?"open":"")+"' data-step='"+idx+"'><div class='step-head'><div class='step-no'>KROK "+(s.step||idx+1)+"</div>"+
   "<div class='chosen'>"+esc(s.selected||"STOP / END")+"</div><div class='step-meta'>"+esc((s.history||[]).join(" → ")||"start")+"</div>"+
   "<div class='roll'>"+(roll==null?"START":"roll "+num(roll,4))+"</div></div>"+
   "<div class='step-body'><div class='source-line'>"+mixHtml+(s.feedback_applied?"<span class='chip feedback'>↻ feedback do connectomu</span>":"")+"</div>"+
   "<div class='explain'>LM base jest wagą po interpolacji pamięci języka. Brain score zmienia ją mnożnikiem. Final share to rzeczywista szansa w losowaniu. "+(roll==null?"Ten krok pochodzi z wyboru początku zdania.":"Los "+num(roll,4)+" wpada w przedział zaznaczonego słowa.")+"</div>"+
   "<div class='tablewrap'><table><thead><tr><th>Kandydat</th><th>LM base</th><th>Brain</th><th>Brain ×</th><th>Final P</th><th>Przedział losu</th><th>Źródła</th></tr></thead><tbody>"+rows+"</tbody></table></div></div></div>"
 }).join("");
 document.querySelectorAll(".step-card .step-head").forEach(h=>h.onclick=()=>h.parentElement.classList.toggle("open"))
}
function renderAssociations(d){
 const nodes=(d.nodes||[]).slice().sort((a,b)=>Number(b.salience||0)-Number(a.salience||0)).slice(0,18);
 $("word-list").innerHTML=nodes.map(n=>"<div class='assoc-row'><div><b>"+esc(n.id)+"</b><small>count "+nfmt(n.count||0)+" • lang reward "+(Number(n.language_reward||0)>=0?"+":"")+num(n.language_reward||0,2)+" • seen "+age(n.last_seen)+"</small></div><span>brain "+num(n.brain_score||0.5,3)+"</span><span>sal "+num(n.salience||0,3)+"</span></div>").join("")||"<div class='empty'>Brak słów do pokazania.</div>";
 const edges=(d.edges||[]).slice().sort((a,b)=>Number(b.weight||0)-Number(a.weight||0)).slice(0,22);
 $("edge-list").innerHTML=edges.map(e=>"<div class='assoc-row'><div><b>"+esc(e.source)+" → "+esc(e.target)+"</b><small>struct "+num(e.structural||0,3)+" • learned "+(Number(e.learned||0)>=0?"+":"")+num(e.learned||0,3)+" • pair a "+num(e.pair_activation||0,3)+"</small></div><span>weight</span><span>"+num(e.weight||0,3)+"</span></div>").join("")||"<div class='empty'>Brak relacji do pokazania.</div>"
}
function renderAll(d){
 raw=d||{};
 const tr=raw.generation_trace||{};
 if(lastTraceStarted!==tr.started_at){
  lastTraceStarted=tr.started_at;
  attemptIndex=acceptedAttempt(tr);
 }
 if(attemptIndex>=(tr.attempts||[]).length)attemptIndex=acceptedAttempt(tr);
 renderHero(raw);renderProcess(raw);renderState(raw);renderSteps(raw);renderAssociations(raw);
 $("output").textContent=tr.result||"Czekam, aż Mucha coś wygeneruje…";
 $("context").textContent=tr.context||"—";
 $("live").textContent="LIVE";
 bindHelp()
}
async function update(){
 try{
  const r=await fetch("/api/associations",{cache:"no-store"});if(r.status===401){location="/login";return}
  if(!r.ok)throw new Error("HTTP "+r.status);renderAll(await r.json())
 }catch(e){$("live").textContent="ROZŁĄCZONO";console.error(e)}
}
bindHelp();setInterval(update,1200);update();
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
.hero{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:12px}
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
.circuits-card{margin-bottom:12px}.circuits-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}.circuit{background:#071019;border:1px solid #172a3b;border-radius:13px;padding:11px;min-width:0;transition:.18s}.circuit.hot{border-color:rgba(85,234,208,.42);box-shadow:0 0 22px rgba(85,234,208,.06)}.circuit-head{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.circuit-title{font-size:12px;font-weight:850;color:#e8f4ff}.circuit-mode{font-size:8px;font-weight:800;letter-spacing:.07em;text-transform:uppercase;padding:4px 6px;border-radius:999px;border:1px solid #284056;color:#8da5b8;white-space:nowrap}.circuit-mode.bio{color:var(--good);border-color:rgba(86,227,154,.32);background:rgba(86,227,154,.06)}.circuit-mode.adaptive{color:var(--cyan);border-color:rgba(85,234,208,.36);background:rgba(85,234,208,.07)}.circuit-mode.fallback{color:var(--warn);border-color:rgba(255,209,102,.28);background:rgba(255,209,102,.05)}.circuit-score{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin:10px 0 7px}.circuit-score .track{height:7px}.circuit-score b{font-size:11px}.circuit-meta{display:flex;gap:9px;flex-wrap:wrap;color:#7790a5;font-size:9px;margin-bottom:7px}.circuit-types{font-size:10px;line-height:1.5;color:#b6c7d5;min-height:30px}.circuit-seeds{margin-top:7px;padding-top:7px;border-top:1px solid rgba(28,43,59,.7);font:9px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:#71899d;max-height:54px;overflow:auto}.circuit-activity{margin-top:6px;color:#91a8b9;font-size:9px}.circuit-activity strong{color:var(--cyan)}
.circuit{cursor:pointer}.circuit:hover{border-color:#365b78;transform:translateY(-1px)}.circuit.selected{border-color:var(--cyan);box-shadow:0 0 0 1px rgba(85,234,208,.18),0 0 26px rgba(85,234,208,.08)}
.path-card{margin-bottom:12px}.path-meta{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px}.path-pill{padding:5px 8px;border:1px solid #233a4e;border-radius:999px;background:#071019;color:#8fa5b8;font-size:9px}.path-list{display:flex;flex-direction:column;gap:9px}.path-row{border:1px solid #172a3b;background:#071019;border-radius:13px;padding:10px}.path-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:8px}.path-head b{font-size:11px}.path-head span{font-size:9px;color:#8298aa}.path-chain{display:flex;align-items:stretch;gap:5px;overflow-x:auto;padding-bottom:4px}.path-node{min-width:150px;max-width:190px;padding:8px;border:1px solid #20374c;background:#08131d;border-radius:10px}.path-node.sensory{border-color:rgba(85,234,208,.35)}.path-node.output{border-color:rgba(255,120,183,.38)}.path-node.modulatory{border-color:rgba(181,140,255,.35)}.path-node small{display:block;color:#70879b;font-size:8px;text-transform:uppercase;letter-spacing:.07em}.path-node b{display:block;margin:3px 0;font-size:10px;word-break:break-all}.path-node em{display:block;color:#93a8ba;font:9px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;font-style:normal}.path-edge{min-width:132px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;color:#718ca2;font:8px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}.path-edge strong{font-size:18px;color:#547a96;line-height:1}.path-edge .learned{color:var(--cyan);font-weight:800}.path-empty{padding:18px;border:1px dashed #284055;border-radius:12px;color:#758ca0;font-size:10px}
@media(max-width:1120px){.grid{grid-template-columns:1fr}.graph-wrap{height:560px}.hero{grid-template-columns:repeat(3,1fr)}.circuits-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:760px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.hero{grid-template-columns:1fr 1fr}.flow{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}.graph-wrap{height:500px}.circuits-grid{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<div class="top">
 <div class="brand"><div class="logo">🧬</div><div><h1>Neural Connectome</h1><div class="sub">Live funkcjonalny widok aktywnej części mózgu Muchy — nie jest to rekonstrukcja anatomiczna.</div></div></div>
 <div class="nav"><a href="/">🏠 Przegląd</a><a href="/autonomy">🧭 Autonomia</a><a href="/details">📋 Szczegóły</a><a class="active" href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a href="/associations">🗣 Mowa</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div>
</div>

<section class="hero">
 <div class="kpi"><small>Neurony</small><strong id="k-neurons">—</strong><em>pełny connectome</em></div>
 <div class="kpi"><small>Połączenia</small><strong id="k-connections">—</strong><em>synapsy w macierzy</em></div>
 <div class="kpi"><small>Aktywne |a| &gt; .1</small><strong id="k-active">—</strong><em>bieżący tick</em></div>
 <div class="kpi"><small>Mean |activation|</small><strong id="k-mean">—</strong><em id="k-backend">backend —</em></div>
 <div class="kpi"><small>Reward trace</small><strong id="k-reward">—</strong><em id="k-tick">tick —</em></div>
 <div class="kpi"><small>Uczone synapsy</small><strong id="k-synapses">—</strong><em id="k-synapse-delta">max |Δ| —</em></div>
</section>

<section class="flow">
 <div class="flowbox"><small>1 • INPUT</small><strong>Discord / Voice / słowa</strong><p>Bodźce trafiają do stabilnych populacji sensorycznych.</p></div>
 <div class="arrow">→</div>
 <div class="flowbox"><small>2 • PROPAGATION</small><strong>139k neuronów + 3.7M połączeń</strong><p>Aktywność rozchodzi się po prawdziwej topologii FlyWire i miesza z pamięcią/plastycznością.</p></div>
 <div class="arrow">→</div>
 <div class="flowbox"><small>3 • READOUT</small><strong>action circuits + competition</strong><p>Akcje mają własne pule wyjściowe; voice wybiera zwycięski readout, a kod tylko sprawdza fizyczną wykonalność i wykonuje ruch.</p></div>
 <div class="arrow">→</div>
 <div class="flowbox"><small>4 • LANGUAGE / ACTION</small><strong>generator + connectome</strong><p>Po rozbudowie słownika connectome może również zmieniać szanse konkretnych słów.</p></div>
</section>

<section class="card circuits-card">
 <div class="card-head">
  <div><h2>Action Circuits • biology + learned</h2><div class="muted" style="margin-top:4px">Biologiczne seedy albo wyuczony adapter → rzeczywiste połączenia FAFB → readout akcji Discord.</div></div>
  <span class="mode">LIVE CIRCUITS</span>
 </div>
 <div class="circuits-grid" id="biological-circuits"></div>
</section>

<section class="card path-card">
 <div class="card-head">
  <div><h2>Path Inspector</h2><div class="muted" style="margin-top:4px">Kliknij akcję powyżej. Ścieżki są śledzone wstecz po realnych krawędziach FAFB aż do sensory, z bazową wagą i learned Δ.</div></div>
  <span class="mode" id="path-action">VOICE_MOVE</span>
 </div>
 <div class="path-meta" id="path-meta"><span class="path-pill">czekam na ścieżkę…</span></div>
 <div class="path-list" id="path-list"><div class="path-empty">Wybierz akcję, aby zobaczyć jej aktualnie najsilniejsze drogi sensory → output.</div></div>
</section>

<section class="card circuits-card">
 <div class="card-head">
  <div><h2>Neuromodulation v2</h2><div class="muted" style="margin-top:4px">Live poziomy monoamin i ich aktualny wpływ na dynamikę connectomu.</div></div>
  <span class="mode" id="neuromod-status">—</span>
 </div>
 <div class="readout">
  <div class="mini"><small>Dopamina</small><strong id="neuromod-da">—</strong><div class="muted" id="neuromod-da-n">—</div></div>
  <div class="mini"><small>Serotonina</small><strong id="neuromod-ser">—</strong><div class="muted" id="neuromod-ser-n">—</div></div>
  <div class="mini"><small>Octopamina</small><strong id="neuromod-oct">—</strong><div class="muted" id="neuromod-oct-n">—</div></div>
  <div class="mini"><small>Plasticity gain</small><strong id="neuromod-plasticity">—</strong><div class="muted">dopamina → uczenie</div></div>
  <div class="mini"><small>Propagation gain</small><strong id="neuromod-gain">—</strong><div class="muted">octopamina → pobudzenie</div></div>
  <div class="mini"><small>Leak / noise</small><strong id="neuromod-dynamics">—</strong><div class="muted" id="neuromod-residual">—</div></div>
 </div>
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
  <div class="legend"><span class="lg"><i class="sens"></i> sensory</span><span class="lg"><i class="internal"></i> internal</span><span class="lg"><i class="mod"></i> modulatory</span><span class="lg"><i class="out"></i> output</span><span style="color:var(--cyan)">◎ wybrana ścieżka</span><span>• rozmiar = |aktywacja| • domyślnie węzły są trzymane ~45 s • zmiany mają fade-in / fade-out</span></div>
 </div>

 <div class="side">
  <div class="card"><div class="card-head"><h2>Action readouts</h2><span class="muted">0 → 1</span></div><div class="actions" id="actions"></div><div class="muted" id="action-pool-detail" style="margin-top:10px">—</div></div>
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
    <div class="mini"><small>połączone outputy</small><strong id="connected-outputs">—</strong></div>
    <div class="mini"><small>izolowane w widoku</small><strong id="isolated-nodes">—</strong></div>
    <div class="mini"><small>okno stabilne</small><strong id="stable-age">—</strong></div>
    <div class="mini"><small>ostatnia podmiana</small><strong id="replacements">—</strong></div>
    <div class="mini"><small>voice policy</small><strong id="voice-policy">—</strong></div>
    <div class="mini"><small>voice winner / margin</small><strong id="voice-winner">—</strong></div>
    <div class="mini"><small>social drive</small><strong id="voice-social-drive">—</strong></div>
    <div class="mini"><small>reward opportunity</small><strong id="voice-reward-opportunity">—</strong></div>
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
let canvasW=1,canvasH=1,canvasDpr=1,lastFrameTs=0,updating=false,maxGraphImportance=.0001;
const CONNECTOME_FRAME_MS=1000/30;
let followActivity=localStorage.getItem("mucha-connectome-follow")==="1";
let selectedAction=localStorage.getItem("mucha-connectome-action")||"voice_move";
let actionPathSnap=null,pathLastRequest=0;
const graphNodes=new Map(),graphEdges=new Map();
const colors={sensory:"#55ead0",internal:"#6da8ff",modulatory:"#b58cff",output:"#ff78b7"};
const nfmt=n=>Number(n||0).toLocaleString("pl-PL");
const esc=v=>String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function hash(s){let h=2166136261;for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)}return (h>>>0)/4294967295}
function resize(){const r=wrap.getBoundingClientRect(),dpr=Math.min(1.5,window.devicePixelRatio||1);canvasW=Math.max(1,r.width);canvasH=Math.max(1,r.height);canvasDpr=dpr;canvas.width=Math.max(1,Math.floor(canvasW*dpr));canvas.height=Math.max(1,Math.floor(canvasH*dpr));canvas.style.width=canvasW+"px";canvas.style.height=canvasH+"px";ctx.setTransform(dpr,0,0,dpr,0,0);updateLayout([...graphNodes.values()].map(x=>x.node))}
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
function visualWithSelectedPath(v,p){
 const nodes=(v.nodes||[]).map(n=>({...n})),edges=(v.edges||[]).map(e=>({...e}));
 const nodeMap=new Map(nodes.map(n=>[String(n.id),n]));
 const edgeMap=new Map(edges.map(e=>[String(e.source)+">"+String(e.target),e]));
 for(const path of ((p&&p.paths)||[]).slice(0,2)){
  for(const n of (path.nodes||[])){
   const id=String(n.id);
   const existing=nodeMap.get(id);
   if(existing){existing.path=true;continue}
   const added={id,role:n.role||"internal",activation:Number(n.activation||0),eligibility:Number(n.eligibility||0),bias:Number(n.bias||0),path:true};
   nodes.push(added);nodeMap.set(id,added);
  }
  for(const e of (path.edges||[])){
   const key=String(e.source)+">"+String(e.target);
   const existing=edgeMap.get(key);
   if(existing){existing.path=true;existing.learned_delta=Number(e.learned_delta||0);continue}
   const added={source:String(e.source),target:String(e.target),weight:Number(e.effective_weight||e.base_weight||0),importance:Math.max(.001,Number(e.importance||0)),learned_delta:Number(e.learned_delta||0),path:true};
   edges.push(added);edgeMap.set(key,added);
  }
 }
 return {...v,nodes,edges};
}
function ingestGraph(v){
 const nowNodes=new Set((v.nodes||[]).map(n=>n.id));
 for(const entry of graphNodes.values())if(!nowNodes.has(entry.node.id))entry.targetAlpha=0;
 for(const n of (v.nodes||[])){
  const old=graphNodes.get(n.id);
  if(old){old.node=n;old.targetAlpha=1}else graphNodes.set(n.id,{node:n,alpha:0,targetAlpha:1});
 }
 const nowEdges=new Set();
 maxGraphImportance=.0001;
 for(const e of (v.edges||[])){
  const key=e.source+">"+e.target;nowEdges.add(key);
  maxGraphImportance=Math.max(maxGraphImportance,Number(e.importance||0));
  const old=graphEdges.get(key);
  if(old){old.edge=e;old.targetAlpha=1}else graphEdges.set(key,{edge:e,alpha:0,targetAlpha:1});
 }
 for(const [key,entry] of graphEdges)if(!nowEdges.has(key))entry.targetAlpha=0;
 updateLayout(v.nodes||[]);
}
function drawGrid(w,h){
 ctx.save();ctx.strokeStyle="rgba(73,108,137,.07)";ctx.lineWidth=1;
 for(let x=24;x<w;x+=48){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke()}
 for(let y=24;y<h;y+=48){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke()}
 ctx.restore();
}
function draw(ts=0){
 requestAnimationFrame(draw);
 if(document.hidden||ts-lastFrameTs<CONNECTOME_FRAME_MS)return;
 lastFrameTs=ts;
 const w=canvasW,h=canvasH;ctx.clearRect(0,0,w,h);drawGrid(w,h);
 const nodeEntries=[...graphNodes.values()];
 for(const e of nodeEntries){e.alpha+=(e.targetAlpha-e.alpha)*.11}
 for(const [id,e] of graphNodes)if(e.targetAlpha===0&&e.alpha<.018){graphNodes.delete(id);delete layout[id]}
 for(const e of graphEdges.values()){e.alpha+=(e.targetAlpha-e.alpha)*.13}
 for(const [id,e] of graphEdges)if(e.targetAlpha===0&&e.alpha<.018)graphEdges.delete(id);
 for(const p of Object.values(layout)){p.x+=(p.tx-p.x)*.075;p.y+=(p.ty-p.y)*.075}
 const t=ts/1000;
 for(const item of graphEdges.values()){
  const e=item.edge,a=layout[e.source],b=layout[e.target];if(!a||!b)continue;
  const srcEntry=graphNodes.get(e.source),dstEntry=graphNodes.get(e.target);if(!srcEntry||!dstEntry)continue;
  const alpha=item.alpha*Math.min(srcEntry.alpha,dstEntry.alpha),q=clamp(Number(e.importance||0)/maxGraphImportance,0,1);
  const target=dstEntry.node,source=srcEntry.node,pathEdge=!!e.path;
  const rgb=pathEdge?"85,234,208":(target.role==="output"?"255,120,183":(Number(e.weight||0)<0?"181,140,255":"90,151,197"));
  ctx.strokeStyle="rgba("+rgb+","+(alpha*(pathEdge?.70:(0.13+q*.52)))+")";ctx.lineWidth=(pathEdge?2.0:.65)+q*(pathEdge?2.2:1.65);
  ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();

  if((pathEdge||q>.18)&&alpha>.10){
   const ang=Math.atan2(b.y-a.y,b.x-a.x),back=8+q*4,wing=3+q*2;
   const ax=b.x-Math.cos(ang)*back,ay=b.y-Math.sin(ang)*back;
   ctx.fillStyle="rgba("+rgb+","+(alpha*(.32+q*.50))+")";
   ctx.beginPath();ctx.moveTo(ax,ay);
   ctx.lineTo(ax-Math.cos(ang-.65)*wing,ay-Math.sin(ang-.65)*wing);
   ctx.lineTo(ax-Math.cos(ang+.65)*wing,ay-Math.sin(ang+.65)*wing);
   ctx.closePath();ctx.fill();
  }

  const act=Math.abs(Number(source.activation||0));
  if((pathEdge||act>.08)&&q>.11&&alpha>.08){
   const phase=(t*(.20+.55*q)+hash(e.source+e.target))%1;
   const x=a.x+(b.x-a.x)*phase,y=a.y+(b.y-a.y)*phase;
   ctx.fillStyle=colors[source.role]||colors.internal;ctx.globalAlpha=alpha*(.42+.48*q);ctx.beginPath();ctx.arc(x,y,1.4+q*1.8,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;
  }
 }
 hover=null;
 for(const item of graphNodes.values()){
  const n=item.node,p=layout[n.id];if(!p)continue;const act=Math.abs(Number(n.activation||0)),rad=3.1+Math.min(8,act*10),c=colors[n.role]||colors.internal;
  const glow=n.path||act>.18;
  ctx.shadowColor=c;ctx.shadowBlur=glow?(4+Math.min(10,act*12))*item.alpha:0;ctx.fillStyle=c;ctx.globalAlpha=item.alpha*(.38+Math.min(.62,act*.75));
  ctx.beginPath();ctx.arc(p.x,p.y,rad,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;ctx.shadowBlur=0;
  if(n.path&&item.alpha>.2){ctx.strokeStyle="#55ead0";ctx.lineWidth=1.8;ctx.globalAlpha=item.alpha*.9;ctx.beginPath();ctx.arc(p.x,p.y,rad+4,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1}
  if(mouse.inside&&item.alpha>.45){const dx=mouse.x-p.x,dy=mouse.y-p.y;if(dx*dx+dy*dy<(rad+8)*(rad+8))hover=n}
 }
 if(hover){const p=layout[hover.id];ctx.strokeStyle="#ffffff";ctx.lineWidth=1;ctx.globalAlpha=.8;ctx.beginPath();ctx.arc(p.x,p.y,12+Math.abs(Number(hover.activation||0))*8,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1}
}
const mouse={x:0,y:0,inside:false};
canvas.addEventListener("mousemove",e=>{const r=canvas.getBoundingClientRect();mouse.x=e.clientX-r.left;mouse.y=e.clientY-r.top;mouse.inside=true;if(hover){tip.style.display="block";tip.style.left=Math.min(r.width-195,mouse.x+14)+"px";tip.style.top=Math.min(r.height-120,mouse.y+14)+"px";tip.innerHTML="<b>"+hover.id+"</b><br><span>"+hover.role+"</span><br>activation "+Number(hover.activation||0).toFixed(5)+"<br>eligibility "+Number(hover.eligibility||0).toFixed(5)+"<br>bias "+Number(hover.bias||0).toFixed(5)}else tip.style.display="none"});
canvas.addEventListener("mouseleave",()=>{mouse.inside=false;tip.style.display="none"});
function renderActions(scores){
 const order=["speak","explore","react","voice_join","voice_move","voice_leave","stay"];
 $("actions").innerHTML=order.map(k=>{const v=clamp(Number(scores[k]||0),0,1);return '<div class="act" style="cursor:pointer" onclick="selectAction(\''+k+'\')"><label>'+k+'</label><div class="track"><div class="fill" style="width:'+(v*100).toFixed(1)+'%"></div></div><b>'+v.toFixed(2)+'</b></div>'}).join("");
}
function selectAction(action){
 selectedAction=String(action||"voice_move");
 localStorage.setItem("mucha-connectome-action",selectedAction);
 pathLastRequest=0;
 $("path-action").textContent=selectedAction.toUpperCase();
 update();
}
function renderBiologicalCircuits(pools,scores){
 const order=["speak","react","voice_join","voice_move","voice_leave","explore","stay"];
 const html=order.map(action=>{
  const p=(pools||{})[action]||{},score=clamp(Number((scores||{})[action]||0),0,1);
  const mode=String(p.mode||""),bio=mode.startsWith("annotated"),adaptive=mode.startsWith("adaptive");
  const types=(p.seed_types||[]).map(x=>esc(x.name)+" ×"+nfmt(x.count)).join(" • ")||(bio?"typ nieopisany":(adaptive?"wyuczony obwód sensoryczny":"brak biologicznych seedów"));
  const seeds=(p.top_seed_activity||[]).slice(0,4).map(x=>{
   const sign=Number(x.activation||0)>=0?"+":"";
   return esc(x.type)+" #"+esc(x.root_id)+" "+sign+Number(x.activation||0).toFixed(3)+(adaptive?" [cue]":(x.in_output_pool?"":" [spoza output]"));
  }).join("<br>");
  const searched=(p.seed_terms||[]).slice(0,7).map(esc).join(", ");
  const external=Number(p.external_seed_count||0);
  const mean=Number(p.mean_abs_activation||0),max=Number(p.max_abs_activation||0);
  return '<div class="circuit '+(score>.58?"hot ":"")+(selectedAction===action?"selected":"")+'" onclick="selectAction(\''+action+'\')">'+
   '<div class="circuit-head"><div class="circuit-title">'+esc(action)+'</div><span class="circuit-mode '+(adaptive?"adaptive":(bio?"bio":"fallback"))+'">'+esc(p.mode||"—")+'</span></div>'+
   '<div class="circuit-score"><div class="track"><div class="fill" style="width:'+(score*100).toFixed(1)+'%"></div></div><b>'+score.toFixed(3)+'</b></div>'+
   '<div class="circuit-meta"><span>'+(adaptive?'sensory cues':'seeds')+' <b>'+nfmt(p.seed_count||0)+'</b></span><span>pool <b>'+nfmt(p.pool_size||0)+'</b></span>'+(!adaptive&&external?'<span>spoza output <b>'+nfmt(external)+'</b></span>':'')+'</div>'+
   '<div class="circuit-types">'+types+'</div>'+
   '<div class="circuit-activity">mean |a| <strong>'+mean.toFixed(4)+'</strong> • max |a| <strong>'+max.toFixed(4)+'</strong></div>'+
   '<div class="circuit-seeds">'+(seeds||(bio?"brak aktywności seedów":"szukano: "+searched))+'</div>'+
  '</div>';
 }).join("");
 $("biological-circuits").innerHTML=html;
}
function renderActionPath(p){
 if(!p){$("path-action").textContent=selectedAction.toUpperCase();return}
 $("path-action").textContent=String(p.action||selectedAction).toUpperCase();
 if(p.error){
  $("path-meta").innerHTML='<span class="path-pill">'+esc(p.error)+'</span>';
  $("path-list").innerHTML='<div class="path-empty">Brak ścieżki do pokazania.</div>';
  return;
 }
 $("path-meta").innerHTML=
  '<span class="path-pill">score <b>'+Number(p.score||0).toFixed(3)+'</b></span>'+
  '<span class="path-pill">'+esc(p.mode||"—")+'</span>'+
  '<span class="path-pill">pool '+nfmt(p.pool_size||0)+'</span>'+
  '<span class="path-pill">pełne sensory paths '+nfmt(p.complete_paths||0)+' / '+nfmt((p.paths||[]).length)+'</span>'+
  '<span class="path-pill">depth ≤ '+nfmt(p.max_depth||0)+'</span>';
 const paths=p.paths||[];
 $("path-list").innerHTML=paths.map(path=>{
  const nodes=path.nodes||[],edges=path.edges||[];
  let chain='';
  nodes.forEach((node,i)=>{
   const label=node.type||node.neuropil||node.role||"neuron";
   chain+='<div class="path-node '+esc(node.role||"internal")+'"><small>'+esc(node.role||"internal")+'</small><b>#'+esc(node.id)+'</b><em>'+esc(label)+'</em><em>a '+Number(node.activation||0).toFixed(4)+' • bias '+Number(node.bias||0).toFixed(5)+' • elig '+Number(node.eligibility||0).toFixed(3)+'</em></div>';
   if(i<edges.length){
    const e=edges[i]||{},delta=Number(e.learned_delta||0),effective=Number(e.effective_weight||0);
    chain+='<div class="path-edge"><strong>→</strong><span>base '+Number(e.base_weight||0).toFixed(4)+'</span><span class="'+(Math.abs(delta)>1e-9?"learned":"")+'">Δ '+(delta>=0?"+":"")+delta.toFixed(5)+'</span><span>eff '+effective.toFixed(4)+'</span></div>';
   }
  });
  return '<div class="path-row"><div class="path-head"><b>PATH '+nfmt(path.rank||0)+' '+(path.complete?"✓ sensory→output":"• partial")+'</b><span>strength '+Number(path.strength||0).toFixed(3)+'</span></div><div class="path-chain">'+chain+'</div></div>';
 }).join("")||'<div class="path-empty">Nie znaleziono ścieżki w aktualnym limicie głębokości. To nie znaczy, że połączenia nie istnieją — mogą być dłuższe lub aktualnie słabe.</div>';
}
function updateModeButton(){
 const b=$("follow-btn");b.textContent="FOLLOW ACTIVITY: "+(followActivity?"ON":"OFF");b.className="follow "+(followActivity?"on":"");
 $("mode-label").textContent=followActivity?"DYNAMIC TOP ACTIVITY":"STABLE WINDOW";
}
$("follow-btn").onclick=()=>{followActivity=!followActivity;localStorage.setItem("mucha-connectome-follow",followActivity?"1":"0");updateModeButton();update()};
function render(s,v){
 stateSnap=s;
 if(v.action_path){actionPathSnap=v.action_path;pathLastRequest=Date.now()}
 const graphView=visualWithSelectedPath(v,actionPathSnap);
 visualSnap=graphView;ingestGraph(graphView);
 renderActionPath(actionPathSnap);
 const d=s.diag||{},ld=s.language_diag||{},cw=ld.connectome_word_control_last||{};
 $("k-neurons").textContent=nfmt(d.neurons);$("k-connections").textContent=nfmt(d.connections);$("k-active").textContent=nfmt(d.active_abs_gt_0_1);$("k-mean").textContent=Number(d.mean_abs||0).toFixed(5);$("k-reward").textContent=Number(d.reward_trace||0).toFixed(3);$("k-backend").textContent=(d.backend||"—")+" • "+(d.device||"");$("k-tick").textContent="tick "+nfmt(d.ticks);$("k-synapses").textContent=nfmt(d.learned_synapses||0);$("k-synapse-delta").textContent="max |Δ| "+Number(d.synaptic_max_abs||0).toFixed(5);
 $("selected").textContent=nfmt(v.selected_neurons);$("edges").textContent=nfmt(v.selected_edges);$("connected-outputs").textContent=nfmt(v.connected_output_neurons||0);$("isolated-nodes").textContent=nfmt(v.isolated_neurons||0);$("event").textContent=s.last_event||"—";$("last-action").textContent=s.last_action||"—";renderActions(s.scores||{});
 const pools=d.action_pools||{};renderBiologicalCircuits(pools,s.scores||{});const poolOrder=["speak","react","voice_join","voice_move","voice_leave","explore","stay"];$("action-pool-detail").textContent=poolOrder.map(k=>{const p=pools[k]||{};return k+": "+(p.mode||"—")+" • seeds "+nfmt(p.seed_count||0)+" • pool "+nfmt(p.pool_size||0)}).join("  |  ");
 const nm=d.neuromodulation||{},da=nm.dopamine||{},ser=nm.serotonin||{},oct=nm.octopamine||{};$("neuromod-status").textContent=nm.enabled?"ACTIVE":"OFF";$("neuromod-da").textContent=Number(da.level||0).toFixed(4);$("neuromod-da-n").textContent=nfmt(da.neurons||0)+" neuronów";$("neuromod-ser").textContent=Number(ser.level||0).toFixed(4);$("neuromod-ser-n").textContent=nfmt(ser.neurons||0)+" neuronów";$("neuromod-oct").textContent=Number(oct.level||0).toFixed(4);$("neuromod-oct-n").textContent=nfmt(oct.neurons||0)+" neuronów";$("neuromod-plasticity").textContent="×"+Number(nm.plasticity_gain||1).toFixed(3);$("neuromod-gain").textContent=Number(nm.effective_gain||0).toFixed(4);$("neuromod-dynamics").textContent="leak "+Number(nm.effective_leak||0).toFixed(4)+" • noise "+Number(nm.effective_noise||0).toFixed(4);$("neuromod-residual").textContent="direct residual "+Number(nm.direct_residual||0).toFixed(2);
 $("stable-age").textContent=followActivity?"FOLLOW":Math.round(Number(v.stable_age_seconds||0))+" s / "+Math.round(Number(v.stable_window_seconds||45))+" s";
 $("replacements").textContent=nfmt(v.replacements||0);
 const voiceDbg=(s.voice_debug||[])[0]||{},voiceBd=voiceDbg.brain_decision||{};$("voice-policy").textContent=voiceDbg.connectome_voice_control?"CONNECTOME":"LEGACY";$("voice-winner").textContent=voiceDbg.connectome_voice_control?((voiceBd.action||"—")+" / "+(voiceBd.margin==null?"—":Number(voiceBd.margin).toFixed(3))):"progi";$("voice-social-drive").textContent=(Number(voiceDbg.social_drive_level||0)*100).toFixed(1)+"% • "+nfmt(voiceDbg.available_humans||0)+" ludzi";$("voice-reward-opportunity").textContent=voiceDbg.reward_opportunity_channel?voiceDbg.reward_opportunity_channel+" • eff "+Number(voiceDbg.reward_opportunity_effective_strength||voiceDbg.reward_opportunity_strength||0).toFixed(2)+" • reach "+Number(voiceDbg.reward_opportunity_guided_reach_max||0).toFixed(3)+" • "+Number(voiceDbg.reward_opportunity_remaining||0).toFixed(0)+"s":"—";
 const vocab=Number(ld.word_vocab||0),min=Number(ld.connectome_word_control_min_vocab||1),ready=!!ld.connectome_word_control_ready,pct=clamp(vocab/min*100,0,100);
 $("word-vocab").textContent=nfmt(vocab)+" / "+nfmt(min);$("word-progress").style.width=pct.toFixed(1)+"%";$("word-badge").textContent=ready?"AKTYWNY":"UCZY SŁOWNIK";$("word-badge").className="badge "+(ready?"on":"wait");
 $("word-eval").textContent=nfmt(cw.evaluated||0);$("word-detail").textContent="Siła wpływu: "+Number(ld.connectome_word_control_strength||0).toFixed(2)+" • średni ostatni score: "+Number(cw.mean_score||.5).toFixed(3)+" • feedback słów: "+nfmt(cw.feedback_words||0)+" • generator: "+(ld.last_generator||"—");
 const nodes=(v.nodes||[]).slice().sort((a,b)=>Math.abs(Number(b.activation))-Math.abs(Number(a.activation))).slice(0,12);
 $("node-table").innerHTML=nodes.map(n=>'<tr><td>'+n.id+'</td><td>'+n.role+'</td><td class="'+(Number(n.activation)>=0?"plus":"minus")+'">'+(Number(n.activation)>=0?"+":"")+Number(n.activation).toFixed(4)+'</td><td>'+Number(n.bias||0).toFixed(5)+'</td></tr>').join("")||'<tr><td colspan="4">Brak danych.</td></tr>';
 $("signal").textContent="MODE  "+(followActivity?"FOLLOW ACTIVITY":"STABLE WINDOW")+"\nINPUT  "+(s.last_event||"—")+"\nCONNECTOME  mean |a| "+Number(d.mean_abs||0).toFixed(5)+" / max "+Number(d.max_abs||0).toFixed(5)+"\nVISIBLE GRAPH  "+nfmt(v.selected_edges||0)+" direct edges • "+nfmt(v.connected_output_neurons||0)+" connected outputs • "+nfmt(v.isolated_neurons||0)+" isolated\nREADOUT  speak "+Number((s.scores||{}).speak||0).toFixed(3)+" / explore "+Number((s.scores||{}).explore||0).toFixed(3)+"\nOUTPUT  "+(s.last_action||"—");
 lastUpdate=Date.now();$("live").textContent="LIVE";
}
async function update(){
 if(updating||document.hidden)return;
 updating=true;
 try{
  const wantPath=Date.now()-pathLastRequest>8000;
  const pathQuery=wantPath?"&action="+encodeURIComponent(selectedAction):"";
  const [stateResp,visualResp]=await Promise.all([
   fetch("/api/state",{cache:"no-store"}),
   fetch("/api/connectome?follow="+(followActivity?"1":"0")+pathQuery,{cache:"no-store"})
  ]);
  if(stateResp.status===401||visualResp.status===401){location="/login";return}
  if(!stateResp.ok)throw new Error("state HTTP "+stateResp.status);
  if(!visualResp.ok)throw new Error("connectome HTTP "+visualResp.status);
  render(await stateResp.json(),await visualResp.json())
 }catch(e){$("live").textContent="ROZŁĄCZONO";console.error(e)}
 finally{updating=false}
}
document.addEventListener("visibilitychange",()=>{if(!document.hidden){lastFrameTs=0;update()}});
window.addEventListener("resize",resize);$("path-action").textContent=selectedAction.toUpperCase();updateModeButton();resize();draw();setInterval(update,1500);update();
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
.hero{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:12px}.kpi,.card{background:linear-gradient(180deg,rgba(11,18,27,.96),rgba(7,13,20,.96));border:1px solid var(--line);border-radius:16px;box-shadow:0 16px 50px rgba(0,0,0,.18)}
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
.signal-list{display:flex;flex-direction:column;gap:6px;max-height:240px;overflow:auto}.signal-row{display:grid;grid-template-columns:54px 1fr 62px;gap:7px;align-items:center;padding:7px;border:1px solid #172637;background:#071019;border-radius:9px;font-size:9px}.signal-row b{font-size:9px}.signal-row span{color:#8298aa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.signal-row em{text-align:right;font-style:normal;font-variant-numeric:tabular-nums}.signal-row.pos em{color:var(--cyan)}.signal-row.neg em{color:var(--pink)}.signal-summary{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-bottom:8px}.signal-summary div{padding:8px;border:1px solid #172637;background:#071019;border-radius:9px}.signal-summary small{display:block;color:#6f879b;font-size:8px;text-transform:uppercase;letter-spacing:.08em}.signal-summary b{display:block;margin-top:4px;font-size:11px}.flow-cue{display:inline-flex;gap:4px;align-items:center;margin:3px 4px 0 0;padding:4px 6px;border:1px solid #244057;border-radius:999px;background:#07131d;color:#9db2c4;font-size:8px}.learn-row{padding:7px;border:1px solid rgba(255,119,183,.18);background:rgba(255,119,183,.04);border-radius:9px;margin-top:6px;font-size:9px;color:#9fb1c1}.flow-note{color:#71879a;font-size:9px;line-height:1.5;margin-top:7px}.flow-history{cursor:pointer}.flow-history:hover,.flow-history.on{border-color:#3f6985;background:#0a1924}.flow-head-actions{display:flex;gap:6px;align-items:center}.flow-head-actions .btn{padding:5px 8px;font-size:8px}.regions{display:flex;flex-direction:column;gap:6px;max-height:310px;overflow:auto}.region{display:grid;grid-template-columns:1fr 72px 44px;gap:8px;align-items:center;padding:8px;border:1px solid #172637;background:#071019;border-radius:10px;cursor:pointer}.region:hover,.region.on{border-color:#34536c;background:#091722}.region b{font-size:10px}.region small{color:#70879b;font-size:9px}.rtrack{height:6px;background:#050b11;border:1px solid #162536;border-radius:999px;overflow:hidden}.rfill{height:100%;background:linear-gradient(90deg,var(--violet),var(--pink));border-radius:999px}
.inspector{min-height:230px}.empty{color:#71879a;font-size:11px;line-height:1.55;padding:10px 0}.ins-title{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.ins-title strong{font-size:14px;word-break:break-all}.role{padding:4px 7px;border-radius:999px;font-size:8px;font-weight:850;letter-spacing:.08em;border:1px solid #274056;color:#a8bac9}
.meta{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:10px}.meta div{padding:8px;background:#071019;border:1px solid #172637;border-radius:9px}.meta small{display:block;color:#6f879b;font-size:8px;text-transform:uppercase;letter-spacing:.09em;margin-bottom:3px}.meta b{font-size:10px;word-break:break-word}
.effects{margin-top:9px;padding:9px;background:#061019;border:1px solid #172637;border-radius:10px}.effects small{display:block;color:#71899e;font-size:8px;text-transform:uppercase;letter-spacing:.1em;margin-bottom:7px}.effect{display:grid;grid-template-columns:80px 1fr auto;gap:7px;align-items:center;font-size:9px;margin-top:5px}.effect .efill{height:6px;background:#08141f;border-radius:999px;overflow:hidden}.effect .efill i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--cyan))}
.note{color:#71879a;font-size:9px;line-height:1.5;margin-top:8px}.legend{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;color:#748b9f;font-size:9px}.legend span{display:inline-flex;align-items:center;gap:5px}.legend i{width:8px;height:8px;border-radius:50%}.sens{background:var(--cyan)}.internal{background:var(--blue)}.mod{background:var(--violet)}.out{background:var(--pink)}
.region-inspector{min-height:300px}.spark-wrap{height:76px;border:1px solid #172637;background:#050c12;border-radius:10px;margin-top:9px;padding:5px}.spark-wrap canvas{width:100%;height:100%}.chips{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}.chip{font-size:8px;padding:4px 6px;border-radius:999px;border:1px solid #263c50;background:#071019;color:#9db0c1}.corr{display:grid;grid-template-columns:82px 1fr 42px;gap:7px;align-items:center;margin-top:6px;font-size:9px}.corrbar{height:6px;background:#07131d;border-radius:999px;overflow:hidden;position:relative}.corrbar:after{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:#28445b}.corrfill{height:100%;position:absolute;top:0}.corrfill.pos{left:50%;background:var(--cyan)}.corrfill.neg{right:50%;background:var(--pink)}.neuron-list{display:flex;flex-direction:column;gap:5px;margin-top:7px}.nrow{display:grid;grid-template-columns:1fr 62px;gap:8px;font-size:9px;padding:6px 7px;border:1px solid #172637;background:#071019;border-radius:8px}.nrow span{color:#9fb1c1}.nrow b{text-align:right}.nrow.clickable{cursor:pointer}.nrow.clickable:hover{border-color:#3a617d;background:#091925}
.why-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.why-col{min-width:0}.why-head{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:6px}.why-head b{font-size:10px}.why-row{display:grid;grid-template-columns:1fr 72px;gap:8px;align-items:center;padding:7px 8px;border:1px solid #172637;background:#071019;border-radius:9px;font-size:9px;margin-top:5px;cursor:pointer}.why-row:hover{border-color:#3a617d}.why-row span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#9fb1c1}.why-row strong{text-align:right;font-variant-numeric:tabular-nums}.why-row.pos strong{color:var(--cyan)}.why-row.neg strong{color:var(--pink)}.why-summary{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-bottom:8px}.why-summary div{padding:8px;border:1px solid #172637;background:#071019;border-radius:9px}.why-summary small{display:block;color:#6f879b;font-size:8px;text-transform:uppercase}.why-summary b{display:block;margin-top:4px;font-size:10px}
.conn-row{display:grid;grid-template-columns:38px 1fr 68px 68px;gap:7px;align-items:center;padding:7px;border:1px solid #172637;background:#071019;border-radius:9px;font-size:9px;margin-top:5px}.conn-row b{font-size:8px}.conn-row span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#9fb1c1}.conn-row em{font-style:normal;text-align:right}.conn-row .drive.pos{color:var(--cyan)}.conn-row .drive.neg{color:var(--pink)}
.replay-controls{display:flex;gap:5px;align-items:center}.replay-controls .btn{padding:5px 8px;font-size:8px}
@media(max-width:1150px){.grid{grid-template-columns:1fr}.map-wrap{height:650px}.hero{grid-template-columns:repeat(3,1fr)}}
@media(max-width:720px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.hero{grid-template-columns:1fr 1fr}.map-wrap{height:540px}.meta{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<div class="top">
 <div class="brand"><div class="logo">🧠</div><div><h1>Fly Brain Neuro-map • Stage 34</h1><div class="sub">Neuro-map 2.0: klikalne neurony i regiony, frame-by-frame replay, realny przepływ po krawędziach FAFB oraz signed attribution bieżącej decyzji.</div></div></div>
 <div class="nav"><a href="/">🏠 Przegląd</a><a href="/autonomy">🧭 Autonomia</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a class="active" href="/neuromap">🧠 Neuro-map</a><a href="/associations">🗣 Mowa</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div>
</div>

<section class="hero">
 <div class="kpi"><small>Pozycje neuronów</small><strong id="coverage">—</strong><em id="coord-count">—</em></div>
 <div class="kpi"><small>Mapa neuropili</small><strong id="neuropil-coverage">—</strong><em id="neuropil-count">—</em></div>
 <div class="kpi"><small>Aktywne |a| &gt; .1</small><strong id="active">—</strong><em>cały runtime</em></div>
 <div class="kpi"><small>Najaktywniejszy region</small><strong id="top-region">—</strong><em id="top-region-detail">—</em></div>
 <div class="kpi"><small>Dominujący readout</small><strong id="dominant">—</strong><em id="dominant-score">—</em></div>
 <div class="kpi"><small>Live signal flow</small><strong id="flow-edge-count">—</strong><em id="flow-frame">brak klatki</em></div>
</section>

<div class="toolbar">
 <div class="toolgroup"><span>Projekcja</span><button class="btn proj on" data-proj="xy">XY</button><button class="btn proj" data-proj="xz">XZ</button><button class="btn proj" data-proj="yz">YZ</button></div>
 <div class="toolgroup"><span>Warstwa</span><button class="btn role on" data-role="all">ALL</button><button class="btn role" data-role="sensory">SENSORY</button><button class="btn role" data-role="internal">INTERNAL</button><button class="btn role" data-role="modulatory">MODULATORY</button><button class="btn role" data-role="output">OUTPUT</button></div>
 <div class="toolgroup"><span>Widok</span><button class="btn on" id="regions-btn">REGION HEAT</button><button class="btn on" id="trail-btn">ACTIVITY TRAIL</button><button class="btn on" id="flow-btn">SIGNAL FLOW</button><button class="btn on" id="learned-btn">LEARNED SYNAPSES</button><button class="btn on" id="attractor-btn">ATTRACTORS</button><button class="btn on" id="follow-btn">FOLLOW DECISION</button><span class="badge" id="region-source">REGIONS</span><span class="badge" id="coord-badge">COORDINATES</span></div>
</div>

<section class="grid">
 <div class="card">
  <div class="card-head"><h2>Brain projection / live activity</h2><div class="live"><i></i><span id="live">LIVE</span></div></div>
  <div class="map-wrap" id="map-wrap">
   <canvas id="brain"></canvas>
   <div class="map-title" id="map-title">XY PROJECTION</div>
   <div class="tooltip" id="tip"></div>
  </div>
  <div class="legend"><span><i class="sens"></i> sensory</span><span><i class="internal"></i> internal</span><span><i class="mod"></i> modulatory</span><span><i class="out"></i> output</span><span>• animowana kropka = kierunek live flow • learned: żółty = FRESH, turkus = CONSOLIDATED, różowy = FADING • krótko-kreskowana = synapsa zmieniona przez ostatni reward/punish • pierścień = output wygrywającego readoutu</span></div>
 </div>

 <div class="side">
  <div class="card"><div class="card-head"><h2>Co robi Mucha teraz</h2><span class="badge" id="source">runtime</span></div>
   <div class="now"><div class="mini"><small>bodziec</small><strong id="event">—</strong></div><div class="mini"><small>akcja</small><strong id="last-action">—</strong></div></div>
   <div class="actions" id="actions" style="margin-top:10px"></div>
  </div>

  <div class="card"><div class="card-head"><h2>🧠 Internal attractors</h2><span class="badge" id="attractor-dominant">czekam</span></div><div id="internal-states" class="empty">Brak odczytu attractorów.</div></div>

  <div class="card"><div class="card-head"><h2>⚡ Live signal flow</h2><div class="flow-head-actions"><div class="replay-controls"><button class="btn" id="flow-prev-btn">◀ FRAME</button><button class="btn on" id="flow-live-btn">LIVE</button><button class="btn" id="flow-next-btn">FRAME ▶</button></div><span class="badge" id="flow-winner">czekam</span></div></div><div id="signal-flow" class="empty">Pierwsza klatka pojawi się po następnym bodźcu i propagacji connectomu.</div></div>

  <div class="card"><div class="card-head"><h2 id="why-title">🔬 Dlaczego ta akcja?</h2><span class="badge" id="why-action">czekam</span></div><div id="decision-explanation" class="empty">Atrybucja neuronów pojawi się po propagacji i odczycie akcji.</div></div>

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
const actionColors={speak:"#b68bff",react:"#ff9f6b",voice_join:"#55ead0",voice_move:"#6ba6ff",voice_leave:"#ff77b7",explore:"#7fd1ff",stay:"#ffd166"};
const stateColors={social_need:"#55ead0",curiosity:"#6ba6ff",stress:"#ff77b7",satiety:"#ffd166",arousal:"#b68bff"};
let projection="xy",roleFilter="all",showRegions=true,showTrail=true,showFlow=true,showLearned=true,showAttractors=true,followDecision=true,flowReplayTick=null,data=null,hover=null,selected=null,selectedRegion="",lastFetch=0;
let mapCanvasW=1,mapCanvasH=1,mapLastFrameTs=0,mapUpdating=false,mapMaxActivation=.0001,mapNodeLookup=new Map(),mapFlowFocus=new Set();
const NEUROMAP_FRAME_MS=1000/24;
const trail=new Map(),mouse={x:0,y:0,inside:false};
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v)),nfmt=n=>Number(n||0).toLocaleString("pl-PL");
const displayAction=a=>String(a||"stay")==="stay"?"NOOP":String(a||"—").toUpperCase();
const esc=v=>String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[m]));
function axes(p){return p==="xy"?["x","y"]:p==="xz"?["x","z"]:["y","z"]}
function point(o,w,h,pad=30){const [a,b]=axes(projection);return {x:pad+clamp(Number(o[a]||0),0,1)*(w-pad*2),y:pad+(1-clamp(Number(o[b]||0),0,1))*(h-pad*2)}}
function resize(){const r=wrap.getBoundingClientRect(),dpr=Math.min(1.5,window.devicePixelRatio||1);mapCanvasW=Math.max(1,r.width);mapCanvasH=Math.max(1,r.height);canvas.width=Math.max(1,Math.floor(mapCanvasW*dpr));canvas.height=Math.max(1,Math.floor(mapCanvasH*dpr));canvas.style.width=mapCanvasW+"px";canvas.style.height=mapCanvasH+"px";ctx.setTransform(dpr,0,0,dpr,0,0)}
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
function currentFlow(){
 if(!data||!data.signal_flow)return null;
 const snap=data.signal_flow;
 if(flowReplayTick!==null){
  const old=(snap.history||[]).find(x=>Number(x.tick)===Number(flowReplayTick));
  if(old)return old;
  flowReplayTick=null
 }
 return snap.latest||null
}
function flowNodeSet(){
 const f=currentFlow(),ids=new Set();if(!f)return ids;
 if(selectedRegion&&data){
  const r=(data.regions||[]).find(x=>x.name===selectedRegion);
  for(const e of ((r||{}).live_flow_edges||[])){ids.add(String(e.source));ids.add(String(e.target))}
  if(ids.size)return ids
 }
 for(const e of (f.edges||[])){ids.add(String(e.source));ids.add(String(e.target))}
 for(const id of (f.output_nodes||[]))ids.add(String(id));
 for(const cue of (f.cues||[]))for(const id of (cue.root_ids||[]))ids.add(String(id));
 return ids
}
function drawAttractors(w,h){
 if(!data||!showAttractors)return;const diag=data.internal_states||{},states=diag.states||{};
 ctx.save();ctx.globalCompositeOperation="lighter";
 for(const n of (data.nodes||[])){const memberships=n.internal_states||[];if(!memberships.length)continue;const p=point(n,w,h,26);
  memberships.slice(0,3).forEach((name,i)=>{const level=clamp(Number((states[name]||{}).level||0),0,1),tone=stateColors[name]||"#fff";ctx.strokeStyle=tone;ctx.globalAlpha=.16+level*.66;ctx.lineWidth=1+level*1.8;ctx.beginPath();ctx.arc(p.x,p.y,8+i*3+level*5,0,Math.PI*2);ctx.stroke()})
 }
 ctx.globalAlpha=1;ctx.restore()
}
function renderInternalStates(){
 const root=$("internal-states"),diag=data&&data.internal_states?data.internal_states:{},states=diag.states||{},order=["social_need","curiosity","stress","satiety","arousal"];
 const dominant=diag.dominant||"—";$("attractor-dominant").textContent=String(dominant).toUpperCase()+" "+(Number(diag.dominant_level||0)*100).toFixed(0)+"%";
 if(!Object.keys(states).length){root.className="empty";root.textContent=diag.enabled===false?"Internal states wyłączone.":"Brak zbudowanych attractorów.";return}
 const labels={social_need:"SOCIAL NEED",curiosity:"CURIOSITY",stress:"STRESS",satiety:"SATIETY",arousal:"AROUSAL"};
 root.className="";root.innerHTML=order.filter(name=>states[name]).map(name=>{const s=states[name],level=clamp(Number(s.level||0),0,1),tone=stateColors[name]||"#fff";return '<div class="signal-row"><b style="color:'+tone+'">'+esc(labels[name]||name)+'</b><span><div class="track"><div class="fill" style="width:'+(level*100).toFixed(1)+'%"></div></div>'+nfmt(s.neurons||0)+' n • '+nfmt(s.recurrent_edges||0)+' recurrent • → '+esc((s.target_actions||[]).join(", "))+'</span><em style="color:'+tone+'">'+(level*100).toFixed(0)+'%</em></div>'}).join("")+'<div class="flow-note">'+esc(diag.method||"")+' • recurrent gain '+Number(diag.recurrent_gain||0).toFixed(2)+'</div>'
}
function drawLearnedSynapses(w,h){
 if(!data||!showLearned)return;const learned=data.learned_synapses||{},rows=learned.edges||[];if(!rows.length)return;
 const maxD=Math.max(.000001,...rows.map(e=>Math.abs(Number(e.learned_delta||0))));
 ctx.save();ctx.globalCompositeOperation="source-over";
 for(const e of rows.slice(0,160)){const a=mapNodeLookup.get(String(e.source))||e.source_position,b=mapNodeLookup.get(String(e.target))||e.target_position;if(!a||!b)continue;const p1=point(a,w,h,26),p2=point(b,w,h,26),q=clamp(Math.abs(Number(e.learned_delta||0))/maxD,0,1),status=String(e.status||"fading"),tone=status==="consolidated"?"#55ead0":status==="fresh"?"#ffd166":"#ff77b7";
  ctx.strokeStyle=tone;ctx.globalAlpha=status==="fading"?.12:(.18+q*.38);ctx.lineWidth=.45+q*1.45;ctx.setLineDash(status==="fading"?[2,5]:[]);ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.stroke()
 }
 ctx.setLineDash([]);ctx.globalAlpha=1;ctx.restore()
}
function drawSignalFlow(w,h){
 if(!data||!showFlow)return;const f=currentFlow();if(!f||!(f.edges||[]).length)return;
 const regionFocus=selectedRegion?flowNodeSet():null;
 const edges=(f.edges||[]).filter(e=>!regionFocus||!regionFocus.size||regionFocus.has(String(e.source))||regionFocus.has(String(e.target))).map(e=>({
  e,
  a:mapNodeLookup.get(String(e.source))||e.source_position,
  b:mapNodeLookup.get(String(e.target))||e.target_position
 })).filter(x=>x.a&&x.b);
 if(!edges.length)return;const maxC=Math.max(.000001,...edges.map(x=>Math.abs(Number(x.e.contribution||0)))),tone=actionColors[f.winner]||"#55ead0",phase=(performance.now()%1100)/1100;
 ctx.save();ctx.globalCompositeOperation="lighter";
 edges.slice(0,140).forEach((row,i)=>{const e=row.e,p1=point(row.a,w,h,26),p2=point(row.b,w,h,26),q=clamp(Math.abs(Number(e.contribution||0))/maxC,0,1),neg=Number(e.contribution||0)<0;
  ctx.strokeStyle=tone;ctx.globalAlpha=.08+q*.48;ctx.lineWidth=.45+q*2.0;ctx.setLineDash(neg?[4,4]:[]);ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.stroke();ctx.setLineDash([]);
  const t=(phase+i*.071)%1,x=p1.x+(p2.x-p1.x)*t,y=p1.y+(p2.y-p1.y)*t;ctx.fillStyle=neg?"#ff77b7":tone;ctx.globalAlpha=.40+q*.60;ctx.shadowColor=ctx.fillStyle;ctx.shadowBlur=5+q*9;ctx.beginPath();ctx.arc(x,y,1.2+q*2.1,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0
 });
 ctx.setLineDash([]);ctx.globalAlpha=.9;ctx.strokeStyle=tone;ctx.lineWidth=1.35;
 for(const out of (f.output_points||[]).slice(0,28)){const p=point(out,w,h,26);ctx.beginPath();ctx.arc(p.x,p.y,8.5,0,Math.PI*2);ctx.stroke()}
 const learning=(data.signal_flow&&data.signal_flow.learning||[]).slice(-1)[0]||null;
 if(learning){
  ctx.globalCompositeOperation="source-over";
  for(const e of (learning.top_synapses||[]).slice(0,20)){const a=mapNodeLookup.get(String(e.source))||e.source_position,b=mapNodeLookup.get(String(e.target))||e.target_position;if(!a||!b)continue;const p1=point(a,w,h,26),p2=point(b,w,h,26),chg=Number(e.change||0);ctx.strokeStyle=chg>=0?"#ffd166":"#ff77b7";ctx.globalAlpha=.60;ctx.lineWidth=1.6;ctx.setLineDash([2,3]);ctx.beginPath();ctx.moveTo(p1.x,p1.y);ctx.lineTo(p2.x,p2.y);ctx.stroke()}
 }
 ctx.setLineDash([]);ctx.globalAlpha=1;ctx.restore()
}
function liveFlowRows(n){
 const rows=n.live_flow_edges||[];if(!rows.length)return '<div class="note">Ten neuron nie uczestniczy w najmocniejszych krawędziach ostatniej klatki.</div>';
 return rows.map(x=>'<div class="signal-row '+(Number(x.contribution||0)>=0?"pos":"neg")+'"><b>'+esc(x.direction==="in"?"IN":"OUT")+'</b><span>'+esc(x.peer||"—")+'</span><em>'+(Number(x.contribution||0)>=0?"+":"")+Number(x.contribution||0).toFixed(4)+'</em></div>').join("")
}
function renderSignalFlow(){
 const root=$("signal-flow"),snap=data&&data.signal_flow?data.signal_flow:{},f=currentFlow(),learning=(snap.learning||[]).slice(-1)[0]||null;
 if(!f){$("flow-winner").textContent="BRAK";$("flow-edge-count").textContent="0";$("flow-frame").textContent="brak klatki";root.className="empty";root.textContent="Pierwsza klatka pojawi się po następnym bodźcu i propagacji connectomu.";return}
 const edges=f.edges||[],cues=f.cues||[],top=edges.slice(0,7),history=(snap.history||[]).slice(-20).reverse(),learned=data.learned_synapses||{};
 $("flow-live-btn").classList.toggle("on",flowReplayTick===null);
 $("flow-live-btn").textContent=flowReplayTick===null?"LIVE":"WRÓĆ LIVE";
 $("flow-winner").textContent=String(f.winner||"—").toUpperCase();$("flow-edge-count").textContent=nfmt(edges.length);$("flow-frame").textContent=(flowReplayTick===null?"LIVE • ":"REPLAY • ")+"tick "+nfmt(f.tick)+" • "+nfmt(f.frame)+"/"+nfmt(f.frames);
 const cueHtml=cues.slice(0,8).map(x=>'<span class="flow-cue">'+esc(x.action?x.action+" • ":"")+esc(x.key||x.kind||"cue")+' ×'+nfmt(x.neurons||0)+'</span>').join("")||'<span class="flow-cue">brak jawnego cue</span>';
 const edgeHtml=top.map(e=>'<div class="signal-row '+(Number(e.contribution||0)>=0?"pos":"neg")+'"><b>'+esc(String(e.source).slice(-6))+'</b><span>→ '+esc(String(e.target).slice(-6))+(Math.abs(Number(e.learned_delta||0))>1e-9?' • learned '+Number(e.learned_delta).toExponential(1):'')+(Math.abs(Number(e.attractor_delta||0))>1e-9?' • attractor '+Number(e.attractor_delta).toExponential(1):'')+'</span><em>'+(Number(e.contribution||0)>=0?"+":"")+Number(e.contribution||0).toFixed(4)+'</em></div>').join("")||'<div class="note">Brak silnych krawędzi w tej klatce.</div>';
 const histHtml=history.map(x=>'<div class="signal-row flow-history '+(Number(x.tick)===Number(flowReplayTick)?"on":"")+'" data-flow-tick="'+Number(x.tick)+'"><b>▶ #'+nfmt(x.tick)+'</b><span>'+esc((x.cues&&x.cues[0]?x.cues[0].key:"propagation"))+'</span><em>'+esc(x.winner||"—")+'</em></div>').join("");
 const learnHtml=learning?'<div class="learn-row"><b>Ostatnia plastyczność:</b> reward '+(Number(learning.amount||0)>=0?"+":"")+Number(learning.amount||0).toFixed(3)+' • '+nfmt(learning.changed_neurons)+' neuronów • '+nfmt(learning.changed_synapses)+' synaps'+((learning.top_synapses||[]).length?' • top Δ '+Number(learning.top_synapses[0].change||0).toExponential(2):'')+'</div>':'<div class="learn-row">Brak reward/punish od startu tej sesji.</div>';
 const learnedHtml='<div class="learn-row"><b>Learned synapses:</b> '+nfmt(learned.total||0)+' • <span style="color:#ffd166">FRESH '+nfmt(learned.fresh||0)+'</span> • <span style="color:#55ead0">CONSOLIDATED '+nfmt(learned.consolidated||0)+'</span> • <span style="color:#ff77b7">FADING '+nfmt(learned.fading||0)+'</span></div>';
 root.className="";root.innerHTML='<div class="signal-summary"><div><small>readout</small><b style="color:'+(actionColors[f.winner]||"#fff")+'">'+esc(f.winner||"—")+'</b></div><div><small>gain</small><b>'+Number(f.propagation_gain||0).toFixed(3)+'</b></div><div><small>edges</small><b>'+nfmt(edges.length)+'</b></div></div><div>'+cueHtml+'</div><div class="flow-note">'+esc(snap.method||"")+'</div><div class="signal-list" style="margin-top:8px">'+edgeHtml+'</div>'+learnHtml+learnedHtml+'<div class="effects"><small>Historia — kliknij, aby odtworzyć przepływ</small><div class="signal-list">'+histHtml+'</div></div>';
 root.querySelectorAll("[data-flow-tick]").forEach(el=>el.onclick=()=>{flowReplayTick=Number(el.dataset.flowTick);mapFlowFocus=flowNodeSet();renderSignalFlow()})
}
function stepFlow(delta){
 const history=data?.signal_flow?.history||[];if(!history.length)return;
 if(flowReplayTick===null){
  if(delta>=0)return;
  const idx=Math.max(0,history.length-2);
  flowReplayTick=Number(history[idx].tick)
 }else{
  let idx=history.findIndex(x=>Number(x.tick)===Number(flowReplayTick));
  if(idx<0)idx=history.length-1;
  const next=idx+Number(delta||0);
  if(next>=history.length){flowReplayTick=null}
  else{flowReplayTick=Number(history[Math.max(0,next)].tick)}
 }
 mapFlowFocus=flowNodeSet();renderSignalFlow()
}
function renderDecisionExplanation(){
 const root=$("decision-explanation"),d=data?.decision_explanation||{},action=String(d.action||"");
 if(!action){$("why-action").textContent="BRAK";$("why-title").textContent="🔬 Dlaczego ta akcja?";root.className="empty";root.textContent="Atrybucja neuronów pojawi się po propagacji i odczycie akcji.";return}
 $("why-title").textContent="🔬 Dlaczego "+displayAction(action)+"?";
 $("why-action").textContent=displayAction(action);
 const pos=d.supporting_neurons||[],neg=d.opposing_neurons||[];
 const rows=(xs,kind)=>xs.slice(0,8).map(x=>'<div class="why-row '+kind+'" data-why-neuron="'+esc(x.id)+'"><span>'+esc(x.id)+' • '+esc(x.type||x.region||"—")+'</span><strong>'+(Number(x.contribution||0)>=0?"+":"")+Number(x.contribution||0).toExponential(2)+'</strong></div>').join("")||'<div class="note">Brak mierzalnych neuronów w tej grupie.</div>';
 root.className="";root.innerHTML=
  '<div class="why-summary">'+
   '<div><small>winner score</small><b>'+Number(d.score||0).toFixed(4)+'</b></div>'+
   '<div><small>runner-up</small><b>'+esc(displayAction(d.runner_up||"stay"))+' '+Number(d.runner_up_score||0).toFixed(4)+'</b></div>'+
   '<div><small>margin</small><b>'+Number(d.margin||0).toFixed(4)+'</b></div>'+
   '<div><small>frame</small><b>'+nfmt(d.frame||0)+' / '+nfmt(d.frames||0)+'</b></div>'+
  '</div>'+
  '<div class="why-grid">'+
   '<div class="why-col"><div class="why-head"><b style="color:var(--cyan)">WSPIERAJĄCE</b><span class="muted">+ drive</span></div>'+rows(pos,"pos")+'</div>'+
   '<div class="why-col"><div class="why-head"><b style="color:var(--pink)">HAMUJĄCE</b><span class="muted">− drive</span></div>'+rows(neg,"neg")+'</div>'+
  '</div>'+
  '<div class="flow-note">'+esc(d.method||"")+'</div>';
 root.querySelectorAll("[data-why-neuron]").forEach(el=>el.onclick=()=>selectNeuronById(el.dataset.whyNeuron))
}
function drawSelectedConnections(w,h){
 if(!selected)return;const rows=selected.structural_connections||[];if(!rows.length)return;
 const p0=point(selected,w,h,26),max=Math.max(.000001,...rows.map(x=>Math.abs(Number(x.current_drive||0))));
 ctx.save();ctx.globalCompositeOperation="source-over";
 for(const row of rows){
  const peer=mapNodeLookup.get(String(row.peer))||row.peer_position;if(!peer)continue;
  const p1=point(peer,w,h,26),q=clamp(Math.abs(Number(row.current_drive||0))/max,0,1),positive=Number(row.current_drive||0)>=0;
  ctx.strokeStyle=positive?"#55ead0":"#ff77b7";ctx.globalAlpha=.18+q*.52;ctx.lineWidth=.6+q*1.7;ctx.setLineDash(row.direction==="in"?[3,3]:[]);
  ctx.beginPath();if(row.direction==="in"){ctx.moveTo(p1.x,p1.y);ctx.lineTo(p0.x,p0.y)}else{ctx.moveTo(p0.x,p0.y);ctx.lineTo(p1.x,p1.y)}ctx.stroke()
 }
 ctx.setLineDash([]);ctx.globalAlpha=1;ctx.restore()
}
function drawNodes(w,h){
 if(!data)return;hover=null;const nodes=(data.nodes||[]),focus=followDecision?mapFlowFocus:null;
 for(const n of nodes){if(!roleVisible(n))continue;if(selectedRegion&&regionOf(n)!==selectedRegion)continue;const p=point(n,w,h,26),a=Math.abs(Number(n.activation||0)),q=clamp(a/mapMaxActivation,0,1),hist=showTrail?(trail.get(n.id)||a):a,c=colors[n.role]||colors.internal,dim=followDecision&&focus&&focus.size&&!focus.has(String(n.id));
  if(showTrail&&hist>a+.015){ctx.strokeStyle=c;ctx.globalAlpha=clamp(hist*.34,0,.28);ctx.lineWidth=1;ctx.beginPath();ctx.arc(p.x,p.y,6+hist*24,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1}
  ctx.shadowColor=c;ctx.shadowBlur=4+q*22;ctx.fillStyle=c;ctx.globalAlpha=dim?.10:(.25+q*.75);ctx.beginPath();ctx.arc(p.x,p.y,2.1+q*5.4,0,Math.PI*2);ctx.fill();ctx.globalAlpha=1;ctx.shadowBlur=0;
  if(n.decision_output){const f=currentFlow(),tone=actionColors[(f||{}).winner]||"#fff";ctx.strokeStyle=tone;ctx.lineWidth=1.5;ctx.globalAlpha=.85;ctx.beginPath();ctx.arc(p.x,p.y,9+q*6,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1}
  if(!n.real_position){ctx.strokeStyle="rgba(255,209,102,.45)";ctx.lineWidth=.6;ctx.beginPath();ctx.arc(p.x,p.y,4+q*5.6,0,Math.PI*2);ctx.stroke()}
  if(mouse.inside){const dx=mouse.x-p.x,dy=mouse.y-p.y;if(dx*dx+dy*dy<120)hover=n}
  if(selected&&selected.id===n.id){ctx.strokeStyle="#fff";ctx.lineWidth=1.2;ctx.globalAlpha=.9;ctx.beginPath();ctx.arc(p.x,p.y,11+q*8,0,Math.PI*2);ctx.stroke();ctx.globalAlpha=1}
 }
}
function draw(ts=0){
 requestAnimationFrame(draw);
 if(document.hidden||ts-mapLastFrameTs<NEUROMAP_FRAME_MS)return;
 mapLastFrameTs=ts;const w=mapCanvasW,h=mapCanvasH;ctx.clearRect(0,0,w,h);
 const grad=ctx.createRadialGradient(w*.5,h*.5,20,w*.5,h*.5,Math.max(w,h)*.55);grad.addColorStop(0,"rgba(34,67,91,.07)");grad.addColorStop(1,"rgba(0,0,0,0)");ctx.fillStyle=grad;ctx.fillRect(0,0,w,h);
 drawReference(w,h);drawRegions(w,h);drawAttractors(w,h);drawLearnedSynapses(w,h);drawSignalFlow(w,h);drawSelectedConnections(w,h);drawNodes(w,h)
}
function renderActions(scores){
 const order=["speak","explore","react","voice_join","voice_move","voice_leave","stay"];
 $("actions").innerHTML=order.map(k=>{const v=clamp(Number(scores[k]||0),0,1);return '<div class="act"><span>'+k+'</span><div class="track"><div class="fill" style="width:'+(v*100).toFixed(1)+'%"></div></div><b>'+v.toFixed(2)+'</b></div>'}).join("");
 const best=order.map(k=>[k,Number(scores[k]||0)]).sort((a,b)=>b[1]-a[1])[0]||["—",0];$("dominant").textContent=best[0];$("dominant-score").textContent="score "+best[1].toFixed(3)
}
function sparkline(canvasEl,values){
 const r=canvasEl.getBoundingClientRect(),dpr=Math.min(1.5,window.devicePixelRatio||1);canvasEl.width=Math.max(1,Math.floor(r.width*dpr));canvasEl.height=Math.max(1,Math.floor(r.height*dpr));const c=canvasEl.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);const w=r.width,h=r.height;c.clearRect(0,0,w,h);
 if(!values||values.length<2)return;const min=Math.min(...values),max=Math.max(...values),span=Math.max(.000001,max-min);
 c.strokeStyle="rgba(85,234,208,.92)";c.lineWidth=1.4;c.shadowColor="rgba(85,234,208,.45)";c.shadowBlur=7;c.beginPath();
 values.forEach((v,i)=>{const x=i/(values.length-1)*w,y=h-5-((v-min)/span)*(h-10);if(i===0)c.moveTo(x,y);else c.lineTo(x,y)});c.stroke();c.shadowBlur=0
}
function corrRows(rows){
 if(!rows||!rows.length)return '<div class="note">Za mało próbek albo brak zmienności. Korelacje pojawią się po kilku sekundach działania Neuro-map.</div>';
 return rows.map(x=>{const c=clamp(Number(x.correlation||0),-1,1),width=Math.abs(c)*50;return '<div class="corr"><span>'+esc(x.action)+'</span><div class="corrbar"><i class="corrfill '+(c>=0?"pos":"neg")+'" style="width:'+width.toFixed(1)+'%"></i></div><b>'+(c>=0?"+":"")+c.toFixed(2)+'</b></div>'}).join("")
}
function selectNeuronById(id){
 const n=(data?.nodes||[]).find(x=>String(x.id)===String(id));
 if(!n)return false;
 selected=n;inspect(n);return true
}
function regionInspector(){
 const root=$("region-inspector");if(!data||!selectedRegion){$("region-picked").textContent="kliknij region";root.className="empty";root.innerHTML="Wybierz region z listy. Zobaczysz historię aktywności od otwarcia Neuro-map, najaktywniejsze neurony, dominujące typy komórek oraz korelacje z readoutami Muchy.";return}
 const r=(data.regions||[]).find(x=>x.name===selectedRegion);if(!r){root.className="empty";root.textContent="Wybrany region nie jest teraz w TOP aktywnych regionów.";return}
 $("region-picked").textContent=data.region_source==="neuropil"?"NEUROPIL":"CLASS GROUP";root.className="";
 const types=(r.dominant_types||[]).map(x=>'<span class="chip">'+esc(x.name)+' ×'+nfmt(x.count)+'</span>').join("")||'<span class="chip">brak typów</span>';
 const neurons=(r.top_neurons||[]).map(n=>'<div class="nrow clickable" data-neuron="'+esc(n.id)+'"><span>'+esc(n.id)+' • '+esc(n.primary_type||"—")+' • '+esc(n.nt_type||"—")+'</span><b class="'+(Number(n.activation)>=0?"plus":"minus")+'">'+(Number(n.activation)>=0?"+":"")+Number(n.activation||0).toFixed(4)+'</b></div>').join("");
 const flowRows=(r.live_flow_edges||[]).slice(0,10).map(e=>'<div class="signal-row '+(Number(e.contribution||0)>=0?"pos":"neg")+'"><b>'+esc(String(e.direction||"").toUpperCase())+'</b><span>'+esc(String(e.source).slice(-7))+' → '+esc(String(e.target).slice(-7))+'</span><em>'+(Number(e.contribution||0)>=0?"+":"")+Number(e.contribution||0).toFixed(4)+'</em></div>').join("")||'<div class="note">Brak live flow przez ten region w aktualnej klatce.</div>';
 root.innerHTML='<div class="ins-title"><div><strong>'+esc(r.name)+'</strong><div class="note">'+nfmt(r.active_count)+' / '+nfmt(r.total)+' neuronów ma |a| &gt; 0.1</div></div><span class="role">mean '+Number(r.mean_abs||0).toFixed(4)+'</span></div>'+
 '<div class="meta"><div><small>max activation</small><b>'+Number(r.max_abs||0).toFixed(4)+'</b></div><div><small>history samples</small><b>'+nfmt(r.correlation_samples||0)+'</b></div></div>'+
 '<div class="spark-wrap"><canvas id="region-spark"></canvas></div>'+
 '<div class="effects"><small>Dominujące typy neuronów</small><div class="chips">'+types+'</div></div>'+
 '<div class="effects"><small>Runtime correlation z readoutami</small>'+corrRows(r.readout_correlations||[])+'<div class="note">To korelacja czasowa aktywności regionu z readoutem Muchy, nie dowód biologicznej funkcji ani przyczynowości.</div></div>'+
 '<div class="effects"><small>Live flow regionu • IN '+Number(r.live_flow_in||0).toFixed(3)+' • INTERNAL '+Number(r.live_flow_internal||0).toFixed(3)+' • OUT '+Number(r.live_flow_out||0).toFixed(3)+'</small><div class="signal-list">'+flowRows+'</div></div>'+
 '<div class="effects"><small>Najaktywniejsze neurony w regionie — kliknij</small><div class="neuron-list">'+neurons+'</div></div>';
 root.querySelectorAll("[data-neuron]").forEach(el=>el.onclick=()=>selectNeuronById(el.dataset.neuron));
 requestAnimationFrame(()=>{const c=$("region-spark");if(c)sparkline(c,r.history||[])})
}
function renderRegions(){
 const regs=data.regions||[],max=Math.max(.0001,...regs.map(r=>Number(r.score||0)));
 $("regions").innerHTML=regs.slice(0,20).map(r=>{const q=clamp(Number(r.score||0)/max,0,1);return '<div class="region '+(selectedRegion===r.name?"on":"")+'" data-region="'+esc(r.name)+'"><div><b>'+esc(r.name)+'</b><br><small>'+nfmt(r.active_count)+' / '+nfmt(r.total)+' active</small></div><div class="rtrack"><div class="rfill" style="width:'+(q*100).toFixed(1)+'%"></div></div><small>'+Number(r.mean_abs||0).toFixed(3)+'</small></div>'}).join("")||'<div class="empty">Brak nazwanych regionów w aktualnym cache.</div>';
 document.querySelectorAll("[data-region]").forEach(el=>el.onclick=()=>{const name=el.dataset.region;selectedRegion=selectedRegion===name?"":name;$("region-filter").textContent=selectedRegion||"ALL";mapFlowFocus=flowNodeSet();renderRegions();regionInspector();renderSignalFlow();updateTrail()})
}
function effectRows(n){
 const xs=n.system_actions||[];if(!xs.length)return '<div class="note">Brak bezpośredniego połączenia tego neuronu do sztucznych populacji action-readout w pokazanym kierunku.</div>';
 const max=Math.max(...xs.map(x=>Number(x.strength||0)),.0001);return xs.map(x=>'<div class="effect"><span>'+esc(x.name)+'</span><div class="efill"><i style="width:'+(Number(x.strength||0)/max*100).toFixed(1)+'%"></i></div><b>'+Number(x.strength||0).toFixed(4)+'</b></div>').join("")
}
function neuropilRows(n){
 const xs=n.neuropils||[];if(!xs.length)return '<div class="note">Brak summary neuropili dla tego neuronu. Przebuduj neuron_meta.npz z plikiem connections_princeton.csv.gz.</div>';
 return xs.map(x=>'<div class="effect"><span>'+esc(x.name)+'</span><div class="efill"><i style="width:'+(clamp(Number(x.share||0),0,1)*100).toFixed(1)+'%"></i></div><b>'+(Number(x.share||0)*100).toFixed(0)+'%</b></div>').join("")
}
function structuralRows(n){
 const rows=n.structural_connections||[];if(!rows.length)return '<div class="note">Brak połączeń w aktualnym runtime matrix.</div>';
 return rows.map(x=>'<div class="conn-row" data-peer-neuron="'+esc(x.peer||"")+'"><b>'+esc(String(x.direction||"").toUpperCase())+'</b><span>'+esc(x.peer||"—")+'</span><em>w '+Number(x.effective_weight||0).toFixed(4)+'</em><em class="drive '+(Number(x.current_drive||0)>=0?"pos":"neg")+'">'+(Number(x.current_drive||0)>=0?"+":"")+Number(x.current_drive||0).toFixed(4)+'</em></div>').join("")
}
function directActionRows(n){
 const rows=n.action_contributions||[];if(!rows.length)return '<div class="note">Brak mierzalnego bezpośredniego wkładu do widocznych action-output pools.</div>';
 const max=Math.max(.000001,...rows.map(x=>Math.abs(Number(x.contribution||0))));
 return rows.map(x=>'<div class="effect"><span>'+esc(x.name)+'</span><div class="efill"><i style="width:'+(Math.abs(Number(x.contribution||0))/max*100).toFixed(1)+'%"></i></div><b class="'+(Number(x.contribution||0)>=0?"plus":"minus")+'">'+(Number(x.contribution||0)>=0?"+":"")+Number(x.contribution||0).toExponential(2)+'</b></div>').join("")
}
function inspect(n){
 selected=n||selected;if(!selected)return;const n0=selected;$("picked").textContent=n0.real_position?"REAL POSITION":"FALLBACK POSITION";
 $("inspector").className="";$("inspector").innerHTML='<div class="ins-title"><div><strong>'+esc(n0.id)+'</strong><div class="note">'+esc(n0.primary_type||n0.sub_class||n0.cell_class||n0.super_class||"brak typu")+'</div></div><span class="role">'+esc(n0.role)+'</span></div>'+
 '<div class="meta"><div><small>activation</small><b>'+(Number(n0.activation)>=0?"+":"")+Number(n0.activation||0).toFixed(5)+'</b></div><div><small>eligibility</small><b>'+Number(n0.eligibility||0).toFixed(5)+'</b></div>'+
 '<div><small>class</small><b>'+esc(n0.cell_class||"—")+'</b></div><div><small>sub_class</small><b>'+esc(n0.sub_class||"—")+'</b></div>'+
 '<div><small>super_class</small><b>'+esc(n0.super_class||"—")+'</b></div><div><small>side / flow</small><b>'+esc((n0.side||"—")+" / "+(n0.flow||"—"))+'</b></div>'+
 '<div><small>neurotransmitter</small><b>'+esc(n0.nt_type||"—")+'</b></div><div><small>primary neuropil</small><b>'+esc(n0.primary_neuropil||"—")+'</b></div>'+
 '<div><small>static in / out</small><b>'+nfmt(n0.incoming_edges||0)+' / '+nfmt(n0.outgoing_edges||0)+'</b></div><div><small>live flow in / out</small><b>'+Number(n0.live_flow_in||0).toFixed(4)+' / '+Number(n0.live_flow_out||0).toFixed(4)+'</b></div>'+
 '<div><small>internal attractor</small><b>'+esc((n0.internal_states||[]).join(", ")||"—")+'</b></div></div>'+
 '<div class="effects"><small>Live incoming / outgoing — ostatnia klatka</small>'+liveFlowRows(n0)+'</div>'+
 '<div class="effects"><small>Najsilniejsze aktualne połączenia strukturalne IN / OUT</small>'+structuralRows(n0)+'</div>'+
 '<div class="effects"><small>Signed wkład do action readoutów</small>'+directActionRows(n0)+'</div>'+
 '<div class="effects"><small>Top neuropile wg incident synapse mass</small>'+neuropilRows(n0)+'</div>'+
 '<div class="effects"><small>Siła bezpośrednich połączeń do systemowych readoutów</small>'+effectRows(n0)+'</div>'+
 '<div class="note">Stage 34 rozdziela: live propagation, strukturalny runtime matrix i signed presynaptic contribution. To telemetryka algorytmu, nie pełny dowód biologicznej przyczynowości.</div>';
 $("inspector").querySelectorAll("[data-peer-neuron]").forEach(el=>el.onclick=()=>selectNeuronById(el.dataset.peerNeuron))
}
canvas.addEventListener("mousemove",e=>{const r=canvas.getBoundingClientRect();mouse.x=e.clientX-r.left;mouse.y=e.clientY-r.top;mouse.inside=true;if(hover){tip.style.display="block";tip.style.left=Math.min(r.width-255,mouse.x+13)+"px";tip.style.top=Math.min(r.height-155,mouse.y+13)+"px";tip.innerHTML='<b>'+esc(hover.id)+'</b><br><span class="mut">'+esc(hover.primary_type||hover.cell_class||hover.super_class||hover.role)+'</span><br><span class="acc">activation '+Number(hover.activation||0).toFixed(5)+'</span><br>flow in/out '+Number(hover.live_flow_in||0).toFixed(3)+' / '+Number(hover.live_flow_out||0).toFixed(3)+'<br>neuropil '+esc(hover.primary_neuropil||"—")+'<br>'+esc(hover.side||"")+' '+esc(hover.nt_type||"")}else tip.style.display="none"});
canvas.addEventListener("mouseleave",()=>{mouse.inside=false;tip.style.display="none"});
canvas.addEventListener("click",()=>{if(hover){selected=hover;inspect(selected)}});

document.querySelectorAll(".proj").forEach(b=>b.onclick=()=>{projection=b.dataset.proj;document.querySelectorAll(".proj").forEach(x=>x.classList.toggle("on",x===b));$("map-title").textContent=projection.toUpperCase()+" PROJECTION";update()});
document.querySelectorAll(".role").forEach(b=>b.onclick=()=>{roleFilter=b.dataset.role;document.querySelectorAll(".role").forEach(x=>x.classList.toggle("on",x===b));selectedRegion="";$("region-filter").textContent=roleFilter.toUpperCase();regionInspector();updateTrail()});
$("regions-btn").onclick=()=>{showRegions=!showRegions;$("regions-btn").classList.toggle("on",showRegions)};
$("trail-btn").onclick=()=>{showTrail=!showTrail;$("trail-btn").classList.toggle("on",showTrail);if(!showTrail)trail.clear()};
$("flow-btn").onclick=()=>{showFlow=!showFlow;$("flow-btn").classList.toggle("on",showFlow)};
$("learned-btn").onclick=()=>{showLearned=!showLearned;$("learned-btn").classList.toggle("on",showLearned)};
$("attractor-btn").onclick=()=>{showAttractors=!showAttractors;$("attractor-btn").classList.toggle("on",showAttractors)};
$("follow-btn").onclick=()=>{followDecision=!followDecision;$("follow-btn").classList.toggle("on",followDecision)};
$("flow-live-btn").onclick=()=>{flowReplayTick=null;mapFlowFocus=flowNodeSet();renderSignalFlow()};
$("flow-prev-btn").onclick=()=>stepFlow(-1);
$("flow-next-btn").onclick=()=>stepFlow(1);

function render(payload){
 const m=payload.brain_map||{};data=m;mapNodeLookup=new Map((m.nodes||[]).map(n=>[String(n.id),n]));mapMaxActivation=.0001;for(const n of (m.nodes||[]))mapMaxActivation=Math.max(mapMaxActivation,Math.abs(Number(n.activation||0)));mapFlowFocus=flowNodeSet();updateTrail();const scores=payload.scores||{};
 const coverage=Number(m.coordinate_coverage||0);$("coverage").textContent=(coverage*100).toFixed(1)+"%";$("coord-count").textContent=nfmt(m.coordinate_neurons)+" / "+nfmt(m.total_neurons)+" neurons";
 $("neuropil-coverage").textContent=(Number(m.neuropil_coverage||0)*100).toFixed(1)+"%";$("neuropil-count").textContent=nfmt(m.neuropil_labels)+" nazwanych neuropili";
 $("active").textContent=nfmt(m.active_abs_gt_0_1);
 const top=(m.regions||[])[0];$("top-region").textContent=top?top.name:"—";$("top-region-detail").textContent=top?(nfmt(top.active_count)+" active • mean "+Number(top.mean_abs||0).toFixed(3)):"brak adnotacji";
 const badge=$("coord-badge"),mode=m.coordinate_mode||"synthetic";badge.textContent=mode==="real"?"REAL FAFB COORDS":mode==="hybrid"?"HYBRID COORDS":"FALLBACK LAYOUT";badge.className="badge "+mode;
 const rs=$("region-source");rs.textContent=m.region_source==="neuropil"?"NAMED NEUROPILS":"CLASS FALLBACK";rs.className="badge "+(m.region_source==="neuropil"?"neuropil":"fallback");$("region-note").textContent=m.region_source_detail||"—";
 $("event").textContent=payload.last_event||"—";$("last-action").textContent=payload.last_action||"—";$("source").textContent=(payload.source||"runtime").includes("FlyWire")?"FAFB v783":"runtime";renderActions(scores);renderInternalStates();renderSignalFlow();renderDecisionExplanation();renderRegions();regionInspector();
 if(selected){const fresh=(m.nodes||[]).find(n=>n.id===selected.id);if(fresh){selected=fresh;inspect(fresh)}}
 $("live").textContent="LIVE";lastFetch=Date.now()
}
async function update(){
 if(mapUpdating||document.hidden)return;
 mapUpdating=true;
 try{const r=await fetch("/api/neuromap?projection="+projection,{cache:"no-store"});if(r.status===401){location="/login";return}if(!r.ok)throw new Error("HTTP "+r.status);render(await r.json())}
 catch(e){$("live").textContent="ROZŁĄCZONO";console.error(e)}
 finally{mapUpdating=false}
}
document.addEventListener("visibilitychange",()=>{if(!document.hidden){mapLastFrameTs=0;update()}});
window.addEventListener("resize",resize);resize();draw();setInterval(update,1800);update();
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
<div class="nav"><a class="active" href="/public">🏠 Podgląd</a><a href="/public/connectome">🧬 Connectome</a><a href="/public/neuromap">🧠 Neuro-map</a><a href="/public/associations">🗣 Mowa</a><a href="/login">🔒 Admin</a></div></div>
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

AUTONOMY_HTML = r"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mucha — One Brain / Autonomia 24E</title>
<style>
:root{--bg:#070b10;--panel:#0f161f;--panel2:#0a1118;--line:#213043;--txt:#edf5fd;--muted:#8190a1;--a:#58dac4;--blue:#6ea8fe;--good:#57db91;--warn:#f0c45b;--bad:#ff7272;--purple:#b995ff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 12% 0%,rgba(88,218,196,.10),transparent 30%),radial-gradient(circle at 88% 0%,rgba(110,168,254,.10),transparent 32%),linear-gradient(180deg,#070b10,#0a1017 60%,#080c11);color:var(--txt);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1540px;margin:auto;padding:22px}.top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:16px}.brand{display:flex;gap:13px;align-items:center}.logo{font-size:37px}h1{margin:0;font-size:24px}.sub{margin-top:4px;color:var(--muted);font-size:12px}.nav{display:flex;gap:8px;flex-wrap:wrap}.nav a{color:#b9c8d7;text-decoration:none;background:#0e1720;border:1px solid var(--line);padding:8px 11px;border-radius:10px;font-size:12px}.nav a.active{color:#07110e;background:var(--a);border-color:var(--a);font-weight:800}
.hero{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:12px}.card,.hero-card{background:rgba(15,22,31,.94);border:1px solid var(--line);border-radius:16px}.hero-card{padding:14px}.hero-card small{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}.hero-card strong{font-size:17px;word-break:break-word}
.pipeline{display:grid;grid-template-columns:repeat(8,1fr);gap:10px;margin-bottom:12px}.stage{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:14px}.stage:not(:last-child):after{content:"→";position:absolute;right:-9px;top:50%;transform:translateY(-50%);z-index:3;color:var(--a);font-size:20px;font-weight:900}.stage b{display:block;font-size:12px;margin-bottom:5px}.stage span{color:var(--muted);font-size:11px;line-height:1.45}
.grid{display:grid;grid-template-columns:1.35fr .65fr;gap:12px}.card{padding:15px;min-width:0}.card h2{margin:0 0 12px;font-size:12px;color:#aebdcb;text-transform:uppercase;letter-spacing:.1em}.span2{grid-column:1/-1}
.guild-tabs{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}.guild-tab{border:1px solid var(--line);background:#0a121a;color:#aebdcb;border-radius:999px;padding:7px 10px;cursor:pointer;font-size:11px}.guild-tab.active{background:rgba(88,218,196,.15);border-color:var(--a);color:#dffff8}
.candidates{display:flex;flex-direction:column;gap:8px}.candidate{display:grid;grid-template-columns:128px 70px minmax(100px,1fr) 110px 110px 126px 110px;gap:8px;align-items:center;padding:10px;background:var(--panel2);border:1px solid #1e2c3b;border-radius:12px;font-size:11px}.candidate.winner{border-color:var(--a);box-shadow:0 0 0 1px rgba(88,218,196,.15),0 0 24px rgba(88,218,196,.08)}.candidate.disabled{opacity:.52}.action-name{font-weight:850;font-size:12px}.pill{display:inline-flex;align-items:center;justify-content:center;border:1px solid #2b4054;border-radius:999px;padding:4px 7px;font-size:9px;white-space:nowrap}.pill.ok{color:var(--good);border-color:rgba(87,219,145,.45)}.pill.no{color:var(--bad);border-color:rgba(255,114,114,.45)}.pill.win{color:#06110e;background:var(--a);border-color:var(--a);font-weight:900}
.metric small{display:block;color:var(--muted);font-size:9px;margin-bottom:3px}.metric b{font-variant-numeric:tabular-nums}.bar{height:5px;background:#071019;border-radius:999px;overflow:hidden;margin-top:4px}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--a));border-radius:999px}.bar.reward i.pos{background:linear-gradient(90deg,#398e65,var(--good))}.bar.reward i.neg{background:linear-gradient(90deg,#8b3f4b,var(--bad))}
.reason{color:#93a5b7;font-size:10px;line-height:1.4}.summary-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.summary{background:var(--panel2);border:1px solid #1e2c3b;border-radius:11px;padding:10px}.summary small{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.07em;margin-bottom:5px}.summary strong{font-size:14px;word-break:break-word}.cue-list{display:flex;flex-direction:column;gap:7px;margin-top:10px}.cue{display:grid;grid-template-columns:115px 1fr 65px;gap:8px;align-items:center;padding:8px 9px;background:#09121a;border:1px solid #1d2b39;border-radius:10px;font-size:10px}.cue span{color:var(--muted)}
.history{display:flex;flex-direction:column;gap:7px;max-height:420px;overflow:auto}.hist{display:grid;grid-template-columns:68px 95px 1fr 82px 72px;gap:8px;align-items:center;background:#09121a;border:1px solid #1c2937;border-radius:10px;padding:9px;font-size:10px}.hist .time{color:var(--muted);font-variant-numeric:tabular-nums}.hist .act{font-weight:800}.hist .detail{color:#b6c4d2;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.hist .rep{color:var(--muted);text-align:right}.hist.external{border-color:rgba(88,218,196,.35)}
.intent-panel{display:grid;grid-template-columns:240px 1fr;gap:14px;align-items:stretch}.intent-main{display:flex;flex-direction:column;justify-content:center;align-items:center;min-height:180px;background:radial-gradient(circle at 50% 15%,rgba(185,149,255,.14),transparent 58%),#09121a;border:1px solid rgba(185,149,255,.30);border-radius:14px;padding:18px}.intent-main .intent-action{font-size:27px;font-weight:900;letter-spacing:.04em}.intent-main .intent-state{font-size:10px;color:var(--purple);text-transform:uppercase;letter-spacing:.12em;margin-bottom:8px}.intent-track{width:100%;height:9px;background:#071019;border:1px solid #26384a;border-radius:999px;overflow:hidden;margin:14px 0 7px}.intent-track i{display:block;height:100%;background:linear-gradient(90deg,var(--purple),var(--a));border-radius:999px}.intent-meta{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;width:100%;margin-top:10px}.intent-k{background:#0b151e;border:1px solid #1d2c3b;border-radius:10px;padding:9px;text-align:center}.intent-k small{display:block;color:var(--muted);font-size:8px;text-transform:uppercase}.intent-k b{font-size:12px}.intent-side{display:flex;flex-direction:column;gap:10px}.intent-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.intent-flow div{background:#09121a;border:1px solid #1d2b39;border-radius:10px;padding:9px;font-size:9px;color:#aebdcb;text-align:center}.intent-flow b{display:block;color:var(--txt);font-size:10px;margin-bottom:3px}.intent-history .hist{grid-template-columns:68px 110px 110px 1fr 72px}
.goal-panel{display:grid;grid-template-columns:260px 1fr;gap:14px}.goal-main{background:radial-gradient(circle at 50% 15%,rgba(88,218,196,.13),transparent 58%),#09121a;border:1px solid rgba(88,218,196,.32);border-radius:14px;padding:18px}.goal-title{font-size:26px;font-weight:900;letter-spacing:.06em}.goal-status{font-size:10px;color:var(--a);text-transform:uppercase;letter-spacing:.12em;margin-bottom:7px}.goal-track{height:10px;background:#071019;border:1px solid #26384a;border-radius:999px;overflow:hidden;margin:14px 0 7px}.goal-track i{display:block;height:100%;background:linear-gradient(90deg,var(--blue),var(--a));border-radius:999px}.goal-kpis{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:12px}.goal-k{background:#0b151e;border:1px solid #1d2c3b;border-radius:10px;padding:9px}.goal-k small{display:block;color:var(--muted);font-size:8px;text-transform:uppercase}.goal-k b{font-size:12px}.goal-sequence{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}.goal-step{display:inline-flex;gap:6px;align-items:center;background:#09121a;border:1px solid #24374a;border-radius:999px;padding:7px 10px;font-size:10px}.goal-step.ok{border-color:rgba(87,219,145,.45)}.goal-step.fail{border-color:rgba(255,114,114,.45)}.goal-step .n{color:var(--muted)}.goal-history .hist{grid-template-columns:68px 100px 110px 1fr 72px}
.personality-panel{display:grid;grid-template-columns:repeat(5,1fr);gap:9px}.trait{background:#09121a;border:1px solid #1d2c3b;border-radius:12px;padding:11px}.trait-head{display:flex;justify-content:space-between;gap:8px;align-items:center}.trait-head b{font-size:10px}.trait-head span{font-size:14px;font-weight:900}.trait-track{height:8px;background:#071019;border:1px solid #26384a;border-radius:999px;overflow:hidden;margin:9px 0 6px;position:relative}.trait-track:after{content:"";position:absolute;left:50%;top:0;bottom:0;width:1px;background:#53677b}.trait-track i{display:block;height:100%;background:linear-gradient(90deg,var(--purple),var(--a));border-radius:999px}.trait small{color:var(--muted);font-size:8px}.personality-meta{display:grid;grid-template-columns:1fr 2fr;gap:9px;margin-top:10px}.personality-meta>div{background:#09121a;border:1px solid #1d2c3b;border-radius:11px;padding:10px;font-size:10px}.personality-cues{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}.personality-cue{border:1px solid #29465c;background:#07131d;border-radius:999px;padding:5px 8px;font-size:9px;color:#a9bdcf}
@media(max-width:900px){.intent-panel,.goal-panel{grid-template-columns:1fr}.intent-flow{grid-template-columns:1fr 1fr}.personality-panel{grid-template-columns:1fr 1fr}.personality-meta{grid-template-columns:1fr}}
.help{display:inline-flex;width:16px;height:16px;align-items:center;justify-content:center;border:1px solid #385069;border-radius:50%;color:#8fa6ba;font-size:9px;cursor:help;position:relative;vertical-align:middle}.help:hover{color:var(--a);border-color:var(--a)}.help:hover:after{content:attr(data-tip);position:absolute;z-index:20;left:50%;top:22px;transform:translateX(-50%);width:270px;background:#05090e;border:1px solid #31465c;border-radius:9px;padding:9px;color:#d5e3f0;font:10px/1.45 Inter,system-ui;box-shadow:0 12px 35px rgba(0,0,0,.45);pointer-events:none}
.live{display:inline-flex;align-items:center;gap:6px}.live:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--good);box-shadow:0 0 10px rgba(87,219,145,.65)}.bad{color:var(--bad)}.good{color:var(--good)}.warn{color:var(--warn)}.muted{color:var(--muted)}.foot{margin-top:12px;text-align:right;color:#5e6e7d;font-size:10px}
@media(max-width:1150px){.hero{grid-template-columns:repeat(3,1fr)}.grid{grid-template-columns:1fr}.candidate{grid-template-columns:120px 65px 1fr 90px 90px}.candidate .optional{display:none}}
@media(max-width:700px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.hero{grid-template-columns:1fr 1fr}.pipeline{grid-template-columns:1fr 1fr}.stage:after{display:none}.candidate{grid-template-columns:1fr 64px}.candidate .metric,.candidate .reason{grid-column:1/-1}.history .hist{grid-template-columns:55px 80px 1fr}.hist .pred,.hist .rep{display:none}}
</style>
</head>
<body><main>
<div class="top">
 <div class="brand"><div class="logo">🪰</div><div><h1>One Brain 25 / Autonomia 24E</h1><div class="sub">Jeden arbiter dla TEXT • REACT • TTS • JOIN • MOVE • EXPLORE • NOOP</div></div></div>
 <div class="nav"><a href="/">🏠 Przegląd</a><a class="active" href="/autonomy">🧭 Autonomia</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a href="/associations">🗣 Mowa</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/logout">Wyloguj</a></div>
</div>

<section class="hero">
 <div class="hero-card"><small>One Brain <span class="help" data-tip="Stage 25: TEXT, REACT, TTS i autonomia korzystają ze wspólnego predicted-reward + connectome arbitration.">?</span></small><strong id="one-brain-enabled">—</strong></div>
 <div class="hero-card"><small>24D loop <span class="help" data-tip="Czy zunifikowana pętla autonomii jest aktywna. Przy OFF działa ścieżka legacy.">?</span></small><strong id="enabled">—</strong></div>
 <div class="hero-card"><small>Final winner <span class="help" data-tip="Akcja wybrana przez końcowe action_competition po propagacji bodźców predicted reward przez connectome.">?</span></small><strong id="winner">—</strong></div>
 <div class="hero-card"><small>Wykonanie <span class="help" data-tip="Czy zwycięska akcja została technicznie wykonana. NOOP jest poprawnym wykonaniem bez efektu na Discordzie.">?</span></small><strong id="execution">—</strong></div>
 <div class="hero-card"><small>Predicted reward <span class="help" data-tip="Oczekiwana nagroda finalnego winnera. Nie wybiera akcji bezpośrednio — staje się signed sensory cue.">?</span></small><strong id="predicted">—</strong></div>
 <div class="hero-card"><small>Odświeżono</small><strong id="age" class="live">—</strong></div>
</section>

<section class="pipeline">
 <div class="stage"><b>24A • Drives</b><span>social_need, curiosity, exploration, caution i boredom pobudzają neuronalne internal-state attractors.</span></div>
 <div class="stage"><b>24B • Candidates</b><span>Do konkurencji trafiają tylko akcje technicznie możliwe. NOOP/STAY pozostaje zawsze.</span></div>
 <div class="stage"><b>24C • Reward model</b><span>Reward EMA + episodic context przewidują wynik i confidence bez ręcznego bonusu akcji.</span></div>
 <div class="stage"><b>24D • Neural winner</b><span>Signed prediction → sensory neurons → FAFB propagation → action_competition → executor.</span></div>
 <div class="stage"><b>25 • One Brain</b><span>TEXT, REACT, TTS i autonomia używają tego samego arbitra oraz wspólnej historii decyzji.</span></div>
 <div class="stage"><b>32 • Persistent intent</b><span>Poprzedni autonomiczny winner może wrócić jako słabnący sensory cue. Nie nadpisuje finalnej konkurencji FAFB.</span></div>
 <div class="stage"><b>33 • Multi-step goal</b><span>Cel SOCIAL / NOVELTY / SAFETY / REST nadaje kierunek wielu kolejnym tickom, ale każdy krok musi ponownie wygrać w FAFB.</span></div>
 <div class="stage"><b>35 • Personality</b><span>Trwały temperament uczy się z outcome'ów i zachowań, a potem wraca tylko przez internal-state sensory cues.</span></div>
</section>

<div class="guild-tabs" id="guild-tabs"></div>

<section class="grid">
 <div class="card">
  <h2>Kandydaci tej decyzji <span class="help" data-tip="Porównanie wszystkich akcji dopuszczonych do bieżącej konkurencji. Feasibility jest twardym ograniczeniem technicznym; reszta pochodzi ze stanu mózgu i nauki.">?</span></h2>
  <div class="candidates" id="candidates"><div class="muted">Czekam na pierwszy tick 24D…</div></div>
 </div>
 <div class="card">
  <h2>Finalna decyzja</h2>
  <div class="summary-grid">
   <div class="summary"><small>Winner</small><strong id="decision-action">—</strong></div>
   <div class="summary"><small>Runner-up</small><strong id="runner-up">—</strong></div>
   <div class="summary"><small>Margin <span class="help" data-tip="Różnica wyniku finalnej konkurencji. Przy tie-break może być 0 mimo wybranego winnera.">?</span></small><strong id="margin">—</strong></div>
   <div class="summary"><small>Tie-break</small><strong id="tie-break">—</strong></div>
   <div class="summary"><small>Prediction source</small><strong id="prediction-source">—</strong></div>
   <div class="summary"><small>Foresight winner <span class="help" data-tip="Diagnostyczny ranking kontrfaktycznej symulacji. Nie jest finalnym wyborem — finalny winner nadal pochodzi z FAFB competition.">?</span></small><strong id="foresight-winner">—</strong></div>
   <div class="summary"><small>Foresight margin</small><strong id="foresight-margin">—</strong></div>
   <div class="summary"><small>Active intent <span class="help" data-tip="Stage 32: poprzedni autonomiczny winner trzymany jako wygasająca pamięć. Przy kolejnym ticku wraca tylko przez sensory cue i nadal musi wygrać w FAFB.">?</span></small><strong id="active-intent">—</strong></div>
   <div class="summary"><small>Intent strength</small><strong id="intent-strength">—</strong></div>
   <div class="summary"><small>NOOP retry <span class="help" data-tip="Jeżeli autonomia najpierw wybrała NOOP mimo wyraźnej niezaspokojonej potrzeby, One Brain może ponownie podać tę potrzebę do internal-state attractorów i jeszcze raz wykonać normalną konkurencję.">?</span></small><strong id="noop-retry">—</strong></div>
   <div class="summary"><small>External effect</small><strong id="external">—</strong></div>
  </div>
  <div class="cue-list" id="cues"></div>
 </div>

 <div class="card span2">
  <h2>Current Intent / Stage 32 <span class="help" data-tip="Aktywny zamiar jest pamięcią poprzedniego neuronalnego winnera. Nie wykonuje akcji sam — wraca jako sensory cue do FAFB.">?</span></h2>
  <div class="intent-panel">
   <div class="intent-main">
    <div class="intent-state" id="intent-event">BRAK INTENCJI</div>
    <div class="intent-action" id="intent-action-big">—</div>
    <div class="intent-track"><i id="intent-bar" style="width:0%"></i></div>
    <div class="muted" id="intent-strength-label">strength 0.000</div>
    <div class="intent-meta">
     <div class="intent-k"><small>Wiek</small><b id="intent-age">0 s</b></div>
     <div class="intent-k"><small>Pozostało</small><b id="intent-remaining">0 s</b></div>
     <div class="intent-k"><small>Sensory cue</small><b id="intent-signal">0.000</b></div>
    </div>
   </div>
   <div class="intent-side">
    <div class="intent-flow">
     <div><b>1 • Winner</b><span id="intent-flow-winner">—</span></div>
     <div><b>2 • Intent memory</b><span id="intent-flow-memory">—</span></div>
     <div><b>3 • Sensory cue</b><span id="intent-flow-cue">—</span></div>
     <div><b>4 • Fresh FAFB</b><span id="intent-flow-result">—</span></div>
    </div>
    <div class="history intent-history" id="intent-history"><div class="muted">Brak historii intencji.</div></div>
   </div>
  </div>
 </div>

 <div class="card span2">
  <h2>Active Goal / Stage 33 <span class="help" data-tip="Cel reprezentuje potrzebę do realnego zmniejszenia. Kolejne akcje nie są z góry wymuszone: Stage 31 ocenia ich możliwy relief, a wskazówka goal wraca do FAFB jako sensory cue.">?</span></h2>
  <div class="goal-panel">
   <div class="goal-main">
    <div class="goal-status" id="goal-status">BRAK CELU</div>
    <div class="goal-title" id="goal-title">—</div>
    <div class="goal-track"><i id="goal-bar" style="width:0%"></i></div>
    <div class="muted" id="goal-progress-label">progress 0%</div>
    <div class="goal-kpis">
     <div class="goal-k"><small>Urgency</small><b id="goal-urgency">0 → 0</b></div>
     <div class="goal-k"><small>Kroki</small><b id="goal-steps-count">0 / 0</b></div>
     <div class="goal-k"><small>Wiek</small><b id="goal-age">0 s</b></div>
     <div class="goal-k"><small>Failed</small><b id="goal-failed">0</b></div>
    </div>
   </div>
   <div>
    <div class="muted" style="font-size:10px;margin-bottom:8px">SEKWENCJA WYKONANYCH KROKÓW</div>
    <div class="goal-sequence" id="goal-sequence"><span class="muted">Brak kroków.</span></div>
    <div class="history goal-history" id="goal-history"><div class="muted">Brak historii celu.</div></div>
   </div>
  </div>
 </div>

 <div class="card span2">
  <h2>Emergent Personality / Stage 35 <span class="help" data-tip="To nie są ręcznie ustawione bonusy. Każda cecha startuje neutralnie przy 0.50, uczy się wolnym EMA z doświadczeń i działa dopiero przez attractory FAFB.">?</span></h2>
  <div class="personality-panel" id="personality-traits"><div class="muted">Czekam na dane personality.</div></div>
  <div class="personality-meta">
   <div><small class="muted">DOMINUJĄCA CECHA</small><br><strong id="personality-dominant">BRAK • NEUTRAL</strong></div>
   <div><small class="muted">OSTATNIA ZMIANA</small><br><strong id="personality-last">Brak doświadczeń.</strong><div class="personality-cues" id="personality-cues"></div></div>
  </div>
 </div>

 <div class="card span2">
  <h2>One Brain timeline <span class="help" data-tip="Wspólna historia decyzji z różnych modalności. kind pokazuje, czy bodziec pochodził z TEXT, VOICE_TTS czy AUTONOMY.">?</span></h2>
  <div class="history" id="one-brain-history"><div class="muted">Brak historii Stage 25.</div></div>
 </div>
 <div class="card span2">
  <h2>Historia autonomii <span class="help" data-tip="Kompaktowy zapis ostatnich decyzji 24D. Powtarzające się NOOP-y są scalane i pokazują licznik ×N.">?</span></h2>
  <div class="history" id="history"><div class="muted">Brak historii.</div></div>
 </div>
</section>
<div class="foot">Stage 35 Emergent Personality + Stage 33 Multi-step Goals + Stage 32 Persistent Intent + Stage 31 Foresight + Stage 30 Motivation + Stage 25 One Brain • dane z /api/state • odświeżanie LIVE_REFRESH_MS ms</div>

<script>
const LIVE_REFRESH_MS=250;
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[m]));
let selectedGuildId=null,lastPayload=null;

function pct(v){return Math.max(0,Math.min(100,Number(v||0)*100))}
function signedWidth(v){return Math.max(0,Math.min(100,Math.abs(Number(v||0))*100))}
function displayAction(a){return String(a||"stay")==="stay"?"NOOP":String(a||"—").toUpperCase()}
function fmt(v,d=3){return Number(v||0).toFixed(d)}
function guildRow(auto){
 const rows=Array.isArray(auto?.guilds)?auto.guilds:[];
 if(!rows.length)return null;
 let row=rows.find(x=>Number(x.guild_id)===Number(selectedGuildId));
 if(!row){
   const selected=Number(auto?.selected_guild_id);
   row=rows.find(x=>Number(x.guild_id)===selected)||rows[0];
   selectedGuildId=Number(row.guild_id);
 }
 return row;
}
function renderTabs(auto){
 const root=$("guild-tabs"),rows=Array.isArray(auto?.guilds)?auto.guilds:[];
 if(!rows.length){root.innerHTML="";return}
 root.innerHTML=rows.map(g=>'<button class="guild-tab '+(Number(g.guild_id)===Number(selectedGuildId)?"active":"")+'" onclick="selectedGuildId='+Number(g.guild_id)+';render(lastPayload)">'+esc(g.guild||g.guild_id)+'</button>').join("");
}
function renderCandidates(row,decision){
 const root=$("candidates"),set=row?.candidate_set||{},rows=set.rows||{};
 const names=["stay","speak","voice_join","voice_move","explore"].filter(name=>rows[name]);
 if(!names.length){root.innerHTML='<div class="muted">Brak kandydatów.</div>';return}
 const winner=String(decision?.action||"");
 root.innerHTML=names.map(name=>{
   const x=rows[name]||{},pred=Number(x.predicted_reward||0),conf=Number(x.prediction_confidence||0),score=Number(x.effective_score||0);
   const sim=x.foresight||{},relief=Number(sim.state_relief||0),risk=Number(sim.risk||0),forecast=Number(sim.forecast_value||0),simConf=Number(sim.simulation_confidence||0);
   const driveChanges=Object.entries(sim.drive_changes||{}).filter(([k,v])=>Number(v.delta||0)<-0.0001).slice(0,3).map(([k,v])=>k+" "+(Number(v.delta||0)*100).toFixed(0)+"pp").join(" • ");
   const predCls=pred>0?"pos":pred<0?"neg":"";
   return '<div class="candidate '+(name===winner?"winner ":"")+(x.feasible===false?"disabled":"")+'">'+
     '<div><div class="action-name">'+(name===winner?'▶ ':'')+esc(displayAction(name))+'</div><div style="margin-top:5px">'+(name===winner?'<span class="pill win">WINNER</span> ':'')+'<span class="pill '+(x.feasible!==false?"ok":"no")+'">'+(x.feasible!==false?"FEASIBLE":"BLOCKED")+'</span></div></div>'+
     '<div class="metric"><small>score <span class="help" data-tip="Effective action score przed finalnym predicted-reward sensory pass.">?</span></small><b>'+fmt(score)+'</b><div class="bar"><i style="width:'+pct(score)+'%"></i></div></div>'+
     '<div class="reason">'+esc(x.technical_reason||"—")+'</div>'+
     '<div class="metric"><small>drive support</small><b>'+fmt(x.drive_support)+'</b><div class="bar"><i style="width:'+pct(x.drive_support)+'%"></i></div></div>'+
     '<div class="metric"><small>state support</small><b>'+fmt(x.state_support)+'</b><div class="bar"><i style="width:'+pct(x.state_support)+'%"></i></div></div>'+
     '<div class="metric optional"><small>predicted reward <span class="help" data-tip="Nauczona prognoza rewardu w zakresie około -1..+1.">?</span></small><b class="'+(pred>0?"good":pred<0?"bad":"")+'">'+(pred>=0?"+":"")+fmt(pred)+'</b><div class="bar reward"><i class="'+predCls+'" style="width:'+signedWidth(pred)+'%"></i></div></div>'+
     '<div class="metric optional"><small>confidence</small><b>'+fmt(conf,2)+'</b><div class="bar"><i style="width:'+pct(conf)+'%"></i></div></div>'+
     '<div class="metric optional"><small>foresight relief <span class="help" data-tip="Przewidywany spadek łącznej motivational urgency po hipotetycznym wykonaniu akcji. Symulacja nie zmienia live state.">?</span></small><b>'+fmt(relief,3)+'</b><div class="bar"><i style="width:'+pct(relief)+'%"></i></div></div>'+
     '<div class="metric optional"><small>foresight risk</small><b class="'+(risk>.55?"bad":"")+'">'+fmt(risk,3)+'</b><div class="bar"><i style="width:'+pct(risk)+'%"></i></div></div>'+
     '<div class="metric optional"><small>forecast / sim conf</small><b class="'+(forecast>0?"good":forecast<0?"bad":"")+'">'+(forecast>=0?"+":"")+fmt(forecast)+' / '+fmt(simConf,2)+'</b></div>'+
     '<div class="reason optional">predicted drives: '+esc(driveChanges||"brak istotnego reliefu")+'</div>'+
   '</div>';
 }).join("");
}
function renderCues(decision){
 const rewardCues=decision?.prediction_cues||{},foresightCues=decision?.foresight_cues||{},goalCues=decision?.goal_cues||{},intentCue=decision?.intention_cue||{},personality=decision?.personality_cues||{},root=$("cues");
 const entries=[
   ...Object.entries(rewardCues).map(([action,x])=>({action,x,kind:"reward",signal:x.magnitude})),
   ...Object.entries(foresightCues).map(([action,x])=>({action,x,kind:"foresight",signal:x.magnitude})),
   ...Object.entries(goalCues).map(([action,x])=>({action,x:x.cue||{},kind:"goal",signal:x.signal,target:x.target})),
   ...(intentCue.active?[{action:intentCue.action,x:intentCue.cue||{},kind:"intent",signal:intentCue.signal}]:[]),
   ...(personality.injected||[]).map(x=>({action:x.state,x:x.cue||{},kind:"personality",signal:x.magnitude,target:x.trait}))
 ];
 if(!entries.length){root.innerHTML='<div class="muted" style="font-size:10px;margin-top:10px">Brak sensory guidance w tej decyzji.</div>';return}
 root.innerHTML='<div class="muted" style="font-size:10px">Reward + foresight + intent + goal + personality → sensory paths → FAFB</div>'+entries.map(row=>
   '<div class="cue"><b>'+esc(displayAction(row.action))+'</b><span>'+esc(row.kind.toUpperCase())+(row.target?' • '+esc(String(row.target).toUpperCase()):'')+' • '+esc(row.x.mode||"sensory")+' • '+Number(row.x.neurons||row.x.entry_neurons||0)+' neuronów</span><strong>'+fmt(row.signal||0)+'</strong></div>'
 ).join("");
}
function renderIntent(intent,decision){
 const state=intent||{},cue=decision?.intention_cue||{},hist=Array.isArray(state.history)?state.history.slice().reverse():[];
 const active=!!state.active,strength=Number(state.strength||0),event=state.last_event||{};
 $("intent-event").textContent=active?String(event.event||"active").replaceAll("-"," ").toUpperCase():"BRAK INTENCJI";
 $("intent-event").className="intent-state "+(active?"good":"muted");
 $("intent-action-big").textContent=active?displayAction(state.action):"—";
 $("intent-bar").style.width=pct(strength)+"%";
 $("intent-strength-label").textContent="strength "+fmt(strength,3);
 $("intent-age").textContent=Math.round(Number(state.age_seconds||0))+" s";
 $("intent-remaining").textContent=Math.round(Number(state.remaining_seconds||0))+" s";
 $("intent-signal").textContent=fmt(cue.signal||0,3);
 $("intent-flow-winner").textContent=displayAction(event.winner||event.action);
 $("intent-flow-memory").textContent=active?(displayAction(state.action)+" • "+fmt(strength,2)):"BRAK";
 $("intent-flow-cue").textContent=cue.active?(displayAction(cue.action)+" +"+fmt(cue.signal,3)):"BRAK";
 $("intent-flow-result").textContent=displayAction(decision?.action);
 const root=$("intent-history");
 if(!hist.length){root.innerHTML='<div class="muted">Brak historii intencji.</div>';return}
 root.innerHTML=hist.slice(0,24).map(x=>{
   const t=new Date(Number(x.time||0)*1000).toLocaleTimeString("pl-PL");
   const ev=String(x.event||"—").replaceAll("-"," ").toUpperCase();
   const winner=x.winner?displayAction(x.winner):"—";
   const action=x.action?displayAction(x.action):"—";
   return '<div class="hist">'+
     '<span class="time">'+esc(t)+'</span>'+
     '<span class="act">'+esc(ev)+'</span>'+
     '<span>'+esc(action)+(winner!=="—"&&winner!==action?' ← '+esc(winner):'')+'</span>'+
     '<span class="detail" title="'+esc(x.reason||"")+'">'+esc(x.reason||"—")+'</span>'+
     '<span class="rep">'+fmt(x.strength||0,2)+'</span>'+
   '</div>';
 }).join("");
}

function renderGoal(goal){
 const g=goal||{},active=!!g.active,progress=Number(g.progress||0),steps=Array.isArray(g.steps)?g.steps:[],hist=Array.isArray(g.history)?g.history.slice().reverse():[],last=g.last_event||{};
 $("goal-status").textContent=active?"ACTIVE • "+String(last.event||"tracking").replaceAll("-"," ").toUpperCase():(String(last.event||"BRAK CELU").replaceAll("-"," ").toUpperCase());
 $("goal-status").className="goal-status "+(active?"good":last.event==="completed"?"good":last.event==="abandoned"?"warn":"muted");
 $("goal-title").textContent=active?String(g.motivation||"—").toUpperCase():"—";
 $("goal-bar").style.width=pct(progress)+"%";
 $("goal-progress-label").textContent="progress "+Math.round(progress*100)+"% • sukces przy "+Math.round(Number(g.success_progress||0)*100)+"%";
 $("goal-urgency").textContent=fmt(g.baseline_urgency,2)+" → "+fmt(g.current_urgency,2);
 $("goal-steps-count").textContent=Number(g.step_count||0)+" / "+Number(g.max_steps||0);
 $("goal-age").textContent=Math.round(Number(g.age_seconds||0))+" s";
 $("goal-failed").textContent=String(Number(g.failed_steps||0));
 $("goal-sequence").innerHTML=steps.length?steps.map(x=>
   '<span class="goal-step '+(x.executed&&x.success?"ok":"fail")+'"><span class="n">#'+Number(x.index||0)+'</span><b>'+esc(displayAction(x.action))+'</b><span>'+(x.executed&&x.success?"✓":"×")+'</span></span>'
 ).join(""):'<span class="muted">Brak wykonanych kroków.</span>';
 const root=$("goal-history");
 if(!hist.length){root.innerHTML='<div class="muted">Brak historii celu.</div>';return}
 root.innerHTML=hist.slice(0,24).map(x=>{
   const t=new Date(Number(x.time||0)*1000).toLocaleTimeString("pl-PL");
   return '<div class="hist">'+
     '<span class="time">'+esc(t)+'</span>'+
     '<span class="act">'+esc(String(x.event||"—").replaceAll("-"," ").toUpperCase())+'</span>'+
     '<span>'+esc(String(x.motivation||"—").toUpperCase())+(x.action?' • '+esc(displayAction(x.action)):'')+'</span>'+
     '<span class="detail" title="'+esc(x.reason||"")+'">'+esc(x.reason||"—")+'</span>'+
     '<span class="rep">'+Math.round(Number(x.progress||0)*100)+'%</span>'+
   '</div>';
 }).join("");
}

function renderPersonality(personality,decision){
 const p=personality||{},traits=p.traits||{},order=["sociability","curiosity","caution","persistence","expressiveness"];
 const labels={sociability:"SOCIABILITY",curiosity:"CURIOSITY",caution:"CAUTION",persistence:"PERSISTENCE",expressiveness:"EXPRESSIVENESS"};
 const root=$("personality-traits");
 root.innerHTML=order.map(name=>{
   const x=traits[name]||{},value=Number(x.value??0.5),conf=Number(x.confidence||0),obs=Number(x.observations||0),expr=Number(x.expressed||0);
   return '<div class="trait"><div class="trait-head"><b>'+labels[name]+'</b><span>'+value.toFixed(3)+'</span></div>'+
     '<div class="trait-track"><i style="width:'+pct(value)+'%"></i></div>'+
     '<small>confidence '+Math.round(conf*100)+'% • '+obs+' obs • neural '+expr.toFixed(3)+'</small></div>';
 }).join("");
 $("personality-dominant").textContent=p.dominant?String(p.dominant).toUpperCase()+" • "+Number(p.dominant_expression||0).toFixed(3):"BRAK • NEUTRAL";
 const ev=p.last_event||{};
 $("personality-last").textContent=ev.event==="startup"||!ev.trait?String(ev.reason||"Brak doświadczeń."):(String(ev.trait).toUpperCase()+" "+(Number(ev.delta||0)>=0?"+":"")+Number(ev.delta||0).toFixed(4)+" • "+String(ev.reason||""));
 const injected=decision?.personality_cues?.injected||[];
 $("personality-cues").innerHTML=injected.length?injected.map(x=>'<span class="personality-cue">'+esc(String(x.trait||"").toUpperCase())+' → '+esc(String(x.state||"").toUpperCase())+' +'+fmt(x.magnitude,3)+'</span>').join(""):'<span class="muted">Brak aktywnego personality cue w tym ticku.</span>'
}

function renderOneBrainHistory(items){
 const root=$("one-brain-history"),rows=Array.isArray(items)?items.slice().reverse():[];
 if(!rows.length){root.innerHTML='<div class="muted">Brak decyzji One Brain — timeline pojawi się po TEXT, TTS albo ticku autonomii.</div>';return}
 root.innerHTML=rows.map(x=>{
   const t=new Date(Number(x.time||0)*1000).toLocaleTimeString("pl-PL");
   const pred=Number(x.predicted_reward||0);
   return '<div class="hist '+(x.external_effect?"external":"")+'">'+
     '<span class="time">'+esc(t)+'</span>'+
     '<span class="act '+(x.success?"good":"")+'">'+esc(displayAction(x.action))+'</span>'+
     '<span class="detail" title="'+esc(x.detail||"")+'"><b>'+esc(String(x.kind||"—").toUpperCase())+'</b> • '+esc(x.guild||"—")+' • '+esc(x.detail||"—")+'</span>'+
     '<span class="pred '+(pred>0?"good":pred<0?"bad":"")+'">'+(pred>=0?"+":"")+fmt(pred,2)+'</span>'+
     '<span class="rep">'+(x.executed?"EXEC":"SKIP")+'</span>'+
   '</div>';
 }).join("");
}

function renderHistory(items){
 const root=$("history"),rows=Array.isArray(items)?items.slice().reverse():[];
 if(!rows.length){root.innerHTML='<div class="muted">Brak historii — pojawi się po pierwszej decyzji 24D.</div>';return}
 root.innerHTML=rows.map(x=>{
   const t=new Date(Number(x.time||0)*1000).toLocaleTimeString("pl-PL");
   const pred=Number(x.predicted_reward||0);
   return '<div class="hist '+(x.external_effect?"external":"")+'">'+
     '<span class="time">'+esc(t)+'</span>'+
     '<span class="act '+(x.success?"good":"")+'">'+esc(displayAction(x.action))+'</span>'+
     '<span class="detail" title="'+esc(x.detail||"")+'">'+esc(x.guild||"—")+' • '+esc(x.detail||"—")+'</span>'+
     '<span class="pred '+(pred>0?"good":pred<0?"bad":"")+'">'+(pred>=0?"+":"")+fmt(pred,2)+'</span>'+
     '<span class="rep">'+(Number(x.repeat_count||1)>1?"×"+Number(x.repeat_count):"")+'</span>'+
   '</div>';
 }).join("");
}
function render(payload){
 lastPayload=payload||{};
 const auto=payload?.autonomous_candidates||{},decision=auto.decision||{},exec=auto.last_execution||{},row=guildRow(auto);
 const ob=payload?.one_brain||{};
 const rowSelected=!!row&&Number(row.guild_id)===Number(auto.selected_guild_id);
 const rowDecision=rowSelected?(row?.candidate_set?.autonomous_decision||decision):{};
 renderTabs(auto);
 $("one-brain-enabled").textContent=ob.enabled?"ON • STAGE 25":"OFF • LEGACY";
 $("one-brain-enabled").className=ob.enabled?"good":"warn";
 $("enabled").textContent=auto.enabled?"ON • 24D":"OFF • legacy";
 $("enabled").className=auto.enabled?"good":"warn";
 $("winner").textContent=displayAction(decision.action);
 $("execution").textContent=exec.executed?(exec.external_effect?"WYKONANO • DISCORD":"WYKONANO • INTERNAL"):(decision.action?"NIEWYKONANO":"—");
 $("execution").className=exec.executed?"good":decision.action?"bad":"";
 const pred=Number(decision.predicted_reward||0);
 $("predicted").textContent=(pred>=0?"+":"")+fmt(pred)+" • conf "+fmt(decision.prediction_confidence,2);
 $("predicted").className=pred>0?"good":pred<0?"bad":"";
 const age=Math.max(0,Date.now()/1000-Number(auto.updated_at||0));
 $("age").textContent=auto.updated_at?age.toFixed(1)+" s":"—";
 const comp=decision.competition||{};
 const rowComp=rowDecision.competition||{};
 $("decision-action").textContent=rowSelected?displayAction(rowDecision.action):"NIE WYBRANO W TYM TICKU";
 $("runner-up").textContent=rowSelected?displayAction(rowComp.runner_up):"—";
 $("margin").textContent=rowSelected?fmt(rowComp.margin):"—";
 $("tie-break").textContent=rowSelected?(rowComp.tie_break||"—"):"—";
 $("prediction-source").textContent=rowSelected?(rowDecision.prediction_source||"—"):"—";
 const foresight=rowDecision.foresight||row?.candidate_set?.foresight||{};
 $("foresight-winner").textContent=rowSelected?displayAction(foresight.winner):"—";
 $("foresight-margin").textContent=rowSelected?fmt(foresight.margin):"—";
 const intent=rowDecision.intention||payload?.intention_state||row?.candidate_set?.intention_state||{};
 $("active-intent").textContent=intent.active?displayAction(intent.action):"BRAK";
 $("active-intent").className=intent.active?"good":"";
 $("intent-strength").textContent=intent.active?(fmt(intent.strength,3)+" • "+Math.round(Number(intent.age_seconds||0))+" s"):"0.000";
 const retry=rowDecision.noop_reafference||{};
 $("noop-retry").textContent=rowSelected
   ?(retry.triggered
      ?("TAK • "+displayAction(retry.winner_before)+" → "+displayAction(retry.winner_after)+" • "+String(retry.candidate||"—").toUpperCase())
      :"NIE")
   :"—";
 $("noop-retry").className=rowSelected&&retry.triggered?(retry.winner_after==="stay"?"warn":"good"):"";
 $("external").textContent=rowSelected?(exec.external_effect?"TAK":"NIE"):"—";
 renderCandidates(row,rowDecision);
 renderCues(rowDecision);
 renderIntent(intent,rowDecision);
 renderGoal(rowDecision.goal||payload?.goal_state||row?.candidate_set?.goal_state||{});
 renderPersonality(rowDecision.personality||payload?.personality_state||row?.candidate_set?.personality_state||{},rowDecision);
 renderOneBrainHistory(payload?.one_brain_history||[]);
 renderHistory(payload?.autonomous_history||[]);
}
async function update(){
 try{
   const r=await fetch("/api/state",{cache:"no-store"});
   if(r.status===401){location="/login";return}
   if(!r.ok)throw new Error("HTTP "+r.status);
   render(await r.json());
 }catch(e){
   $("age").textContent="ROZŁĄCZONO";
   $("age").className="bad";
   console.error(e);
 }
}
setInterval(update,LIVE_REFRESH_MS);update();
</script>
</main></body></html>"""

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
.progress>div{height:100%;background:linear-gradient(90deg,var(--a),var(--blue));transition:width .18s linear}
.gpu-list{display:flex;flex-direction:column;gap:6px;margin-top:9px}.gpu-chip{display:grid;grid-template-columns:minmax(120px,1fr) auto auto;gap:10px;align-items:center;padding:8px 9px;border:1px solid #1e2d3d;background:#09121a;border-radius:10px;font-size:10px}.gpu-chip b{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.gpu-chip span{color:var(--muted);font-variant-numeric:tabular-nums}
.footer{text-align:right;color:#5e6e7d;font-size:11px;margin-top:12px}
@media(max-width:900px){.hero{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.span2{grid-column:auto}}
@media(max-width:560px){main{padding:12px}.top{align-items:flex-start;flex-direction:column}.hero{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}}
</style></head>
<body><main>
<div class="top">
  <div class="brand"><div class="logo">🪰</div><div><h1>Mucha Control Center</h1><div class="sub">Windows / VPS • Discord • Connectome • Chaser • Audio</div></div></div>
  <div class="nav"><a class="active" href="/">🏠 Przegląd</a><a href="/autonomy">🧭 Autonomia</a><a href="/details">📋 Szczegóły</a><a href="/connectome">🧬 Connectome</a><a href="/neuromap">🧠 Neuro-map</a><a href="/associations">🗣 Mowa</a><a href="/affinity">🤝 Affinity</a><a href="/config">⚙ Konfiguracja</a><a href="/api/state">JSON</a><a href="/logout">Wyloguj</a></div>
</div>

<section class="hero">
 <div class="hero-card"><small>Mucha</small><strong id="hero-mucha">łączenie…</strong></div>
 <div class="hero-card"><small>Chaser</small><strong id="hero-chaser">łączenie…</strong></div>
 <div class="hero-card"><small>Voice</small><strong id="hero-voice">—</strong></div>
 <div class="hero-card"><small>Następny pościg</small><strong id="hero-next">—</strong></div>
</section>

<section class="grid">
 <div class="card">
  <h2>🪰 Mucha Process / Service</h2>
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
  <h2>🖥 System — live</h2>
  <div class="kpis">
   <div class="k"><small>CPU system</small><strong id="sys-cpu">—</strong></div>
   <div class="k"><small>RAM system</small><strong id="vps-ram">—</strong></div>
   <div class="k"><small>Dysk</small><strong id="vps-disk">—</strong></div>
  </div>
  <div class="progress"><div id="cpu-bar" style="width:0"></div></div>
  <div class="progress"><div id="ram-bar" style="width:0"></div></div>
  <div class="progress"><div id="disk-bar" style="width:0"></div></div>
  <div class="kpis" style="margin-top:10px">
   <div class="k"><small>CPU Mucha</small><strong id="proc-cpu">—</strong></div>
   <div class="k"><small>RAM Mucha</small><strong id="proc-ram">—</strong></div>
   <div class="k"><small>Wątki Muchy</small><strong id="proc-threads">—</strong></div>
  </div>
  <div class="row"><span>Uptime systemu</span><strong id="vps-up">—</strong></div>
  <div class="row"><span>Load 1 / 5 / 15</span><strong id="vps-load">—</strong></div>
  <div class="row"><span>Wolny RAM</span><strong id="vps-free">—</strong></div>
  <div class="row"><span>GPU</span><strong id="gpu-main">—</strong></div>
  <div class="progress"><div id="gpu-bar" style="width:0"></div></div>
  <div class="row"><span>VRAM</span><strong id="gpu-vram">—</strong></div>
  <div class="progress"><div id="vram-bar" style="width:0"></div></div>
  <div class="row"><span>Temperatura GPU</span><strong id="gpu-temp">—</strong></div>
  <div class="gpu-list" id="gpu-list"></div>
  <div class="row"><span>Aktualizacja telemetryki</span><strong id="updated">—</strong></div>
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
  <div class="row"><span>Uczone synapsy</span><strong id="learned-synapses">—</strong></div>
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
const LIVE_REFRESH_MS=250;
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
function renderSystem(sys){
 sys=sys||{};const proc=sys.process||{},gpuState=sys.gpu||{},gpus=gpuState.gpus||[],gpu=gpus[0]||null;
 const cpu=Math.max(0,Number(sys.cpu_percent||0)),ram=Math.max(0,Number(sys.mem_percent||0)),disk=Math.max(0,Number(sys.disk_percent||0));
 $("sys-cpu").textContent=cpu.toFixed(1)+"% • "+Number(sys.cpu_count||0)+" CPU";
 $("vps-ram").textContent=ram.toFixed(1)+"% • "+fmtBytes(sys.mem_used)+" / "+fmtBytes(sys.mem_total);
 $("vps-disk").textContent=disk.toFixed(1)+"% • "+fmtBytes(sys.disk_used)+" / "+fmtBytes(sys.disk_total);
 $("cpu-bar").style.width=Math.min(100,cpu)+"%";$("ram-bar").style.width=Math.min(100,ram)+"%";$("disk-bar").style.width=Math.min(100,disk)+"%";
 $("proc-cpu").textContent=Number(proc.cpu_percent||0).toFixed(1)+"%";$("proc-ram").textContent=fmtBytes(proc.memory_bytes);$("proc-threads").textContent=String(Number(proc.threads||0));
 $("vps-up").textContent=dur(sys.uptime_seconds);$("vps-load").textContent=(sys.load||[]).length?(sys.load||[]).map(x=>Number(x).toFixed(2)).join(" / "):"—";
 $("vps-free").textContent=fmtBytes(sys.mem_available);
 if(gpu){
  $("gpu-main").textContent=(gpu.name||"GPU")+" • "+Number(gpu.utilization_percent||0).toFixed(0)+"%";
  $("gpu-vram").textContent=fmtBytes(gpu.memory_used_bytes)+" / "+fmtBytes(gpu.memory_total_bytes)+" • "+Number(gpu.memory_percent||0).toFixed(0)+"%";
  $("gpu-temp").textContent=Number(gpu.temperature_c||0).toFixed(0)+" °C";
  $("gpu-bar").style.width=Math.min(100,Number(gpu.utilization_percent||0))+"%";
  $("vram-bar").style.width=Math.min(100,Number(gpu.memory_percent||0))+"%";
  $("gpu-list").innerHTML=gpus.map(g=>'<div class="gpu-chip"><b>#'+Number(g.index||0)+' '+esc(g.name||"GPU")+'</b><span>GPU '+Number(g.utilization_percent||0).toFixed(0)+'%</span><span>VRAM '+Number(g.memory_percent||0).toFixed(0)+'%</span></div>').join("");
 }else{
  $("gpu-main").textContent="GPU unavailable";$("gpu-vram").textContent="—";$("gpu-temp").textContent="—";$("gpu-bar").style.width="0%";$("vram-bar").style.width="0%";
  $("gpu-list").innerHTML='<div class="gpu-chip"><b>Brak telemetryki GPU</b><span></span><span>'+esc(gpuState.error||"nvidia-smi unavailable")+'</span></div>';
 }
 $("updated").textContent=new Date().toLocaleTimeString("pl-PL",{hour12:false})+" • "+LIVE_REFRESH_MS+" ms";
}
function renderLive(s,sys){
 s=s||{};const diag=s.diag||{},a=s.audio_debug||{},stt=s.stt_debug||{};
 $("hero-voice").textContent=s.voice||"poza voice";
 $("connectome").textContent=nfmt(diag.neurons)+" / "+nfmt(diag.connections);$("backend").textContent=(diag.backend||"cpu").toUpperCase()+" • "+(diag.device||"CPU");
 $("last-event").textContent=s.last_event||"—";$("last-action").textContent=s.last_action||"—";
 $("audio-status").textContent=a.status||"—";$("audio-stage").textContent=a.stage||"—";$("audio-target").textContent=(a.guild||"—")+" / "+(a.channel||"—");
 $("audio-file").textContent=a.file||"—";$("audio-text").textContent=a.text||"—";$("stt-status").textContent=(stt.status||"—")+" • "+(stt.model||"—");$("stt-heard").textContent=stt.text?((stt.user||"ktoś")+": "+stt.text):"—";
 $("neurons").textContent=nfmt(diag.neurons);$("connections").textContent=nfmt(diag.connections);$("learned-synapses").textContent=nfmt(diag.learned_synapses||0);$("active-neurons").textContent=nfmt(diag.active_abs_gt_0_1);
 $("mean-a").textContent=Number(diag.mean_abs||0).toFixed(5);$("reward-trace").textContent=Number(diag.reward_trace||0).toFixed(4);$("ticks").textContent=nfmt(diag.ticks);
 renderActions(s.scores||{});renderSystem(sys);
}
function render(d){
 last=d;const s=d.snapshot||{},diag=s.diag||{},m=d.services?.mucha||{},ch=d.services?.chaser||{},sys=d.system||{},cs=d.chaser_status||{},cg=firstGuild(cs),a=s.audio_debug||{},stt=s.stt_debug||{};
 renderLive(s,sys);
 $("hero-mucha").innerHTML=serviceLabel(m);$("hero-chaser").innerHTML=serviceLabel(ch);
 $("hero-voice").textContent=s.voice||"poza voice";$("hero-next").textContent=nextText(cg.next_round_at);
 $("mucha-ram").textContent=fmtBytes(m.memory_bytes);$("mucha-up").textContent=dur(m.uptime_seconds);$("mucha-pid").textContent=m.pid||"—";$("mucha-state").innerHTML=serviceLabel(m);
 $("connectome").textContent=nfmt(diag.neurons)+" / "+nfmt(diag.connections);$("backend").textContent=(diag.backend||"cpu").toUpperCase()+" • "+(diag.device||"CPU");
 $("last-event").textContent=s.last_event||"—";$("last-action").textContent=s.last_action||"—";
 $("chaser-ram").textContent=fmtBytes(ch.memory_bytes);$("chaser-cycle").textContent=dur(cs.interval_seconds||0);$("chaser-duration").textContent=dur(cs.duration_seconds||0);
 $("chaser-state").innerHTML=serviceLabel(ch)+" • <span class='"+(cg.active_chase?"badc":"accent")+"'>"+esc(cg.state||"—")+"</span>";
 $("chaser-channel").textContent=cg.current_voice_channel||"poza voice";$("chaser-next").textContent=nextText(cg.next_round_at);$("chaser-event").textContent=cg.last_event||"—";
 $("audio-status").textContent=a.status||"—";$("audio-stage").textContent=a.stage||"—";$("audio-target").textContent=(a.guild||"—")+" / "+(a.channel||"—");
 $("audio-file").textContent=a.file||"—";$("audio-text").textContent=a.text||"—";
 $("stt-status").textContent=(stt.status||"—")+" • "+(stt.model||"—");
 $("stt-heard").textContent=stt.text?((stt.user||"ktoś")+": "+stt.text):"—";
 $("neurons").textContent=nfmt(diag.neurons);$("connections").textContent=nfmt(diag.connections);$("learned-synapses").textContent=nfmt(diag.learned_synapses||0);$("active-neurons").textContent=nfmt(diag.active_abs_gt_0_1);
 $("mean-a").textContent=Number(diag.mean_abs||0).toFixed(5);$("reward-trace").textContent=Number(diag.reward_trace||0).toFixed(4);$("ticks").textContent=nfmt(diag.ticks);
 renderActions(s.scores||{});
 $("mucha-logs").textContent=(d.logs?.mucha||[]).join("\n")||"brak logów";$("chaser-logs").textContent=(d.logs?.chaser||[]).join("\n")||"brak logów";
}
let liveBusy=false,slowBusy=false;
async function updateLive(){
 if(liveBusy)return;liveBusy=true;
 try{
  const [sr,yr]=await Promise.all([fetch("/api/state",{cache:"no-store"}),fetch("/api/system",{cache:"no-store"})]);
  if(sr.status===401||yr.status===401){location="/login";return}
  if(!sr.ok||!yr.ok)throw new Error("live HTTP "+sr.status+"/"+yr.status);
  const [s,sys]=await Promise.all([sr.json(),yr.json()]);renderLive(s,sys)
 }catch(e){console.error(e)}
 finally{liveBusy=false}
}
async function updateSlow(){
 if(slowBusy)return;slowBusy=true;
 try{const r=await fetch("/api/overview",{cache:"no-store"});if(r.status===401){location="/login";return}if(!r.ok)throw new Error("HTTP "+r.status);render(await r.json())}
 catch(e){console.error(e);$("hero-mucha").innerHTML='<span class="badc">BRAK POŁĄCZENIA</span>'}
 finally{slowBusy=false}
}
setInterval(updateLive,LIVE_REFRESH_MS);setInterval(updateSlow,2500);setInterval(()=>{if(last){const g=firstGuild(last.chaser_status||{});$("hero-next").textContent=nextText(g.next_round_at);$("chaser-next").textContent=nextText(g.next_round_at)}},1000);updateSlow();updateLive();
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
.top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:12px}
.brand{display:flex;align-items:center;gap:12px}.fly{font-size:32px}.title h1{font-size:23px;margin:0}.title p{margin:4px 0 0;color:var(--muted);font-size:13px}
.badges{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}.badge{border:1px solid var(--line);background:#0f161f;border-radius:999px;padding:7px 10px;font-size:12px}

/* Details UI v2 — guided diagnostics first, raw telemetry second. */
.guide-hero{grid-column:1/-1;background:
  radial-gradient(circle at 0 0,rgba(85,211,195,.10),transparent 34%),
  radial-gradient(circle at 100% 100%,rgba(110,168,254,.08),transparent 34%),
  rgba(14,21,30,.98);border:1px solid #294153;border-radius:20px;padding:16px;box-shadow:0 18px 50px rgba(0,0,0,.18)}
.guide-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:13px}
.guide-head h2{margin:0;font-size:15px;letter-spacing:-.01em}.guide-head p{margin:4px 0 0;color:var(--muted);font-size:11px;line-height:1.45}
.guide-mode{display:flex;gap:6px;background:#091018;border:1px solid #1c2b39;border-radius:10px;padding:4px;flex:0 0 auto}
.guide-mode button{border:0;background:transparent;color:#8fa0b0;border-radius:7px;padding:6px 9px;font-size:10px;font-weight:800;cursor:pointer}
.guide-mode button.active{background:#18303a;color:#8ff2e4}
.decision-flow{display:grid;grid-template-columns:1.25fr 24px 1.1fr 24px 1.05fr 24px 1.05fr 24px 1.2fr;gap:6px;align-items:stretch}
.decision-trace-card{background:linear-gradient(135deg,rgba(85,211,195,.055),rgba(110,168,254,.035));border-color:#294153}
.trace-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:11px}.trace-head h2{margin:0}.trace-meta{color:var(--muted);font-size:10px;text-align:right;line-height:1.45}
.trace-verdict{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(220px,.65fr);gap:9px;margin-bottom:10px}.trace-verdict>div{background:#0a1118;border:1px solid #1d2936;border-radius:12px;padding:11px}.trace-verdict small{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px}.trace-verdict strong{font-size:14px;line-height:1.35;word-break:break-word}.trace-reason{margin-top:5px;color:#9fb0c0;font-size:10px;line-height:1.45}
.trace-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px}.trace-stage{background:#081018;border:1px solid #192735;border-radius:11px;padding:9px;min-width:0}.trace-stage small{display:block;color:#7890a5;font-size:8px;text-transform:uppercase;letter-spacing:.09em;margin-bottom:5px}.trace-stage b{display:block;font-size:11px;line-height:1.35;word-break:break-word}.trace-stage em{display:block;color:#7f93a5;font-size:9px;font-style:normal;line-height:1.45;margin-top:5px}
.trace-chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:9px}.trace-chip{border:1px solid #213548;background:#09121a;border-radius:999px;padding:5px 7px;font-size:9px;color:#a9bac9}.trace-chip.good{border-color:#285747;color:#74e8ac}.trace-chip.warn{border-color:#5b4a29;color:#ffd277}.trace-chip.bad{border-color:#63363d;color:#ff9099}
.trace-empty{color:var(--muted);font-size:11px;padding:10px 0}
.trace-history-card{background:#0d151e;border-color:#273d50}.trace-history-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap}.trace-history-head h2{margin:0}.trace-history-controls{display:flex;gap:6px;flex-wrap:wrap}.trace-history-btn{border:1px solid #294153;background:#09121a;color:#9fb2c2;border-radius:8px;padding:6px 9px;font-size:9px;font-weight:800;cursor:pointer}.trace-history-btn:hover{border-color:#4a6b83;color:#e6f5ff}.trace-history-btn.active{background:#17303a;border-color:#3b8076;color:#a5fff0}.trace-history-list{display:flex;flex-direction:column;gap:6px;max-height:330px;overflow:auto}.trace-history-item{display:grid;grid-template-columns:82px 70px 145px minmax(0,1fr);gap:8px;align-items:center;width:100%;border:1px solid #1c2c3a;background:#081018;border-radius:10px;padding:8px 9px;color:#c9d8e4;text-align:left;cursor:pointer;font:10px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace}.trace-history-item:hover{border-color:#3b5a70;background:#0b151e}.trace-history-item.selected{border-color:var(--accent);background:#0e2225;box-shadow:inset 3px 0 0 var(--accent)}.trace-history-time{color:#75899b}.trace-history-kind{font-weight:800;text-transform:uppercase;color:#73b8e8}.trace-history-item[data-kind="text"] .trace-history-kind{color:#7be2bd}.trace-history-decision{font-weight:800;color:#edf6fc;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.trace-history-action{color:#93a6b6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.trace-history-empty{padding:12px;color:var(--muted);font-size:10px;border:1px dashed #223646;border-radius:10px}
.flow-step{min-width:0;background:#091119;border:1px solid #1d2d3b;border-radius:13px;padding:11px}
.flow-step small{display:flex;align-items:center;gap:6px;color:#758697;font-size:9px;text-transform:uppercase;letter-spacing:.09em;margin-bottom:6px}
.flow-step strong{display:block;font-size:13px;line-height:1.35;word-break:break-word}.flow-step em{display:block;margin-top:5px;color:#7f91a2;font-size:9px;font-style:normal;line-height:1.4}
.flow-arrow{display:grid;place-items:center;color:#416071;font-size:17px}
.flow-step.attention{border-color:#315064}.flow-step.brain{border-color:#31465f}.flow-step.decision{border-color:#3a4f4b}.flow-step.action{border-color:#31584f}
.panel-explainer{grid-column:1/-1;display:flex;align-items:flex-start;gap:10px;background:#0a1219;border:1px dashed #294052;border-radius:13px;padding:10px 12px;color:#9cadbc;font-size:10px;line-height:1.5}
.panel-explainer b{color:#cfdee9}.panel-explainer .help-dot{margin-top:1px}
.card{transition:border-color .18s ease,background .18s ease,transform .18s ease}.card:hover{border-color:#30475d}
.card>h2{display:flex;align-items:center;gap:7px}
.advanced-card{opacity:.88}.guided-simple .advanced-card:not(.is-collapsible){display:none}
.help-dot{display:inline-grid;place-items:center;width:16px;height:16px;border-radius:50%;border:1px solid #3a556c;background:#0a131b;color:#8fded3;font:800 9px/1 ui-monospace,SFMono-Regular,Consolas,monospace;cursor:help;vertical-align:middle;flex:0 0 auto;outline:0}
.help-dot:hover,.help-dot:focus{border-color:var(--accent);color:#d7fff9;background:#11252a}
.help-tooltip{position:fixed;z-index:9999;width:min(360px,calc(100vw - 24px));background:#071019;border:1px solid #355168;border-radius:13px;padding:12px 13px;box-shadow:0 18px 55px rgba(0,0,0,.55);pointer-events:none;opacity:0;transform:translateY(4px);transition:opacity .12s ease,transform .12s ease}
.help-tooltip.show{opacity:1;transform:none}.help-tooltip .tt-title{font-weight:800;color:#eaf5fc;font-size:12px;margin-bottom:5px}.help-tooltip .tt-body{color:#a9b9c7;font-size:10px;line-height:1.5}.help-tooltip .tt-read{margin-top:7px;padding-top:7px;border-top:1px solid #1d2d3a;color:#77daca;font-size:10px;line-height:1.45}
.metric>span:first-child,.kpi small,.voice-kpi small,.voice-box h3,.attention-panel h3,th{display:flex;align-items:center;gap:5px}
.system-strip{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:8px;margin:0 0 12px}.system-live-card{background:#0b1219;border:1px solid #203142;border-radius:12px;padding:9px 10px;min-width:0}.system-live-card small{display:block;color:#758697;font-size:8px;text-transform:uppercase;letter-spacing:.09em;margin-bottom:4px}.system-live-card strong{font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;display:block}.system-live-bar{height:4px;border-radius:99px;background:#071019;overflow:hidden;margin-top:6px}.system-live-bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .18s linear}
.detail-nav{position:sticky;top:8px;z-index:30;display:flex;align-items:center;gap:7px;overflow-x:auto;margin:0 0 18px;padding:8px;background:rgba(10,15,21,.88);border:1px solid var(--line);border-radius:14px;backdrop-filter:blur(14px);box-shadow:0 10px 30px rgba(0,0,0,.18)}
.detail-nav a{flex:0 0 auto;color:#aebdcb;text-decoration:none;font-size:11px;font-weight:700;padding:8px 10px;border-radius:9px;border:1px solid transparent}
.detail-nav a:hover{color:var(--text);background:#111b25;border-color:#26384a}.detail-nav a:first-child{color:#07110e;background:var(--accent)}
.grid{display:grid;grid-template-columns:1.05fr 1.35fr .9fr;gap:14px}
.section-heading{grid-column:1/-1;display:flex;align-items:flex-end;justify-content:space-between;gap:14px;margin:10px 2px -2px;padding-top:4px;scroll-margin-top:76px}
.section-heading:first-child{margin-top:0}.section-heading h2{margin:0;font-size:17px;letter-spacing:-.01em}.section-heading p{margin:3px 0 0;color:var(--muted);font-size:11px;line-height:1.45}.section-no{font:700 10px/1 ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--accent);letter-spacing:.12em}
.snapshot-card{background:linear-gradient(145deg,rgba(85,211,195,.06),rgba(17,24,33,.94) 42%)}.focus-card{border-color:#2c4557;box-shadow:0 12px 32px rgba(0,0,0,.14)}
.card.is-collapsible>h2{cursor:pointer;margin:-3px -3px 0;padding:3px 3px 12px;display:flex;align-items:center;justify-content:space-between;gap:10px}.card.is-collapsible>h2::after{content:"SCHOWAJ";font-size:9px;color:var(--muted);letter-spacing:.08em}.card.is-collapsible.collapsed>h2{padding-bottom:3px;margin-bottom:0}.card.is-collapsible.collapsed>h2::after{content:"POKAŻ";color:var(--accent)}.card.is-collapsible.collapsed>:not(h2){display:none}
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
.voice-shell{display:flex;flex-direction:column;gap:12px}
.voice-hero{display:grid;grid-template-columns:minmax(260px,1.5fr) repeat(3,minmax(120px,.65fr));gap:10px}
.voice-hero-main,.voice-box{background:#0c131b;border:1px solid #1d2936;border-radius:14px;padding:13px;min-width:0}
.voice-hero-main{background:linear-gradient(135deg,rgba(85,211,195,.08),rgba(110,168,254,.04));border-color:#294153}
.voice-eyebrow{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.09em;margin-bottom:5px}
.voice-hero-title{font-size:20px;font-weight:800;line-height:1.25;word-break:break-word}
.voice-hero-sub{margin-top:5px;color:var(--muted);font-size:11px;line-height:1.45}
.voice-hero-stat{display:flex;flex-direction:column;justify-content:center}
.voice-hero-stat strong{font-size:18px;font-variant-numeric:tabular-nums}.voice-hero-stat small{color:var(--muted);margin-top:5px}
.voice-groups{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}
.voice-groups>.voice-box:nth-child(4){grid-column:span 2}.voice-groups>.voice-box:nth-child(7){grid-column:span 2}
.mini-details{margin-top:8px;border:1px solid #192735;border-radius:10px;background:#081018;overflow:hidden}.mini-details summary{cursor:pointer;padding:8px 9px;color:#b9c9d6;font-size:10px;font-weight:700;list-style:none}.mini-details summary::-webkit-details-marker{display:none}.mini-details[open] summary{border-bottom:1px solid #192735}.mini-details .memory-list{padding:0 8px 8px}
.voice-box h3{margin:0 0 10px;font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:#a9b8c8}
.voice-kpis{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}
.voice-kpi{background:#0a1118;border:1px solid #192735;border-radius:10px;padding:9px;min-width:0}
.voice-kpi small{display:block;color:var(--muted);font-size:10px;margin-bottom:5px}.voice-kpi strong{font-size:13px;word-break:break-word}
.voice-score{display:grid;grid-template-columns:92px 1fr 48px;gap:8px;align-items:center;margin:7px 0;font-size:11px}
.voice-score .track{height:7px}.voice-score .val{font-size:11px}
.drive-row{display:grid;grid-template-columns:112px 1fr 48px;gap:8px;align-items:center;margin:8px 0}
.drive-row label{color:#b7c5d2;font-size:11px}.drive-row output{text-align:right;font-size:11px;font-variant-numeric:tabular-nums}
.drive-track{height:8px;background:#071019;border:1px solid #1d2b39;border-radius:999px;overflow:hidden}
.drive-fill{height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent));border-radius:999px}
.drive-fill.warn-fill{background:linear-gradient(90deg,#d79e32,var(--warn))}
.drive-fill.bad-fill{background:linear-gradient(90deg,#d75252,var(--bad))}
.voice-note{font-size:11px;color:var(--muted);line-height:1.5;margin-top:8px}
.memory-list{display:flex;flex-direction:column;gap:6px;margin-top:9px}
.memory-row{display:grid;grid-template-columns:72px 1fr auto;gap:8px;align-items:center;background:#071019;border:1px solid #172635;border-radius:9px;padding:7px 8px;font-size:10px}
.memory-row b{color:#cfe1ef}.memory-row span{color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.memory-row em{font-style:normal;font-variant-numeric:tabular-nums}
.voice-decision{padding:11px 12px;border-radius:12px;background:#0a1118;border:1px solid #1d2936;font-size:12px;line-height:1.55}
.voice-decision b{color:var(--accent)}
.voice-technical{border:1px solid #1d2936;border-radius:12px;background:#0a1118;overflow:hidden}
.voice-technical summary{cursor:pointer;padding:10px 12px;color:#a9b8c8;font-size:11px;font-weight:700}
.voice-technical-body{padding:0 12px 12px}
.voice-tech-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}
.voice-table-wrap{overflow-x:auto;border:1px solid #1d2936;border-radius:12px}
.voice-table-wrap table{min-width:820px}.voice-table-wrap th{background:#0a1118;position:sticky;top:0}
.voice-server-sep{height:1px;background:linear-gradient(90deg,transparent,#294153,transparent);margin:2px 0}

.attention-shell{display:grid;grid-template-columns:minmax(220px,.8fr) minmax(0,1.4fr) minmax(0,1.2fr);gap:10px}
.attention-panel{background:#0c131b;border:1px solid #1d2936;border-radius:12px;padding:11px;min-width:0}
.attention-panel h3{margin:0 0 9px;font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#a9b8c8}
.attention-focus{font-size:20px;font-weight:800;line-height:1.25;word-break:break-word}.attention-focus small{display:block;color:var(--muted);font-size:10px;font-weight:500;margin-top:6px}
.attention-row{display:grid;grid-template-columns:95px 1fr 54px;gap:8px;align-items:center;margin:7px 0;font-size:10px}.attention-row b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.attention-row output{text-align:right;color:#c8d7e4;font-variant-numeric:tabular-nums}
.attention-memory{display:flex;flex-direction:column;gap:6px;max-height:230px;overflow:auto}.attention-memory-row{background:#081018;border:1px solid #192735;border-radius:9px;padding:7px 8px;font-size:10px;line-height:1.45}.attention-memory-row b{color:#cfe1ef}.attention-memory-row small{color:var(--muted);display:block;margin-bottom:3px}.attention-memory-row span{color:#b6c5d1}
.policy-wrap{overflow-x:auto;border:1px solid #1d2936;border-radius:12px;background:#081018}.policy-table{width:100%;min-width:860px;border-collapse:collapse;font-size:10px}.policy-table th,.policy-table td{padding:8px 9px;border-bottom:1px solid #172735;text-align:right}.policy-table th:first-child,.policy-table td:first-child{text-align:left}.policy-table th{color:#8296a8;background:#0b141d;text-transform:uppercase;letter-spacing:.06em;font-size:8px}.policy-action{font-weight:800;color:#d8e8f3}.policy-pos{color:var(--good)}.policy-neg{color:var(--bad)}.policy-gates{display:flex;gap:7px;flex-wrap:wrap;margin:0 0 9px}.policy-gate{border:1px solid #274056;background:#0b141d;border-radius:999px;padding:5px 8px;font-size:9px;color:#9eb1c2}.policy-gate.pass{border-color:#2f6550;color:#79dfa3}.policy-gate.fail{border-color:#653b42;color:#e9979f}
.reason{padding:11px 12px;border-radius:12px;background:#0c131b;border:1px solid #1d2936;margin-bottom:12px}
.reason b{color:var(--accent)}
.ok{color:var(--good)}.no{color:var(--bad)}.warn{color:var(--warn)}
.people-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.person-card{background:#0a1118;border:1px solid #1d2d3a;border-radius:13px;padding:11px;min-width:0}.person-head{display:flex;justify-content:space-between;gap:10px;align-items:flex-start}.person-name{font-weight:800;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.person-id{display:block;color:var(--muted);font:9px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;margin-top:2px}.person-valence{font-size:9px;font-weight:800;border:1px solid #2a4052;border-radius:999px;padding:4px 7px;white-space:nowrap}.person-valence.positive{color:#79dfa3;border-color:#2f6550}.person-valence.negative{color:#ff9aa2;border-color:#653b42}.person-valence.mixed{color:#ffd277;border-color:#5b4a29}.person-kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px;margin:9px 0}.person-kpi{background:#071019;border:1px solid #172635;border-radius:9px;padding:7px;min-width:0}.person-kpi small{display:block;color:#6f8395;font-size:8px;text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}.person-kpi b{font-size:11px;word-break:break-word}.person-lines{font-size:9px;color:#98aaba;line-height:1.55}.person-lines b{color:#d4e3ee}.person-tags{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}.person-tag{border:1px solid #213548;background:#08121a;border-radius:999px;padding:4px 6px;font-size:8px;color:#a8bac8}
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
@media(max-width:1180px){.decision-flow{grid-template-columns:1fr 20px 1fr 20px 1fr}.decision-flow .flow-arrow:nth-of-type(6),.decision-flow .flow-arrow:nth-of-type(8){display:none}.decision-flow .flow-step:nth-of-type(7),.decision-flow .flow-step:nth-of-type(9){grid-column:span 2}}
@media(max-width:1050px){.grid{grid-template-columns:1fr 1fr}.span3{grid-column:span 2}.voice-hero{grid-template-columns:1fr 1fr}.voice-hero-main{grid-column:span 2}.voice-groups{grid-template-columns:1fr 1fr}.voice-groups>.voice-box:nth-child(4),.voice-groups>.voice-box:nth-child(7){grid-column:span 2}.trace-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.trace-verdict{grid-template-columns:1fr}}
@media(max-width:1100px){.system-strip{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:900px){.voice-summary,.learning-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.voice-tech-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.attention-shell{grid-template-columns:1fr 1fr}.attention-shell>.attention-panel:last-child{grid-column:span 2}}
@media(max-width:700px){.system-strip{grid-template-columns:1fr 1fr}
main{padding:12px}.guide-head{flex-direction:column}.decision-flow{grid-template-columns:1fr}.trace-grid{grid-template-columns:1fr}.trace-head{flex-direction:column}.trace-meta{text-align:left}.flow-arrow{transform:rotate(90deg);height:14px}.decision-flow .flow-arrow:nth-of-type(6),.decision-flow .flow-arrow:nth-of-type(8){display:grid}.decision-flow .flow-step:nth-of-type(7),.decision-flow .flow-step:nth-of-type(9){grid-column:auto}.top{align-items:flex-start;flex-direction:column}.badges{justify-content:flex-start}.detail-nav{top:4px;margin-bottom:14px}.grid{grid-template-columns:1fr}.span2,.span3{grid-column:auto}.section-heading{grid-column:auto}.section-heading p{max-width:46ch}.events{grid-template-columns:1fr}.attention-shell,.voice-summary,.learning-grid,.voice-groups,.voice-hero,.voice-tech-grid,.people-grid{grid-template-columns:1fr}.trace-history-item{grid-template-columns:65px 55px minmax(90px,.7fr) minmax(0,1fr);gap:5px;font-size:9px}.attention-shell>.attention-panel:last-child{grid-column:auto}.voice-hero-main,.voice-groups>.voice-box:nth-child(4),.voice-groups>.voice-box:nth-child(7){grid-column:auto}.log-item{grid-template-columns:62px 72px 1fr}}
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

  <div class="system-strip" aria-label="Live system telemetry">
    <div class="system-live-card"><small>CPU system</small><strong id="detail-cpu">—</strong><div class="system-live-bar"><i id="detail-cpu-bar"></i></div></div>
    <div class="system-live-card"><small>CPU Mucha</small><strong id="detail-proc-cpu">—</strong><div class="system-live-bar"><i id="detail-proc-cpu-bar"></i></div></div>
    <div class="system-live-card"><small>RAM system</small><strong id="detail-ram">—</strong><div class="system-live-bar"><i id="detail-ram-bar"></i></div></div>
    <div class="system-live-card"><small>RAM Mucha</small><strong id="detail-proc-ram">—</strong></div>
    <div class="system-live-card"><small>Dysk</small><strong id="detail-disk">—</strong><div class="system-live-bar"><i id="detail-disk-bar"></i></div></div>
    <div class="system-live-card"><small>GPU / VRAM</small><strong id="detail-gpu">—</strong><div class="system-live-bar"><i id="detail-gpu-bar"></i></div></div>
  </div>

  <nav class="detail-nav" aria-label="Sekcje szczegółów">
    <a href="#teraz">● Teraz</a>
    <a href="#voice-section">Voice i decyzje</a>
    <a href="#learning-section">Uczenie i pamięć</a>
    <a href="#social-section">Relacje</a>
    <a href="#audio-section">Audio / STT</a>
    <a href="#neurons-section">Neurony</a>
  </nav>

  <section class="grid">
    <div class="section-heading" id="teraz"><div><span class="section-no">01 / TERAZ</span><h2>Co dzieje się w tej chwili</h2><p>Najpierw zobacz ścieżkę decyzji. Dopiero potem schodź do parametrów technicznych.</p></div></div>

    <div class="guide-hero">
      <div class="guide-head">
        <div><h2>Jak Mucha doszła do tego, co robi teraz?</h2><p>Pięć kroków od bodźca do akcji. Najedź na <span class="help-dot" data-help-key="decision-flow" tabindex="0">?</span>, aby zobaczyć jak czytać ten widok.</p></div>
        <div class="guide-mode" aria-label="Poziom szczegółowości">
          <button type="button" id="mode-simple" class="active">PROSTY</button>
          <button type="button" id="mode-full">PEŁNY</button>
        </div>
      </div>
      <div class="decision-flow">
        <div class="flow-step"><small>1 • BODZIEC <span class="help-dot" data-help-key="last-event" tabindex="0">?</span></small><strong id="flow-event">—</strong><em>Co właśnie dotarło do Muchy.</em></div>
        <div class="flow-arrow">→</div>
        <div class="flow-step attention"><small>2 • UWAGA <span class="help-dot" data-help-key="attention-score" tabindex="0">?</span></small><strong id="flow-attention">—</strong><em>Na czym utrzymuje się working memory.</em></div>
        <div class="flow-arrow">→</div>
        <div class="flow-step brain"><small>3 • CONNECTOME <span class="help-dot" data-help-key="brain-state" tabindex="0">?</span></small><strong id="flow-brain">—</strong><em>Dominujący stan neuronalny.</em></div>
        <div class="flow-arrow">→</div>
        <div class="flow-step decision"><small>4 • READOUT <span class="help-dot" data-help-key="readout" tabindex="0">?</span></small><strong id="flow-readout">—</strong><em>Najsilniejsze wyjście zachowania.</em></div>
        <div class="flow-arrow">→</div>
        <div class="flow-step action"><small>5 • AKCJA <span class="help-dot" data-help-key="last-action" tabindex="0">?</span></small><strong id="flow-action">—</strong><em>Co faktycznie wykonała.</em></div>
      </div>
    </div>

    <div class="card span3 decision-trace-card">
      <div class="trace-head">
        <div><h2>🔎 Dlaczego zrobiła X? <span class="help-dot" data-help-key="decision-trace" tabindex="0">?</span></h2><div class="footer">Faktyczny trace algorytmu z chwili decyzji: sensory → stan → readout/policy → ograniczenia → akcja.</div></div>
        <div class="trace-meta" id="decision-trace-meta">czekam na pierwszy cykl decyzyjny…</div>
      </div>
      <div id="decision-trace"><div class="trace-empty">Brak trace. Pojawi się po pierwszej decyzji tekstowej albo voice.</div></div>
    </div>

    <div class="card span3 trace-history-card">
      <div class="trace-history-head">
        <div><h2>🕘 Historia Decision Trace <span class="help-dot" data-help-key="decision-trace-history" tabindex="0">?</span></h2><div class="footer" id="decision-history-status">LIVE • ostatnie decyzje</div></div>
        <div class="trace-history-controls">
          <button type="button" class="trace-history-btn active" id="trace-live-btn" onclick="selectDecisionTrace(null)">● LIVE</button>
          <button type="button" class="trace-history-btn active" id="trace-filter-all" onclick="setDecisionTraceFilter('all')">WSZYSTKIE</button>
          <button type="button" class="trace-history-btn" id="trace-filter-voice" onclick="setDecisionTraceFilter('voice')">VOICE</button>
          <button type="button" class="trace-history-btn" id="trace-filter-text" onclick="setDecisionTraceFilter('text')">TEXT</button>
        </div>
      </div>
      <div class="trace-history-list" id="decision-trace-history"><div class="trace-history-empty">Historia zapełni się po kolejnych decyzjach.</div></div>
    </div>

    <div class="panel-explainer"><span class="help-dot" data-help-key="simple-mode" tabindex="0">?</span><div><b>Tryb PROSTY</b> ukrywa część surowej telemetrii. Niczego nie wyłącza w Musze — zmienia tylko sposób wyświetlania. Tryb PEŁNY pokazuje cały debug.</div></div>

    <div class="card snapshot-card">
      <h2>Stan mózgu <span class="help-dot" data-help-key="brain-state" tabindex="0">?</span></h2>
      <div class="metric"><span data-help-label="neurons">Neurony</span><strong id="neurons">—</strong></div>
      <div class="metric"><span data-help-label="connections">Połączenia</span><strong id="connections">—</strong></div>
      <div class="metric"><span data-help-label="active-neurons">Aktywne |a| &gt; 0.1</span><strong id="active">—</strong></div>
      <div class="metric"><span data-help-label="mean-activation">Średnia |a|</span><strong id="mean">—</strong></div>
      <div class="metric"><span data-help-label="max-activation">Maks. |a|</span><strong id="max">—</strong></div>
      <div class="metric"><span data-help-label="reward-trace">Reward trace</span><strong id="reward">—</strong></div>
      <div class="metric"><span data-help-label="tick">Tick</span><strong id="ticks">—</strong></div>
    </div>

    <div class="card">
      <h2>Wyjścia connectome <span class="help-dot" data-help-key="readout" tabindex="0">?</span></h2>
      <div id="actions"></div>
    </div>

    <div class="card advanced-card">
      <h2>Środowisko <span class="help-dot" data-help-key="environment" tabindex="0">?</span></h2>
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

    <div class="card span3 focus-card">
      <h2>Attention / Working Memory <span class="help-dot" data-help-key="attention" tabindex="0">?</span></h2>
      <div id="attention-debug"><div class="reason">Czekam na pierwszy kontekst tekstowy lub voice…</div></div>
    </div>

    <div class="card span3 focus-card">
      <h2>Learned Action Policy <span class="help-dot" data-help-key="action-policy" tabindex="0">?</span></h2>
      <div id="action-policy-debug"><div class="reason">Policy czeka na dane connectomu i pierwszy reward/punish…</div></div>
    </div>

    <div class="card span3 focus-card">
      <h2>🧠 Affective State / Stage 27 <span class="help-dot" data-help-key="affective-state" tabindex="0">?</span></h2>
      <div class="reason" id="affective-summary">Czekam na pierwszy slow affect tick…</div>
      <div class="learning-grid" id="affective-state-grid" style="grid-template-columns:repeat(5,minmax(0,1fr));margin-top:10px"></div>
    </div>

    <div class="card span3 focus-card">
      <h2>🔥 Natural Motivation / Stage 30 <span class="help-dot" data-help-key="motivation-state" tabindex="0">?</span></h2>
      <div class="reason" id="motivation-summary">Czekam na pierwszy tick motywacji…</div>
      <div class="learning-grid" id="motivation-state-grid" style="grid-template-columns:repeat(4,minmax(0,1fr));margin-top:10px"></div>
    </div>

    <div class="section-heading" id="voice-section"><div><span class="section-no">02 / VOICE</span><h2>Voice i decyzje</h2><p>Najpierw decyzja, potem jej przyczyna: readouty, potrzeby, pamięć, reward i zagrożenia.</p></div></div>

    <div class="card span3 focus-card">
      <h2>Voice / Zachowanie <span class="help-dot" data-help-key="voice-decision" tabindex="0">?</span></h2>
      <div id="voice-debug"><div class="reason">Czekam na pierwszy cykl voice…</div></div>
    </div>

    <div class="card span3 focus-card">
      <h2>📍 Long-term Channel / Place Memory <span class="help-dot" data-help-key="channel-memory" tabindex="0">?</span></h2>
      <div class="reason" id="channel-memory-summary">Czekam na trwałe doświadczenia z kanałami voice…</div>
      <div class="people-grid" id="channel-memory-grid"><div class="trace-empty">Brak profili miejsc.</div></div>
    </div>

    <div class="card span3 focus-card">
      <h2>🎚 Reward-learned Conversation Dynamics <span class="help-dot" data-help-key="voice-dynamics-memory" tabindex="0">?</span></h2>
      <div class="reason" id="voice-dynamics-summary">Czekam na reward/punish przypisany do dynamiki rozmowy…</div>
      <div class="people-grid" id="voice-dynamics-grid"><div class="trace-empty">Brak wyuczonych wzorców dynamiki.</div></div>
    </div>

    <div class="section-heading" id="learning-section"><div><span class="section-no">03 / UCZENIE</span><h2>Uczenie i pamięć</h2><p>Zmiany wynikające z rewardu, plastyczność, historia aktywności i to, co utrwaliło się od startu.</p></div></div>

    <div class="card span3 focus-card">
      <h2>💤 Sleep / Offline Consolidation <span class="help-dot" data-help-key="sleep" tabindex="0">?</span></h2>
      <div class="learning-grid" style="grid-template-columns:repeat(6,minmax(0,1fr))">
        <div class="kpi"><small>stan snu</small><strong id="sleep-state">—</strong></div>
        <div class="kpi"><small>stan dobowy</small><strong id="circadian-state">—</strong></div>
        <div class="kpi"><small>fatigue</small><strong id="circadian-fatigue">—</strong></div>
        <div class="kpi"><small>cisza</small><strong id="sleep-quiet">—</strong></div>
        <div class="kpi"><small>cykl</small><strong id="sleep-cycle">—</strong></div>
        <div class="kpi"><small>replay epizodów</small><strong id="sleep-episodes">—</strong></div>
        <div class="kpi"><small>zmienione synapsy</small><strong id="sleep-synapses">—</strong></div>
        <div class="kpi"><small>zmienione neurony</small><strong id="sleep-neurons">—</strong></div>
        <div class="kpi"><small>semantic rehearsal</small><strong id="sleep-semantic">—</strong></div>
        <div class="kpi"><small>Δ siły wspomnień</small><strong id="sleep-memory-delta">—</strong></div>
        <div class="kpi"><small>utrwalone sceny</small><strong id="sleep-scenes">—</strong></div>
        <div class="kpi"><small>utrwalone synapsy</small><strong id="sleep-consolidated-synapses">—</strong></div>
      </div>
      <div style="margin:10px 0 8px"><div class="track"><div class="fill" id="sleep-progress" style="width:0%"></div></div></div>
      <div class="reason" id="sleep-summary">Czekam na warunki snu…</div>
      <div id="sleep-last" class="voice-note"></div>
    </div>

    <div class="card span2 advanced-card">
      <h2>Aktywność w czasie <span class="help-dot" data-help-key="activity-chart" tabindex="0">?</span></h2>
      <canvas id="chart" width="1000" height="220" aria-label="Wykres aktywności mózgu"></canvas>
    </div>

    <div class="card">
      <h2>Ostatnie zdarzenia <span class="help-dot" data-help-key="events" tabindex="0">?</span></h2>
      <div class="events">
        <div class="event"><small>bodziec</small><div id="event">—</div></div>
        <div class="event"><small>akcja</small><div id="lastaction">—</div></div>
      </div>
    </div>

    <div class="card span2">
      <h2>Learning Debug <span class="help-dot" data-help-key="learning-debug" tabindex="0">?</span></h2>
      <div class="learning-grid">
        <div class="kpi"><small>ostatni reward</small><strong id="learn-reward">—</strong></div>
        <div class="kpi"><small>target action</small><strong id="learn-action">—</strong></div>
        <div class="kpi"><small>zmienione neurony</small><strong id="learn-count">—</strong></div>
        <div class="kpi"><small>max |Δ bias|</small><strong id="learn-max">—</strong></div>
        <div class="kpi"><small>zmienione synapsy</small><strong id="learn-synapses">—</strong></div>
        <div class="kpi"><small>uczone synapsy razem</small><strong id="learn-synapses-total">—</strong></div>
      </div>
      <div class="reason" id="learn-summary">Czekam na pierwszy reward…</div>
      <div id="learning-impact"></div>
    </div>

    <div class="card span3">
      <h2>Learning Since Startup <span class="help-dot" data-help-key="learning-startup" tabindex="0">?</span></h2>
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

    <div class="section-heading" id="social-section"><div><span class="section-no">04 / RELACJE</span><h2>Relacje i reakcje</h2><p>Jak Mucha reaguje na ludzi, słowa i feedback społeczny.</p></div></div>

    <div class="card span3 focus-card">
      <h2>🧠 Long-term People Memory <span class="help-dot" data-help-key="person-memory" tabindex="0">?</span></h2>
      <div class="reason" id="people-memory-summary">Czekam na trwałe doświadczenia z ludźmi…</div>
      <div class="people-grid" id="people-memory-grid"><div class="trace-empty">Brak profili osób.</div></div>
    </div>

    <div class="card span3 focus-card">
      <h2>🎭 Long-term Social Situations <span class="help-dot" data-help-key="social-scene-memory" tabindex="0">?</span></h2>
      <div class="reason" id="social-scene-summary">Czekam na powtarzalne sytuacje społeczne…</div>
      <div class="people-grid" id="social-scene-grid"><div class="trace-empty">Brak profili sytuacji.</div></div>
    </div>

    <div class="card span2">
      <h2>Social Learning / Relacje <span class="help-dot" data-help-key="social-learning" tabindex="0">?</span></h2>
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
      <h2>Reaction Debug <span class="help-dot" data-help-key="reaction-debug" tabindex="0">?</span></h2>
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
      <h2>Plasticity <span class="help-dot" data-help-key="plasticity" tabindex="0">?</span></h2>
      <div class="metric"><span>Średni bias</span><strong id="bias-mean">—</strong></div>
      <div class="metric"><span>Średni |bias|</span><strong id="bias-mean-abs">—</strong></div>
      <div class="metric"><span>Max |bias|</span><strong id="bias-max">—</strong></div>
      <div class="metric"><span>Dodatnie neurony</span><strong id="bias-pos">—</strong></div>
      <div class="metric"><span>Ujemne neurony</span><strong id="bias-neg">—</strong></div>
      <div class="metric"><span>Uczone synapsy</span><strong id="synapse-count">—</strong></div>
      <div class="metric"><span>Średni |Δ synapsy|</span><strong id="synapse-mean">—</strong></div>
      <div class="metric"><span>Max |Δ synapsy|</span><strong id="synapse-max">—</strong></div>
      <canvas id="bias-chart" width="520" height="150" aria-label="Histogram plastic bias" style="height:150px;margin-top:12px"></canvas>
    </div>

    <div class="card span2">
      <h2>Reward timeline <span class="help-dot" data-help-key="reward-timeline" tabindex="0">?</span></h2>
      <canvas id="reward-chart" width="1000" height="220" aria-label="Historia reward trace"></canvas>
      <div class="legend"><span class="trace-line">reward trace</span><span class="reward-line">zdarzenie reward</span></div>
    </div>

    <div class="card span2">
      <h2>Action History <span class="help-dot" data-help-key="action-history" tabindex="0">?</span></h2>
      <div class="log-list" id="action-history"><div class="reason">Brak akcji.</div></div>
    </div>

    <div class="card">
      <h2>Top changed neurons <span class="help-dot" data-help-key="changed-neurons" tabindex="0">?</span></h2>
      <table><thead><tr><th>root_id</th><th>Δ bias</th><th>activation</th></tr></thead><tbody id="changed-neurons"></tbody></table>
    </div>

    <div class="card span3">
      <h2>Server Learning Context <span class="help-dot" data-help-key="server-learning" tabindex="0">?</span></h2>
      <table>
        <thead><tr><th>Serwer</th><th>Ostatnia nagradzalna akcja</th><th>Szczegół</th><th>Wiek</th></tr></thead>
        <tbody id="guild-learning-context"></tbody>
      </table>
    </div>

    <div class="section-heading" id="audio-section"><div><span class="section-no">05 / AUDIO</span><h2>Audio i rozpoznawanie mowy</h2><p>Diagnostyka TTS, odtwarzania i STT — zwykle potrzebna dopiero przy problemie.</p></div></div>

    <div class="card span3">
      <h2>Audio Debug <span class="help-dot" data-help-key="audio-debug" tabindex="0">?</span></h2>
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
      <h2>Voice Recognition / STT <span class="help-dot" data-help-key="stt" tabindex="0">?</span></h2>
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

    <div class="section-heading" id="neurons-section"><div><span class="section-no">06 / NEURONY</span><h2>Neurony i surowy stan</h2><p>Najbardziej aktywne komórki do głębszej analizy w Connectome i Neuro-map.</p></div></div>

    <div class="card span3">
      <h2>Najbardziej aktywne neurony <span class="help-dot" data-help-key="top-neurons" tabindex="0">?</span></h2>
      <table><thead><tr><th>#</th><th>FlyWire root_id</th><th>activation</th><th>|a|</th></tr></thead><tbody id="top"></tbody></table>
      <div class="footer">Przy prawdziwym FAFB v783 root_id odpowiada identyfikatorowi neuronu FlyWire.</div>
    </div>
  </section>
  <div id="help-tooltip" class="help-tooltip" role="tooltip" aria-hidden="true"></div>
</main>
<script>
const actionOrder=["speak","react","voice_join","voice_move","voice_leave","explore","stay"];
const history=[];
const maxHistory=180;
const $=id=>document.getElementById(id);
function fmt(n,d=3){return Number(n).toFixed(d)}
function nfmt(n){return Number(n).toLocaleString("pl-PL")}
function fmtBytes(n){n=Number(n||0);if(!n)return "0 B";const u=["B","KB","MB","GB","TB"];let i=0;while(n>=1024&&i<u.length-1){n/=1024;i++}return n.toFixed(i>1?2:1)+" "+u[i]}
function renderSystemLive(sys){
  sys=sys||{};const proc=sys.process||{},gpu=((sys.gpu||{}).gpus||[])[0]||null;
  const cpu=Math.max(0,Number(sys.cpu_percent||0)),pcpu=Math.max(0,Number(proc.cpu_percent||0));
  const ram=Math.max(0,Number(sys.mem_percent||0)),disk=Math.max(0,Number(sys.disk_percent||0));
  $("detail-cpu").textContent=cpu.toFixed(1)+"% • "+Number(sys.cpu_count||0)+" CPU";
  $("detail-proc-cpu").textContent=pcpu.toFixed(1)+"%";
  $("detail-ram").textContent=ram.toFixed(1)+"% • "+fmtBytes(sys.mem_used)+" / "+fmtBytes(sys.mem_total);
  $("detail-proc-ram").textContent=fmtBytes(proc.memory_bytes)+" • "+Number(proc.threads||0)+" th";
  $("detail-disk").textContent=disk.toFixed(1)+"% • "+fmtBytes(sys.disk_free)+" wolne";
  $("detail-gpu").textContent=gpu?(String(gpu.name||"GPU")+" • "+Number(gpu.utilization_percent||0).toFixed(0)+"% • "+Number(gpu.memory_percent||0).toFixed(0)+"% VRAM"):"brak danych GPU";
  $("detail-cpu-bar").style.width=Math.min(100,cpu)+"%";
  $("detail-proc-cpu-bar").style.width=Math.min(100,pcpu)+"%";
  $("detail-ram-bar").style.width=Math.min(100,ram)+"%";
  $("detail-disk-bar").style.width=Math.min(100,disk)+"%";
  $("detail-gpu-bar").style.width=Math.min(100,gpu?Number(gpu.utilization_percent||0):0)+"%";
}
function renderActions(scores){
  const dom=Object.entries(scores||{}).sort((a,b)=>Number(b[1])-Number(a[1]))[0]?.[0];
  $("actions").innerHTML=actionOrder.map(k=>{
    const v=Number((scores||{})[k]??0);
    return '<div class="action"><div class="'+(k===dom?'dominant':'')+'">'+(k===dom?'▶ ':'')+esc(k)+' '+helpDot(k)+'</div>'+
      '<div class="track"><div class="fill" style="width:'+Math.max(0,Math.min(100,v*100))+'%"></div></div>'+
      '<div class="val">'+v.toFixed(3)+'</div></div>';
  }).join("");
  enhanceHelp();
}
function esc(v){
  return String(v??"").replace(/[&<>"']/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[m]));
}

const HELP={
  "decision-flow":{title:"Jak czytać ścieżkę decyzji?",body:"To skrót jednego cyklu: bodziec → uwaga/pamięć → stan connectomu → readout zachowania → faktyczna akcja.",read:"Czytaj od lewej do prawej. Jeśli readout i akcja się różnią, niżej szukaj cooldownu, blokady albo ograniczenia operacyjnego."},
  "decision-trace":{title:"Decision Trace",body:"To telemetryczny zapis danych faktycznie użytych przez kod przy ostatnim cyklu decyzyjnym. Nie jest opisem ukrytego toku rozumowania ani interpretacją zachowania.",read:"Porównaj raw score z effective score, następnie sprawdź pamięć/sygnały i listę ograniczeń. Pole AKCJA mówi co runtime faktycznie zdecydował wykonać."},
  "decision-trace-history":{title:"Historia Decision Trace",body:"Runtime przechowuje w RAM do 48 pełnych zamrożonych trace’ów. Dashboard pokazuje ostatnie 40. Kliknięcie wpisu otwiera stan dokładnie z tamtej decyzji zamiast bieżącego LIVE.",read:"Użyj filtrów VOICE/TEXT. Przycisk LIVE wraca do najnowszego trace. Historia zeruje się po restarcie Muchy."},
  "simple-mode":{title:"PROSTY vs PEŁNY",body:"PROSTY pokazuje tylko elementy potrzebne do zrozumienia bieżącego zachowania. PEŁNY odsłania telemetrię techniczną.",read:"To zmienia wyłącznie interfejs. Nie zmienia configu ani zachowania Muchy."},
  "last-event":{title:"Ostatni bodziec",body:"Ostatnie zdarzenie zapisane jako wejście dla runtime'u, np. tekst, STT, reward, voice albo threat.",read:"To punkt startowy, gdy chcesz sprawdzić co poprzedziło zmianę zachowania."},
  "attention":{title:"Attention / Working Memory",body:"Krótkotrwały kontekst osób, kanałów i tematów. Ślad zanika, a aktywny focus jest ponownie podawany do connectomu.",read:"Attention jest chwilowe. Affinity jest relacją długoterminową. Working memory pokazuje ostatnie sceny nadal dostępne jako kontekst."},
  "attention-score":{title:"Attention score",body:"Połączenie świeżości krótkotrwałego śladu z neuronalnym attention_score odczytanym z connectomu.",read:"Wyższy score = większa aktualna dominacja tego elementu w kontekście. To nie jest reward."},
  "action-policy":{title:"Learned Action Policy",body:"Warstwa ucząca się na reward/punish, która przesuwa efektywne readouty. Przy Connectome behavior competition SPEAK/REACT nie przechodzą już przez ręczny próg — konkurują ze STAY.",read:"raw = sam connectome, bias = doświadczenie, effective = raw po policy. W trybie competition patrz na winner/runner-up i margin. Progi są tylko legacy fallback."},
  "affective-state":{title:"Affective State / Stage 27",body:"Wolnozmienny stan afektywny wyliczany z realnej aktywności attractorów FAFB, homeostatic drives i reward trace. Nie jest osobnym systemem decyzji. Jego pamięć jest zapisywana w brain_state.npz i wraca jako delikatny sensory feedback do tych samych attractorów.",read:"value = utrwalony stan, target = to, do czego pcha go bieżąca aktywność neuronalna. CONTENTMENT, TENSION, CURIOSITY, SOCIAL_LONGING i ACTIVATION mogą utrzymywać się między eventami, ale każda akcja nadal wygrywa w One Brain."},
  "motivation-state":{title:"Natural Motivation / Stage 30",body:"Cztery wspólne motywy — SOCIAL, NOVELTY, SAFETY i REST — łączą surowe homeostatic drives z wolnym affectem. Jeśli potrzeba długo pozostaje wysoka, rośnie frustration. Po zaspokojeniu rośnie satiation, która chwilowo tłumi ponowną presję.",read:"Pressure = bieżąca potrzeba. Frustration = historia niezaspokojenia. Satiation = chwilowe nasycenie po sukcesie. Urgency = wynik tych trzech. Urgency nie dodaje punktów do akcji; tylko skaluje drive sensory input do attractorów FAFB."},
  "brain-state":{title:"Stan connectomu",body:"Bieżąca aktywność całej sieci po bodźcach, propagacji, plastyczności i internal states. Stan nie resetuje się po każdym evencie.",read:"Globalna aktywność mówi jak mocno sieć pracuje, ale do konkretnej decyzji patrz na readouty."},
  "readout":{title:"Readout",body:"Wartość 0–1 z populacji neuronów wyjściowych przypisanej do akcji speak/react/join/move/leave/explore/stay.",read:"Najsilniejszy readout jest kandydatem. Wykonanie może być zablokowane przez cooldown, permissions albo warunki bezpieczeństwa."},
  "last-action":{title:"Faktyczna akcja",body:"Ostatnie zachowanie naprawdę wykonane przez bota, a nie sam zamiar connectomu.",read:"Porównaj ją z najsilniejszym readoutem i z polem 'powód decyzji'."},
  "neurons":{title:"Neurony",body:"Liczba neuronów załadowanych do runtime connectome.",read:"Pełny przygotowany FAFB v783 ma około 139 tys. neuronów."},
  "connections":{title:"Połączenia",body:"Liczba kierunkowych połączeń w bazowej sparse matrix FAFB.",read:"Uczone synaptic delta są nakładką i nie zmieniają tej bazowej liczby."},
  "active-neurons":{title:"Aktywne |a| > 0.1",body:"Liczba neuronów, których bezwzględna aktywacja przekracza diagnostyczny próg 0.1.",read:"To szerokość pobudzenia sieci, nie biologiczna liczba spike'ów."},
  "mean-activation":{title:"Średnia |a|",body:"Średnia bezwzględna aktywacja wszystkich neuronów.",read:"Najbardziej użyteczna przy porównywaniu chwil przed i po bodźcu."},
  "max-activation":{title:"Maks. |a|",body:"Największa bezwzględna aktywacja pojedynczego neuronu.",read:"Może być wysoka nawet wtedy, gdy średnia aktywność całej sieci jest niska."},
  "reward-trace":{title:"Reward trace",body:"Krótkotrwały ślad niedawnych nagród i kar używany w uczeniu i lekko w readoutach.",read:"Dodatni = niedawny pozytywny feedback, ujemny = kara. Z czasem zanika."},
  "tick":{title:"Tick connectomu",body:"Licznik kroków propagacji wykonanych przez runtime.",read:"Tick nie oznacza sekundy. Jedno zdarzenie może wykonać kilka ticków."},
  "environment":{title:"Środowisko",body:"Stan modelu języka, voice i gotowość generatora.",read:"To głównie diagnostyka techniczna, dlatego w trybie PROSTYM jest ukryta."},
  "voice-decision":{title:"Voice / Zachowanie",body:"Zbiera readouty voice, homeostazę, internal states, pamięć epizodyczną, reward opportunity, threat i Chasera.",read:"Najpierw czytaj duży wynik i 'reason'. Dopiero potem rozwijaj techniczne sekcje."},
  "activity-chart":{title:"Aktywność w czasie",body:"Historia średniej i maksymalnej aktywacji connectomu.",read:"Pozwala zobaczyć odpowiedź na bodźce i tempo wygaszania stanu."},
  "events":{title:"Ostatnie zdarzenia",body:"Ostatni bodziec oraz ostatnia wykonana akcja.",read:"Do pełnej kolejności użyj Action History."},
  "learning-debug":{title:"Learning Debug",body:"Ostatni reward() i jego wpływ na bias neuronów, synaptic delta oraz readouty przed/po.",read:"Dodatnie Δ wzmacnia, ujemne osłabia. Target action mówi czego dotyczył ślad."},
  "sleep":{title:"Sleep / Circadian / Offline Consolidation",body:"Stage 26 łączy sen z trwałym fatigue. Podczas czuwania fatigue narasta i przez istniejące attractory SATIETY/STRESS wpływa na One Brain. Każdy prawdziwy cykl replay spłaca fatigue, a pełny sen uruchamia krótki POST-SLEEP.",read:"AWAKE = zwykłe czuwanie, TIRED = fatigue przekroczył próg, SLEEP = trwa replay/konsolidacja, POST-SLEEP = okres po pełnej sesji. Sen nie tworzy nowych zdarzeń Discord."},
  "learning-startup":{title:"Learning Since Startup",body:"Liczniki uczenia od uruchomienia procesu: język, reward events i skumulowane zmiany.",read:"Te liczniki resetują się po restarcie, nawet jeśli trwały stan został zapisany."},
  "social-learning":{title:"Social Learning / Relacje",body:"Długoterminowe sygnały społeczne i affinity użytkowników.",read:"Nie myl z Attention: affinity opisuje relację, Attention opisuje to, co zajmuje Muchę teraz."},
  "person-memory":{title:"Long-term People Memory / Stage 28",body:"Profil osoby łączy trwałą pamięć semantyczną z chronologiczną historią kontaktów. Mucha pamięta teraz nie tylko średnią relacji, ale też kiedy kogo spotkała, gdzie, jakie akcje wykonywała i czy ostatnio relacja się poprawia czy pogarsza.",read:"Familiarity = ilość doświadczenia. Recent valence = ostatnie realne outcome. Trend = kierunek zmian w nowszych vs starszych zdarzeniach. Stability = jak przewidywalna jest relacja. Te cechy wracają jako sensory do connectomu; nie ustawiają akcji bezpośrednio."},
  "channel-memory":{title:"Long-term Channel / Place Memory / Stage 29",body:"Trwały model kanału łączy teraz agregaty z chronologią miejsca: wizyty, dynamikę rozmowy, realne outcome akcji, typowych ludzi i kierunek zmian reputacji kanału.",read:"Recent valence opisuje ostatnie realne outcome. Trend porównuje nowsze i starsze zdarzenia, stability mówi jak przewidywalne są wyniki, a recent occupancy pokazuje ilu ludzi zwykle było ostatnio. Całość wraca jako sensory do connectomu, bez ręcznego bonusowania JOIN/MOVE/STAY."},
  "voice-dynamics-memory":{title:"Reward-learned Conversation Dynamics",body:"Ten model generalizuje ponad ludźmi i kanałami. Rozpoznaje wzorce typu DIALOGUE/CROSSTALK, tempo zmian mówców, overlap, handoff, długość tur, dominację, ciszę i tempo mowy, a realny reward/punish uczy wyników akcji w takich warunkach.",read:"Seen zwiększa tylko familiarity. Historyczne wyniki SPEAK/STAY/JOIN/MOVE/LEAVE wracają jako sensory connectomu, nie jako bezpośredni bonus do action score."},
  "social-scene-memory":{title:"Long-term Social Situations",body:"Scena łączy KTO + GDZIE + dynamikę rozmowy + dominujący stan wewnętrzny Muchy. Neutralne widzenie sceny zwiększa familiarity, a reward/punish zapisuje wynik konkretnych akcji.",read:"Klikalnego wyboru akcji tu nie ma: znana scena wraca jako sensory do connectomu. Dobra/zła akcja opisuje pamięć historyczną, a nie ręcznie ustawiony bonus."},
  "reaction-debug":{title:"Reaction Debug",body:"Readout react, próg, cooldown, kandydaci emoji i wynik próby reakcji.",read:"Jeśli score jest wysoki, ale brak reakcji, sprawdź cooldown i Discord permissions."},
  "plasticity":{title:"Plasticity",body:"Trwałe zmiany bias neuronów i wag synaptycznych nałożone na bazowy FAFB.",read:"Bazowy connectome pozostaje nienaruszony; uczenie jest nakładką."},
  "reward-timeline":{title:"Reward timeline",body:"Historia reinforcement events wraz z bieżącym reward trace.",read:"Porównuj znaczniki nagród/kar z Action History i Learning Debug."},
  "action-history":{title:"Action History",body:"Chronologiczny log zachowań i zdarzeń runtime'u.",read:"Najlepsze miejsce do ustalenia dokładnej kolejności tego, co się wydarzyło."},
  "changed-neurons":{title:"Top changed neurons",body:"Neurony z największą zmianą plastic bias po rewardzie.",read:"root_id możesz potem odszukać na Neuro-map."},
  "server-learning":{title:"Server Learning Context",body:"Ostatnia akcja możliwa do nagrodzenia zapisana osobno dla każdego serwera.",read:"Chroni przed przypisaniem rewardu z jednego guild do akcji z innego."},
  "audio-debug":{title:"Audio Debug",body:"Stan odtwarzania TTS/random audio, FFmpeg, voice connection i błędy.",read:"Używaj gdy Mucha miała coś powiedzieć, ale nic nie słychać."},
  "stt":{title:"Voice Recognition / STT",body:"Stan pipeline'u faster-whisper i ostatnia transkrypcja.",read:"HEARD = rozpoznano tekst; NO_SPEECH = brak użytecznej mowy; ERROR = błąd pipeline'u."},
  "top-neurons":{title:"Najbardziej aktywne neurony",body:"Neuronowe root_id z największą chwilową bezwzględną aktywacją.",read:"Wysoka aktywacja nie oznacza automatycznie, że neuron sam spowodował decyzję."},
  "speak":{title:"speak",body:"Readout skłonności do wygenerowania tekstu lub TTS.",read:"Odpowiedź tekstowa porównuje go ze speak_threshold; spontaniczne pisanie używa dodatkowo +0.08."},
  "react":{title:"react",body:"Readout skłonności do reakcji emoji.",read:"Musi pokonać reaction_threshold i cooldown."},
  "voice_join":{title:"voice_join",body:"Readout skłonności do wejścia na voice.",read:"Sam wybór kanału uwzględnia dodatkowo pamięć, affinity, exploration i dostępność."},
  "voice_move":{title:"voice_move",body:"Readout skłonności do zmiany kanału voice.",read:"Może rosnąć po threat, fatigue lub habituation przez wpływ sensoryczny na connectome."},
  "voice_leave":{title:"voice_leave",body:"Readout skłonności do opuszczenia voice.",read:"Konkuruje z join/move/stay."},
  "explore":{title:"explore",body:"Readout eksploracyjny wpływający na bardziej nowe/niepewne wybory.",read:"Wyższy wynik oznacza większą tendencję do eksploracji zamiast utrwalonego zachowania."},
  "stay":{title:"stay",body:"Readout pozostania w obecnym stanie/kanał voice.",read:"Może być osłabiany przez overstay, threat i social fatigue."},
  "attention-neural":{title:"Neural attention",body:"Czysty neuronalny odczyt tego elementu uwagi z przypisanych populacji sensory/output.",read:"0.5 jest w przybliżeniu neutralne. Wynik powyżej 0.5 wzmacnia chwilowy ślad, poniżej 0.5 go osłabia."},
  "attention-age":{title:"Age uwagi",body:"Czas od ostatniego odświeżenia elementu attention.",read:"Im starszy element, tym bardziej jego ślad zanika zgodnie z attention_half_life_seconds."},
  "social-need":{title:"SOCIAL NEED",body:"Wewnętrzny stan reprezentujący brak kontaktu społecznego / motywację do szukania ludzi.",read:"To aktywność attractora po propagacji, nie ręczny bonus do voice_join."},
  "curiosity":{title:"CURIOSITY",body:"Wewnętrzny stan ciekawości i eksploracji.",read:"Może być pobudzany przez nowe sytuacje, pytania i brak znajomości środowiska."},
  "stress":{title:"STRESS",body:"Wewnętrzny stan związany z threat, odrzuceniem i niekorzystnymi bodźcami.",read:"Wyższy poziom oznacza silniejszą aktywację zespołu STRESS, nie diagnozę emocji."},
  "satiety":{title:"SATIETY",body:"Wewnętrzny stan nasycenia/stabilizacji po kontakcie i nagrodzie.",read:"Może przeciwdziałać ciągłemu szukaniu nowych bodźców."},
  "arousal":{title:"AROUSAL",body:"Globalny stan pobudzenia neuronalnego używany także przy generowaniu języka.",read:"Wyższy poziom może zwiększać reaktywność i losowość generatora."},
  "social-drive-cue":{title:"Social need cue",body:"Sensoryczny bodziec informujący connectome, że Mucha długo pozostaje poza voice przy dostępnych ludziach.",read:"To wejście do sieci. Nie wymusza join."},
  "social-fatigue-cue":{title:"Social fatigue cue",body:"Sensoryczny sygnał zmęczenia długim pobytem z ludźmi na voice.",read:"Ma zwiększać szansę move/leave poprzez connectome, a nie przez bezpośredni if."},
  "habituation-cue":{title:"Habituation cue",body:"Sygnał przyzwyczajenia do powtarzającej się sceny/kanału.",read:"Rośnie przy braku zmian i może osłabiać atrakcyjność pozostawania."},
  "exploration-cue":{title:"Exploration cue",body:"Sensoryczny sygnał zachęcający do sprawdzenia innych kanałów lub nowych sytuacji.",read:"Jego wpływ zależy od rzeczywistych ścieżek connectome do readoutów."},
  "predicted-reward":{title:"Przewidywany reward",body:"Oczekiwana wartość nagrody dla bieżącej sceny/akcji wyliczona z pamięci epizodycznej.",read:"Dodatni oznacza, że podobne sytuacje wcześniej kończyły się korzystniej."},
  "prediction-error":{title:"Prediction error",body:"Różnica między faktycznym rewardem a tym, czego Mucha spodziewała się na podstawie pamięci.",read:"Dodatni = było lepiej niż oczekiwano; ujemny = gorzej. To ważny sygnał uczenia."},
  "prediction-correction":{title:"Korekta connectomu",body:"Wielkość korekty neuronalnej zastosowanej po błędzie predykcji.",read:"Im większy błąd i dostępny ślad, tym mocniej doświadczenie może zmienić przyszłe zachowanie."},
  "episodic":{title:"Epizody",body:"Liczba zapisanych doświadczeń/scen w pamięci epizodycznej.",read:"Epizod łączy kontekst, akcję, przewidywany i faktyczny reward."},
  "credit-queue":{title:"Credit queue",body:"Kolejka niedawnych decyzji czekających na późniejszy reward/punish.",read:"Pozwala przypisać feedback do akcji, która wydarzyła się wcześniej, zamiast tylko do ostatniego ticka."},
  "memory-replay":{title:"Memory Replay",body:"Odtwarzanie ważnych epizodów podczas ciszy jako słabszych bodźców dla connectomu.",read:"Ma utrwalać powtarzające się doświadczenia i wygaszać przypadkowe ślady."},
  "semantic-memory":{title:"Pamięć semantyczna",body:"Uogólnienie z wielu epizodów. Zamiast pamiętać tylko konkretną scenę, Mucha zbiera statystyki typu użytkownik→akcja, kanał→akcja, stan→akcja i użytkownik+kanał→akcja.",read:"Expected reward mówi kierunek doświadczenia, confidence mówi jak wiarygodne jest uogólnienie, a signal jest tym, co faktycznie trafia jako signed cue do connectomu."},
  "semantic-signal":{title:"Semantic signal",body:"Iloczyn oczekiwanego rewardu i confidence uogólnienia.",read:"Dodatni pobudza sensoryczną drogę do akcji, ujemny ją hamuje. Sam action score nie jest edytowany bezpośrednio."},
  "uncertainty":{title:"Semantic uncertainty",body:"Miara 0–1 opisująca jak mało Mucha wie o kombinacji kanału, użytkowników, stanu i możliwych akcji.",read:"1.0 = prawie brak doświadczenia. 0.0 = wysoka znajomość. Niepewność pobudza attractor CURIOSITY."},
  "information-gain":{title:"Information gain",body:"Spadek semantic uncertainty po nowym doświadczeniu.",read:"Jeżeli niepewność realnie spadła, decyzja może dostać mały intrinsic reward."},
  "curiosity-cue":{title:"Uncertainty → CURIOSITY",body:"Niepewność jest zamieniana na sensory cue wejściowy attractora CURIOSITY.",read:"Cue przechodzi przez neuronalny attractor i connectome do EXPLORE/MOVE. To nie jest ręczny bonus do action score."},
  "memory-scenes":{title:"Memory scenes",body:"Liczba unikalnych scen/kontekstów utrzymywanych przez pamięć epizodyczną.",read:"Kilka epizodów może należeć do tej samej sceny."},
  "consolidated":{title:"Consolidated",body:"Liczba scen, których strength przekroczył próg konsolidacji.",read:"Takie wspomnienia są bardziej odporne na zapominanie i częściej trafiają do replay."},
  "reward-opportunity":{title:"Reward opportunity",body:"Kanał/scena oznaczona jako potencjalna możliwość zdobycia pozytywnego reinforcement.",read:"To zachęta sensoryczna; nadal connectome musi wyprodukować odpowiednią decyzję."},
  "cue-effective":{title:"Cue effective",body:"Efektywna siła bodźca reward opportunity po uwzględnieniu bieżącego stanu i parametrów.",read:"Wyższa wartość oznacza silniejszy sygnał wejściowy skierowany w stronę JOIN."},
  "threat":{title:"Threat",body:"Poziom aktywnego zagrożenia voice, np. overstay albo Chaser.",read:"Wyższy threat pobudza ścieżki ucieczki i może zwiększać move/leave."},
  "overstay":{title:"Overstay",body:"Czas przebywania na jednym voice ponad limit komfortowego dwell.",read:"Po przekroczeniu limitu może pojawić się kara za stay i rosnący threat."},
  "chaser":{title:"Chaser",body:"Stan wykrytego bota-predatora Mucha Chaser.",read:"ACTIVE oznacza aktywną pogoń/panic; wtedy część normalnych ograniczeń ruchu jest rozluźniana."},
  "propagation":{title:"Propagation",body:"Liczba ticków connectomu wykonanych po podaniu danego bodźca motywacyjnego.",read:"Więcej ticków daje sygnałowi więcej czasu na przejście realnymi krawędziami FAFB."},
  "guided-reach":{title:"Guided reach",body:"Miara strukturalnego dotarcia wybranego sensory input do konkretnego readoutu przez połączenia connectomu.",read:"Wyższe reach sugeruje, że bodziec ma biologicznie-spójną drogę do danego outputu w modelu."},
  "effective-move":{title:"Effective move",body:"Końcowy wynik używany przy decyzji o zmianie voice po uwzględnieniu dynamicznych modyfikatorów.",read:"Porównaj z surowym voice_move, żeby zobaczyć wpływ threat/homeostazy."},
  "effective-margin":{title:"Effective margin",body:"Aktualny wymagany margines przewagi move nad pozostaniem.",read:"Niższy margines ułatwia ucieczkę w sytuacji zagrożenia."}
};

const HELP_LABEL_KEYS={
  "SOCIAL NEED":"social-need","CURIOSITY":"curiosity","STRESS":"stress","SATIETY":"satiety","AROUSAL":"arousal",
  "Social need cue":"social-drive-cue","Social fatigue cue":"social-fatigue-cue","Habituation cue":"habituation-cue","Exploration cue":"exploration-cue",
  "Przewidywany reward":"predicted-reward","Prediction error":"prediction-error","Korekta connectomu":"prediction-correction",
  "Epizody":"episodic","Credit queue":"credit-queue","Replay":"memory-replay","Memory scenes":"memory-scenes","Consolidated":"consolidated","Semantic entries":"semantic-memory","Semantic signal":"semantic-signal","Uncertainty":"uncertainty","Information gain":"information-gain","Curiosity cue":"curiosity-cue",
  "Reward opportunity":"reward-opportunity","Cue effective":"cue-effective","Threat":"threat","Overstay":"overstay","Chaser":"chaser",
  "Propagation":"propagation","Guided reach":"guided-reach","Effective move":"effective-move","Effective margin":"effective-margin"
};
function helpKeyForLabel(label){return HELP_LABEL_KEYS[String(label||"").trim()]||null}

function helpDot(key){
  return HELP[key]?'<span class="help-dot" data-help-key="'+esc(key)+'" tabindex="0">?</span>':'';
}
function showHelp(dot){
  const row=HELP[dot?.dataset?.helpKey],tip=$("help-tooltip");
  if(!row||!tip)return;
  tip.innerHTML='<div class="tt-title">'+esc(row.title)+'</div><div class="tt-body">'+esc(row.body)+'</div>'+
    (row.read?'<div class="tt-read"><b>Jak czytać:</b> '+esc(row.read)+'</div>':'');
  tip.classList.add("show");tip.setAttribute("aria-hidden","false");
  const r=dot.getBoundingClientRect(),pad=10,tw=Math.min(360,window.innerWidth-24);
  tip.style.width=tw+"px";
  tip.style.left=Math.min(window.innerWidth-tw-pad,Math.max(pad,r.left+r.width/2-tw/2))+"px";
  tip.style.top=(r.bottom+8)+"px";
  const tr=tip.getBoundingClientRect();
  if(tr.bottom>window.innerHeight-pad)tip.style.top=Math.max(pad,r.top-tr.height-8)+"px";
}
function hideHelp(){const t=$("help-tooltip");if(t){t.classList.remove("show");t.setAttribute("aria-hidden","true")}}
function enhanceHelp(){
  document.querySelectorAll("[data-help-label]").forEach(label=>{
    const key=label.dataset.helpLabel;
    if(HELP[key]&&!label.querySelector(".help-dot"))label.insertAdjacentHTML("beforeend",helpDot(key));
  });
  document.querySelectorAll(".help-dot").forEach(dot=>{
    if(dot.dataset.helpBound)return;dot.dataset.helpBound="1";
    dot.addEventListener("mouseenter",()=>showHelp(dot));dot.addEventListener("mouseleave",hideHelp);
    dot.addEventListener("focus",()=>showHelp(dot));dot.addEventListener("blur",hideHelp);
    dot.addEventListener("click",e=>{e.stopPropagation();showHelp(dot)});
  });
}
function setDetailsMode(mode){
  const full=mode==="full";
  document.body.classList.toggle("guided-simple",!full);
  $("mode-simple")?.classList.toggle("active",!full);$("mode-full")?.classList.toggle("active",full);
  try{localStorage.setItem("mucha-details-mode",full?"full":"simple")}catch(_){}
}
function initGuide(){
  let mode="simple";try{mode=localStorage.getItem("mucha-details-mode")||"simple"}catch(_){}
  setDetailsMode(mode);
  $("mode-simple")?.addEventListener("click",()=>setDetailsMode("simple"));
  $("mode-full")?.addEventListener("click",()=>setDetailsMode("full"));
  enhanceHelp();
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

function renderActionPolicy(p){
  const root=$("action-policy-debug");
  p=p||{};
  if(!p.enabled){
    root.innerHTML='<div class="reason">Learned Action Policy jest wyłączone. Decyzje używają surowych readoutów connectomu.</div>';
    return;
  }
  const gates=p.gates||{};
  const gateRows=Object.entries(gates).map(([name,g])=>{
    const competition=g.competition||{};
    if(g.decision_mode==="connectome-competition"){
      return '<span class="policy-gate '+(g.passed?'pass':'fail')+'">'+esc(name)+
        ' • COMPETE vs STAY'+
        ' • raw '+Number(g.raw_score||0).toFixed(3)+
        ' • effective '+Number(g.effective_score||0).toFixed(3)+
        ' • winner '+esc(competition.action||"—")+
        ' • margin '+Number(competition.margin||0).toFixed(3)+
        ' • '+(g.passed?'WIN':'LOSE')+'</span>';
    }
    return '<span class="policy-gate '+(g.passed?'pass':'fail')+'">'+esc(name)+
      ' • LEGACY threshold'+
      ' • raw '+Number(g.raw_score||0).toFixed(3)+
      ' • effective '+Number(g.effective_score||0).toFixed(3)+
      ' • próg '+Number(g.base_threshold||0).toFixed(3)+
      ' • '+(g.passed?'PASS':'BLOCK')+'</span>';
  }).join("");
  const rows=actionOrder.map(name=>{
    const a=(p.actions||{})[name]||{};
    const bias=Number(a.bias||0), raw=Number(a.raw_score||0), eff=Number(a.effective_score||0);
    const cls=bias>0.0005?'policy-pos':bias<-.0005?'policy-neg':'';
    return '<tr><td class="policy-action">'+esc(name)+'</td>'+
      '<td>'+raw.toFixed(3)+'</td>'+
      '<td class="'+cls+'">'+(bias>=0?'+':'')+bias.toFixed(3)+'</td>'+
      '<td><b>'+eff.toFixed(3)+'</b></td>'+
      '<td>'+(Number(a.reward_ema||0)>=0?'+':'')+Number(a.reward_ema||0).toFixed(3)+'</td>'+
      '<td>'+Number(a.updates||0).toLocaleString("pl-PL")+'</td></tr>';
  }).join("");
  const last=p.last_update||{};
  root.innerHTML=
    '<div class="policy-gates">'+gateRows+'</div>'+
    '<div class="voice-note" style="margin:0 0 8px">Ostatnia aktualizacja: <b>'+esc(last.action||"brak")+'</b> • reward '+
      (Number(last.reward||0)>=0?'+':'')+Number(last.reward||0).toFixed(3)+
      ' • Δ bias '+(Number(last.delta||0)>=0?'+':'')+Number(last.delta||0).toFixed(4)+
      ' • lr '+Number(p.learning_rate||0).toFixed(3)+
      ' • max |bias| '+Number(p.max_bias||0).toFixed(2)+'</div>'+
    '<div class="policy-wrap"><table class="policy-table"><thead><tr><th>Akcja</th><th>Raw connectome</th><th>Learned bias</th><th>Effective</th><th>Reward EMA</th><th>Updates</th></tr></thead><tbody>'+rows+'</tbody></table></div>';
}

function renderDecisionTrace(t,viewMode="live"){
  const root=$("decision-trace"),meta=$("decision-trace-meta");
  t=t||{};
  if(!t.checked_at){
    meta.textContent="czekam na pierwszy cykl decyzyjny…";
    root.innerHTML='<div class="trace-empty">Brak trace. Pojawi się po pierwszej decyzji tekstowej albo voice.</div>';
    return;
  }
  const age=Math.max(0,Date.now()/1000-Number(t.checked_at||0));
  meta.textContent=(viewMode==="history"?"HISTORIA • ":"LIVE • ")+String(t.source_label||t.kind||"decision")+" • "+String(t.guild||"—")+" • "+age.toFixed(age<10?1:0)+" s temu";
  const focus=t.attention_focus||{};
  const states=t.internal_states||{},stateMap=states.states||{};
  const dominant=states.dominant||"—",dominantLevel=Number(states.dominant_level||0);
  const nm=t.neuromodulators||{};
  const read=t.readout||{},mem=t.memory||{},sig=t.signals||{};
  const raw=read.raw_score==null?"—":Number(read.raw_score).toFixed(3);
  const effective=read.effective_score==null?"—":Number(read.effective_score).toFixed(3);
  const threshold=read.threshold==null?"—":Number(read.threshold).toFixed(3);
  const learned=read.learned_raw_threshold==null?"—":Number(read.learned_raw_threshold).toFixed(3);
  const policyBias=Number(read.policy_bias||0);
  const readExtra=read.threshold==null
    ? "runner-up "+esc(read.runner_up||"—")+" "+(read.runner_up_score==null?"":Number(read.runner_up_score).toFixed(3))+(read.margin==null?"":" • margin "+Number(read.margin).toFixed(3))
    : "threshold "+threshold+" • learned raw "+learned+" • bias "+(policyBias>=0?"+":"")+policyBias.toFixed(3);
  const focusText=focus.label||focus.key
    ? esc(focus.label||focus.key)+" • "+Number(focus.score||0).toFixed(3)
    : "brak zapisanego focusu";
  const neuralText=dominant!=="—"
    ? esc(dominant)+" • "+(dominantLevel*100).toFixed(0)+"%"
    : "brak dominującego attractora";
  const modParts=["dopamine","serotonin","octopamine"].filter(k=>nm[k]).map(k=>k+" "+Number((nm[k]||{}).level||0).toFixed(2));
  const memoryParts=[];
  if(mem.predicted_reward!=null) memoryParts.push("pred reward "+Number(mem.predicted_reward||0).toFixed(3));
  if(Number(mem.semantic_signal||0)!==0) memoryParts.push("semantic "+(Number(mem.semantic_signal)>=0?"+":"")+Number(mem.semantic_signal).toFixed(3));
  if(mem.uncertainty!=null) memoryParts.push("uncertainty "+Number(mem.uncertainty||0).toFixed(3));
  if(mem.affinity!=null) memoryParts.push("affinity "+(Number(mem.affinity)>=0?"+":"")+Number(mem.affinity).toFixed(2));
  if(Number(mem.information_gain||0)!==0) memoryParts.push("info gain "+Number(mem.information_gain).toFixed(3));
  const constraints=(t.constraints||[]);
  const chips=[];
  chips.push('<span class="trace-chip '+(read.passed===false?'bad':'good')+'">readout '+esc(read.action||"—")+' raw '+raw+' → effective '+effective+'</span>');
  if(modParts.length) chips.push('<span class="trace-chip">neuromod: '+esc(modParts.join(" • "))+'</span>');
  if(memoryParts.length) chips.push('<span class="trace-chip">memory: '+esc(memoryParts.join(" • "))+'</span>');
  Object.entries(sig).slice(0,7).forEach(([k,v])=>{
    if(v==null||typeof v==="object")return;
    chips.push('<span class="trace-chip">'+esc(k)+' '+esc(typeof v==="number"?Number(v).toFixed(3):v)+'</span>');
  });
  constraints.forEach(x=>chips.push('<span class="trace-chip bad">'+esc(x)+'</span>'));
  if(!constraints.length)chips.push('<span class="trace-chip good">brak blokad wykonania</span>');
  root.innerHTML=
    '<div class="trace-verdict">'+
      '<div><small>Decyzja runtime</small><strong>'+esc(t.decision||"—")+'</strong><div class="trace-reason">'+esc(t.reason||"—")+'</div></div>'+
      '<div><small>Faktyczna akcja</small><strong class="'+(String(t.actual_action||"").includes("BRAK")?"warn":"ok")+'">'+esc(t.actual_action||"—")+'</strong><div class="trace-reason">'+esc(t.source_label||t.kind||"—")+'</div></div>'+
    '</div>'+
    '<div class="trace-grid">'+
      '<div class="trace-stage"><small>1 • BODZIEC</small><b>'+esc(t.stimulus||"—")+'</b><em>Wejście, które rozpoczęło ten trace.</em></div>'+
      '<div class="trace-stage"><small>2 • UWAGA</small><b>'+focusText+'</b><em>Najsilniejszy zapis working memory przy decyzji.</em></div>'+
      '<div class="trace-stage"><small>3 • STAN NEURALNY</small><b>'+neuralText+'</b><em>'+esc(modParts.length?modParts.join(" • "):"connectome state po propagacji")+'</em></div>'+
      '<div class="trace-stage"><small>4 • READOUT / POLICY</small><b>'+esc(read.action||"—")+' • '+raw+' → '+effective+'</b><em>'+esc(readExtra)+'</em></div>'+
      '<div class="trace-stage"><small>5 • PAMIĘĆ / BLOKADY</small><b>'+esc(memoryParts.length?memoryParts.join(" • "):"brak silnego dodatkowego sygnału")+'</b><em>'+esc(constraints.length?constraints.join(" • "):"brak blokad")+'</em></div>'+
    '</div>'+
    '<div class="trace-chips">'+chips.join("")+'</div>';
}
let decisionTraceHistory=[];
let decisionTraceLive={};
let selectedDecisionTraceId=null;
let decisionTraceFilter="all";

function decisionTraceById(id){
  return decisionTraceHistory.find(x=>Number(x.history_id)===Number(id))||null;
}
function selectDecisionTrace(id){
  selectedDecisionTraceId=id==null?null:Number(id);
  const selected=selectedDecisionTraceId==null?null:decisionTraceById(selectedDecisionTraceId);
  if(selected){
    renderDecisionTrace(selected,"history");
    $("decision-history-status").textContent="ZAMROŻONY • #"+selected.history_id+" • kliknij LIVE, aby wrócić";
  }else{
    selectedDecisionTraceId=null;
    renderDecisionTrace(decisionTraceLive||{},"live");
    $("decision-history-status").textContent="LIVE • ostatnie decyzje";
  }
  renderDecisionTraceHistoryList();
}
function setDecisionTraceFilter(filter){
  decisionTraceFilter=["voice","text"].includes(filter)?filter:"all";
  ["all","voice","text"].forEach(name=>$("trace-filter-"+name)?.classList.toggle("active",name===decisionTraceFilter));
  renderDecisionTraceHistoryList();
}
function renderDecisionTraceHistoryList(){
  const root=$("decision-trace-history");
  $("trace-live-btn")?.classList.toggle("active",selectedDecisionTraceId==null);
  const rows=decisionTraceHistory
    .filter(x=>decisionTraceFilter==="all"||String(x.kind||"")===decisionTraceFilter)
    .slice()
    .reverse();
  if(!rows.length){
    root.innerHTML='<div class="trace-history-empty">Brak decyzji dla tego filtra.</div>';
    return;
  }
  root.innerHTML=rows.map(x=>{
    const id=Number(x.history_id||0);
    const time=new Date(Number(x.checked_at||0)*1000).toLocaleTimeString("pl-PL");
    const kind=String(x.kind||"—");
    const selected=id===selectedDecisionTraceId?" selected":"";
    return '<button type="button" class="trace-history-item'+selected+'" data-kind="'+esc(kind)+'" onclick="selectDecisionTrace('+id+')">'+
      '<span class="trace-history-time">'+esc(time)+'</span>'+
      '<span class="trace-history-kind">'+esc(kind)+'</span>'+
      '<span class="trace-history-decision">'+esc(x.decision||"—")+'</span>'+
      '<span class="trace-history-action">'+esc(x.actual_action||x.reason||"—")+'</span>'+
    '</button>';
  }).join("");
}
function renderDecisionTraceHistory(items,latest){
  decisionTraceHistory=Array.isArray(items)?items:[];
  decisionTraceLive=latest||{};
  if(
    selectedDecisionTraceId!=null
    && !decisionTraceById(selectedDecisionTraceId)
  ){
    selectedDecisionTraceId=null;
  }
  if(selectedDecisionTraceId==null){
    renderDecisionTrace(decisionTraceLive,"live");
    $("decision-history-status").textContent="LIVE • "+decisionTraceHistory.length+" zapisanych decyzji";
  }else{
    const selected=decisionTraceById(selectedDecisionTraceId);
    if(selected)renderDecisionTrace(selected,"history");
    $("decision-history-status").textContent="ZAMROŻONY • #"+selectedDecisionTraceId+" • "+decisionTraceHistory.length+" w buforze";
  }
  renderDecisionTraceHistoryList();
}
function renderAttention(a){
  const root=$("attention-debug");
  a=a||{};
  if(!a.enabled){
    root.innerHTML='<div class="reason">Attention / Working Memory jest wyłączone w konfiguracji.</div>';
    return;
  }
  const guilds=(a.guilds||[]).filter(g=>(g.items||[]).length||(g.working_memory||[]).length);
  if(!guilds.length){
    root.innerHTML='<div class="reason">Brak aktywnego kontekstu. Pierwsza wiadomość tekstowa albo transkrypcja voice utworzy ślad uwagi.</div>';
    return;
  }
  root.innerHTML=guilds.map(g=>{
    const focus=g.focus||null;
    const items=(g.items||[]).slice(0,8);
    const memories=(g.working_memory||[]).slice(0,5);
    const focusText=focus
      ? '<div class="attention-focus">'+esc(focus.label||focus.key||"—")+
        '<small>'+esc(focus.kind||"item")+' • score '+Number(focus.score||0).toFixed(3)+' '+helpDot("attention-score")+
        ' • neural '+Number(focus.neural||0).toFixed(3)+' '+helpDot("attention-neural")+
        ' • age '+Number(focus.age||0).toFixed(0)+' s '+helpDot("attention-age")+'</small></div>'
      : '<div class="attention-focus">—<small>brak dominującego focusu</small></div>';
    const itemRows=items.map(x=>{
      const width=Math.max(0,Math.min(100,Number(x.score||0)*100));
      return '<div class="attention-row"><b title="'+esc(x.key||"")+'">'+esc(x.label||x.key||"—")+'</b>'+
        '<div class="track"><div class="fill" style="width:'+width.toFixed(1)+'%"></div></div>'+
        '<output>'+Number(x.score||0).toFixed(2)+'</output></div>';
    }).join("")||'<div class="voice-note">Brak elementów.</div>';
    const memoryRows=memories.map(m=>
      '<div class="attention-memory-row"><small>'+esc(m.source||"—")+' • '+esc(m.user||"—")+
      ' • '+esc(m.channel||"—")+' • '+Number(m.age||0).toFixed(0)+' s</small><span>'+esc(m.text||"—")+'</span>'+
      ((m.topics||[]).length?'<div class="voice-note">topics: '+esc((m.topics||[]).join(" • "))+'</div>':'')+'</div>'
    ).join("")||'<div class="voice-note">Brak working memory.</div>';
    return '<div class="voice-note" style="margin:0 0 7px"><b>'+esc(g.guild||"serwer")+'</b> • half-life '+Number(a.half_life_seconds||0).toFixed(0)+' s • memory '+Number(a.working_memory_seconds||0).toFixed(0)+' s</div>'+
      '<div class="attention-shell">'+
        '<div class="attention-panel"><h3>🎯 Focus</h3>'+focusText+'</div>'+
        '<div class="attention-panel"><h3>🧠 Aktywna uwaga</h3>'+itemRows+'</div>'+
        '<div class="attention-panel"><h3>🗂 Working memory</h3><div class="attention-memory">'+memoryRows+'</div></div>'+
      '</div>';
  }).join('<div class="voice-server-sep" style="margin:12px 0"></div>');
}

function renderVoiceDebug(items){
  const root=$("voice-debug");
  const openDetailKeys=new Set(
    [...root.querySelectorAll("details[open][data-detail-key]")]
      .map(el=>el.dataset.detailKey)
      .filter(Boolean)
  );
  if(!Array.isArray(items)||!items.length){
    root.innerHTML='<div class="reason">Brak danych. Jeśli voice jest wyłączony w configu, pętla diagnostyczna nie wystartuje.</div>';
    return;
  }

  const pct=v=>Math.max(0,Math.min(100,Number(v||0)*100));
  const drive=(label,value,tone="")=>{
    const hk=helpKeyForLabel(label);
    return '<div class="drive-row"><label>'+esc(label)+(hk?' '+helpDot(hk):'')+'</label>'+
      '<div class="drive-track"><div class="drive-fill '+tone+'" style="width:'+pct(value).toFixed(1)+'%"></div></div>'+
      '<output>'+pct(value).toFixed(0)+'%</output></div>';
  };
  const score=(label,value,active=false)=>
    '<div class="voice-score"><span class="'+(active?'dominant':'')+'">'+(active?'▶ ':'')+esc(label)+'</span>'+
    '<div class="track"><div class="fill" style="width:'+pct(value).toFixed(1)+'%"></div></div>'+
    '<span class="val">'+Number(value||0).toFixed(3)+'</span></div>';
  const kpi=(label,value,cls="")=>{
    const hk=helpKeyForLabel(label);
    return '<div class="voice-kpi"><small>'+esc(label)+(hk?' '+helpDot(hk):'')+'</small><strong class="'+cls+'">'+value+'</strong></div>';
  };

  root.innerHTML=items.map(v=>{
    const s=v.scores||{},bd=v.brain_decision||{},neural=!!v.connectome_voice_control;
    const vs=v.voice_sensory||{},vsBrain=vs.brain||{};
    const winner=String(bd.action||"—");
    const tie=bd.tie_evidence||{};
    const tieSummary=Object.entries(tie).map(([name,row])=>
      name+" "+Number((row||{}).support||0).toFixed(4)
    ).join(" • ");
    const onVoice=!!v.current;
    const overstay=Number(v.overstay_seconds||0);
    const punished=Boolean(v.overstay_punished);
    const expected=Object.entries(v.prediction_expected||{}).map(([k,x])=>
      k+" "+Number(x||0).toFixed(2)
    ).join(" • ")||"—";
    const guided=Object.entries(v.homeostasis_guided||{}).map(([k,x])=>
      k+" "+Number((x||{}).reach_max||0).toFixed(3)
    ).join(" • ")||"—";
    const predErr=v.prediction_error==null?null:Number(v.prediction_error);
    const predClass=predErr==null?"":predErr>0?"ok":predErr<0?"no":"";
    const replay=v.memory_replay||{};
    const neuralInternal=(v.internal_states||{}).states||{};
    const internalLevel=name=>Math.max(0,Math.min(1,Number((neuralInternal[name]||{}).level||0)));
    const replayLast=(replay.last||[]).slice(-1)[0]||null;
    const replaySummary=replayLast
      ? (String(replayLast.action||"—")+" • "+String(replayLast.channel_name||"poza VC")+" • reward "+Number(replayLast.replay_reward||0).toFixed(3)+" • memory "+(Math.max(0,Math.min(1,Number(replayLast.memory_strength||0)))*100).toFixed(0)+"% • "+Number(replayLast.steps||0)+" tick • "+Number(replayLast.changed_synapses||0)+" synaps")
      : (replay.reason||"brak replay");
    const memoryRows=(v.episodic_recent||[]).slice(-3).reverse().map(ep=>{
      const people=(ep.user_names||[]).join(", ")||(ep.user_ids||[]).join(", ")||"—";
      const place=ep.channel_name||"poza VC";
      const err=Number(ep.prediction_error||0);
      return '<div class="memory-row"><b>'+esc(ep.action||"—")+'</b><span>'+esc(place)+' • '+esc(people)+'</span><em class="'+(err>0?"ok":err<0?"no":"")+'">'+(err>=0?"+":"")+err.toFixed(2)+'</em></div>';
    }).join("")||'<div class="voice-note">Brak zapisanych epizodów.</div>';
    const consolidatedRows=(v.episodic_top_memories||[]).slice(0,5).map(m=>{
      const people=(m.user_names||[]).join(", ")||(m.user_ids||[]).join(", ")||"—";
      const place=m.channel_name||"poza VC";
      const strength=Math.max(0,Math.min(1,Number(m.strength||0)));
      const status=String(m.status||"forming");
      return '<div class="memory-row"><b>'+esc(m.action||"—")+'</b><span>'+esc(place)+' • '+esc(people)+' • '+nfmt(m.event_count||0)+' evt / '+nfmt(m.replay_count||0)+' replay</span><em class="'+(status==="consolidated"?"ok":"")+'">'+(strength*100).toFixed(0)+'%</em></div>';
    }).join("")||'<div class="voice-note">Brak utrwalanych scen.</div>';
    const recallSummary=Object.entries(v.episodic_recall||{}).map(([k,x])=>
      k+" "+Number((x||{}).magnitude||0).toFixed(2)+" / "+Number((x||{}).reach_max||0).toFixed(3)
    ).join(" • ")||"—";
    const semanticSummary=Object.entries(v.semantic_recall||{}).map(([k,x])=>
      k+" reward "+Number((x||{}).expected_reward||0).toFixed(2)+
      " • conf "+(Number((x||{}).confidence||0)*100).toFixed(0)+"%"+
      " • signal "+Number((x||{}).signal||0).toFixed(2)
    ).join(" | ")||"—";
    const semanticGuidedSummary=Object.entries(v.semantic_guided||{}).map(([k,x])=>
      k+" "+String((x||{}).valence||"neutral")+" "+
      Number((x||{}).magnitude||0).toFixed(2)+
      " / reach "+Number((x||{}).reach_max||0).toFixed(3)
    ).join(" • ")||"—";
    const semanticRows=(v.semantic_top||[]).slice(0,6).map(m=>{
      const value=Number(m.expected_reward||0),conf=Number(m.confidence||0);
      const cls=value>0?"ok":value<0?"no":"";
      return '<div class="memory-row"><b>'+esc(m.action||"—")+'</b><span>'+
        esc(m.concept_type||"concept")+': '+esc(m.concept_key||"—")+
        ' • '+nfmt(m.observations||0)+' obs • conf '+(conf*100).toFixed(0)+'%</span>'+
        '<em class="'+cls+'">'+(value>=0?"+":"")+value.toFixed(2)+'</em></div>';
    }).join("")||'<div class="voice-note">Brak uogólnień z wystarczającą siłą.</div>';
    const uncertaintyRows=(v.uncertainty_channels||[]).slice(0,6).map(row=>{
      const u=Math.max(0,Math.min(1,Number(row.uncertainty||0)));
      return '<div class="memory-row"><b>'+esc(row.channel||"—")+'</b><span>'+
        nfmt(row.observations||0)+' obs • '+nfmt(row.known_pairs||0)+'/'+nfmt(row.possible_pairs||0)+' znanych par • '+nfmt(row.human_count||0)+' osób</span>'+
        '<em class="'+(u>.65?"warn":u<.25?"ok":"")+'">'+(u*100).toFixed(0)+'%</em></div>';
    }).join("")||'<div class="voice-note">Brak kanałów do oceny niepewności.</div>';
    const curiosityCue=v.uncertainty_curiosity_cue||{};
    const infoGain=v.information_gain_last||{};
    const currentTime=onVoice
      ? Number(v.dwell_elapsed||0).toFixed(0)+" / "+Number(v.maximum_dwell_seconds||0).toFixed(0)+" s"
      : Number(v.outside_seconds||0).toFixed(0)+" s poza VC";

    const channels=(v.channels||[]).map(ch=>{
      const aff=ch.affinity==null?"—":Number(ch.affinity).toFixed(3);
      const targetScore=neural
        ? (ch.neural_target_score==null?"—":Number(ch.neural_target_score).toFixed(3))
        : (ch.exploration_score==null?"—":Number(ch.exploration_score).toFixed(3));
      const novelty=ch.novelty==null?"—":(Number(ch.novelty)*100).toFixed(0)+"%";
      const uncertainty=ch.semantic_uncertainty==null?"—":(Number(ch.semantic_uncertainty)*100).toFixed(0)+"%";
      const visitAge=ch.visit_age==null?"never":Number(ch.visit_age).toFixed(0)+"s";
      const status=ch.eligible?(ch.reward_opportunity?"🎯 REWARD?":"OK"):ch.status;
      const cls=ch.eligible?"ok":(ch.status==="AFK"?"warn":"no");
      return '<tr>'+
        '<td>'+(ch.current?"▶ ":"")+esc(ch.name)+'</td>'+
        '<td>'+Number(ch.humans||0)+'</td>'+
        '<td>'+aff+'</td>'+
        '<td>'+targetScore+'</td>'+
        '<td>'+novelty+'</td>'+
        '<td>'+uncertainty+'</td>'+
        '<td>'+visitAge+'</td>'+
        '<td class="'+cls+'">'+esc(status)+'</td>'+
      '</tr>';
    }).join("");

    const actionScores=["voice_join","voice_move","voice_leave","stay"]
      .map(name=>score(name,Number(s[name]??0),winner===name)).join("");

    const rewardMode=String(v.reward_opportunity_guided_mode||"");
    const socialMode=String(v.social_drive_guided_mode||"");

    return '<div class="voice-shell">'+
      '<div class="voice-hero">'+
        '<div class="voice-hero-main">'+
          '<div class="voice-eyebrow">'+esc(v.guild||"serwer")+' • '+(neural?'CONNECTOME':'LEGACY')+'</div>'+
          '<div class="voice-hero-title">'+(onVoice?'🔊 '+esc(v.current):'🌙 Poza voice')+'</div>'+
          '<div class="voice-hero-sub">'+esc(v.decision||"ANALIZA")+' — '+esc(v.reason||"oczekiwanie na wynik")+'</div>'+
        '</div>'+
        '<div class="voice-box voice-hero-stat"><div class="voice-eyebrow">Decyzja</div><strong class="'+(winner==="stay"?"warn":"ok")+'">'+esc(winner)+'</strong><small>margin '+(bd.margin==null?"—":Number(bd.margin).toFixed(3))+'</small></div>'+
        '<div class="voice-box voice-hero-stat"><div class="voice-eyebrow">Czas</div><strong>'+esc(currentTime)+'</strong><small>scena '+Number(v.voice_scene_age||0).toFixed(0)+' s</small></div>'+
        '<div class="voice-box voice-hero-stat"><div class="voice-eyebrow">Ludzie</div><strong>'+Number(v.available_humans||0)+'</strong><small>dostępnych na VC</small></div>'+
      '</div>'+

      '<div class="voice-groups">'+
        '<div class="voice-box">'+
          '<h3>🧠 Decyzja connectomu</h3>'+
          actionScores+
          '<div class="voice-note">Runner-up: <b>'+esc(bd.runner_up||"—")+'</b> '+(bd.runner_up_score==null?"":Number(bd.runner_up_score).toFixed(3))+
          (bd.tie_break?' • tie-break: '+esc(bd.tie_break):'')+'</div>'+
        '</div>'+

        '<div class="voice-box">'+
          '<h3>🫀 Homeostaza sensoryczna</h3>'+
          drive("Social need cue",v.social_drive_level,"")+
          drive("Social fatigue cue",v.social_fatigue_level,"warn-fill")+
          drive("Habituation cue",v.habituation_level,"warn-fill")+
          drive("Exploration cue",v.exploration_drive_level,"")+
          '<div class="voice-note">To są wejścia sensoryczne. Poniżej widać faktyczny stan zespołów neuronów po propagacji.</div>'+
        '</div>'+

        '<div class="voice-box">'+
          '<h3>🧠 Neural internal states</h3>'+
          drive("SOCIAL NEED",internalLevel("social_need"),"")+
          drive("CURIOSITY",internalLevel("curiosity"),"")+
          drive("STRESS",internalLevel("stress"),"warn-fill")+
          drive("SATIETY",internalLevel("satiety"),"warn-fill")+
          drive("AROUSAL",internalLevel("arousal"),"")+
          '<div class="voice-note">Dominujący attractor: <b>'+esc((v.internal_states||{}).dominant||"—")+'</b> '+(Number((v.internal_states||{}).dominant_level||0)*100).toFixed(0)+'% • brak bezpośredniego action-score bonusu.</div>'+
        '</div>'+

        '<div class="voice-box">'+
          '<h3>🎙 Voice Sensory Bus — LIVE</h3>'+
          '<div class="voice-kpis">'+
            kpi("Status",esc(vs.status||"—"),vs.status==="SPEAKING"?"ok":vs.status==="SILENCE"?"warn":"")+
            kpi("PCM capture",vs.pcm_capture?"LIVE":"brak / poza VC",vs.pcm_capture?"ok":"warn")+
            kpi("Tryb rozmowy",esc(vs.conversation_mode||"—"),vs.conversation_mode==="CROSSTALK"?"warn":vs.conversation_mode==="DIALOGUE"?"ok":"")+
            kpi("Intensywność",(Number(vs.conversation_intensity||0)*100).toFixed(0)+"%")+
            kpi("Mówi teraz",String(Number(vs.speaker_count||0)))+
            kpi("Overlap teraz",String(Number(vs.overlap_count||0)),Number(vs.overlap_count||0)>0?"warn":"")+
            kpi("Speech ratio 60s",(Number(vs.speech_ratio_60s||0)*100).toFixed(0)+"%")+
            kpi("Unikalni mówcy 60s",String(Number(vs.unique_speakers_60s||0)))+
            kpi("Zmiany mówcy 60s",String(Number(vs.speaker_switches_60s||0)))+
            kpi("Overlap events 60s",String(Number(vs.overlap_events_60s||0)),Number(vs.overlap_events_60s||0)>1?"warn":"")+
            kpi("Śr. handoff",vs.mean_handoff_seconds==null?"—":Number(vs.mean_handoff_seconds).toFixed(2)+" s")+
            kpi("Śr. tura",Number(vs.mean_turn_seconds||0).toFixed(2)+" s")+
            kpi("Najdłuższa tura",Number(vs.longest_turn_seconds||0).toFixed(2)+" s",Number(vs.longest_turn_seconds||0)>=8?"warn":"")+
            kpi("Dominacja mówcy",(Number(vs.speaker_dominance||0)*100).toFixed(0)+"%")+
            kpi("Cisza",Number(vs.silence_seconds||0).toFixed(1)+" s")+
            kpi("Tempo",Number(vs.turns_per_minute||0)+" turn/min")+
            kpi("Ludzie tutaj",String(Number(vs.human_count||0)))+
            kpi("Znajomi gdzie indziej",Number(vs.other_familiar_humans||0)+" / "+Number(vs.other_voice_humans||0))+
            kpi("Reply po TTS",vs.reply_after_tts?(esc(vs.reply_user_name||vs.reply_user_id||"tak")+" • "+Number(vs.reply_tts_age_seconds||vs.reply_age_seconds||0).toFixed(1)+" s"):(vs.tts_pending_reply?("czeka • "+Number(vs.tts_age_seconds||0).toFixed(1)+" s"):"nie"),vs.reply_after_tts?"ok":vs.tts_pending_reply?"warn":"")+
            kpi("Raw cues",String(Number(vsBrain.cue_count||0)),vsBrain.mode==="raw-sensory-only"?"ok":"")+
          '</div>'+
          '<div class="voice-note"><b>Mówcy:</b> '+((vs.speakers||[]).length?(vs.speakers||[]).map(x=>esc(x.name||x.id)+" "+Number(x.speaking_for||0).toFixed(1)+"s • aff "+(Number(x.affinity||0)>=0?"+":"")+Number(x.affinity||0).toFixed(2)).join(" | "):"nikt")+'</div>'+
          '<div class="voice-note"><b>Dynamika 60s:</b> top '+esc(vs.top_speaker_name||"—")+" • "+(Number(vs.speaker_dominance||0)*100).toFixed(0)+"% speaker-time • speech "+Number(vs.speech_seconds_60s||0).toFixed(1)+"s • handoffów "+Number(vs.handoff_count_60s||0)+(vs.mean_handoff_seconds==null?"":" • avg "+Number(vs.mean_handoff_seconds).toFixed(2)+"s")+'</div>'+
          '<div class="voice-note"><b>Ostatni transcript:</b> '+(vs.last_transcript?(esc(vs.last_transcript.user_name||vs.last_transcript.user_id||"—")+" • "+Number(vs.last_transcript.word_count||0)+" słów • "+Number(vs.last_transcript.words_per_second||0).toFixed(2)+" sł/s • "+Number(vs.last_transcript.age_seconds||0).toFixed(1)+"s temu"):"—")+'</div>'+
          '<div class="voice-note"><b>Wejścia neuronalne:</b> '+((vsBrain.cues||[]).length?(vsBrain.cues||[]).slice(0,14).map(x=>esc(x.key)+" "+Number(x.magnitude||0).toFixed(2)).join(" • "):"—")+'</div>'+
          '<div class="voice-note"><b>Tryb:</b> '+esc(vsBrain.mode||"—")+' • action-guided: '+(vsBrain.action_guided?"TAK":"NIE")+' • direct action bias: '+(vsBrain.direct_action_bias?"TAK":"NIE")+'. Surowe sensory trafiają do connectomu; nie są ręcznym JOIN/MOVE/LEAVE score.</div>'+
        '</div>'+

        '<div class="voice-box">'+
          '<h3>🧪 Prediction / pamięć epizodyczna</h3>'+
          '<div class="voice-kpis">'+
            kpi("Przewidywany reward",Number(v.predicted_reward||0).toFixed(3))+
            kpi("Prediction error",predErr==null?"—":predErr.toFixed(3),predClass)+
            kpi("Korekta connectomu",Number(v.prediction_correction_applied||0).toFixed(3))+
            kpi("Epizody",String(Number(v.episodic_memory_size||0)))+
            kpi("SQLite",v.episodic_persistent?"ON":"OFF",v.episodic_persistent?"ok":"warn")+
            kpi("Predykcje",String(Number(v.episodic_prediction_count||0)))+
            kpi("Credit queue",String(Number(v.prediction_credit_queue_depth||0)))+
            kpi("Replay",esc(replay.state||"—"),replay.state==="REPLAY"?"ok":"")+
            kpi("Replay count",String(Number(replay.count||0)))+
            kpi("Memory scenes",String(Number(v.episodic_memory_scenes||0)))+
            kpi("Consolidated",String(Number(v.episodic_consolidated_scenes||0)),Number(v.episodic_consolidated_scenes||0)>0?"ok":"")+
          '</div>'+
          '<div class="voice-note"><b>Expected:</b> '+esc(expected)+'</div>'+
          '<div class="voice-note"><b>Recall → connectome:</b> '+esc(recallSummary)+'</div>'+
          '<div class="voice-note"><b>MEMORY REPLAY:</b> '+esc(replaySummary)+'</div>'+
          '<details class="mini-details" data-detail-key="memory-top-'+esc(v.guild||"server")+'"><summary>Najsilniejsze sceny • '+Number(v.episodic_consolidated_scenes||0)+' utrwalonych</summary><div class="memory-list">'+consolidatedRows+'</div></details>'+
          '<details class="mini-details" data-detail-key="memory-recent-'+esc(v.guild||"server")+'"><summary>Ostatnie epizody • pokaż 3 najnowsze</summary><div class="memory-list">'+memoryRows+'</div></details>'+
        '</div>'+

        '<div class="voice-box">'+
          '<h3>🧩 Pamięć semantyczna '+helpDot("semantic-memory")+'</h3>'+
          '<div class="voice-kpis">'+
            kpi("Semantic entries",String(Number(v.semantic_entries||0)))+
            kpi("Semantic signal",Object.values(v.semantic_recall||{}).length?String(Object.values(v.semantic_recall||{}).map(x=>Number((x||{}).signal||0).toFixed(2)).join(" / ")):"—")+
            kpi("Recall",v.semantic_memory_enabled?"ON":"OFF",v.semantic_memory_enabled?"ok":"warn")+
          '</div>'+
          '<div class="voice-note"><b>Uogólniony recall:</b> '+esc(semanticSummary)+'</div>'+
          '<div class="voice-note"><b>Signed cue → connectome:</b> '+esc(semanticGuidedSummary)+'</div>'+
          '<details class="mini-details" data-detail-key="semantic-top-'+esc(v.guild||"server")+'"><summary>Najsilniejsze uogólnienia</summary><div class="memory-list">'+semanticRows+'</div></details>'+
        '</div>'+

        '<div class="voice-box">'+
          '<h3>🔎 Curiosity / Uncertainty '+helpDot("uncertainty")+'</h3>'+
          '<div class="voice-kpis">'+
            kpi("Uncertainty",(Number(v.uncertainty_overall||0)*100).toFixed(0)+"%",Number(v.uncertainty_overall||0)>.65?"warn":"")+
            kpi("Curiosity cue",curiosityCue.magnitude==null?"—":Number(curiosityCue.magnitude||0).toFixed(3),Number(curiosityCue.magnitude||0)>0?"ok":"")+
            kpi("Information gain",infoGain.information_gain==null?"—":Number(infoGain.information_gain||0).toFixed(3),Number(infoGain.information_gain||0)>0?"ok":"")+
            kpi("Intrinsic reward",infoGain.intrinsic_reward==null?"—":("+"+Number(infoGain.intrinsic_reward||0).toFixed(3)),Number(infoGain.intrinsic_reward||0)>0?"ok":"")+
          '</div>'+
          '<div class="voice-note"><b>Ścieżka:</b> uncertainty '+Number(v.uncertainty_overall||0).toFixed(3)+
            ' → CURIOSITY cue '+Number(curiosityCue.magnitude||0).toFixed(3)+
            ' → '+Number(curiosityCue.entry_neurons||0)+' input n → attractor '+Number(curiosityCue.attractor_neurons||0)+' n → '+
            esc((curiosityCue.target_actions||[]).join(" / ")||"—")+'</div>'+
          '<div class="voice-note"><b>Ostatni information gain:</b> '+
            (infoGain.information_gain==null?'brak':
              Number(infoGain.uncertainty_before||0).toFixed(3)+' → '+Number(infoGain.uncertainty_after||0).toFixed(3)+
              ' • gain '+Number(infoGain.information_gain||0).toFixed(3)+
              ' • reward +'+Number(infoGain.intrinsic_reward||0).toFixed(3))+
          '</div>'+
          '<details class="mini-details" data-detail-key="uncertainty-top-'+esc(v.guild||"server")+'"><summary>Najbardziej nieznane kanały</summary><div class="memory-list">'+uncertaintyRows+'</div></details>'+
        '</div>'+

        '<div class="voice-box">'+
          '<h3>🎯 Social / reward</h3>'+
          '<div class="voice-kpis">'+
            kpi("Reward opportunity",v.reward_opportunity_channel?esc(v.reward_opportunity_channel):"—",v.reward_opportunity_channel?"ok":"")+
            kpi("Cue effective",Number(v.reward_opportunity_effective_strength||0).toFixed(2))+
            kpi("JOIN reward",Number(v.social_join_reward||0).toFixed(3),Number(v.social_join_reward||0)>0?"ok":"")+
            kpi("STAY punish",v.social_drive_stay_punished?Number(v.social_drive_stay_punish_amount||0).toFixed(3):"nie",v.social_drive_stay_punished?"no":"")+
          '</div>'+
          '<div class="voice-note">Reward znaleziony: <b class="'+(v.reward_opportunity_found?"ok":"")+'">'+(v.reward_opportunity_found?"TAK +"+Number(v.reward_opportunity_reward||0).toFixed(3):"nie")+'</b> • TTL '+Number(v.reward_opportunity_remaining||0).toFixed(0)+' s</div>'+
        '</div>'+

        '<div class="voice-box">'+
          '<h3>⚠ Threat / chaser</h3>'+
          '<div class="voice-kpis">'+
            kpi("Threat",(Number(v.threat_level||0)*100).toFixed(0)+"%",v.threat_active?"no":"ok")+
            kpi("Overstay",onVoice?overstay.toFixed(0)+" s":"—",overstay>0?"no":"")+
            kpi("Chaser",v.chaser_active?"ACTIVE":"idle",v.chaser_active?"no":"ok")+
            kpi("Escape target",esc(v.escape_target||"—"))+
          '</div>'+
          '<div class="voice-note">threat magnitude '+Number(v.threat_magnitude||0).toFixed(3)+
          (punished?' • overstay punish '+Number(v.overstay_punish_amount||0).toFixed(2):'')+
          (v.chaser_active?' • panic '+Number(v.chaser_remaining||0).toFixed(1)+' s':'')+'</div>'+
        '</div>'+

        '<div class="voice-box">'+
          '<h3>🔬 Co faktycznie dochodzi do readoutów</h3>'+
          '<div class="voice-kpis">'+
            kpi("Propagation",Number(v.motivation_propagation_steps||0)+" tick")+
            kpi("Action selection",neural?"NEURAL":"LEGACY",neural?"ok":"warn")+
            kpi("Target selection",esc(v.target_selection_source||"—"),neural?"ok":"warn")+
            kpi("Learning timing",v.learning_updates_do_not_override_current_decision?"NEXT TICK":"normal",v.learning_updates_do_not_override_current_decision?"ok":"")+
            kpi("Social guided",socialMode.startsWith("connectome-guided")?"OK":"—",socialMode.startsWith("connectome-guided")?"ok":"")+
            kpi("Reward guided",rewardMode.startsWith("connectome-guided")?"OK":"—",rewardMode.startsWith("connectome-guided")?"ok":"")+
            kpi("Tie evidence",tieSummary?"jest":"—",tieSummary?"ok":"")+
          '</div>'+
          '<div class="voice-note"><b>Homeostasis reach:</b> '+esc(guided)+'</div>'+
        '</div>'+
      '</div>'+

      '<div class="voice-decision"><b>'+esc(v.decision||"—")+'</b> — '+esc(v.reason||"—")+
        (neural?'<br><span style="color:var(--muted)">Connectome candidates:</span> '+Object.entries(bd.candidates||{}).map(([k,x])=>esc(k)+' '+Number(x).toFixed(3)).join(' • '):'')+
      '</div>'+

      '<details class="voice-technical" data-detail-key="technical-'+esc(v.guild||"server")+'">'+
        '<summary>▸ Szczegóły techniczne</summary>'+
        '<div class="voice-technical-body">'+
          '<div class="voice-tech-grid">'+
            kpi("Prediction context",esc(v.prediction_context||"—"))+
            kpi("Scene key",esc(v.prediction_scene_key||"—"))+
            kpi("Episodic DB",esc(v.episodic_database||"—"))+
            kpi("Temporal queue",Number(v.prediction_credit_queue_depth||0)+" / corrections "+Number(v.prediction_correction_queue_depth||0))+
            kpi("Replay reason",esc(replay.reason||"—"))+
            kpi("Semantic entries",String(Number(v.semantic_entries||0)))+
            kpi("Semantic recall",esc(semanticSummary))+
            kpi("Semantic guided",esc(semanticGuidedSummary))+
            kpi("Uncertainty",(Number(v.uncertainty_overall||0)*100).toFixed(0)+"%")+
            kpi("Curiosity cue",Number(curiosityCue.magnitude||0).toFixed(3))+
            kpi("Information gain",infoGain.information_gain==null?"—":Number(infoGain.information_gain||0).toFixed(3))+
            kpi("Info reward applied",Number(v.information_gain_reward_applied||0).toFixed(3))+
            kpi("neural tie-break",esc(bd.tie_break||"niepotrzebny"))+
            kpi("Tie evidence",esc(tieSummary||"—"))+
            kpi("Homeostasis paths",esc(guided))+
            kpi("Reward cue base",Number(v.reward_opportunity_strength||0).toFixed(2))+
            kpi("Guided JOIN",esc(rewardMode||"—")+" • "+Number(v.reward_opportunity_guided_neurons||0)+" n")+
            kpi("Guided reach",Number(v.reward_opportunity_guided_reach_mean||0).toFixed(4)+" / "+Number(v.reward_opportunity_guided_reach_max||0).toFixed(4))+
            kpi("Social guided",esc(socialMode||"—")+" • "+Number(v.social_drive_guided_neurons||0)+" n")+
            kpi("Ignored cue punish",v.reward_opportunity_stay_punished?Number(v.reward_opportunity_stay_punish_amount||0).toFixed(3):"nie",v.reward_opportunity_stay_punished?"no":"")+
            kpi("Minimum dwell",onVoice?Number(v.dwell_remaining||0).toFixed(1)+" s":"—")+
            kpi("Effective move",v.effective_move_score==null?"—":Number(v.effective_move_score).toFixed(3))+
            kpi("Effective margin",v.effective_move_margin==null?"—":Number(v.effective_move_margin).toFixed(3))+
            kpi("Chaser ID",esc(v.chaser_id||"—"))+
          '</div>'+
        '</div>'+
      '</details>'+

      '<details class="voice-technical" data-detail-key="channels-'+esc(v.guild||"server")+'"><summary>▸ Kanały głosowe ('+Number((v.channels||[]).length)+')</summary><div class="voice-technical-body"><div class="voice-table-wrap"><table><thead><tr><th>Kanał</th><th>Ludzie</th><th>Affinity</th><th>Target score</th><th>Novelty</th><th>Uncertainty</th><th>Last visit</th><th>Status</th></tr></thead><tbody>'+channels+'</tbody></table></div></div></details>'+
    '</div>';
  }).join('<div class="voice-server-sep"></div>');
  root.querySelectorAll("details[data-detail-key]").forEach(el=>{
    if(openDetailKeys.has(el.dataset.detailKey)) el.open=true;
  });
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
function renderAffective(a){
  a=a||{};
  const values=a.values||{},targets=a.targets||{};
  const labels={
    contentment:"CONTENTMENT",
    tension:"TENSION",
    curiosity:"CURIOSITY",
    social_longing:"SOCIAL LONGING",
    activation:"ACTIVATION"
  };
  const dominant=String(a.dominant||"—");
  const dominantValue=Number(a.dominant_value||0);
  $("affective-summary").innerHTML=
    '<b>'+esc(dominant.toUpperCase())+'</b> • '+(dominantValue*100).toFixed(0)+'%'+
    ' • feedback '+Number(a.feedback_gain||0).toFixed(2)+
    ' • reward trace '+(Number(a.reward_trace||0)>=0?"+":"")+Number(a.reward_trace||0).toFixed(3);
  $("affective-state-grid").innerHTML=Object.keys(labels).map(key=>{
    const value=Math.max(0,Math.min(1,Number(values[key]||0)));
    const target=Math.max(0,Math.min(1,Number(targets[key]||0)));
    const delta=target-value;
    return '<div class="kpi"><small>'+esc(labels[key])+'</small>'+
      '<strong>'+(value*100).toFixed(0)+'%</strong>'+
      '<div class="track" style="margin-top:7px"><div class="fill" style="width:'+(value*100).toFixed(1)+'%"></div></div>'+
      '<div class="footer">target '+(target*100).toFixed(0)+'% • Δ '+(delta>=0?"+":"")+(delta*100).toFixed(0)+' pp</div></div>';
  }).join("");
}
function renderMotivation(m){
  m=m||{};
  const rows=m.motivations||{};
  const order=["social","novelty","safety","rest"];
  const labels={social:"SOCIAL",novelty:"NOVELTY",safety:"SAFETY",rest:"REST"};
  const dominant=String(m.dominant||"—");
  const urgency=Number(m.dominant_urgency||0);
  const last=m.last_event||{};
  $("motivation-summary").innerHTML=
    '<b>'+esc(dominant.toUpperCase())+'</b> • urgency '+(urgency*100).toFixed(0)+'%'+
    ' • neural gain '+Number(m.neural_gain||0).toFixed(2)+
    (last.event&&last.event!=="startup"?' • last '+esc(last.event):'');
  $("motivation-state-grid").innerHTML=order.map(key=>{
    const x=rows[key]||{};
    const pressure=Math.max(0,Math.min(1,Number(x.pressure||0)));
    const frustration=Math.max(0,Math.min(1,Number(x.frustration||0)));
    const satiation=Math.max(0,Math.min(1,Number(x.satiation||0)));
    const u=Math.max(0,Math.min(1.5,Number(x.urgency||0)));
    return '<div class="kpi"><small>'+esc(labels[key]||key)+'</small>'+
      '<strong>'+(u*100).toFixed(0)+'%</strong>'+
      '<div class="track" style="margin-top:7px"><div class="fill" style="width:'+Math.min(100,u*100).toFixed(1)+'%"></div></div>'+
      '<div class="footer">pressure '+(pressure*100).toFixed(0)+'% • frustration '+(frustration*100).toFixed(0)+'% • satiation '+(satiation*100).toFixed(0)+'%</div></div>';
  }).join("");
}
function renderSleep(s){
  s=s||{};
  const state=String(s.state||"AWAKE");
  $("sleep-state").textContent=state;
  $("sleep-state").className=s.active?"warn":state==="COMPLETE"?"ok":"";
  const circadian=String(s.circadian_state||"AWAKE");
  const fatigue=Number(s.fatigue||0);
  const tiredThreshold=Number(s.tired_threshold||0);
  $("circadian-state").textContent=circadian;
  $("circadian-state").className=circadian==="TIRED"?"warn":circadian==="POST-SLEEP"?"ok":"";
  $("circadian-fatigue").textContent=(fatigue*100).toFixed(0)+"%";
  $("circadian-fatigue").className=fatigue>=tiredThreshold&&tiredThreshold>0?"warn":"";
  $("sleep-quiet").textContent=sessionDuration(s.quiet_for||0);
  $("sleep-cycle").textContent=Number(s.cycle||0)+" / "+Number(s.max_cycles||0);
  $("sleep-episodes").textContent=nfmt(s.episodes_replayed||0);
  $("sleep-synapses").textContent=nfmt(s.changed_synapses||0);
  $("sleep-neurons").textContent=nfmt(s.changed_neurons||0);
  $("sleep-semantic").textContent=nfmt(s.semantic_rehearsed||0);
  const md=Number(s.memory_strength_delta||0);
  $("sleep-memory-delta").textContent=(md>=0?"+":"")+md.toFixed(4);
  $("sleep-scenes").textContent=nfmt(s.consolidated_scenes||0);
  $("sleep-consolidated-synapses").textContent=nfmt(s.consolidated_synapses||0);
  $("sleep-progress").style.width=Math.max(0,Math.min(100,Number(s.progress||0)*100))+"%";
  const next=Number(s.next_cycle_in||0);
  const idle=Number(s.idle_required||0);
  $("sleep-summary").innerHTML=
    '<b>'+esc(state)+'</b> • '+esc(s.reason||"—")+
    ' • quiet '+sessionDuration(s.quiet_for||0)+
    (state==="AWAKE"&&idle?(' / sleep po '+sessionDuration(idle)):"")+
    (s.active&&next?(' • następny cykl za '+next.toFixed(0)+' s'):"")+
    (Number(s.post_sleep_remaining||0)>0?(' • post-sleep '+sessionDuration(s.post_sleep_remaining)):"")+
    ' • fatigue '+(fatigue*100).toFixed(0)+'%'+
    ' • fading synapses '+nfmt(s.fading_synapses||0);
  const last=(s.last||[]);
  $("sleep-last").innerHTML=last.length
    ? '<b>Ostatni cykl:</b> '+last.slice(0,5).map(x=>
        esc(x.action||"—")+
        ' • reward '+(Number(x.replay_reward||0)>=0?"+":"")+Number(x.replay_reward||0).toFixed(3)+
        ' • memory '+Number(x.memory_strength_before||0).toFixed(3)+'→'+Number(x.memory_strength||0).toFixed(3)+
        ' • syn '+nfmt(x.changed_synapses||0)+
        ' • semantic '+nfmt(x.semantic_rehearsed||0)
      ).join(' | ')
    : 'Brak replay w bieżącej/ostatniej sesji.';
}
function renderLearning(l){
  l=l||{};
  const amount=Number(l.amount||0);
  $("learn-reward").textContent=(amount>=0?"+":"")+amount.toFixed(2);
  $("learn-reward").className=amount>0?"ok":amount<0?"no":"";
  $("learn-action").textContent=l.action||"global / brak";
  $("learn-count").textContent=nfmt(l.changed_neurons||0);
  $("learn-max").textContent=Number(l.max_delta||0).toExponential(3);
  $("learn-synapses").textContent=nfmt(l.changed_synapses||0);
  $("learn-synapses-total").textContent=nfmt(l.learned_synapses||0);
  $("learn-summary").innerHTML='<b>'+esc(l.action||"brak targetu")+'</b> • średnie Δ bias: '+
    (Number(l.mean_delta||0)>=0?"+":"")+Number(l.mean_delta||0).toExponential(3)+
    ' • synapsy Δ max: '+Number(l.synaptic_max_delta||0).toExponential(3);

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
function renderSocialScenes(items){
  const root=$("social-scene-grid"),summary=$("social-scene-summary");
  items=Array.isArray(items)?items:[];
  if(!items.length){
    summary.innerHTML='<b>Brak dojrzałych scen.</b> Sytuacje pojawią się po powtarzalnych interakcjach voice.';
    root.innerHTML='<div class="trace-empty">Brak profili sytuacji.</div>';
    return;
  }
  const mature=items.filter(x=>Number(x.observations||0)>=2);
  summary.innerHTML='<b>'+nfmt(items.length)+' scen</b> • '+nfmt(mature.length)+' powtarzalnych • model łączy ludzi, miejsce, dynamikę i stan wewnętrzny';

  root.innerHTML=items.slice(0,16).map(p=>{
    const fam=Math.max(0,Math.min(1,Number(p.familiarity||0)));
    const conf=Math.max(0,Math.min(1,Number(p.confidence||0)));
    const val=Math.max(-1,Math.min(1,Number(p.valence||0)));
    const label=String(p.valence_label||"neutral");
    const preferred=p.preferred_action||null,avoided=p.avoided_action||null;
    const people=(p.people||[]).map(x=>esc(x.display_name||x.user_id)).join(' + ')||'bez ludzi';
    const inj=p.last_injection||{},brain=inj.brain||{};
    const actions=(p.actions||[]).slice(0,5).map(x=>
      '<span class="person-tag">'+esc(x.action||"—")+' '+(Number(x.signal||0)>=0?"+":"")+Number(x.signal||0).toFixed(2)+' ×'+nfmt(x.observations||0)+'</span>'
    ).join('');
    return '<div class="person-card">'+
      '<div class="person-head"><div><div class="person-name">🎭 '+esc(p.channel_name||"poza voice")+'</div><span class="person-id">'+esc(p.guild||"")+' • '+people+'</span></div>'+
      '<span class="person-valence '+esc(label)+'">'+esc(label.toUpperCase())+' '+(val>=0?"+":"")+val.toFixed(2)+'</span></div>'+
      '<div class="person-kpis">'+
        '<div class="person-kpi"><small>Familiarity</small><b>'+(fam*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Confidence</small><b>'+(conf*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Seen / outcomes</small><b>'+nfmt(p.seen_observations||0)+' / '+nfmt(p.outcome_observations||0)+'</b></div>'+
        '<div class="person-kpi"><small>Ludzie</small><b>'+nfmt(p.human_count||0)+'</b></div>'+
      '</div>'+
      '<div class="person-lines">'+
        '<div><b>Rozmowa:</b> '+esc(p.conversation_mode||"UNKNOWN")+' • intensity '+nfmt(p.intensity_bucket||0)+'/3 • speech '+nfmt(p.speech_bucket||0)+'/3</div>'+
        '<div><b>Stan Muchy:</b> '+esc(p.dominant_state||"none")+' '+nfmt(p.dominant_state_bucket||0)+'/3 • '+esc(p.context_signature||"")+'</div>'+
        '<div><b>Dobra akcja:</b> '+(preferred?esc(preferred.action)+' '+(Number(preferred.signal||0)>=0?"+":"")+Number(preferred.signal||0).toFixed(2):'—')+'</div>'+
        '<div><b>Zła akcja:</b> '+(avoided?esc(avoided.action)+' '+Number(avoided.signal||0).toFixed(2):'—')+'</div>'+
        '<div><b>Ostatnie wejście do connectomu:</b> '+(inj.checked_at?(nfmt(brain.cue_count||0)+' cues • '+Math.max(0,Date.now()/1000-Number(inj.checked_at)).toFixed(0)+'s temu'):'jeszcze nie użyta w bieżącej sesji')+'</div>'+
      '</div>'+
      '<div class="person-tags">'+(actions||'<span class="person-tag">brak wyników akcji</span>')+'</div>'+
    '</div>';
  }).join("");
}
function renderVoiceDynamicsProfiles(items){
  const root=$("voice-dynamics-grid"),summary=$("voice-dynamics-summary");
  items=Array.isArray(items)?items:[];
  if(!items.length){
    summary.innerHTML='<b>Brak wyuczonych wzorców.</b> Potrzebny jest realny reward/punish po decyzjach voice lub TTS.';
    root.innerHTML='<div class="trace-empty">Brak profili dynamiki rozmowy.</div>';
    return;
  }
  const learned=items.filter(x=>Number(x.outcome_observations||0)>0);
  summary.innerHTML='<b>'+nfmt(items.length)+' wzorców</b> • '+nfmt(learned.length)+' z realnym outcome • generalizacja niezależna od osoby i kanału';

  root.innerHTML=items.slice(0,16).map(p=>{
    const fam=Math.max(0,Math.min(1,Number(p.familiarity||0)));
    const conf=Math.max(0,Math.min(1,Number(p.confidence||0)));
    const val=Math.max(-1,Math.min(1,Number(p.valence||0)));
    const label=String(p.valence_label||"neutral");
    const preferred=p.preferred_action||null,avoided=p.avoided_action||null;
    const inj=p.last_injection||{},brain=inj.brain||{};
    const actions=(p.actions||[]).slice(0,6).map(x=>
      '<span class="person-tag">'+esc(x.action||"—")+' '+(Number(x.signal||0)>=0?"+":"")+Number(x.signal||0).toFixed(2)+' ×'+nfmt(x.observations||0)+'</span>'
    ).join('');
    return '<div class="person-card">'+
      '<div class="person-head"><div><div class="person-name">🎚 '+esc(p.conversation_mode||"UNKNOWN")+'</div><span class="person-id">'+
        'int '+nfmt(p.intensity_bucket||0)+'/3 • speech '+nfmt(p.speech_bucket||0)+'/3 • switches '+nfmt(p.switch_bucket||0)+'/3 • overlap '+nfmt(p.overlap_bucket||0)+'/3</span></div>'+
      '<span class="person-valence '+esc(label)+'">'+esc(label.toUpperCase())+' '+(val>=0?"+":"")+val.toFixed(2)+'</span></div>'+
      '<div class="person-kpis">'+
        '<div class="person-kpi"><small>Familiarity</small><b>'+(fam*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Confidence</small><b>'+(conf*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Seen / outcome</small><b>'+nfmt(p.seen_observations||0)+' / '+nfmt(p.outcome_observations||0)+'</b></div>'+
        '<div class="person-kpi"><small>Handoff / turn</small><b>'+esc(p.handoff_bucket||"none")+' / '+esc(p.turn_bucket||"none")+'</b></div>'+
      '</div>'+
      '<div class="person-lines">'+
        '<div><b>Dominacja:</b> '+nfmt(p.dominance_bucket||0)+'/3 • <b>cisza:</b> '+esc(p.silence_bucket||"—")+' • <b>speech rate:</b> '+esc(p.speech_rate_bucket||"—")+'</div>'+
        '<div><b>Najlepszy historyczny wynik:</b> '+(preferred?esc(preferred.action)+' '+(Number(preferred.signal||0)>=0?"+":"")+Number(preferred.signal||0).toFixed(2):'—')+'</div>'+
        '<div><b>Najgorszy historyczny wynik:</b> '+(avoided?esc(avoided.action)+' '+Number(avoided.signal||0).toFixed(2):'—')+'</div>'+
        '<div><b>Ostatnie wejście do connectomu:</b> '+(inj.checked_at?(nfmt(brain.cue_count||0)+' cues • '+Math.max(0,Date.now()/1000-Number(inj.checked_at)).toFixed(0)+'s temu'):'jeszcze nie użyty w bieżącej sesji')+'</div>'+
      '</div>'+
      '<div class="person-tags">'+(actions||'<span class="person-tag">brak wyników akcji</span>')+'</div>'+
    '</div>';
  }).join("");
}
function renderChannelProfiles(items){
  const root=$("channel-memory-grid"),summary=$("channel-memory-summary");
  items=Array.isArray(items)?items:[];
  if(!items.length){
    summary.innerHTML='<b>Brak dojrzałych profili miejsc.</b> Pojawią się po wizytach i doświadczeniach na voice.';
    root.innerHTML='<div class="trace-empty">Brak profili miejsc.</div>';
    return;
  }
  const mature=items.filter(x=>Number(x.observations||0)>=2);
  const avgFam=mature.length
    ? mature.reduce((a,x)=>a+Number(x.familiarity||0),0)/mature.length
    : 0;
  summary.innerHTML='<b>'+nfmt(items.length)+' profili miejsc</b> • '+nfmt(mature.length)+' dojrzałych • średnia familiarity '+(avgFam*100).toFixed(0)+'% • pamięć trwała';

  root.innerHTML=items.slice(0,16).map(p=>{
    const fam=Math.max(0,Math.min(1,Number(p.familiarity||0)));
    const conf=Math.max(0,Math.min(1,Number(p.confidence||0)));
    const val=Math.max(-1,Math.min(1,Number(p.valence||0)));
    const label=String(p.valence_label||"neutral");
    const preferred=p.preferred_action||null,avoided=p.avoided_action||null;
    const people=(p.people||[]).slice(0,5);
    const modes=(p.conversation_modes||[]).slice(0,4);
    const recent=(p.recent_episodes||[])[0]||null;
    const history=(p.interaction_history||[]).slice(0,4);
    const actionOutcomes=(p.action_outcomes||[]).slice(0,3);
    const trend=String(p.place_trend||"stable");
    const recentValence=Math.max(-1,Math.min(1,Number(p.recent_valence||0)));
    const stability=Math.max(0,Math.min(1,Number(p.place_stability||0)));
    const inj=p.last_injection||{},brain=inj.brain||{};
    const tags=[];
    modes.forEach(x=>tags.push('<span class="person-tag">'+esc(x.mode||"UNKNOWN")+' ×'+nfmt(x.observations||0)+'</span>'));
    people.forEach(x=>tags.push('<span class="person-tag">👤 '+esc(x.display_name||x.user_id||"—")+' ×'+nfmt(x.observations||0)+'</span>'));
    return '<div class="person-card">'+
      '<div class="person-head"><div><div class="person-name">📍 '+esc(p.channel_name||p.channel_id||"—")+'</div><span class="person-id">'+esc(p.guild||"")+' • '+esc(p.channel_id||"")+'</span></div>'+
      '<span class="person-valence '+esc(label)+'">'+esc(label.toUpperCase())+' '+(val>=0?"+":"")+val.toFixed(2)+'</span></div>'+
      '<div class="person-kpis">'+
        '<div class="person-kpi"><small>Familiarity</small><b>'+(fam*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Confidence</small><b>'+(conf*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Obserwacje</small><b>'+nfmt(p.observations||0)+'</b></div>'+
        '<div class="person-kpi"><small>Historia</small><b>'+nfmt(p.history_observations||0)+'</b></div>'+
        '<div class="person-kpi"><small>Recent valence</small><b class="'+(recentValence>0.05?'ok':recentValence<-0.05?'no':'')+'">'+(recentValence>=0?"+":"")+recentValence.toFixed(2)+'</b></div>'+
        '<div class="person-kpi"><small>Trend</small><b class="'+(trend==="improving"?'ok':trend==="worsening"?'no':'')+'">'+esc(trend.toUpperCase())+'</b></div>'+
        '<div class="person-kpi"><small>Stability</small><b>'+(stability*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Recent humans</small><b>'+Number(p.recent_human_density||0).toFixed(1)+'</b></div>'+
        '<div class="person-kpi"><small>Visit / dynamics</small><b>'+nfmt(p.visit_observations||0)+' / '+nfmt(p.dynamics_observations||0)+'</b></div>'+
      '</div>'+
      '<div class="person-lines">'+
        '<div><b>Typ miejsca:</b> '+esc(p.dominant_mode||"UNKNOWN")+' • intensity '+(Number(p.mean_intensity||0)*100).toFixed(0)+'% • speech '+(Number(p.mean_speech_ratio||0)*100).toFixed(0)+'% • avg humans '+Number(p.mean_human_density||0).toFixed(1)+'</div>'+
        '<div><b>Dobra akcja:</b> '+(preferred?esc(preferred.action)+' '+(Number(preferred.signal||0)>=0?"+":"")+Number(preferred.signal||0).toFixed(2):'—')+'</div>'+
        '<div><b>Zła akcja:</b> '+(avoided?esc(avoided.action)+' '+Number(avoided.signal||0).toFixed(2):'—')+'</div>'+
        '<div><b>Chronologia:</b> '+(history.length?history.map(x=>{
          const amount=Number(x.amount||0);
          return esc(x.kind||"event")+' '+esc(x.action||x.source||"")+(Math.abs(amount)>0.0001?' '+(amount>=0?"+":"")+amount.toFixed(2):'');
        }).join(' • '):'—')+'</div>'+
        '<div><b>Outcome akcji:</b> '+(actionOutcomes.length?actionOutcomes.map(x=>{
          const reward=Number(x.mean_reward||0);
          return esc(x.action||"—")+' '+(reward>=0?"+":"")+reward.toFixed(2)+' ×'+nfmt(x.observations||0);
        }).join(' • '):'—')+'</div>'+
        '<div><b>Wiek miejsca:</b> '+Number(p.place_age_days||0).toFixed(1)+' d • ostatnio '+(p.last_seen?sessionDuration(Math.max(0,Date.now()/1000-Number(p.last_seen)))+' temu':'—')+'</div>'+
        '<div><b>Ostatni epizod:</b> '+(recent?(esc(recent.action||"—")+' • '+nfmt(recent.human_count||0)+' ludzi • reward '+(Number(recent.actual_reward||0)>=0?"+":"")+Number(recent.actual_reward||0).toFixed(2)):'—')+'</div>'+
        '<div><b>Ostatnie wejście do connectomu:</b> '+(inj.checked_at?(nfmt(brain.cue_count||0)+' cues • '+Math.max(0,Date.now()/1000-Number(inj.checked_at)).toFixed(0)+'s temu'):'jeszcze nie użyty w bieżącej sesji')+'</div>'+
      '</div>'+
      '<div class="person-tags">'+(tags.join('')||'<span class="person-tag">brak dodatkowych danych</span>')+'</div>'+
    '</div>';
  }).join("");
}
function renderPersonProfiles(items){
  const root=$("people-memory-grid"),summary=$("people-memory-summary");
  items=Array.isArray(items)?items:[];
  if(!items.length){
    summary.innerHTML='<b>Brak dojrzałych profili.</b> Profile zaczną rosnąć od kontaktów tekstowych, STT i epizodów voice.';
    root.innerHTML='<div class="trace-empty">Brak profili osób.</div>';
    return;
  }
  const mature=items.filter(x=>Number(x.observations||0)>=2);
  const avgFam=mature.length
    ? mature.reduce((a,x)=>a+Number(x.familiarity||0),0)/mature.length
    : 0;
  summary.innerHTML='<b>'+nfmt(items.length)+' profili</b> • '+nfmt(mature.length)+' z min. 2 obserwacjami • średnia familiarity '+(avgFam*100).toFixed(0)+'% • dane trwałe w voice_episodes.sqlite3';

  root.innerHTML=items.slice(0,16).map(p=>{
    const fam=Math.max(0,Math.min(1,Number(p.familiarity||0)));
    const conf=Math.max(0,Math.min(1,Number(p.confidence||0)));
    const val=Math.max(-1,Math.min(1,Number(p.valence||0)));
    const label=String(p.valence_label||"neutral");
    const preferred=p.preferred_action||null, avoided=p.avoided_action||null;
    const contacts=(p.contact_sources||[]).slice(0,4);
    const social=(p.social_events||[]).slice(0,4);
    const channels=(p.channels||[]).slice(0,3);
    const recent=(p.recent_episodes||[])[0]||null;
    const history=(p.interaction_history||[]).slice(0,4);
    const actionOutcomes=(p.action_outcomes||[]).slice(0,3);
    const trend=String(p.relationship_trend||"stable");
    const recentValence=Math.max(-1,Math.min(1,Number(p.recent_valence||0)));
    const stability=Math.max(0,Math.min(1,Number(p.relationship_stability||0)));
    const inj=p.last_injection||{},brain=inj.brain||{};
    const tags=[];
    contacts.forEach(x=>tags.push('<span class="person-tag">'+esc(x.source||"contact")+' ×'+nfmt(x.observations||0)+'</span>'));
    social.forEach(x=>{
      const sig=Number(x.signal||0);
      tags.push('<span class="person-tag">'+esc(x.event||"social")+' '+(sig>=0?"+":"")+sig.toFixed(2)+'</span>');
    });
    return '<div class="person-card">'+
      '<div class="person-head"><div><div class="person-name">'+esc(p.display_name||p.user_id||"—")+'</div><span class="person-id">'+esc(p.user_id||"")+'</span></div>'+
      '<span class="person-valence '+esc(label)+'">'+esc(label.toUpperCase())+' '+(val>=0?"+":"")+val.toFixed(2)+'</span></div>'+
      '<div class="person-kpis">'+
        '<div class="person-kpi"><small>Familiarity</small><b>'+(fam*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Confidence</small><b>'+(conf*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Doświadczenia</small><b>'+nfmt(p.observations||0)+'</b></div>'+
        '<div class="person-kpi"><small>Historia</small><b>'+nfmt(p.history_observations||0)+'</b></div>'+
        '<div class="person-kpi"><small>Recent valence</small><b class="'+(recentValence>0.05?'ok':recentValence<-0.05?'no':'')+'">'+(recentValence>=0?"+":"")+recentValence.toFixed(2)+'</b></div>'+
        '<div class="person-kpi"><small>Trend</small><b class="'+(trend==="improving"?'ok':trend==="worsening"?'no':'')+'">'+esc(trend.toUpperCase())+'</b></div>'+
        '<div class="person-kpi"><small>Stability</small><b>'+(stability*100).toFixed(0)+'%</b></div>'+
        '<div class="person-kpi"><small>Contact / social</small><b>'+nfmt(p.contact_observations||0)+' / '+nfmt(p.social_observations||0)+'</b></div>'+
      '</div>'+
      '<div class="person-lines">'+
        '<div><b>Typowa dobra akcja:</b> '+(preferred?esc(preferred.action)+' '+(Number(preferred.signal||0)>=0?"+":"")+Number(preferred.signal||0).toFixed(2):'—')+'</div>'+
        '<div><b>Typowa zła akcja:</b> '+(avoided?esc(avoided.action)+' '+Number(avoided.signal||0).toFixed(2):'—')+'</div>'+
        '<div><b>Kanały:</b> '+(channels.length?channels.map(x=>esc(x.channel_name||x.channel_id)+' ('+nfmt(x.observations||0)+')').join(' • '):'—')+'</div>'+
        '<div><b>Chronologia:</b> '+(history.length?history.map(x=>{
          const amount=Number(x.amount||0);
          return esc(x.kind||"event")+' '+esc(x.action||x.source||"")+(Math.abs(amount)>0.0001?' '+(amount>=0?"+":"")+amount.toFixed(2):'');
        }).join(' • '):'—')+'</div>'+
        '<div><b>Outcome akcji:</b> '+(actionOutcomes.length?actionOutcomes.map(x=>{
          const reward=Number(x.mean_reward||0);
          return esc(x.action||"—")+' '+(reward>=0?"+":"")+reward.toFixed(2)+' ×'+nfmt(x.observations||0);
        }).join(' • '):'—')+'</div>'+
        '<div><b>Wiek relacji:</b> '+Number(p.relationship_age_days||0).toFixed(1)+' d • ostatnio '+(p.last_seen?sessionDuration(Math.max(0,Date.now()/1000-Number(p.last_seen)))+' temu':'—')+'</div>'+
        '<div><b>Ostatni epizod:</b> '+(recent?(esc(recent.action||"—")+' • '+esc(recent.channel_name||"poza VC")+' • reward '+(Number(recent.actual_reward||0)>=0?"+":"")+Number(recent.actual_reward||0).toFixed(2)):'—')+'</div>'+
        '<div><b>Ostatnie wejście do connectomu:</b> '+(inj.checked_at?(esc(inj.source||"—")+' • '+nfmt(brain.cue_count||0)+' cues • '+Math.max(0,Date.now()/1000-Number(inj.checked_at)).toFixed(0)+'s temu'):'jeszcze nie użyty w bieżącej sesji')+'</div>'+
      '</div>'+
      '<div class="person-tags">'+(tags.join('')||'<span class="person-tag">brak dodatkowych sygnałów</span>')+'</div>'+
    '</div>';
  }).join("");
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
function initDetailsLayout(){
  const collapsedByDefault=new Set(["Learning Since Startup","Action History","Server Learning Context","Audio Debug","Voice Recognition / STT","Najbardziej aktywne neurony"]);
  document.querySelectorAll(".card").forEach(card=>{
    const h=card.querySelector(":scope > h2");
    if(!h||!collapsedByDefault.has(h.textContent.trim()))return;
    card.classList.add("is-collapsible","collapsed");
    h.setAttribute("role","button");
    h.setAttribute("tabindex","0");
    h.setAttribute("aria-expanded","false");
    const toggle=()=>{
      card.classList.toggle("collapsed");
      h.setAttribute("aria-expanded",String(!card.classList.contains("collapsed")));
    };
    h.addEventListener("click",toggle);
    h.addEventListener("keydown",e=>{
      if(e.key==="Enter"||e.key===" "){e.preventDefault();toggle();}
    });
  });
}
initDetailsLayout();
initGuide();
let detailsBusy=false;
async function update(){
  if(detailsBusy)return;
  detailsBusy=true;
  try{
    const [r,sysr]=await Promise.all([
      fetch("/api/state",{cache:"no-store"}),
      fetch("/api/system",{cache:"no-store"})
    ]);
    if(r.status===401||sysr.status===401){location="/login";return}
    if(!r.ok)throw new Error("state HTTP "+r.status);
    if(!sysr.ok)throw new Error("system HTTP "+sysr.status);
    const [s,sys]=await Promise.all([r.json(),sysr.json()]),d=s.diag;
    renderSystemLive(sys);
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
    $("synapse-count").textContent=nfmt(d.learned_synapses||0);
    $("synapse-mean").textContent=Number(d.synaptic_mean_abs||0).toExponential(3);
    $("synapse-max").textContent=Number(d.synaptic_max_abs||0).toExponential(3);
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
    $("paused").textContent=s.paused?"PAUZA":((s.sleep||{}).active?"SLEEP":"aktywny");$("event").textContent=s.last_event||"—";$("lastaction").textContent=s.last_action||"—";
    const scores=s.scores||{};
    const rankedActions=Object.entries(scores).sort((a,b)=>Number(b[1])-Number(a[1]));
    const dominantAction=rankedActions[0]||["—",0];
    const attGuild=((s.attention||{}).guilds||[]).find(g=>g.focus)||((s.attention||{}).guilds||[])[0]||{};
    const focus=attGuild.focus||null;
    const firstVoice=(s.voice_debug||[])[0]||{};
    const internal=firstVoice.internal_states||{};
    const trace=s.decision_trace||{};
    const traceFocus=trace.attention_focus||focus;
    const traceInternal=trace.internal_states||internal;
    const traceRead=trace.readout||{};
    $("flow-event").textContent=trace.stimulus||s.last_event||"—";
    $("flow-attention").textContent=traceFocus
      ? String(traceFocus.label||traceFocus.key||"—")+" • "+Number(traceFocus.score||0).toFixed(2)
      : "brak aktywnego focusu";
    $("flow-brain").textContent=traceInternal.dominant
      ? String(traceInternal.dominant)+" • "+(Number(traceInternal.dominant_level||0)*100).toFixed(0)+"%"
      : "mean |a| "+Number(d.mean_abs||0).toFixed(4);
    $("flow-readout").textContent=traceRead.action
      ? String(traceRead.action)+" • "+Number(traceRead.raw_score||0).toFixed(3)+(traceRead.effective_score==null?"":" → "+Number(traceRead.effective_score).toFixed(3))
      : String(dominantAction[0])+" • "+Number(dominantAction[1]||0).toFixed(3);
    $("flow-action").textContent=trace.actual_action||s.last_action||"brak wykonanej akcji";
    renderDecisionTraceHistory(s.decision_trace_history||[],trace);
    renderActions(scores);
    renderReaction(s.reaction_debug||{});
    renderLearning(s.learning_debug||{});
    renderSleep(s.sleep||{});
    renderAffective(s.affective_state||{});
    renderMotivation(s.motivation_state||{});
    renderLearningSinceStart(s.learning_since_start||{});
    renderSocialScenes(s.social_scene_profiles||[]);
    renderVoiceDynamicsProfiles(s.voice_dynamics_profiles||[]);
    renderChannelProfiles(s.channel_profiles||[]);
    renderPersonProfiles(s.person_profiles||[]);
    renderSocial(s);
    renderActionHistory(s.action_history||[]);
    renderGuildLearningContext(s.guild_learning_context||[]);
    drawRewardChart(s.reward_history||[],d.reward_trace);
    renderAudioDebug(s.audio_debug||{});
    renderSttDebug(s.stt_debug||{});
    renderAttention(s.attention||{});
    renderActionPolicy(s.action_policy||{});
    renderVoiceDebug(s.voice_debug||[]);
    enhanceHelp();
    $("top").innerHTML=(s.top_neurons||[]).map((x,i)=>'<tr><td>'+(i+1)+'</td><td>'+x[0]+'</td><td>'+(x[1]>=0?"+":"")+Number(x[1]).toFixed(5)+'</td><td>'+Math.abs(x[1]).toFixed(5)+'</td></tr>').join("");
    history.push({mean:Number(d.mean_abs),max:Number(d.max_abs)});while(history.length>maxHistory)history.shift();draw();
    $("live").textContent="LIVE";
  }catch(e){$("live").textContent="ROZŁĄCZONO";console.error(e)}
  finally{detailsBusy=false}
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
        refresh_ms: int = 250,
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
        configured_chaser_status = Path(chaser_status_file)
        if (
            os.name == "nt"
            and not configured_chaser_status.exists()
            and str(chaser_status_file).replace("\\", "/").startswith(
                "/opt/mucha-chaser/"
            )
        ):
            configured_chaser_status = (
                Path(__file__).resolve().parents[2]
                / "mucha-chaser"
                / "state"
                / "chaser_status.json"
            )
        self.chaser_status_file = configured_chaser_status
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
        self._process_started_monotonic = time.monotonic()
        self._gpu_monitor_task: asyncio.Task | None = None
        self._gpu_status_cache: dict = {
            "available": False,
            "backend": "nvidia-smi",
            "gpus": [],
            "updated_at": 0.0,
            "error": "not checked yet",
        }
        self._process_metrics = None
        if psutil is not None:
            try:
                self._process_metrics = psutil.Process(os.getpid())
            except Exception:
                self._process_metrics = None
            try:
                if self._process_metrics is not None:
                    self._process_metrics.cpu_percent(interval=None)
            except Exception:
                pass
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

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
        app.router.add_get("/autonomy", self._autonomy_page)
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
        app.router.add_get("/api/system", self._system_state)
        app.router.add_get("/api/config", self._config_get)
        app.router.add_post("/api/config", self._config_post)
        app.router.add_get("/health", self._health)

        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self._bind_host, self.port)
        await self.site.start()
        log.info("Web UI: http://%s:%s", self._bind_host, self.port)
        if self._gpu_monitor_task is None:
            self._gpu_monitor_task = asyncio.create_task(
                self._gpu_monitor_loop()
            )

        if self.auto_open and self._bind_host in {"127.0.0.1", "localhost"}:
            url = f"http://127.0.0.1:{self.port}"
            asyncio.get_running_loop().call_later(1.0, webbrowser.open, url)

    async def stop(self) -> None:
        if self._gpu_monitor_task is not None:
            self._gpu_monitor_task.cancel()
            try:
                await self._gpu_monitor_task
            except asyncio.CancelledError:
                pass
            self._gpu_monitor_task = None
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None
            self.site = None

    async def _index(self, request: web.Request) -> web.Response:
        html = OVERVIEW_HTML.replace(
            "const LIVE_REFRESH_MS=250;",
            f"const LIVE_REFRESH_MS={self.refresh_ms};",
        )
        return web.Response(text=html, content_type="text/html")

    async def _autonomy_page(self, request: web.Request) -> web.Response:
        html = AUTONOMY_HTML.replace(
            "const LIVE_REFRESH_MS=250;",
            f"const LIVE_REFRESH_MS={self.refresh_ms};",
        )
        return web.Response(text=html, content_type="text/html")

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
            '<a href="/autonomy">🧭 Autonomia</a>',
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
        action = str(request.query.get("action", "")).strip()
        snap = await self.connectome_provider(
            raw in {"1", "true", "yes", "on"},
            action or None,
        )
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

    async def _system_state(self, request: web.Request) -> web.Response:
        return web.json_response(
            self._system_status(),
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
        action = str(request.query.get("action", "")).strip()
        snap = await self.connectome_provider(
            follow_activity,
            action or None,
        )
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
        if os.name == "nt":
            if unit == "mucha.service":
                memory_bytes = 0
                try:
                    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                        _fields_ = [
                            ("cb", ctypes.c_ulong),
                            ("PageFaultCount", ctypes.c_ulong),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t),
                        ]

                    counters = PROCESS_MEMORY_COUNTERS()
                    counters.cb = ctypes.sizeof(counters)
                    process = ctypes.windll.kernel32.GetCurrentProcess()
                    if ctypes.windll.psapi.GetProcessMemoryInfo(
                        process,
                        ctypes.byref(counters),
                        counters.cb,
                    ):
                        memory_bytes = int(counters.WorkingSetSize)
                except (AttributeError, OSError, ValueError):
                    memory_bytes = 0

                return {
                    "unit": unit,
                    "active": True,
                    "active_state": "active",
                    "sub_state": "local-windows",
                    "pid": os.getpid(),
                    "memory_bytes": memory_bytes,
                    "cpu_seconds": float(time.process_time()),
                    "uptime_seconds": max(
                        0.0,
                        time.monotonic() - self._process_started_monotonic,
                    ),
                    "error": "",
                }

            if unit == "mucha-chaser.service":
                status = self._chaser_status()
                active = bool(status.get("online")) if status else False
                try:
                    started_at = float(status.get("started_at") or 0.0)
                except (TypeError, ValueError):
                    started_at = 0.0
                return {
                    "unit": unit,
                    "active": active,
                    "active_state": "active" if active else "inactive",
                    "sub_state": "local-windows" if active else "status-file",
                    "pid": 0,
                    "memory_bytes": 0,
                    "cpu_seconds": 0.0,
                    "uptime_seconds": (
                        max(0.0, time.time() - started_at)
                        if started_at
                        else 0.0
                    ),
                    "error": (
                        ""
                        if status
                        else f"Brak statusu: {self.chaser_status_file}"
                    ),
                }

            return {
                "unit": unit,
                "active": False,
                "active_state": "unknown",
                "sub_state": "windows",
                "pid": 0,
                "memory_bytes": 0,
                "cpu_seconds": 0.0,
                "uptime_seconds": 0.0,
                "error": "Usługa systemd niedostępna w Windows.",
            }

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

    async def _query_gpu_status(self) -> dict:
        executable = shutil.which("nvidia-smi")
        now = time.time()
        if not executable:
            return {
                "available": False,
                "backend": "nvidia-smi",
                "gpus": [],
                "updated_at": now,
                "error": "nvidia-smi not found",
            }
        try:
            proc = await asyncio.create_subprocess_exec(
                executable,
                (
                    "--query-gpu=index,name,utilization.gpu,"
                    "memory.used,memory.total,temperature.gpu"
                ),
                "--format=csv,noheader,nounits",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=1.5,
            )
            if proc.returncode != 0:
                return {
                    "available": False,
                    "backend": "nvidia-smi",
                    "gpus": [],
                    "updated_at": now,
                    "error": stderr.decode(
                        "utf-8",
                        "replace",
                    ).strip() or f"exit {proc.returncode}",
                }

            rows = []
            for line in stdout.decode(
                "utf-8",
                "replace",
            ).splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 6:
                    continue

                def number(value: str) -> float:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return 0.0

                used_mb = number(parts[3])
                total_mb = number(parts[4])
                rows.append({
                    "index": int(number(parts[0])),
                    "name": parts[1],
                    "utilization_percent": number(parts[2]),
                    "memory_used_bytes": int(
                        used_mb * 1024 * 1024
                    ),
                    "memory_total_bytes": int(
                        total_mb * 1024 * 1024
                    ),
                    "memory_percent": (
                        used_mb / total_mb * 100.0
                        if total_mb > 0.0
                        else 0.0
                    ),
                    "temperature_c": number(parts[5]),
                })

            return {
                "available": bool(rows),
                "backend": "nvidia-smi",
                "gpus": rows,
                "updated_at": now,
                "error": "" if rows else "no NVIDIA GPU data",
            }
        except Exception as exc:
            return {
                "available": False,
                "backend": "nvidia-smi",
                "gpus": [],
                "updated_at": now,
                "error": f"{type(exc).__name__}: {exc}",
            }

    async def _gpu_monitor_loop(self) -> None:
        while True:
            self._gpu_status_cache = await self._query_gpu_status()
            delay = (
                1.0
                if self._gpu_status_cache.get("available")
                else 5.0
            )
            await asyncio.sleep(delay)

    def _system_status(self) -> dict:
        now = time.time()
        try:
            disk = shutil.disk_usage(Path.cwd())
        except OSError:
            disk_root = Path.cwd().anchor or ("/" if os.name != "nt" else "C:\\")
            disk = shutil.disk_usage(disk_root)

        total = available = used = 0
        mem_percent = 0.0
        cpu_percent = 0.0
        cpu_count = os.cpu_count() or 0
        process_rss = 0
        process_cpu_percent = 0.0
        process_threads = 0

        if psutil is not None:
            try:
                vm = psutil.virtual_memory()
                total = int(vm.total)
                available = int(vm.available)
                used = int(vm.used)
                mem_percent = float(vm.percent)
                cpu_percent = float(
                    psutil.cpu_percent(interval=None)
                )
                cpu_count = int(psutil.cpu_count() or cpu_count)
            except Exception:
                pass
            try:
                process = self._process_metrics
                if process is None:
                    # A failed CPU priming call must not permanently disable
                    # process metrics. Re-create the current-process handle.
                    process = psutil.Process(os.getpid())
                    self._process_metrics = process
                process_rss = int(process.memory_info().rss)
                try:
                    process_cpu_percent = float(
                        process.cpu_percent(interval=None)
                    )
                except Exception:
                    process_cpu_percent = 0.0
                try:
                    process_threads = int(process.num_threads())
                except Exception:
                    process_threads = 0
            except Exception:
                pass

        # psutil is optional in the local Windows launcher. If it is missing,
        # broken, or returned an invalid RSS value, query the current process
        # through Win32. Explicit pointer-sized signatures matter on 64-bit
        # Python; relying on ctypes defaults can truncate HANDLE values.
        if process_rss <= 0 and os.name == "nt":
            try:
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                kernel32 = ctypes.WinDLL(
                    "kernel32",
                    use_last_error=True,
                )
                kernel32.GetCurrentProcess.argtypes = []
                kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                process_handle = kernel32.GetCurrentProcess()

                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)

                query = getattr(
                    kernel32,
                    "K32GetProcessMemoryInfo",
                    None,
                )
                if query is None:
                    psapi = ctypes.WinDLL(
                        "psapi",
                        use_last_error=True,
                    )
                    query = psapi.GetProcessMemoryInfo

                query.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                    ctypes.c_ulong,
                ]
                query.restype = ctypes.c_int
                if query(
                    process_handle,
                    ctypes.byref(counters),
                    ctypes.c_ulong(counters.cb),
                ):
                    process_rss = int(counters.WorkingSetSize)
            except Exception:
                process_rss = 0

        if total <= 0:
            if os.name == "nt":
                try:
                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]
                    status = MEMORYSTATUSEX()
                    status.dwLength = ctypes.sizeof(status)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(
                        ctypes.byref(status)
                    ):
                        total = int(status.ullTotalPhys)
                        available = int(status.ullAvailPhys)
                except Exception:
                    pass
            else:
                mem: dict[str, int] = {}
                try:
                    for line in Path("/proc/meminfo").read_text(
                        encoding="utf-8",
                    ).splitlines():
                        if ":" not in line:
                            continue
                        key, value = line.split(":", 1)
                        mem[key] = int(
                            value.strip().split()[0]
                        ) * 1024
                    total = int(mem.get("MemTotal", 0))
                    available = int(
                        mem.get("MemAvailable", 0)
                    )
                except Exception:
                    pass
            used = max(0, total - available)
            mem_percent = (
                used / total * 100.0
                if total
                else 0.0
            )

        try:
            if psutil is not None:
                uptime = max(
                    0.0,
                    now - float(psutil.boot_time()),
                )
            elif os.name == "nt":
                get_tick_count_64 = ctypes.windll.kernel32.GetTickCount64
                get_tick_count_64.restype = ctypes.c_ulonglong
                uptime = float(get_tick_count_64()) / 1000.0
            else:
                uptime = float(
                    Path("/proc/uptime").read_text(
                        encoding="utf-8",
                    ).split()[0]
                )
        except Exception:
            uptime = 0.0

        try:
            load = list(os.getloadavg())
        except (OSError, AttributeError):
            load = []

        disk_percent = (
            disk.used / disk.total * 100.0
            if disk.total
            else 0.0
        )
        gpu = dict(self._gpu_status_cache)
        return {
            "platform": "windows" if os.name == "nt" else "linux",
            "updated_at": now,
            "uptime_seconds": uptime,
            "load": load,
            "cpu_percent": cpu_percent,
            "cpu_count": cpu_count,
            "mem_total": total,
            "mem_available": available,
            "mem_used": used,
            "mem_percent": mem_percent,
            "disk_total": int(disk.total),
            "disk_used": int(disk.used),
            "disk_free": int(disk.free),
            "disk_percent": disk_percent,
            "process": {
                "pid": os.getpid(),
                "cpu_percent": process_cpu_percent,
                "memory_bytes": process_rss,
                "threads": process_threads,
                "uptime_seconds": max(
                    0.0,
                    time.monotonic()
                    - self._process_started_monotonic,
                ),
            },
            "gpu": gpu,
        }

    def _chaser_status(self) -> dict:
        try:
            return json.loads(
                self.chaser_status_file.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return {}

    async def _journal_tail(self, unit: str, lines: int = 18) -> list[str]:
        if os.name == "nt":
            if unit == "mucha.service":
                return [
                    "Tryb lokalny Windows — logi Muchy są w oknie launchera."
                ]
            if unit == "mucha-chaser.service":
                return [
                    "Tryb lokalny Windows — logi Chasera są w oknie launchera."
                ]
            return []

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
