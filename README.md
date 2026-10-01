# 🪰 Mucha

**Mucha** to eksperymentalny autonomiczny bot Discord sterowany przez ciągły stan sieci o topologii **FlyWire FAFB v783**.

Projekt nie używa ChatGPT ani innego gotowego LLM do generowania wypowiedzi. Mucha uczy się języka z wiadomości i transkrypcji, a decyzje o zachowaniu są powiązane ze stanem connectomu, pamięcią, rewardem, relacjami i wewnętrznymi stanami neuronalnymi.

> To nie jest dokładna emulacja świadomości ani biologicznego mózgu muchy. Projekt łączy prawdziwą topologię connectome z własnym runtime'em, sztucznymi wejściami Discord/voice oraz sztucznymi readoutami zachowania.

---

## Repozytoria

Projekt jest rozdzielony na dwie wersje:

- **`Davidgracz/mucha`** — główna wersja rozwijana lokalnie na Windowsie.
- **`Davidgracz/much_vps`** — osobna wersja przeznaczona do działania na VPS.

Dzięki temu wersja lokalna może być rozwijana i testowana niezależnie od instancji działającej 24/7.

---

## Co Mucha potrafi

### Connectome

Runtime obsługuje pełny przygotowany dataset FAFB v783:

- około **139 255 neuronów**,
- około **3 732 460 połączeń synaptycznych**,
- kierunek i siłę realnych połączeń,
- klasy neuronów,
- predicted neurotransmitter,
- neuropile,
- opcjonalne rzeczywiste współrzędne FlyWire,
- ciągły stan aktywacji między kolejnymi zdarzeniami.

Stan nie jest resetowany po każdej wiadomości. Nowe bodźce trafiają do istniejącej aktywności sieci.

### Plastyczność

Mucha ma kilka warstw trwałego uczenia:

- neuronalny `plastic bias`,
- sparse overlay uczonych zmian istniejących synaps,
- eligibility trace,
- reward / punish,
- konsolidację,
- czasowe zapominanie,
- ochronę utrwalonych synaps,
- oznaczenia `CONSOLIDATED` / `FADING`.

Bazowy connectome FAFB pozostaje niezmieniony. Uczenie działa jako nakładka na biologiczną sieć.

### Neuromodulacja

Runtime posiada globalne stany:

- dopamina,
- serotonina,
- octopamina.

Wpływają one m.in. na:

- tempo plastyczności,
- stabilność stanu,
- propagation gain,
- poziom pobudzenia i szumu.

### Neural internal states

W connectomie istnieją wewnętrzne attractory:

- **SOCIAL NEED**
- **CURIOSITY**
- **STRESS**
- **SATIETY**
- **AROUSAL**

Nie są to ręcznie dodawane punkty do action score. Stan jest odczytywany z aktywności odpowiednich zespołów neuronów po propagacji po istniejących krawędziach FAFB.

---

## Decyzje i zachowanie

Aktualne readouty connectomu obejmują:

- `speak`
- `react`
- `voice_join`
- `voice_move`
- `voice_leave`
- `explore`
- `stay`

Mucha może autonomicznie:

- odpowiedzieć na wiadomość,
- napisać spontanicznie,
- dodać reakcję emoji,
- wejść na kanał voice,
- zostać,
- zmienić kanał,
- wyjść z voice,
- reagować na nagrodę i karę,
- uciekać przed Mucha Chaser.

Przy odpowiedzi na wiadomość decyzja zależy od `speak_threshold`, gotowości modelu języka, cooldownu, affinity i blokad kanałów.

Spontaniczne pisanie używa obecnie:

```text
speak >= speak_threshold + 0.08
```

Przy domyślnym `speak_threshold = 0.50` oznacza to próg **0.58**.

---

## Język

Mucha zaczyna bez gotowego modelu językowego.

Uczy się online z:

- wiadomości Discord,
- transkrypcji voice STT,
- późniejszego feedbacku.

Generator jest hybrydowy:

- char unigram / bigram / trigram,
- word unigram / bigram / trigram,
- początki wypowiedzi,
- zakończenia wypowiedzi,
- recent boost,
- reward dla użytych słów i przejść,
- opcjonalne sterowanie kandydatami słów przez aktualny stan connectomu.

Dane językowe są zapisywane w:

```text
state/language.sqlite3
```

Przykładowa konfiguracja:

```toml
[language]
min_chars_before_speaking = 800
min_unique_chars_before_speaking = 16
max_generated_chars = 120
spontaneous_text = true
reply_cooldown_seconds = 1
spontaneous_cooldown_seconds = 180

hybrid_word_enabled = true
word_model_probability = 0.90
word_max_tokens = 14

connectome_word_control_enabled = true
connectome_word_control_min_vocab = 1500
connectome_word_control_strength = 0.35
connectome_word_control_candidates = 24

connectome_word_feedback_enabled = true
connectome_word_feedback_steps = 2
connectome_word_feedback_magnitude = 0.18
```

---

## Voice

Mucha analizuje dostępne kanały voice i wybiera zachowanie na podstawie konkurencji readoutów connectomu.

Uwzględniane są m.in.:

- liczba użytkowników,
- affinity,
- historia odwiedzin,
- novelty,
- exploration,
- homeostaza,
- social need,
- fatigue,
- habituation,
- curiosity,
- threat,
- remembered reward opportunity,
- pamięć epizodyczna,
- Chaser.

### STT

Mucha może słuchać użytkowników na voice:

```text
Discord PCM
→ bufor wypowiedzi
→ wykrycie ciszy
→ 16 kHz mono
→ faster-whisper
→ tekst
→ language.learn()
→ connectome
```

Surowe segmenty audio nie są zapisywane jako trwałe pliki.

### TTS

Mucha może odpowiadać głosowo korzystając z tego samego modelu języka co dla tekstu.

Obsługiwane są lokalne silniki TTS, w tym Piper.

---

## Pamięć epizodyczna

Voice i zachowanie korzystają z trwałej pamięci scen.

Mucha zapisuje m.in.:

- akcję,
- serwer i kanał,
- użytkowników,
- przewidywany reward,
- faktyczny reward,
- prediction error,
- siłę wspomnienia,
- liczbę powtórzeń i replay.

Pamięć może później wpływać na connectome przez recall.

### MEMORY REPLAY

Po okresie ciszy Mucha może odtworzyć ważne wspomnienia jako słabszy bodziec i ponownie przepuścić je przez sieć.

Replay może:

- wzmacniać lub osłabiać ślady,
- utrwalać powtarzające się doświadczenia,
- zwiększać consolidation strength,
- wygaszać przypadkowe wspomnienia.

---

## Relacje / affinity

Mucha utrzymuje trwałą relację z użytkownikami.

Affinity ma zakres:

```text
-1.0 ... +1.0
```

Wpływają na nią m.in.:

- pozytywne i negatywne reakcje,
- reply do Muchy,
- mention,
- kontynuowanie rozmowy,
- ponowne użycie jej słów lub fraz,
- wspólne przebywanie na voice,
- zachowanie po TTS,
- opuszczanie kanału po wejściu Muchy,
- naturalne sygnały odrzucenia.

Dodatkowo działa **Neural Social Memory** — użytkownik może mieć stabilną reprezentację neuronalną i trwałe zmiany synaps.

---

## Mucha Chaser

Projekt posiada obsługę osobnego bota-predatora.

Po wykryciu Chasera Mucha może:

- otrzymać sensory threat,
- wejść w stan panic,
- ominąć zwykły dwell,
- uciec na inny kanał,
- krzyknąć `AAAAAAAA!`,
- zapamiętać niebezpieczny kanał,
- dostać reward za udaną ucieczkę.

---

# Dashboard WWW

Domyślny port:

```text
http://127.0.0.1:8765
```

Najważniejsze widoki:

| Ścieżka | Zawartość |
|---|---|
| `/` | Control Center |
| `/details` | szczegółowy stan Muchy |
| `/connectome` | graf connectome i action circuits |
| `/neuromap` | mapa mózgu, signal flow i attractory |
| `/associations` | **Mowa / Language Brain** — live trace doboru słów, wpływu connectomu i recurrent feedback |
| `/affinity` | relacje i Neural Social Memory |
| `/config` | edytor konfiguracji |
| `/public` | publiczny widok read-only |

## Szczegóły

Zakładka `/details` jest podzielona na:

1. **Teraz** — bieżący stan connectomu i readouty.
2. **Voice i decyzje** — decyzja oraz przyczyny.
3. **Uczenie i pamięć** — reward, plasticity, replay.
4. **Relacje** — social learning i affinity.
5. **Audio / STT** — diagnostyka voice.
6. **Neurony** — surowy stan aktywnych neuronów.

Rozwinięte sekcje diagnostyczne zachowują stan podczas live refreshu.

## Mowa / Language Brain

Zakładka `/associations` została przebudowana z samej mapy skojarzeń na pełny podgląd procesu generacji języka.

Pokazuje na żywo ostatnią wygenerowaną wypowiedź oraz każdy krok modelu słów:

```text
kontekst / working memory
→ trigram + bigram + unigram
→ znormalizowany mix kandydatów
→ language_word_score() z connectomu
→ mnożnik brain score
→ ważone losowanie 0–1
→ wybrane słowo
→ recurrent feedback do connectomu
→ brain.step()
→ kolejny wybór z nowego stanu mózgu
```

Dla każdego kroku można zobaczyć:

- top kandydatów,
- bazowy udział modelu języka,
- częstotliwość, reward, recent boost i repeat penalty,
- `brain_score` konkretnego słowa,
- mnożnik connectomu,
- finalne prawdopodobieństwo wyboru,
- dokładny los `0..1` i przedział, w który trafił,
- informację czy wybrane słowo zostało odesłane jako recurrent feedback.

Connectome nie tworzy słów spoza słownika. Najpierw model online wyznacza kandydatów, a aktualny stan connectomu może zwiększyć lub zmniejszyć ich szanse.

Stara mapa skojarzeń nadal jest dostępna niżej na tej samej stronie jako pomocniczy widok pamięci słów.

## Neuro-map

Neuro-map pokazuje m.in.:

- projekcje XY / XZ / YZ,
- biologiczne regiony,
- aktywność neuronów,
- learned synapses,
- consolidated synapses,
- attractory internal states,
- live signal flow,
- historię przepływu,
- Path Inspector,
- ścieżki od sensory input do action readout.

---

# Uruchomienie na Windows

Zalecany Python: **3.12**.

## Instalacja

```bat
install_windows.bat
```

Następnie dodaj token Discord do `.env`:

```env
DISCORD_TOKEN=...
```

Uruchom:

```bat
run_windows.bat
```

lub ręcznie:

```powershell
.\.venv\Scripts\Activate.ps1
python bot.py
```

## Test

```powershell
.\.venv\Scripts\Activate.ps1
python tests\smoke_test.py
```

Oczekiwany wynik:

```text
SMOKE TEST OK
```

---

# Pełny FAFB v783

Przygotowane dane runtime znajdują się w:

```text
data/connectome/
    matrix.npz
    pools.npz
    neuron_meta.npz
    manifest.json
```

Jeżeli przygotowujesz connectome od zera, potrzebne są dane FlyWire/Codex, np.:

```text
raw_flywire/
    classification.csv.gz
    neurons.csv.gz
    connections_princeton.csv.gz
    coordinates.csv.gz
    consolidated_cell_types.csv.gz
```

Budowa:

```powershell
python tools\prepare_connectome.py --input raw_flywire --output data\connectome
```

Wzbogacenie Neuro-map:

```powershell
python tools\augment_connectome_map.py --raw raw_flywire --connectome data\connectome
```

---

# GPU / CUDA

Backend:

```toml
[brain]
backend = "auto"
gpu_device = 0
```

Tryby:

- `auto` — CUDA jeśli dostępna, inaczej CPU,
- `cuda` — wymaga CUDA,
- `cpu` — wymusza NumPy/SciPy.

Windows:

```bat
install_gpu_windows.bat
```

Test GPU:

```powershell
python tools\check_gpu.py
```

Stan mózgu pozostaje przenośny pomiędzy CPU i GPU.

---

# Konfiguracja

Bazowe ustawienia:

```text
config.toml
```

Lokalne nadpisania:

```text
config.local.toml
```

`config.local.toml` nie powinien być wrzucany do Git.

Panel `/config` zapisuje lokalne ustawienia i pozwala zmieniać m.in.:

- parametry mózgu,
- plastyczność,
- neuromodulację,
- język,
- reaction thresholds,
- social learning,
- voice,
- homeostazę,
- episodic memory,
- replay,
- TTS / STT,
- wykluczenia kanałów.

---

# Ważne pliki stanu

```text
state/brain_state.npz
state/language.sqlite3
state/voice_episodes.sqlite3
```

To właśnie te pliki zawierają dużą część indywidualnego doświadczenia konkretnej instancji Muchy:

- `brain_state.npz` — trwały stan mózgu, plastic bias, learned synapses, konsolidacja, action policy i internal drives,
- `language.sqlite3` — wyuczony język, skojarzenia, social learning i affinity,
- `voice_episodes.sqlite3` — pamięć epizodyczna, prediction error, replay, osoby/miejsca/sytuacje i długoterminowe outcome.

Dwie instancje uruchomione z osobnymi katalogami `state/` mogą z czasem wykształcić różne zachowania mimo identycznego kodu i bazowego connectome.

## Automatyczny backup wyuczonego stanu

Repo zawiera bezpieczny backup stanu do synchronizowanego Google Drive:

```text
tools/backup_learned_state.py
tools/backup_state_to_drive.ps1
tools/install_drive_backup_task.ps1
BACKUP_NOW.bat
SETUP_DRIVE_BACKUP.bat
```

Backup nie kopiuje działających baz SQLite zwykłym `Copy-Item`. Dla `language.sqlite3` i `voice_episodes.sqlite3` używany jest `sqlite3.backup()`, więc archiwum jest spójne również wtedy, gdy Mucha działa.

Archiwum zawiera tylko dane potrzebne do zachowania wyuczonego charakteru instancji:

```text
state/brain_state.npz
state/language.sqlite3
state/voice_episodes.sqlite3
backup_manifest.json
```

Pliki `.env`, token Discorda, hasła i modele Whisper nie są kopiowane.

Ręczny backup można uruchomić dwuklikiem:

```text
BACKUP_NOW.bat
```

albo z PowerShell:

```powershell
.\tools\backup_state_to_drive.ps1
```

Jednorazowa instalacja automatycznego zadania Windows co 6 godzin:

```text
SETUP_DRIVE_BACKUP.bat
```

albo:

```powershell
.\tools\install_drive_backup_task.ps1
```

Skrypt próbuje automatycznie znaleźć Google Drive for desktop. Jeżeli folder synchronizowany jest w niestandardowym miejscu, ustaw go raz:

```powershell
[Environment]::SetEnvironmentVariable(
  "MUCHA_BACKUP_DRIVE_DIR",
  "G:\My Drive\ChatGPT\Mucha\Backups",
  "User"
)
```

Domyślnie zachowywanych jest 40 najnowszych archiwów. Każda kopia ma manifest i sumę SHA-256, a wynik wykonywania jest dopisywany do:

```text
state/drive_backup.log
```

---

# Bezpieczeństwo i udostępnianie dashboardu

Lokalny dashboard nasłuchuje wyłącznie na:

```toml
host = "127.0.0.1"
```

Nie trzeba przekierowywać portu `8765` na routerze.

Dashboard ma dwa poziomy dostępu:

- `/public` oraz `/public/*` — publiczny tryb **read-only** dla znajomych,
- prywatne widoki, `/config` i prywatne `/api/*` — wymagają sesji administratora.

Hasło administratora ustaw w lokalnym pliku `.env`:

```env
MUCHA_DASHBOARD_PASSWORD=TU_MOCNE_HASLO
```

Plik `.env` jest ignorowany przez Git i nie powinien być publikowany.

## Szybkie udostępnienie znajomym

Na Windows uruchom:

```bat
SHARE_DASHBOARD.bat
```

Skrypt sprawdza, czy Mucha działa lokalnie, a następnie uruchamia Cloudflare Quick Tunnel do `http://127.0.0.1:8765`.

Cloudflare wypisze tymczasowy adres w stylu:

```text
https://random-words.trycloudflare.com
```

Znajomym podawaj wyłącznie adres z `/public`:

```text
https://random-words.trycloudflare.com/public
```

Quick Tunnel działa tylko tak długo, jak działa proces `cloudflared`, i jest przeznaczony głównie do szybkiego/testowego udostępniania. Do stałego adresu można później utworzyć nazwany Cloudflare Tunnel i podpiąć własną domenę.

---

# Attention + Working Memory

Mucha posiada krótkotrwały stan uwagi utrzymujący aktywny kontekst:

- użytkowników,
- kanały tekstowe i voice,
- najważniejsze słowa / tematy,
- ostatnie sceny tekstowe i transkrypcje STT.

Każdy element uwagi ma zanikającą siłę. Jego końcowy score jest dodatkowo modulowany przez bieżący stan connectomu poprzez stabilne populacje `attention:<item>`.

Aktywny kontekst jest:

- podawany do connectomu natychmiast po nowym bodźcu,
- okresowo reiniektowany podczas idle ticków,
- wygaszany wykładniczo,
- usuwany po wygaśnięciu okna working memory,
- używany jako kontekst przy spontanicznym generowaniu tekstu.

Domyślne ustawienia:

```toml
[behavior]
attention_enabled = true
attention_half_life_seconds = 45.0
working_memory_seconds = 120
attention_max_items = 8
attention_reinject_magnitude = 0.32
attention_topic_words = 5
attention_mention_boost = 0.35
```

Stan jest widoczny na żywo w `/details` jako **Attention / Working Memory**.

# Learned Action Policy

Reward i punish uczą teraz trwałą preferencję każdej akcji:

```text
raw readout connectomu
→ learned action bias
→ effective score
→ decyzja / konkurencja akcji
```

Policy obejmuje:

- `speak`,
- `react`,
- `voice_join`,
- `voice_move`,
- `voice_leave`,
- `explore`,
- `stay`.

Bias jest ograniczony i nie zastępuje connectomu. Surowy readout nadal pochodzi z aktywności neuronalnej, a warstwa policy jedynie przesuwa jego skuteczną wartość na podstawie wcześniejszych nagród i kar.

Dla akcji progowych, takich jak `speak` i `react`, dashboard pokazuje również **learned raw threshold** — czyli jaki surowy readout connectomu jest aktualnie potrzebny, aby przejść bazowy próg po uwzględnieniu doświadczenia.

Stan policy zapisuje się w:

```text
state/brain_state.npz
```

i jest widoczny na żywo w `/details` jako **Learned Action Policy**.

# Semantic Memory

Powtarzające się epizody voice są teraz uogólniane do pamięci semantycznej.

Zamiast przechowywać wyłącznie konkretne sceny:

```text
kanał X + użytkownicy A/B + akcja JOIN + reward
```

Mucha buduje też bardziej ogólne relacje:

```text
użytkownik → akcja → typowy rezultat
kanał → akcja → typowy rezultat
stan homeostatyczny → akcja → typowy rezultat
użytkownik + kanał → akcja → typowy rezultat
```

Recall używa confidence zależnego od liczby obserwacji i zgodności doświadczeń. Dopiero po minimalnej liczbie podobnych zdarzeń uogólnienie może wrócić do mózgu.

```text
semantic expected reward
× confidence
= signed semantic signal
→ action-guided sensory neurons
→ real FAFB propagation
→ action readout
```

Dodatni sygnał pobudza sensoryczną drogę do danej akcji, a ujemny ją hamuje. Kod nie dodaje semantycznego bonusu bezpośrednio do action score.

Dane semantyczne są zapisywane w tej samej bazie SQLite co epizody:

```text
state/voice_episodes.sqlite3
```

Po pierwszej aktualizacji pusta tabela semantyczna jest bootstrapowana z części istniejących epizodów, więc wcześniejsze doświadczenia mogą od razu zacząć tworzyć uogólnienia.

# Curiosity / Uncertainty Exploration

Mucha wykorzystuje pamięć semantyczną także do oceny **niepewności**.

Dla kanałów, użytkowników i bieżącego kontekstu obliczana jest znajomość sytuacji na podstawie:

- liczby obserwacji,
- pokrycia znanych kombinacji akcja/kontekst,
- confidence pamięci semantycznej.

```text
mało doświadczenia
→ semantic uncertainty
→ sensory cue
→ attractor CURIOSITY
→ propagacja po connectomie
→ EXPLORE / MOVE / JOIN / STAY
```

Niepewność nie jest dodawana bezpośrednio do action score. Pobudza neuronalny attractor `CURIOSITY`, który dopiero przez sieć wpływa na readouty.

Po podjęciu decyzji JOIN/MOVE wybór konkretnego kanału może dodatkowo uwzględniać uncertainty, aby spośród dostępnych miejsc częściej wybierać te słabiej poznane.

### Information Gain

Po nowym doświadczeniu porównywana jest niepewność:

```text
uncertainty_before - uncertainty_after = information_gain
```

Jeżeli wiedza realnie wzrosła, decyzja może dostać mały intrinsic reward. Dzięki temu eksploracja jest nagradzana za **zdobycie informacji**, a nie samo przypadkowe przemieszczanie się.

Domyślne parametry:

```toml
uncertainty_exploration_enabled = true
uncertainty_curiosity_magnitude = 1.10
uncertainty_curiosity_steps = 3
uncertainty_target_weight = 0.30
information_gain_reward_scale = 0.20
information_gain_reward_max = 0.08
information_gain_min_delta = 0.01
```

Stan jest widoczny w `/details` jako **Curiosity / Uncertainty**, razem z najbardziej nieznanymi kanałami, cue do attractora i ostatnim information gain.

# Rich Voice Sensory Dynamics

Voice Sensory Bus mierzy teraz nie tylko obecność ludzi i bieżących mówców, ale również dynamikę rozmowy z realnych callbacków PCM.

W oknie ostatnich ~60 sekund liczone są m.in.:

- udział czasu, w którym faktycznie trwała mowa,
- liczba unikalnych mówców,
- zmiany mówcy,
- zdarzenia overlap / crosstalk,
- średni czas przekazania tury między osobami,
- średnia i najdłuższa długość tury,
- dominacja jednego rozmówcy,
- intensywność rozmowy,
- tryb sceny: QUIET / CONVERSATION / DIALOGUE / MONOLOGUE / CROSSTALK,
- tempo mowy ostatniej transkrypcji,
- oczekiwanie na odpowiedź po TTS Muchy oraz realny reply latency.

Te wartości nie są bezpośrednimi bonusami do JOIN/MOVE/LEAVE/STAY. Są kodowane jako `voice:sensory:...` i trafiają do zwykłych populacji sensorycznych connectomu. Wpływ na zachowanie pojawia się dopiero po propagacji i przez wyuczone ścieżki.

Panel `/details` pokazuje te dane live w sekcji **Voice Sensory Bus — LIVE**, a najważniejsze sygnały trafiają również do Decision Trace.

---

# Sleep / Offline Consolidation

Mucha ma teraz osobny stan **SLEEP**, uruchamiany po dłuższym okresie realnej ciszy na Discordzie.

Domyślnie:

```toml
sleep_enabled = true
sleep_idle_seconds = 900
sleep_cycle_interval_seconds = 15
sleep_max_cycles = 8
sleep_replay_batch_size = 4
sleep_replay_magnitude_multiplier = 1.60
sleep_reward_scale_multiplier = 1.50
sleep_steps_multiplier = 2.00
```

Warunki wejścia w sen:

```text
brak nowych wiadomości / zmian voice
+ Mucha nie jest połączona z VC
+ brak aktywnego Chasera
+ istnieją epizody nadające się do replay
→ SLEEP
```

Podczas snu zwykłe autonomiczne decyzje voice, spontaniczne pisanie, TTS i rare audio są wstrzymane. Każdy cykl:

```text
ważny epizod
→ reaktywacja sensoryczna sceny
→ action-guided sensory
→ propagacja przez FAFB
→ capture learning trace
→ mały replay reward/punish
→ neuronalna + synaptyczna plastyczność
→ episodic consolidation
→ semantic rehearsal
```

Semantic rehearsal nie udaje nowego doświadczenia: nie zwiększa licznika realnych obserwacji. Stabilizuje tylko istniejące uogólnienie wyprowadzone z prawdziwych epizodów.

Każda nowa wiadomość lub zmiana voice natychmiast wybudza Muchę. Po zakończeniu pełnej sesji kolejny sen wymaga nowej aktywności i ponownego okresu ciszy.

Stan snu jest widoczny na żywo w `/details` jako **Sleep / Offline Consolidation** razem z:

- numerem cyklu i postępem,
- liczbą replayowanych epizodów,
- liczbą zmienionych neuronów i synaps,
- semantic rehearsal,
- zmianą siły wspomnień,
- liczbą utrwalonych scen i synaps.

---

# Decision Trace — „Dlaczego zrobiła X?”

Prywatny widok `/details` posiada teraz spójny **Decision Trace** zapisujący faktyczne dane użyte przez runtime przy decyzji.

Dla decyzji tekstowych trace zapisuje m.in.:

- bodziec tekstowy i aktywny Attention / Working Memory,
- dominujący internal state oraz poziomy neuromodulatorów,
- surowy `speak` readout,
- wynik po Learned Action Policy,
- bazowy i learned raw threshold,
- affinity użytkownika,
- cooldown, gotowość języka i blokady kanału,
- wynik równoległego gate'a reakcji,
- informację, czy wiadomość została rzeczywiście wysłana.

Dla voice trace składa istniejący bogaty debug decyzji w jeden łańcuch:

```text
voice sensory / context
→ episodic + semantic recall
→ uncertainty / curiosity
→ internal states
→ connectome propagation
→ raw action candidates
→ learned policy
→ winner / runner-up / margin
→ ograniczenia wykonania
→ realna decyzja
```

To jest telemetryczny zapis działania algorytmu, a nie deklaracja ukrytego „toku myślenia”.

## Decision Trace History

Runtime przechowuje w RAM do **48 pełnych, zamrożonych trace'ów**. Prywatny `/details` pokazuje ostatnie 40 jako klikalną oś czasu.

Każdy wpis zachowuje stan z chwili decyzji:

```text
timestamp
+ źródło TEXT / VOICE
+ bodziec
+ attention
+ internal states
+ neuromodulatory
+ memory / uncertainty
+ raw + effective readout
+ policy bias
+ ograniczenia
+ faktyczna akcja
```

Kliknięcie wpisu przełącza panel **„Dlaczego zrobiła X?”** w tryb zamrożony. Przycisk **LIVE** wraca do najnowszej decyzji. Dostępne są filtry `VOICE` i `TEXT`.

Historia jest celowo pamięcią diagnostyczną RAM i zeruje się po restarcie procesu. Nie jest publikowana przez `/api/public/state`.

---

# Live System Telemetry

Prywatny dashboard ma szybki kanał telemetryczny odświeżany domyślnie co **250 ms**.

Na `/` i `/details` są widoczne:

- użycie CPU całego systemu,
- użycie CPU procesu Muchy,
- RAM całego systemu,
- RAM procesu Muchy,
- liczba wątków procesu,
- zajętość i wolne miejsce na dysku,
- GPU utilization,
- zajętość VRAM,
- temperatura GPU.

CPU/RAM/dysk są odczytywane przez `psutil`. Telemetria NVIDIA jest pobierana przez `nvidia-smi` w osobnej pętli i cache'owana, więc szybkie odświeżanie dashboardu nie uruchamia osobnego procesu `nvidia-smi` przy każdym requestcie.

Jeżeli NVIDIA / `nvidia-smi` nie jest dostępne, panel pokazuje `GPU unavailable` i pozostała telemetria działa normalnie.

Cięższe dane, takie jak logi systemd i status usług, pozostają na wolniejszym interwale, aby sam dashboard nie powodował niepotrzebnego obciążenia.

# Long-term People Memory

Mucha posiada teraz trwałe profile konkretnych ludzi, budowane z realnych interakcji zamiast wyłącznie z jednej liczby affinity.

Profil osoby łączy:

- kontakty tekstowe,
- wypowiedzi STT na voice,
- wyniki wcześniejszych akcji przy tej osobie,
- trwałe semantic memory typu użytkownik → akcja,
- użytkownik + kanał,
- użytkownik + stan sytuacji,
- signed social events, np. pozytywne odpowiedzi, reuse słów/fraz i odrzucenia,
- ostatnie epizody voice,
- legacy affinity i neural social memory.

Kontakt sam w sobie jest zapisywany jako **neutralne doświadczenie**: zwiększa familiarity, ale nie udaje rewardu. Dopiero realne pozytywne lub negatywne sygnały społeczne wpływają na valence.

Profil wraca do mózgu jako zwykłe sensory:

```text
social:person-profile:user:<id>
social:person-profile:familiarity:<bucket>
social:person-profile:valence:<positive|neutral|mixed|negative>
social:person-contact:<text|voice_speech>:<bucket>
social:person-event:<event>:<positive|negative>
social:person-history:action:<action>:<positive|negative>
```

Te sygnały **nie zmieniają action score bezpośrednio**. Są wstrzykiwane do sensorycznych populacji connectomu i muszą przejść przez normalną propagację FAFB.

Prywatny `/details` ma panel **Long-term People Memory**, który pokazuje m.in.:

- familiarity i confidence,
- liczbę trwałych doświadczeń,
- liczbę kontaktów i signed social events,
- valence,
- typową korzystną i niekorzystną akcję przy tej osobie,
- najczęstsze kanały,
- ostatni epizod,
- ostatnią reiniekcję profilu do connectomu.

Dane są przechowywane w istniejącym:

```text
state/voice_episodes.sqlite3
```

---

# Long-term Channel / Place Memory

Mucha buduje teraz trwały model kanałów voice jako **miejsc**, a nie tylko identyfikatorów Discorda.

Profil miejsca łączy:

- realne wizyty JOIN/MOVE,
- wcześniejsze epizody i ich reward/punish,
- osoby spotykane w danym miejscu,
- tryby rozmowy z Voice Sensory Bus,
- średnią intensywność rozmów,
- typowy speech ratio,
- typową liczbę ludzi,
- historię akcji, które w tym miejscu kończyły się dobrze albo źle.

Starsze dane z `voice_episodes.sqlite3` są automatycznie używane do backfillu podstawowej familiarity oraz skojarzeń kanał ↔ ludzie.

Dynamika miejsca jest aktualizowana przy realnych transkrypcjach STT, dzięki czemu polling dashboardu nie sztucznie nie zwiększa liczby obserwacji.

Profil kanału wraca do mózgu jako sensory:

```text
voice:place-profile:channel:<id>
voice:place-profile:presence:<current|candidate>
voice:place-profile:familiarity:<bucket>
voice:place-profile:confidence:<bucket>
voice:place-profile:valence:<positive|neutral|negative>
voice:place-profile:mode:<dialogue|monologue|crosstalk|quiet|...>
voice:place-profile:intensity:<bucket>
voice:place-profile:speech-ratio:<bucket>
voice:place-profile:human-density:<bucket>
voice:place-history:person:<user>:<bucket>
voice:place-history:action:<action>:<positive|negative>
```

To nadal **nie jest ręczny bonus do JOIN/MOVE/STAY**. Pamięć miejsca trafia do sensorycznych populacji, propaguje się przez FAFB, a dopiero stan connectomu wpływa na readouty oraz istniejący neural channel affinity.

Prywatny `/details` ma panel **Long-term Channel / Place Memory**, pokazujący:

- familiarity i confidence,
- visit / dynamics observations,
- valence,
- dominujący tryb rozmowy,
- typową intensywność, speech ratio i liczbę ludzi,
- osoby najczęściej kojarzone z miejscem,
- korzystne i niekorzystne akcje,
- ostatni epizod,
- ostatnią reiniekcję pamięci miejsca do connectomu.

---

# Long-term Social Situations

Mucha posiada teraz trwałą pamięć **powtarzalnych sytuacji społecznych**, która łączy kilka warstw naraz:

```text
KTO
+ GDZIE
+ conversation mode
+ intensity
+ speech ratio
+ liczba ludzi
+ dominujący internal state
+ social need / fatigue / habituation / exploration
```

Scena dostaje stabilny, bucketowany klucz. Neutralne ponowne zobaczenie tej samej sceny zwiększa tylko familiarity. Dopiero realny reward/punish zapisany przez temporal credit aktualizuje wynik konkretnej akcji w tej sytuacji.

Przykład:

```text
ASG
+ Dawid + Stivi
+ DIALOGUE
+ intensity 2/3
+ speech 2/3
+ social_need 2/3
→ STAY +0.31
→ VOICE_LEAVE -0.18
```

Znana scena wraca do connectomu jako sensory:

```text
social:scene-profile:<hash>
social:scene-profile:familiarity:<bucket>
social:scene-profile:confidence:<bucket>
social:scene-profile:valence:<positive|neutral|mixed|negative>
social:scene-profile:mode:<mode>
social:scene-profile:state:<internal-state>
social:scene-profile:humans:<n>
social:scene-profile:intensity:<bucket>
social:scene-profile:speech:<bucket>
social:scene-history:action:<action>:<positive|negative>
```

Nie ma bezpośredniego `+score` do JOIN/MOVE/STAY. Reiniekcja sytuacji wchodzi do sensorycznych populacji FAFB, a wpływ na zachowanie musi przejść przez normalną propagację i uczenie.

Starsze epizody w `state/voice_episodes.sqlite3` są automatycznie backfillowane jako uproszczone sceny z trybem `UNKNOWN`. Nowe sceny zawierają bogatszą dynamikę oraz stan wewnętrzny.

Prywatny `/details` pokazuje panel **Long-term Social Situations** z familiarity, confidence, ludźmi, miejscem, dynamiką, stanem Muchy oraz historią dobrych i złych akcji.

---

# Reward-learned Conversation Dynamics

Mucha uczy się teraz **wyników akcji dla powtarzalnych wzorców rozmowy**, niezależnie od konkretnej osoby i kanału.

Klucz dynamiki łączy:

```text
conversation mode
+ intensity
+ speech ratio
+ speaker switches
+ overlap / crosstalk
+ handoff latency
+ średnia długość tury
+ speaker dominance
+ długość ciszy
+ speech rate
```

Przykładowy wzorzec:

```text
DIALOGUE
intensity 2/3
speech 2/3
switches 2/3
overlap 1/3
handoff normal
turn medium
dominance 2/3
silence active
speech-rate normal
```

Neutralne widzenie wzorca zwiększa tylko familiarity i jest ograniczone cooldownem, aby szybki polling nie pompował obserwacji.

Realny reward/punish przez temporal credit zapisuje historyczny outcome:

```text
DIALOGUE + niski overlap
→ STAY +0.55
→ VOICE_MOVE -0.40
→ SPEAK +0.30
```

Akcje voice `STAY / MOVE / LEAVE` uczą się z istniejącego temporal credit. `SPEAK` dostaje outcome z realnego feedbacku po TTS, m.in. kontynuacji rozmowy, reuse słowa/frazy albo werbalnego odrzucenia.

Wyuczona dynamika wraca do connectomu jako sensory:

```text
voice:dynamics-memory:<hash>
voice:dynamics-memory:familiarity:<bucket>
voice:dynamics-memory:confidence:<bucket>
voice:dynamics-memory:valence:<...>
voice:dynamics-memory:mode:<...>
voice:dynamics-memory:handoff:<...>
voice:dynamics-memory:turn:<...>
voice:dynamics-memory:silence:<...>
voice:dynamics-memory:speech-rate:<...>
voice:dynamics-memory:intensity:<bucket>
voice:dynamics-memory:speech:<bucket>
voice:dynamics-memory:switches:<bucket>
voice:dynamics-memory:overlap:<bucket>
voice:dynamics-memory:dominance:<bucket>
voice:dynamics-history:action:<action>:<positive|negative>
```

To nie jest `if CROSSTALK: speak -= 0.3`. Historyczny wynik jest reiniektowany jako sensoryczny kontekst, a wpływ na zachowanie musi przejść przez connectome i jego wyuczoną plastyczność.

Prywatny `/details` ma panel **Reward-learned Conversation Dynamics**, który pokazuje familiarity, confidence, parametry wzorca, liczbę realnych outcome oraz historycznie najlepsze i najgorsze wyniki akcji.

Decision Trace przechowuje również `voice_dynamics_key`, familiarity i valence z chwili decyzji.

---

# Reduced-hardcode Voice Decisions

Kolejna warstwa ręcznych preferencji została usunięta z autonomii voice.

## SPEAK / REACT vs STAY

Ręczne progi zachowania nie są już domyślną ścieżką dla tekstu, reakcji ani TTS.

W trybie:

```toml
connectome_behavior_competition_enabled = true
```

runtime używa wspólnej konkurencji:

```text
TEXT reply:        SPEAK vs STAY
REACTION:          REACT vs STAY
SPONTANEOUS TEXT:  SPEAK vs STAY
VOICE TTS:         SPEAK vs STAY
```

Każda konkurencja przechodzi przez `action_competition()`, który używa:

- surowych readoutów connectomu,
- learned action policy,
- neural tie evidence,
- unbiased fallback tylko przy praktycznie płaskim stanie sieci.

Dawne `speak_threshold` i `reaction_threshold` pozostają wyłącznie jako **legacy fallback**, gdy competition mode zostanie ręcznie wyłączony. Cooldowny pozostają ograniczeniami technicznymi, a nie preferencją decyzyjną.

Decision Trace i panel Learned Action Policy pokazują teraz `winner / runner-up / margin / source`, zamiast opisywać neural competition jako „przejście progu”.

## Wybór konkretnego kanału

Przy `connectome_voice_control_enabled = true` cel JOIN/MOVE nie jest już wybierany przez ręczne:

```text
affinity
+ novelty bonus
- recent penalty
+ uncertainty bonus
+ forced reward-opportunity target
```

Każdy kandydat dostaje własny sensoryczny kontekst:

```text
voice:target:<guild>:<channel>:novelty:<bucket>
voice:target:<guild>:<channel>:recent:<bucket>
voice:target:<guild>:<channel>:uncertainty:<bucket>
voice:target:<guild>:<channel>:reward-opportunity
voice:target:<guild>:<channel>:current
```

Po propagacji wybór celu odbywa się przez kanałowe neural readouty `voice-affinity:<guild>:<channel>`. Ręcznie ważony exploration score pozostaje tylko w trybie legacy.

Losowa reward opportunity nadal jest bodźcem, ale nie wymusza już konkretnego kanału przez `preferred_channel_id`.

## Reward nie zmienia decyzji po fakcie

Social-drive punish, missed reward-opportunity punish i overstay punish uczą ślad neuronalny, ale **nie przeliczają ponownie bieżącej decyzji w tym samym ticku**.

Przebieg jest teraz:

```text
sensory
→ connectome
→ decyzja
→ wykonanie decyzji
→ outcome / reward
→ plastyczność
→ wpływ na następne decyzje
```

zamiast:

```text
decyzja STAY
→ skrypt daje punish
→ natychmiastowe ponowne losowanie decyzji
→ skrypt próbuje wymusić zmianę
```

## Co celowo pozostaje twarde

Nie są usuwane ograniczenia, które nie są preferencją Muchy:

- Discord permissions / brak możliwości połączenia,
- blocked guild/channel,
- kanał AFK, jeśli został wyłączony,
- motor refractory / minimum dwell jako fizyczna blokada ponownego ruchu,
- Chaser panic i escape,
- brak dostępnego celu.

Prywatny `/details` pokazuje teraz **Target selection = neural-channel-readout**, target score kanałów oraz informację, gdy reward został zapisany wyłącznie dla następnych decyzji.

---

# Kierunek po integracji One Brain

Stage 25 realizuje wspólny neural arbitration layer dla tekstu, reakcji, TTS i autonomii. Dalsze kierunki nie wymagają dokładania kolejnych niezależnych systemów decyzyjnych:

- pogłębianie biologicznego wpływu connectomu na istniejący One Brain,
- lepsze uczenie expected reward i temporal credit z długiego doświadczenia,
- bogatsza pamięć osób, miejsc i sytuacji społecznych,
- dalsza redukcja ręcznych wyjątków do bezpieczeństwa, permissions i fizycznych ograniczeń Discorda,
- lepsza obserwowalność przepływu sensory → connectome → motor output.

---

## Założenie projektu

Najważniejsza zasada Muchy:

**jak najwięcej zachowania ma wynikać z connectomu, jego aktualnego stanu i doświadczenia, a jak najmniej z ręcznie zakodowanych decyzji.**

---

# Stage 24A — Homeostatic Internal Drives

Mucha ma teraz pięć wolnozmiennych potrzeb wewnętrznych:

- `social_need` — potrzeba kontaktu,
- `curiosity` — ciekawość,
- `exploration` — potrzeba eksploracji,
- `caution` — ostrożność po zagrożeniu lub negatywnym rewardzie,
- `boredom` — nuda narastająca przy braku bodźców.

Drive'y nie wybierają akcji bezpośrednio. Ich wartości są mapowane na istniejące neuronalne attractory (`social_need`, `curiosity`, `stress`, `satiety`, `arousal`) i wchodzą do connectomu przez sensory entry pools. Dopiero aktywność sieci wpływa później na readouty akcji.

Homeostaza jest aktualizowana w `idle_loop`: cisza podnosi potrzebę kontaktu, ciekawość, eksplorację i nudę, kontakt z ludźmi obniża potrzebę społeczną, aktywność ogranicza nudę, a ostrożność samoistnie wygasa. Chaser i negatywny reward zwiększają ostrożność. Pozytywny reward, udana interakcja i eksploracja zaspokajają odpowiednie potrzeby.

Stan drive'ów jest zapisywany w `brain_state.npz`, więc nie zeruje się po restarcie procesu.

Konfiguracja:

```toml
internal_drives_enabled = true
internal_drive_neural_gain = 0.42
internal_drive_social_need_per_minute = 0.035
internal_drive_curiosity_per_minute = 0.022
internal_drive_exploration_per_minute = 0.016
internal_drive_boredom_per_minute = 0.045
internal_drive_caution_decay_per_minute = 0.055
```

To jest warstwa 24A. Następny etap 24B może użyć tych potrzeb do autonomicznego generowania kandydatów akcji, przy zachowaniu `STAY/NOOP` jako normalnej konkurującej możliwości.

---

# Stage 24B — Autonomous Action Candidates

Mucha tworzy teraz własny zestaw kandydatów akcji podczas każdego `idle_loop`.

To nadal **nie wykonuje akcji**. 24B odpowiada wyłącznie za pytanie:

```text
co w tej chwili ma sens rozważyć?
```

Generator bierze pod uwagę:

- techniczną wykonalność akcji,
- bieżące homeostatic drives z 24A,
- aktywność neuronalnych internal-state attractors,
- aktualne readouty connectomu,
- learned action policy.

Typowy zestaw poza voice:

```text
NOOP / STAY
SPEAK
VOICE_JOIN
EXPLORE
```

Po wejściu na voice:

```text
NOOP / STAY
SPEAK
VOICE_MOVE
EXPLORE
```

`STAY` jest wystawiany na zewnątrz jako `NOOP` i nigdy nie jest usuwany z zestawu. Autonomia nie oznacza więc obowiązku wykonania akcji.

`SPEAK` pozostaje kandydatem, jeżeli istnieje dozwolony ostatni kanał tekstowy i model języka jest gotowy. Drive'y nie mają prawa wyciszyć tej akcji na stałe.

JOIN/MOVE pojawiają się tylko wtedy, gdy istnieje technicznie dostępny kanał voice: z respektowaniem blokad, AFK, permissions, deadly-channel oraz aktywnego Chasera.

Każdy kandydat zawiera diagnostykę:

```text
raw_score
effective_score
drive_support
state_support
supporting_drives
supporting_states
technical_reason
```

Następnie wykonywany jest wyłącznie **competition preview** przez istniejące `action_competition()`. Wynik jest zapisywany do dashboard snapshot jako `autonomous_candidates`, ale ma `executed = false`.

To celowo oddziela:

```text
24B: wygeneruj możliwości
24C: przewidź reward możliwości
24D: pozwól autonomicznej pętli wykonać zwycięzcę
```

---

# Stage 24C — Predicted Reward

Każdy autonomiczny kandydat z 24B ma teraz przewidywany reward przed wykonaniem akcji.

Predykcja nie jest ręcznym bonusem przypisanym do typu zachowania. Jest uczona z rzeczywistych wyników:

```text
global action history:
reward_ema + liczba reward update'ów

voice context:
episodic prediction dla konkretnej sceny / ludzi / kanału
```

Brak wcześniejszych doświadczeń daje neutralny prior:

```text
predicted_reward = 0
confidence = 0
```

Wraz z kolejnymi doświadczeniami rośnie `prediction_confidence`. Jeżeli istnieje zarówno ogólna historia akcji, jak i kontekstowa pamięć voice, oba przewidywania są łączone proporcjonalnie do confidence wynikającego z liczby obserwacji.

Każdy kandydat zawiera teraz m.in.:

```text
predicted_reward
prediction_confidence
prediction_source
global_reward
global_observations
contextual_reward
contextual_observations
```

24C tworzy też:

```text
predicted_reward_order
predicted_reward_winner
```

To nadal wyłącznie preview. `prediction_executed = false` i `executed = false`.

Rozdział odpowiedzialności pozostaje świadomy:

```text
24A — wewnętrzne potrzeby
24B — jakie akcje są możliwe
24C — czego Mucha spodziewa się po każdej akcji
24D — autonomicznie wybierz i wykonaj akcję
```

Dla voice 24C wykorzystuje istniejącą trwałą pamięć epizodyczną i jej prediction error / temporal credit zamiast budować drugi niezależny model.

---

# Stage 24D — Autonomous Loop

24D uruchamia właściwą pętlę autonomiczną. Mucha nie tylko generuje kandydatów i przewiduje reward, ale wykonuje finalnego zwycięzcę neuronalnej konkurencji.

Pipeline:

```text
24A internal drives
→ 24B feasible action candidates
→ 24C learned predicted reward
→ predicted reward jako signed sensory cue
→ propagacja przez connectome
→ action_competition()
→ finalny winner
→ wykonanie akcji
```

Predicted reward nie wybiera akcji przez osobny ranking. Dodatni lub ujemny expected reward jest podawany do sensorycznych ścieżek prowadzących do danego action readoutu. Dopiero po propagacji przez FAFB wykonywane jest zwykłe `action_competition()`.

24D może wykonać:

```text
NOOP / STAY
SPEAK
VOICE_JOIN
VOICE_MOVE
EXPLORE
```

`NOOP` jest pełnoprawną decyzją i oznacza brak zewnętrznej akcji.

`SPEAK` respektuje gotowość modelu, blocked text channels, user avoidance oraz istniejący spontaneous cooldown.

`VOICE_MOVE` nie pojawia się podczas minimum dwell / motor refractory. JOIN/MOVE przed wykonaniem ponownie sprawdzają permissions, blocked/AFK/deadly channel i aktywnego Chasera. Konkretny kanał nadal wybiera neural channel targeting.

`EXPLORE` jest na tym etapie wewnętrznym skanem środowiska: Mucha podaje aktualne możliwe kanały voice / ostatni kanał tekstowy do sensory context i pobudza curiosity. Nie generuje przez to dodatkowego spamu na Discordzie.

Stary `voice_loop` nie podejmuje już zwykłych JOIN/MOVE, kiedy 24D jest włączone. Pozostaje aktywny tylko jako warstwa awaryjna dla:

```text
blocked voice guild
Chaser panic / escape
maximum dwell / emergency voice handling
```

Dzięki temu `NOOP` z 24D nie może zostać chwilę później nadpisany przez drugi zwykły voice decision.

Losowe voice reward opportunity oraz social-drive reward są zachowane również dla JOIN wykonanego przez 24D. Decyzje voice są nadal zapisywane do episodic prediction queue, więc późniejszy reward może aktualizować prediction error i temporal credit.

Konfiguracja:

```toml
autonomous_loop_enabled = true
autonomous_predicted_reward_gain = 0.85
autonomous_prediction_steps = 2
autonomous_explore_cooldown_seconds = 30
```

Przy `autonomous_loop_enabled = false` pozostaje legacy spontaneous text path oraz zwykły legacy/connectome voice loop.

Następny etap 24E dotyczy dashboardu i pełnej obserwowalności tego pipeline'u.

---

# Stage 24E — Autonomy Dashboard

24E dodaje prywatną zakładkę:

```text
/autonomy
```

która pokazuje na żywo cały pipeline 24A → 24D bez mieszania go z ogólnym dashboardem.

Widok zawiera:

- status zunifikowanej pętli 24D,
- finalnego winnera,
- wynik wykonania i informację, czy akcja miała efekt zewnętrzny,
- predicted reward oraz confidence,
- wszystkie akcje z 24B, również technicznie zablokowane,
- `technical_reason`, np. `motor-refractory` albo brak celu voice,
- `effective_score`,
- `drive_support`,
- `state_support`,
- predicted reward i confidence każdego kandydata,
- signed sensory prediction cues,
- finalny runner-up, margin i tie-break,
- historię decyzji autonomii.

Powtarzające się decyzje `NOOP / STAY` bez efektu zewnętrznego są kompresowane w historii do jednego wpisu z licznikiem `×N`, dzięki czemu łatwo zobaczyć okresy bezczynności bez zalewania interfejsu.

Przy wielu serwerach można przełączać zakładki guildów. Finalna decyzja jest oznaczana wyłącznie przy serwerze faktycznie wybranym przez scheduler w danym ticku; pozostałe pokazują swój zestaw kandydatów bez fałszywego przypisywania winnera.

Najważniejsze pola mają tooltipy po najechaniu kursorem, wyjaśniające m.in. różnicę między feasibility, score, predicted reward, confidence i finalnym neural winnerem.

Dashboard odświeża się z częstotliwością `web_ui.refresh_ms` i korzysta z istniejącego prywatnego `/api/state`.

Dodatkowo ustawienia 24D są dostępne z edytora konfiguracji dashboardu:

```toml
behavior.autonomous_loop_enabled
behavior.autonomous_predicted_reward_gain
behavior.autonomous_prediction_steps
behavior.autonomous_explore_cooldown_seconds
```

Snapshot runtime zawiera też `autonomous_history`, a smoke test pilnuje obecności strony, routingu, historii, tooltipowego pipeline'u oraz kontrolek konfiguracji.

24E zamyka etap 24 i przygotowuje projekt do Stage 25 — integracji One Brain.

---

# Stage 25 — One Brain

Stage 25 scala dobrowolne zachowania Muchy pod jeden wspólny neural arbitration core.

Wspólny rdzeń:

```text
one_brain_candidate_set()
→ learned predicted reward
→ signed sensory cues
→ FAFB propagation
→ one_brain_action_decision()
→ action_competition()
→ dokładnie jeden motor winner
→ wykonanie
→ outcome / reward / plasticity
```

## Jedna decyzja zamiast kilku niezależnych bramek

Dla wiadomości tekstowej poprzednia architektura mogła osobno policzyć:

```text
REACT vs STAY
SPEAK vs STAY
```

Stage 25 buduje jeden zestaw:

```text
STAY / NOOP
REACT
SPEAK
```

i wykonuje jedną konkurencję One Brain.

Jeżeli REACT wygra, dopiero potem osobny neural readout wybiera konkretne emoji.
Jeżeli SPEAK wygra, dopiero potem language brain dobiera konkretne słowa.
To rozdziela decyzję motorową od wyboru szczegółowego celu/treści.

## Wspólny rdzeń dla modalności

Przez `one_brain_action_decision()` przechodzą teraz:

```text
TEXT:
  NOOP / REACT / SPEAK

VOICE TTS:
  NOOP / SPEAK

AUTONOMY:
  NOOP / SPEAK / VOICE_JOIN / VOICE_MOVE / EXPLORE
```

Stage 24D pozostaje kompatybilny przez `autonomous_action_decision()`, ale ta metoda jest już wrapperem wspólnego One Brain core.

Ordinary JOIN/MOVE nadal wybierają konkretny kanał przez neural channel targeting po tym, jak One Brain wybierze typ akcji.

## Predicted reward

Każda modalność korzysta z tego samego mechanizmu:

```text
expected reward × confidence × gain
→ signed sensory input
→ realne ścieżki connectomu
→ propagacja
→ action competition
```

Predicted reward nie dopisuje punktów bezpośrednio do winner score.

Konfiguracja:

```toml
one_brain_enabled = true
one_brain_predicted_reward_gain = 0.85
one_brain_prediction_steps = 2
```

Przy włączonym One Brain parametry te są używane również przez autonomię 24D. Starsze `autonomous_predicted_reward_gain` i `autonomous_prediction_steps` pozostają jako fallback kompatybilności.

## Technical feasibility

Akcja zablokowana technicznie nie bierze udziału w konkurencji.

Przykłady:

```text
SPEAK:
blocked text / language not ready / reply cooldown / social avoid

REACT:
reaction cooldown / brak dostępnego emoji / social avoid

VOICE_JOIN:
brak dozwolonego targetu

VOICE_MOVE:
poza VC / minimum dwell / brak alternatywy

EXPLORE:
explore cooldown
```

NOOP / STAY pozostaje zawsze dostępny.

## Brak martwych mini-konkurencji

W trybie One Brain stare `_behavior_gate()` dla TEXT działają jako `one-brain-deferred`: raportują score/policy do diagnostyki, ale nie wykonują własnego `action_competition()`.

Dzięki temu jedno zdarzenie tekstowe ma dokładnie jedną właściwą neural competition.

## Cross-modal timeline

Runtime przechowuje:

```text
one_brain
one_brain_history
```

Historia łączy w jednym miejscu decyzje:

```text
text
voice_tts
autonomy
```

Każdy wpis zawiera m.in.:

```text
winner
runner-up
margin
candidate actions
technical feasibility
predicted reward
prediction confidence
execution success
external effect
decision context
```

Zakładka `/autonomy` pokazuje teraz **One Brain 25 / Autonomia 24E** oraz wspólny One Brain timeline.

## Co celowo pozostaje poza One Brain

Nie są to preferencje zachowania, tylko kontrola techniczna, bezpieczeństwo albo jawne polecenie administratora:

```text
Discord permissions
blocked guild/channel/user
AFK exclusions
Chaser panic / emergency escape
maximum-dwell emergency handling
admin commands
explicit audiotest
rare random audio reflex
reconnect / playback / STT plumbing
```

Chaser escape może więc przerwać zwykłą decyzję One Brain. Jest to safety/reflex path, a nie drugi system preferencji.

## Stan końcowy

Po Stage 25 architektura wysokiego poziomu wygląda tak:

```text
sensory events
+ attention / working memory
+ people / place / social memory
+ internal drives
+ voice sensory state
+ learned reward expectations
        ↓
     ONE BRAIN
        ↓
 shared neural arbitration
        ↓
 motor action
        ↓
 target/content selection
        ↓
 Discord outcome
        ↓
 reward + synaptic / policy / memory learning
        ↺
```

Stage 25 zamyka główny roadmap autonomii i integracji mózgu. Dalsze prace mogą rozwijać biologiczną jakość connectome, uczenie, pamięć i zachowanie bez dokładania kolejnych niezależnych systemów decyzyjnych.



---

# Stage 26 — Circadian Fatigue / Sleep Pressure

Stage 26 rozbudowuje istniejący Sleep / Offline Consolidation o trwały homeostatic drive `fatigue`.

Pipeline:

```text
czas czuwania + aktywność
→ persistent fatigue
→ SATIETY / STRESS attractors
→ realne ścieżki FAFB
→ One Brain action competition
```

`fatigue` nie wybiera akcji bezpośrednio. Wysokie zmęczenie pobudza istniejące attractory wspierające `STAY`, `VOICE_LEAVE` i `VOICE_MOVE`, więc wpływ nadal musi przejść przez connectome.

Każdy prawdziwy cykl snu wykonujący replay wywołuje zdarzenie `rest`, które spłaca część fatigue. Pełna sesja snu uruchamia krótki stan `POST-SLEEP` oraz łagodny cue SATIETY/AROUSAL przez connectome.

Stany cyklu:

```text
AWAKE
TIRED
SLEEP
POST-SLEEP
```

Nowe parametry:

```toml
circadian_enabled = true
circadian_fatigue_per_minute = 0.006
circadian_activity_fatigue_per_minute = 0.003
circadian_sleep_recovery_per_cycle = 0.16
circadian_tired_threshold = 0.65
circadian_post_sleep_seconds = 600
```

Fatigue jest zapisywany razem z innymi homeostatic drives w `brain_state.npz`. Loader pozostaje kompatybilny ze starszymi plikami stanu zawierającymi tylko pięć drive'ów.

Panel Sleep w `/details` pokazuje teraz osobno stan snu, stan dobowy i poziom fatigue.


---

# Stage 27 — Persistent Affective State

Stage 27 dodaje wolnozmienny stan afektywny wynikający z realnej aktywności connectomu, homeostatic drives i reward trace.

Stany:

```text
CONTENTMENT
TENSION
CURIOSITY
SOCIAL_LONGING
ACTIVATION
```

To nie jest osobny system decyzji ani ręczne `mood += 1`.

Pipeline:

```text
FAFB internal attractors
+ homeostatic drives
+ reward trace
        ↓
slow affective EMA
        ↓
persistent affective state
        ↓
sensory feedback do tych samych attractorów
        ↓
FAFB propagation
        ↓
One Brain
```

Przykłady:

- `CONTENTMENT` bazuje głównie na SATIETY i dodatnim reward trace,
- `TENSION` bazuje na STRESS, caution i ujemnym reward trace,
- `CURIOSITY` bazuje na attractorze CURIOSITY oraz drives curiosity/exploration,
- `SOCIAL_LONGING` bazuje na SOCIAL_NEED,
- `ACTIVATION` bazuje głównie na AROUSAL.

Stan afektywny ma pamięć czasową przez EMA, więc nie znika po pojedynczym ticku. Jest zapisywany w `brain_state.npz` jako `affective_state_values` i pozostaje kompatybilny ze starszymi plikami stanu.

Nowe ustawienia:

```toml
affective_state_enabled = true
affective_state_smoothing = 0.96
affective_state_feedback_gain = 0.18
affective_state_reward_gain = 0.28
```

Dashboard `/details` pokazuje osobno aktualną wartość i target każdego stanu. Dzięki temu można zobaczyć nie tylko „co Mucha czuje teraz”, ale też w którą stronę przesuwa ją bieżąca aktywność neuronalna.


---

# Stage 28 — Durable Person Relationship Memory

Stage 28 rozbudowuje Long-term People Memory z profilu agregatowego do trwałej historii relacji.

Nowa tabela w `state/voice_episodes.sqlite3`:

```text
person_interaction_history
```

Każdy wpis może przechować:

```text
czas
user_id / user_name
rodzaj zdarzenia: contact / social / episode
źródło
akcję Muchy
reward / signed social outcome
guild
kanał
kontekst
```

Przy pierwszym uruchomieniu Stage 28 istniejące zapisane `voice_episodes` są automatycznie używane do zainicjalizowania historii osób. Migracja jest wykonywana hurtowo i przycinana dopiero na końcu, żeby nie blokować startu przy dużej bazie.

Profil osoby zawiera teraz m.in.:

```text
familiarity
valence
confidence
first_seen / last_seen
relationship_age_days
history_observations
recent_valence
relationship_trend = improving / stable / worsening
relationship_stability
action_outcomes
dominant_history_channel
interaction_history
preferred_action
avoided_action
```

`recent_valence` opisuje ostatnie realne signed outcomes, a `relationship_trend` porównuje nowszą część historii ze starszą. `relationship_stability` mierzy, jak zmienne są ostatnie outcomes.

Chronologia jest ograniczona w panelu do najnowszych wpisów, ale licznik i wiek relacji są liczone po całej zachowanej historii osoby.

## Wpływ na connectome

Stage 28 nie dodaje bezpośredniego bonusu do action score.

Profil osoby jest zamieniany na sensoryczne cues, m.in.:

```text
social:person-history:trend:<state>
social:person-history:recent:<positive|negative>
social:person-history:stability:<stable|variable|volatile>
social:person-history:action-outcome:<action>:<positive|negative>
social:person-history:density:<bucket>
social:person-history:recent-contact
```

Dopiero potem sygnał propaguje się przez connectome i bierze udział w One Brain.

Dashboard `/details` pokazuje teraz trend relacji, recent valence, stability, wiek relacji, chronologię i historyczne outcome konkretnych akcji.


---

# Stage 29 — Durable Channel / Place Memory

Stage 29 rozbudowuje model kanału voice z agregatów semantycznych do trwałej chronologii miejsca.

Nowa tabela w `state/voice_episodes.sqlite3`:

```text
channel_interaction_history
```

Chronologia może zapisać:

```text
czas
channel_id / channel_name
rodzaj: visit / dynamics / episode
źródło
akcję Muchy
reward / outcome
liczbę ludzi
user_ids
conversation mode
intensity
speech ratio
context
```

Istniejące zapisane `voice_episodes` są przy pierwszym uruchomieniu używane do zainicjalizowania historii kanałów.

Profil miejsca zawiera teraz m.in.:

```text
familiarity
valence
confidence
dominant_mode
mean_intensity
mean_speech_ratio
mean_human_density
history_observations
first_seen / last_seen
place_age_days
recent_valence
place_trend = improving / stable / worsening
place_stability
recent_human_density
action_outcomes
interaction_history
people
conversation_modes
preferred_action
avoided_action
```

`recent_valence` bazuje na najnowszych realnych outcome akcji w danym miejscu. `place_trend` porównuje nowszą część historii ze starszą, a `place_stability` opisuje przewidywalność ostatnich wyników. `recent_human_density` opisuje ostatnią typową liczbę ludzi.

## Wpływ na connectome

Stage 29 nie dodaje bezpośrednich punktów do `VOICE_JOIN`, `VOICE_MOVE`, `STAY` ani `VOICE_LEAVE`.

Historia miejsca jest zamieniana na sensoryczne cues:

```text
voice:place-history:density:<bucket>
voice:place-history:trend:<state>
voice:place-history:recent:<positive|negative>
voice:place-history:stability:<stable|variable|volatile>
voice:place-history:recent-human-density:<bucket>
voice:place-history:recent-visit
voice:place-history:action-outcome:<action>:<positive|negative>
```

Dopiero potem sygnał propaguje się przez connectome i uczestniczy w One Brain.

Dashboard `/details` pokazuje teraz chronologię miejsca, recent valence, trend, stability, recent occupancy, wiek miejsca i historyczne outcome akcji.

Nowa tabela jest częścią `voice_episodes.sqlite3`, więc istniejący automatyczny backup Google Drive zabezpiecza ją bez dodatkowej konfiguracji.


---

# Stage 30 — Natural Drives and Motivation

Stage 30 łączy pojedyncze homeostatic drives w cztery wspólne motywy:

```text
SOCIAL
NOVELTY
SAFETY
REST
```

Motywy nie są nowymi akcjami i nie dodają punktów bezpośrednio do `SPEAK`, `VOICE_JOIN`, `VOICE_MOVE`, `EXPLORE` ani `STAY`.

Pipeline:

```text
homeostatic drives
+ slow affect
      ↓
motivation pressure
      ↓
frustration / satiation
      ↓
urgency
      ↓
skalowanie istniejących drive sensory inputs
      ↓
internal-state attractors
      ↓
FAFB propagation
      ↓
One Brain
```

## Motywy

`SOCIAL` korzysta głównie z `social_need`, części `boredom` i `social_longing`.

`NOVELTY` korzysta z `curiosity`, `exploration`, `boredom` oraz affective curiosity/activation.

`SAFETY` korzysta z `caution` oraz `tension`.

`REST` korzysta z `fatigue`.

## Pressure, frustration i satiation

Każdy motyw ma trzy ważne składniki:

- **pressure** — aktualna potrzeba wynikająca z drives i affectu,
- **frustration** — rośnie, gdy pressure długo pozostaje powyżej progu bez zaspokojenia,
- **satiation** — rośnie po realnym zaspokojeniu potrzeby i z czasem zanika.

Urgency powstaje z tych trzech wartości. Wysoka frustracja wzmacnia urgency, a wysokie satiation chwilowo ją tłumi.

Przykładowe zdarzenia:

```text
social_contact / social_success → SOCIAL satiation
novelty / exploration_complete → NOVELTY satiation
threat / negative_reward → SAFETY frustration
rest → REST satiation
```

Frustration i satiation są zapisywane w `state/brain_state.npz`, więc stan motywacyjny przeżywa restart.

Nowe ustawienia:

```toml
motivation_enabled = true
motivation_frustration_threshold = 0.42
motivation_frustration_per_minute = 0.035
motivation_frustration_decay_per_minute = 0.08
motivation_satiation_decay_per_minute = 0.10
motivation_frustration_gain = 0.60
motivation_satiation_gain = 0.70
motivation_neural_gain = 0.55
```

Dashboard `/details` pokazuje dla każdego motywu pressure, frustration, satiation i finalne urgency. Te same parametry są dostępne w edytorze konfiguracji.

Stage 30 współpracuje z mechanizmem NOOP anti-lock: przy autonomii `drive_support` jest teraz liczony z uwzględnieniem aktualnej motywacyjnej urgency, więc drugi neuronalny przebieg po NOOP bierze pod uwagę nie tylko surową wartość drive, ale również historię niezaspokojenia i chwilowe nasycenie.


---

# Stage 31 — Internal Counterfactual Foresight

Stage 31 rozbudowuje istniejące learned reward prediction o wewnętrzną symulację kandydatów przed finalną decyzją.

Mucha nie próbuje przewidywać całego Discorda ani zachowania konkretnych ludzi. Symuluje tylko ten fragment przyszłości, który może oszacować z własnego stanu:

```text
bieżące drives
+ bieżąca motivation
+ action ↔ internal-state alignment
+ learned expected reward
        ↓
counterfactual copy
        ↓
przewidywany relief drives
        ↓
przewidywane SOCIAL / NOVELTY / SAFETY / REST
        ↓
state relief + risk + confidence
        ↓
foresight sensory cue
        ↓
FAFB propagation
        ↓
One Brain final competition
```

## Brak mutacji live state

Symulacja działa na kopii wartości. Samo „wyobrażenie” `VOICE_JOIN`, `EXPLORE`, `SPEAK` itd. nie zmniejsza prawdziwego `social_need`, `curiosity`, `fatigue` ani innych drives.

Realny stan zmienia się dopiero po rzeczywistym zdarzeniu i jego outcome.

## Skąd bierze się przewidywany relief

Stage 31 nie ma tabeli typu „VOICE_JOIN = +0.5”.

Możliwy relief jest wyprowadzany z istniejących map:

```text
homeostatic drive
→ internal-state attractor
→ akcje wspierane przez attractor
```

Przykładowo `social_need` prowadzi do SOCIAL_NEED/AROUSAL, a te już wcześniej były powiązane z odpowiednimi readoutami. Foresight tylko sprawdza, jak hipotetyczne rozładowanie zgodnego drive zmieniłoby późniejszą motivational urgency.

## Dane dla każdego kandydata

Każda symulowana akcja dostaje:

```text
drive_alignment
drive_changes
motivation_changes
state_relief
predicted_reward
reward_confidence
simulation_confidence
risk
forecast_value
state_signal
```

`risk` łączy niepewność historii z przewidywanym negatywnym rewardem. Jest przede wszystkim diagnostyką.

`forecast_value` służy do pokazania kontrfaktycznego rankingu, ale **nie jest finalnym action score**.

Finalny winner nadal pochodzi z:

```text
foresight sensory cue
+ learned reward sensory cue
+ cały bieżący stan sieci
→ FAFB propagation
→ action_competition()
```

## Foresight ranking vs final winner

Dashboard może pokazać np.:

```text
Foresight winner: VOICE_JOIN
Final winner: EXPLORE
```

To jest prawidłowe. Foresight jest jednym z wejść, a nie osobnym arbitrem.

## Ustawienia

```toml
foresight_enabled = true
foresight_drive_relief_scale = 0.38
foresight_state_signal_gain = 0.65
foresight_base_confidence = 0.55
foresight_uncertainty_weight = 0.35
```

Panel `/autonomy` pokazuje dla kandydatów foresight relief, risk, forecast, simulation confidence i przewidywane zmiany drives. Pokazuje też osobno `Foresight winner` oraz `Foresight margin`.

Stage 31 nie dodaje nowego trwałego pliku. Korzysta z istniejących drives, motivation, affect oraz learned reward memory, które już są objęte backupem.


---

# Stage 32 — Persistent Intent / Goal Holding

Stage 32 dodaje krótkotrwałą pamięć zamiaru dla autonomii. Zamiar nie jest osobnym systemem decyzyjnym i nie nadpisuje wyniku One Brain.

Pipeline:

```text
finalny winner FAFB
+ neural margin
+ Stage 31 foresight
+ learned reward confidence
        ↓
intention evidence
        ↓
decaying persistent intent
        ↓
action-guided sensory cue przy kolejnym autonomous ticku
        ↓
FAFB propagation
        ↓
świeże action_competition()
```

## Najważniejsza zasada

Intencja **nie dodaje punktów bezpośrednio do action score**.

Jeżeli np. wcześniejsza decyzja utworzyła zamiar `VOICE_JOIN`, kolejny tick nie wykonuje JOIN automatycznie. Stage 32 podaje jedynie dodatni sensory cue do istniejących ścieżek `VOICE_JOIN`. Dopiero bieżący stan connectomu może ponownie wybrać tę akcję albo ją przegłosować.

Intencja działa tylko dla kontekstu autonomicznego `autonomous-idle`, więc nie wymusza kolejnych odpowiedzi tekstowych, reakcji ani TTS po pojedynczym zdarzeniu.

## Formowanie i utrzymanie

Evidence intencji powstaje z:

```text
siła finalnego neuronalnego winnera
+ margin nad runner-upem
+ zgodny przewidywany state relief z Stage 31
+ dodatni learned reward przy istniejącej confidence
```

Jeżeli kolejny finalny winner jest taki sam, intencja jest wzmacniana. Jeżeli winner się zmienia, nowa intencja zastępuje starą dopiero wtedy, gdy jej evidence przekroczy aktualną siłę o `intention_switch_margin`. Dzięki temu pojedyncze małe wahanie sieci nie powoduje ciągłego przeskakiwania zamiaru.

NOOP nie kasuje intencji automatycznie. Zamiar może przeżyć chwilowy `STAY`, ale cały czas naturalnie słabnie.

## Wygaszanie i outcome

Siła intencji ma half-life i maksymalny wiek.

Dodatni reward dla tej samej akcji wzmacnia intencję. Ujemny reward ją osłabia, a silny punish może ją natychmiast wyczyścić.

Jeżeli zamierzona akcja przestaje być technicznie możliwa w bieżącym kontekście, intencja jest usuwana zamiast omijać feasibility.

## Persistence

Akcja intencji, jej siła i timestamps są zapisywane w `state/brain_state.npz`. Po restarcie zostaje odtworzona z uwzględnieniem czasu, który upłynął — stary zamiar może więc wygasnąć podczas wyłączenia procesu.

## Ustawienia

```toml
intention_enabled = true
intention_half_life_seconds = 90.0
intention_max_age_seconds = 300.0
intention_signal_gain = 0.45
intention_reinforcement_gain = 0.30
intention_switch_margin = 0.10
intention_min_evidence = 0.10
intention_outcome_gain = 0.75
```

Dashboard `/autonomy` pokazuje teraz `Active intent` oraz `Intent strength`, a `/config` pozwala edytować parametry Stage 32.


---

# Stage 33 — Multi-step Motivational Goals

Stage 33 rozwija Stage 32 z krótkiego zamiaru do celu, który może przeżyć wiele kolejnych autonomicznych ticków i kilka różnych wykonanych działań.

Cel nie jest listą ręcznie narzuconych komend. Reprezentuje jedną aktualną potrzebę:

```text
SOCIAL
NOVELTY
SAFETY
REST
```

Pipeline:

```text
Stage 32 persistent intent
+ Stage 31 counterfactual relief
+ aktualna motivational urgency
        ↓
utworzenie celu
        ↓
persistent target motivation
        ↓
dla każdego kandydata:
przewidywany relief celu
        ↓
goal sensory cue
        ↓
FAFB propagation
        ↓
fresh action_competition()
        ↓
wykonany krok
        ↓
realna zmiana drives / motivation
        ↓
progress / success / abandon
```

## Dynamiczna sekwencja zamiast skryptu

Stage 33 nie posiada tabeli w rodzaju:

```text
SOCIAL = VOICE_JOIN -> SPEAK -> STAY
```

Jeżeli aktywny jest cel `SOCIAL`, każdy bieżący kandydat jest oceniany przez już istniejący Stage 31 pod kątem tego, ile może zmniejszyć urgency SOCIAL.

Dlatego rzeczywista sekwencja może wyglądać np.:

```text
VOICE_JOIN
→ STAY
→ SPEAK
→ VOICE_MOVE
```

albo zupełnie inaczej. Każdy krok nadal musi wygrać w normalnym FAFB `action_competition()`.

## Tworzenie celu

Nowy cel może powstać tylko podczas autonomii, gdy:

- istnieje aktywna intencja Stage 32,
- finalny winner nie jest NOOP,
- Stage 31 przewiduje wystarczający relief jednej z motywacji,
- aktualna urgency tej motywacji jest wystarczająco wysoka.

Motywacja o największym przewidywanym reliefie staje się targetem celu.

## Progress

Przy starcie zapisywana jest początkowa urgency celu.

Progress jest liczony z realnego spadku urgency względem tego punktu:

```text
progress = (baseline urgency - current urgency) / baseline urgency
```

Progress jest monotoniczny w obrębie jednego celu, żeby krótkotrwały szum drives nie kasował już osiągniętego postępu.

Cel kończy się sukcesem, gdy progress osiągnie `goal_success_progress`.

## Kroki i porzucenie celu

Każde rzeczywiste wykonanie autonomicznego winnera jest rejestrowane jako kolejny krok:

```text
index
action
executed
success
detail
progress before
progress after
urgency after
```

Cel może zostać porzucony po:

- przekroczeniu maksymalnego wieku,
- przekroczeniu maksymalnej liczby kroków,
- zbyt wielu nieudanych wykonaniach,
- ręcznym wyłączeniu Stage 33.

## Persistence

Aktywny target, baseline urgency, progress, timestamps, liczniki oraz ostatnia sekwencja kroków są zapisywane w `state/brain_state.npz`.

Po restarcie Mucha może kontynuować niedokończony cel, o ile nie wygasł przez limit wieku.

## Ustawienia

```toml
goal_enabled = true
goal_signal_gain = 0.40
goal_min_relief = 0.01
goal_min_start_urgency = 0.20
goal_success_progress = 0.45
goal_max_age_seconds = 900.0
goal_max_steps = 12
goal_max_failed_steps = 3
```

## Dashboard

`/autonomy` pokazuje teraz dwa osobne poziomy pamięci:

```text
CURRENT INTENT
  konkretna akcja, np. VOICE_JOIN
  strength / age / sensory cue
  historia FORMED / REINFORCED / SWITCHED / CLEARED

ACTIVE GOAL
  target, np. SOCIAL
  baseline urgency -> current urgency
  progress
  liczba kroków / failed steps
  sekwencja wykonanych akcji
  historia FORMED / STEP / COMPLETED / ABANDONED
```

Dzięki temu można odróżnić krótkie „co chcę zrobić teraz” od dłuższego „jaką potrzebę próbuję rozwiązać”.
