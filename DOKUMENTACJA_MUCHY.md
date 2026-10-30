# 🪰 MUCHA — kompletna dokumentacja projektu

**Wersja dokumentacji:** 1.0 — 30 października 2026  
**Zakres:** oryginalna Mucha, repozytorium **Davidgracz/mucha**, etapy do **Stage 36 — Autobiographical Memory** włącznie.  
**Punkt odniesienia kodu:** gałąź **main**, commit **07d3f511ad** z 2 października 2026 (przed dodaniem niniejszego dokumentu).  
**System docelowy:** przede wszystkim Windows, lokalna instancja Muchy.  
**Oddzielne projekty:** *mucha-selfaware* i *mucha-chaser* nie są tym samym repozytorium. Mucha Chaser jest osobnym botem współpracującym z Muchą.

> **Cel dokumentu:** po kilku miesiącach bez pracy nad projektem można tu sprawdzić, jak Mucha działa, która część kodu odpowiada za daną funkcję, co wolno zmieniać, jakie pliki zachować, jak wykonać diagnostykę oraz jak kontynuować rozwój.  
> **Źródło prawdy:** dla dokładnej implementacji zawsze obowiązuje kod danej wersji. Ta dokumentacja opisuje sprawdzony stan z podanego commita. Konfiguracja z **config.local.toml** może nadpisywać wartości domyślne z **config.toml**.

---

## 1. Projekt w 60 sekund

**Mucha** to eksperymentalny bot Discord, którego zachowanie jest sterowane przez utrzymywany w czasie stan sztucznej sieci zbudowanej na **rzeczywistej topologii połączeń FAFB v783** mózgu dorosłej muszki owocowej (*Drosophila melanogaster*).

Docelowy, przygotowany zestaw danych zawiera około **139 255 neuronów** i **3 732 460 połączeń**. Nie są to parametry wyuczone przez model językowy, tylko przetworzona mapa połączeń biologicznych. Program nadaje tym połączeniom runtime, syntetyczne bodźce z Discorda, pamięć, uczenie i odczyty zachowań.

Mucha potrafi pisać, reagować emoji, mówić i słuchać na VC, wchodzić na kanały głosowe i zmieniać je, uczyć się ze zdarzeń, utrzymywać relacje, odczuwać symulowane potrzeby i zmęczenie, odpoczywać, budować cele i autobiograficzną pamięć wydarzeń.

**To nie jest:**
- dokładna emulacja biologicznego mózgu, zmysłów lub świadomości;
- gotowy model LLM/ChatGPT generujący wypowiedzi;
- system, w którym każda decyzja jest bezpośrednio narzucona przez ręcznie napisane warunki;
- gwarantowane odtworzenie rzeczywistych zachowań muszki.

**To jest:** prawdziwa struktura connectome + własny sztuczny model aktywacji i plastyczności + połączenia między Discordem, pamięcią, językiem, celami, wewnętrznymi stanami i neuronowymi odczytami akcji.

### Cztery filary projektu

| Filar | Co obejmuje | Najważniejszy kod |
|---|---|---|
| **Mózg** | FAFB, propagacja, sensory, attractory, reward, plastyczność, action readout | **mucha/brain.py** |
| **Środowisko** | Discord, tekst, voice, Chaser, akcje, pętle i ograniczenia techniczne | **mucha/discord_bot.py** |
| **Doświadczenie** | język, osoby, kanały, sceny, autobiografia, prediction/replay | **mucha/language.py**, **mucha/episodic.py** |
| **Obserwowalność** | dashboard, Neuro-map, decision trace, konfiguracja, testy | **mucha/web_ui.py**, **tests/smoke_test.py** |

---

## 2. Najważniejsza zasada architektury: FAFB ma naprawdę decydować

Poprawna, preferowana ścieżka decyzji:

~~~text
Discord / voice / otoczenie / upływ czasu
                │
                ▼
      sensory inputs + internal drives
                │
                ▼
   pamięć • reward prediction • foresight
   intent • goal • personality • autobiography
                │
                ▼
       słabe, kierunkowe sensory cues
                │
                ▼
      rzeczywista sieć FAFB + learned overlay
                │
                ▼
          propagation / brain.step()
                │
                ▼
        neuronalne action readouts
                │
                ▼
          action_competition()
                │
                ▼
      winner → executor Discord / NOOP
                │
                ▼
         real outcome / reward / punish
                │
                └──────► plasticity + memory
~~~

**Przykład właściwego wzorca:** przypomnienie dobrego spotkania na VC generuje słaby podpisany sensory cue skojarzony z VOICE_JOIN. Sygnał przechodzi przez FAFB; ostatecznie VOICE_JOIN wcale nie musi wygrać.

**Przykład niepożądanego wzorca:** wpisanie w Pythonie „jeśli SOCIAL > 0.8, to action = VOICE_JOIN” albo dodawanie do wyniku akcji bezpośredniego bonusu +0.5 bez propagacji. To odbiera sieci rolę arbitra.

**Wyjątek:** ograniczenia techniczne i procedury awaryjne mogą działać twardo (brak uprawnień Discord, blokada serwera/kanału, zagrożenie ze strony Chasera, krytyczne zmęczenie przed snem, minimum czasu pobytu, brak odbiornika voice). One nie reprezentują swobodnej preferencji bota.

**Uwaga:** istnieje także **Learned Action Policy**, historyczna warstwa umiarkowanego biasu readoutów ucząca się z reward/punish. Nie należy mylić jej z sensorycznym prowadzeniem Stage 30–36. Projekt ma oba mechanizmy; przy zmianach trzeba świadomie sprawdzać, który odpowiada za efekt.

---

## 3. Mapa repozytorium — który plik do czego służy

~~~text
mucha/
├─ bot.py                         punkt wejścia; .env, konfiguracja, uruchomienie
├─ config.toml                    domyślne parametry wszystkich modułów
├─ config.local.toml              lokalne nadpisania (nie commitować)
├─ .env                           token Discorda i hasło panelu (nie commitować)
├─ .env.example                   bezpieczny wzór .env
├─ requirements.txt               zależności Python / Discord / STT / TTS
├─ requirements-gpu.txt           zależności CUDA / CuPy
├─ README.md                      historyczny i szczegółowy opis etapów
├─ SOURCES.md                     źródła FlyWire / publikacje / discord.py
├─ DOKUMENTACJA_MUCHY.md          ten podręcznik
├─ START_MUCHA.bat                start lokalnej Muchy z walidacją plików
├─ START_BOTH.bat                 start Muchy i osobnego Mucha Chaser
├─ START_MUCHA_PUBLIC.bat         start + próba wystawienia publicznego dashboardu
├─ SHARE_DASHBOARD.bat            tymczasowy tunel tylko do dashboardu
├─ install_windows.bat            Python 3.12, .venv i dependencies
├─ install_gpu_windows.bat        dodatkowe dependencies GPU
├─ run_windows.bat                prosty punkt uruchomienia
├─ BACKUP_NOW.bat                 ręczny backup wyuczonej pamięci
├─ SETUP_DRIVE_BACKUP.bat         setup cyklicznego backupu
├─ mucha/
│  ├─ brain.py                    centralny silnik mózgu i decyzji
│  ├─ connectome.py               czytanie i walidacja przygotowanego FAFB
│  ├─ compute.py                  backend CPU / CUDA
│  ├─ config.py                   dataclasses i loader TOML
│  ├─ discord_bot.py              MuchaClient i obsługa zdarzeń Discord
│  ├─ episodic.py                 trwała pamięć scen, osób, miejsc, autobiografia
│  ├─ language.py                 generator i ucząca się pamięć języka
│  ├─ voice_sensory.py            sygnały z głosu, kto mówi/kiedy/overlap
│  ├─ web_ui.py                   dashboard HTML, API, auth, konfiguracja
│  └─ console_ui.py               terminalowy panel stanu
├─ tools/
│  ├─ prepare_connectome.py       budowa runtime FAFB z danych źródłowych
│  ├─ augment_connectome_map.py   dodatkowe metadane mapy/neuroanatomii
│  ├─ make_demo_connectome.py     sztuczny zestaw testowy, NIE produkcyjny
│  ├─ check_gpu.py                test CUDA
│  ├─ affinity_cli.py             narzędzia relacji
│  ├─ backup_learned_state.py     spójny ZIP stanu + manifest SHA256
│  ├─ backup_state_to_drive.ps1   kopia ZIP do folderu synchronizowanego
│  └─ install_drive_backup_task.ps1 harmonogram Windows
├─ tests/
│  └─ smoke_test.py               test integracyjny modułów na małym connectomie
├─ .github/workflows/smoke.yml    automatyczny compile + smoke po push/PR
├─ data/connectome/              przygotowane FAFB, pliki binarne ignorowane przez Git
└─ state/                        wyuczone doświadczenia (ignorowane przez Git)
~~~

**Jak znaleźć miejsce zmiany?**

| Chcesz zmienić… | Zacznij od… | Dodatkowo sprawdź… |
|---|---|---|
| neural action winner / sygnał | **brain.py** | **discord_bot.py** (wywołanie i feasibility) |
| wejście lub wyjście na VC | **discord_bot.py** | **brain.py**, **config.toml** [voice] |
| fatigue, TIRED, SLEEP | **discord_bot.py**: _sleep_tick, _sleep_pressure_snapshot | **brain.py**: internal drives |
| generowanie tekstu | **language.py** | **brain.py**: language_word_score, advance_language_word |
| STT | **discord_bot.py** i **voice_sensory.py** | **config.toml** [voice] |
| TTS | **discord_bot.py** | model Piper, FFmpeg |
| relacje z osobami | **discord_bot.py**, **episodic.py** | **language.py**, neural social memory |
| cele i intencje | **brain.py** | **/autonomy** i testy |
| trwałe wspomnienia | **episodic.py** | **discord_bot.py**: _record/_inject_autobiographical_* |
| nowe panele WWW | **web_ui.py** | **discord_bot.py**: snapshot API |
| default konfiguracji | **config.toml**, **config.py** | edytor konfiguracji w **web_ui.py** i walidator w **discord_bot.py** |
| baza FAFB | **connectome.py**, narzędzia **tools/** | manifest i osobny backup datasetu |
| test / regresję | **tests/smoke_test.py** | GitHub Actions |

---

## 4. Jak uruchomić Muchę — Windows

### 4.1 Wymagania

- Python **3.12** zalecany dla lokalnego instalatora.
- Git; opcjonalnie GitHub CLI.
- Token własnego bota Discord i odpowiednie uprawnienia oraz intents.
- FFmpeg dostępny w PATH do audio.
- Prawdziwy przygotowany dataset FAFB w katalogu **data/connectome/**.
- Dla STT/TTS: odpowiednie dependencies i modele (faster-whisper / Piper).
- Opcjonalnie karta NVIDIA z CUDA i CuPy.

**Pierwsza instalacja:**

~~~powershell
cd F:\mucha
.\install_windows.bat
~~~

Jeżeli to nowa kopia repozytorium:

~~~powershell
git clone https://github.com/Davidgracz/mucha.git F:\mucha
cd F:\mucha
.\install_windows.bat
~~~

Instalator tworzy **.venv**, instaluje **requirements.txt** i kopiuje wzór **.env**, ale **celowo nie generuje ani nie zastępuje prawdziwego connectome**.

Ustaw w lokalnym **.env**:

~~~dotenv
DISCORD_TOKEN=WKLEJ_WLASNY_TOKEN
MUCHA_DASHBOARD_PASSWORD=USTAW_SILNE_HASLO
~~~

Nie publikuj tego pliku ani jego zawartości.

**Start:**

~~~powershell
cd F:\mucha
.\START_MUCHA.bat
~~~

Alternatywnie:

~~~powershell
cd F:\mucha
.\.venv\Scripts\python.exe bot.py
~~~

**Mucha + Chaser:** uruchom **START_BOTH.bat**. Skrypt zakłada, że obok folderu Muchy istnieje folder **mucha-chaser** ze skryptem **START_CHASER.bat**. Chaser jest osobną aplikacją.

**Test integracyjny:**

~~~powershell
cd F:\mucha
.\.venv\Scripts\python.exe tests\smoke_test.py
~~~

Oczekiwane: **SMOKE TEST OK**. Test tworzy własną, tymczasową małą sieć; nie dowodzi, że wszystkie funkcje realnego Discorda zadziałają bez błędu w produkcji.

### 4.2 Aktualizacja bez utraty pamięci

1. Wykonaj backup wyuczonego stanu: **BACKUP_NOW.bat**.
2. Upewnij się, że nie edytujesz ręcznie tego samego pliku co działająca instancja.
3. Zatrzymaj Muchę.
4. Pobierz aktualny kod i uruchom test.
5. Uruchom ponownie Muchę.

~~~powershell
cd F:\mucha
git status
git pull
.\.venv\Scripts\python.exe tests\smoke_test.py
.\START_MUCHA.bat
~~~

Jeżeli **git status** pokazuje lokalne zmiany — nie wykonuj pochopnie **git reset --hard**; najpierw sprawdź, co jest zmienione.

**Git nie zawiera wyuczonej osobowości ani całego connectome.** Samo **git clone** nie odtwarza konkretnej Muchy.

### 4.3 GPU

~~~powershell
cd F:\mucha
.\install_gpu_windows.bat
.\.venv\Scripts\python.exe tools\check_gpu.py
~~~

Konfiguracja:

~~~toml
[brain]
backend = "auto"
gpu_device = 0
~~~

Tryby: **auto** (CUDA gdy dostępna), **cuda** (wymagana), **cpu** (NumPy/SciPy). Przy braku CuPy tryb auto może pracować na CPU. Aktywne urządzenie można sprawdzić w diagnostyce.

---

## 5. Jakie dane są naprawdę niezbędne

### 5.1 Bazowy connectome (anatomia)

Przygotowana lokalna struktura:

~~~text
data/connectome/
├─ matrix.npz
├─ root_ids.npy
├─ pools.npz
├─ manifest.json
└─ neuron_meta.npz        opcjonalne/anatomiczne metadane
~~~

- **matrix.npz** — skierowana rzadka macierz połączeń, importowana przez SciPy w formacie CSR.
- **root_ids.npy** — identyfikator biologiczny neuronu ↔ indeks w macierzy.
- **pools.npz** — podzbiory sensory / output / modulatory.
- **manifest.json** — metadane przygotowania datasetu.
- **neuron_meta.npz** — typy, regiony, koordynaty i inne adnotacje wykorzystywane np. przez Neuro-map.

Jeżeli trzeba odtworzyć cache z danych FlyWire/Codex:

~~~powershell
python tools\prepare_connectome.py --input raw_flywire --output data\connectome
python tools\augment_connectome_map.py --raw raw_flywire --connectome data\connectome
~~~

Źródła i licencje: **SOURCES.md**. Dane źródłowe i duże cache binarne są poza repozytorium. Plik **make_demo_connectome.py** służy wyłącznie do testów i tworzy sieć syntetyczną — **nie zamieniaj nim pełnego FAFB w instalacji, którą chcesz zachować**.

### 5.2 Wyuczona pamięć konkretnej instancji

~~~text
state/
├─ brain_state.npz
├─ language.sqlite3
├─ voice_episodes.sqlite3
├─ whisper/                  opcjonalny cache modelu STT
└─ drive_backup.log          lokalny log backupów, jeżeli skonfigurowano
~~~

| Plik | Zawartość | Co tracisz po usunięciu |
|---|---|---|
| **brain_state.npz** | aktywność, plastic bias, learned synapses, polityka akcji, homeostaza, affect, motywacje, intencje, goal, personality | dużą część neuronalnej historii i wyuczonego temperamentu |
| **language.sqlite3** | słowa, n-gramy, feedback, części relacji i pamięci językowej | wyuczony język, częstości i relacje przechowywane tam |
| **voice_episodes.sqlite3** | prognozy, outcome, historia osób/miejsc/scen, semantic memory, replay i Stage 36 | długoterminowe sceny, autobiografia i wyuczone modele sytuacji |

**Najważniejsze:** te trzy pliki stanowią kluczowy zestaw do przeniesienia wyuczonej Muchy. Osobno trzeba zachować **data/connectome/**, konfigurację lokalną, tokeny i potrzebne modele/głosy. **.env** nie jest częścią archiwum pamięci.

### 5.3 Backup bez uszkadzania baz SQLite

- **BACKUP_NOW.bat** — ręcznie.
- **SETUP_DRIVE_BACKUP.bat** — instalacja zadania Windows.
- **tools/backup_learned_state.py** — tworzy ZIP z manifestem i SHA256.
- **tools/backup_state_to_drive.ps1** — przenosi ZIP do synchronizowanego folderu; domyślna retencja **40** archiwów.
- **tools/install_drive_backup_task.ps1** — harmonogram co **6 godzin**.

Przykład ręcznego, niezależnego backupu:

~~~powershell
cd F:\mucha
.\.venv\Scripts\python.exe tools\backup_learned_state.py --output F:\mucha-backup.zip
~~~

Dla baz SQLite narzędzie używa **sqlite3.backup()**, zamiast kopiować żywą bazę zwykłym Copy-Item. W manifest wpisuje datę, SHA256, nazwy i wielkości plików oraz commit źródła.

**Przywracanie:** zatrzymaj Muchę, zweryfikuj archiwum, zachowaj kopię aktualnego **state/**, odtwórz do niego trzy pliki z ZIP i uruchom instancję na zgodnym kodzie / connectome. Gdy migrowano schemat bazy, najpierw sprawdź wersję kodu.

---

## 6. Cykl pracy procesu i główne pętle

**bot.py** czyta **.env**, ładuje **config.toml** z lokalnym nadpisaniem i tworzy **MuchaClient**.

**MuchaClient** (discord_bot.py) inicjuje connectome, FlyBrain, OnlineLanguage, VoiceEpisodicMemory, sensory bus i dashboard.

Główne zdarzenia / cykle:

| Zdarzenie lub pętla | Odpowiedzialność |
|---|---|
| **setup_hook / on_ready** | start usług, zadań, panelu, przygotowanie runtime |
| **on_message** | sensory text, attention, social feedback, language.learn, One Brain dla TEXT |
| **on_raw_reaction_add** | feedback reakcji i relacji |
| **on_voice_state_update** | użytkownicy VC, Chaser, wizyty i relacje |
| **idle_loop** | tyknięcie mózgu, drive/affect/motivation, cele, autonomia i save |
| **voice_loop** | niezależna kontrola techniczna/awaryjna VC; legacy voice tam, gdzie potrzebne |
| **voice_sensory_loop** | live sygnały PCM, poziom aktywności rozmowy, turn taking |
| **stt_segment_loop** | wycinanie i rozpoznawanie wypowiedzi |
| **tts_loop** | neuralny wybór / wypowiedzenie tekstu głosem |
| **random_audio_loop** | rzadki odruch odtwarzania audio |
| **presence_loop** | status Discord na podstawie mózgu |
| **console_loop** | terminalowa telemetria |

Zadania używają blokad asynchronicznych, zwłaszcza **_brain_lock**, aby nie wywoływać równoległych mutacji stanu mózgu w niekontrolowanej kolejności.

**Uwaga:** nie każda akcja na Discord przechodzi identyczną ścieżkę. Zwykły wybór akcji i techniczne procedury awaryjne mają różne role.

---

## 7. Mózg: connectome, neuron, synapsa i propagacja

### 7.1 Stan i obliczanie

Klasa **FlyBrain** ładuje **Connectome** i **ComputeBackend**.

Najważniejsze składniki:
- **state** — bieżąca aktywność neuronów, zachowana między zdarzeniami;
- **matrix** — bazowa anatomia FAFB;
- **synaptic overlay** — nauczone przyrosty/zmiany połączeń;
- **plastic_bias** — wolniejsza neuronalna pamięć podatności;
- **eligibility trace** — pamięć, które neurony/połączenia mogły odpowiadać za ostatni efekt;
- **neuromodulatory state** — globalne regulatory;
- **action output pools** — sztuczne przyporządkowanie aktywności do zachowań Discord.

W uproszczeniu:

~~~text
wejście sensoryczne
  + stan z poprzedniego kroku
  + propagacja przez bazowe FAFB
  + learned synaptic overlay
  + attractory / neuromodulacja / noise
                  ↓
             nowy state
                  ↓
         odczyty + plastyczność
~~~

Parametry **leak**, **propagation_gain**, **noise**, **steps_per_event** i **idle_steps** wpływają na dynamikę. Nie są biologicznie „świętymi” stałymi; są parametrami eksperymentalnego runtime'u.

### 7.2 Dlaczego neuron ma „akcję”?

FAFB nie zna Discorda. W projekcie grupy neuronów są przypisywane do readoutów. Bieżące akcje:

~~~text
speak • react • voice_join • voice_move
voice_leave • explore • stay (NOOP)
~~~

**action_scores()** czyta siłę aktywacji, a **action_competition()** porównuje dozwolone, technicznie wykonalne działania. **STAY/NOOP** jest rzeczywistym kandydatem — brak zewnętrznego działania też jest wynikiem.

**action_policy_scores()** to osobna wyuczona korekta readoutów, jeżeli włączona jest Learned Action Policy.

### 7.3 Sensory, attractory i action-guided input

Kluczowe metody w **brain.py**:

| Metoda | Po co istnieje |
|---|---|
| **inject()** | bazowe mapowanie tekstowego klucza bodźca na sensoryczną aktywację |
| **inject_text()** | bodźce z wiadomości |
| **inject_voice_snapshot()** | kto jest na VC i w jakim kontekście |
| **inject_voice_sensory_bus()** | bieżące cechy sceny mówionej |
| **inject_internal_state_cue()** | sprzęgnięcie potrzeby/stanu z istniejącym attractorem |
| **inject_action_guided_sensory()** | dodatni sensory cue związany z action readout |
| **inject_action_guided_signed_sensory()** | dodatni lub ujemny cue, np. z autobiografii |
| **step()** | krok propagacji i aktualizacja aktywności |
| **action_scores()** | odczyty akcji |
| **action_competition()** | finalna konkurencja akcji |

**Sensory cue** nie jest tożsame z wynikiem akcji. Sygnał może zaniknąć, przejść przez inne regiony, zostać przeważony przez pozostałą aktywność lub ograniczenia techniczne.

### 7.4 Neuromodulacja

Runtime utrzymuje symulowane poziomy:
- **dopamine** — przede wszystkim związana z plastycznością/nagrodą;
- **serotonin** — stabilność;
- **octopamine** — pobudzenie/arousal.

Sterowane parametrami **dopamine_plasticity_gain**, **serotonin_stability_gain**, **octopamine_arousal_gain**, **neuromodulator_smoothing**.

Nie są to odtworzone w pełni układy chemiczne biologicznego mózgu.

### 7.5 Plastyczność i zapominanie

**reward()** przypisuje dodatnią lub ujemną informację zwrotną, opcjonalnie do konkretnej akcji i zapamiętanego learning trace. **_reinforce_synapses()** modyfikuje sparse overlay, a **consolidate_and_forget()** odpowiada za czasowe wygaszanie i utrwalanie.

**CONSOLIDATED** = ślad lepiej utrwalony; **FADING** = ślad zanikający. Bazowego FAFB nie należy nadpisywać wyuczonym overlay.

Metody do diagnostyki:
- **capture_learning_trace()**,
- **learned_synapses_snapshot()**,
- **learning_diagnostics()**,
- **neuromodulator_diagnostics()**,
- **action_policy_diagnostics()**.

---

## 8. Język — skąd Mucha bierze zdania

**OnlineLanguage** w **mucha/language.py** używa własnych statystyk, a nie LLM.

Buduje rozkłady:
- znaków (unigram/bigram/trigram),
- słów (unigram/bigram/trigram),
- początków i zakończeń wypowiedzi,
- świeżości użycia,
- nagród i kar za konkretne słowa/przejścia.

**Uczy się** z tekstu Discord i transkrypcji STT, zgodnie z konfiguracją (np. blokada uczenia od botów). Rozróżnia gotowość do mówienia od samej chęci mówienia.

Przepływ:

~~~text
wiadomość / transkrypcja
        ↓
OnlineLanguage.learn()
        ↓
SQLite: wystąpienia znaków, słów, n-gramów
        ↓
One Brain wybiera, czy SPEAK jest możliwe/wygra
        ↓
OnlineLanguage.generate()
        ↓
kandydaci słów + kontekst + częstość + recent + reward
        ↓
brain.language_word_score() [opcjonalnie]
        ↓
ważone losowanie konkretnego słowa
        ↓
brain.advance_language_word() [feedback]
        ↓
gotowa wypowiedź
~~~

**Rola connectome:** może zmieniać prawdopodobieństwo wyboru słowa *spośród znanych kandydatów*. Nie wymyśla słów spoza słownika i nie dostaje w tle gotowej odpowiedzi od dużego modelu.

Ustawienia kluczowe:
- **min_chars_before_speaking**, **min_unique_chars_before_speaking** — warunki gotowości;
- **max_generated_chars** — limit wypowiedzi;
- **spontaneous_text**, **spontaneous_cooldown_seconds** — samodzielne pisanie;
- **hybrid_word_enabled**, **word_model_probability**, **word_max_tokens** — dobór słów;
- **connectome_word_control_enabled**, **connectome_word_control_min_vocab**, **connectome_word_control_strength**, **connectome_word_control_candidates** — wpływ neuronów;
- **connectome_word_feedback_enabled**, **connectome_word_feedback_steps**, **connectome_word_feedback_magnitude** — sprzężenie zwrotne.

Panel **/associations** pokazuje śledzenie poszczególnych słów, top kandydatów, ich wagi i losowanie.

---

## 9. Voice: VC, STT, TTS, homeostaza i Chaser

### 9.1 Dwie warstwy voice

**Percepcja i motywacja:** sygnały o osobach, dźwięku, nowości, znajomości miejsca, czasie pobytu, social need, fatigue, reward/opportunity i threat.

**Wykonanie i ograniczenia:** znalezienie kanału, uprawnienia Discord, dozwolony cel, AFK, blokady, minimum-dwell, timeouty API, reconnect i awaryjna ucieczka.

W trybie normalnej autonomii **Stage 24D / One Brain** odpowiada za zwykłe wybory JOIN/MOVE. Osobna **voice_loop** zachowuje nadzór techniczny i procedury awaryjne. Nie należy bezwiednie tworzyć drugiego konkurencyjnego arbitra dla tych samych działań.

### 9.2 STT

~~~text
Discord voice PCM
→ voice receive
→ VoiceSensoryBus / segmentacja ciszy
→ mono 16 kHz
→ faster-whisper
→ transkrypcja
→ language.learn() + sensory context
→ decyzje One Brain
~~~

**voice_sensory.py** mierzy m.in. kto mówił, przerwy, nakładanie się wypowiedzi, odpowiedzi po TTS i dynamikę rozmowy. **Nie podejmuje sam decyzji**. Surowe audio nie jest trwałą pamięcią autobiograficzną; pamięć przechowuje wybrane metadane, wyniki i tekst.

### 9.3 TTS

~~~text
One Brain SPEAK winner
→ OnlineLanguage.generate()
→ Piper / inny skonfigurowany lokalny silnik
→ źródło audio / FFmpeg
→ odtworzenie na bieżącym kanale Discord
→ outcome i nauka
~~~

Kluczowe ustawienia: **tts_enabled**, **tts_engine**, **tts_piper_model**, **tts_rate**, **tts_volume**, **tts_interval_seconds**, **tts_max_chars**, **stt_enabled**, **stt_model**, **stt_device**, **stt_compute_type**.

### 9.4 Kiedy wchodzi na VC?

Istotne wejścia:
- obecność ludzi na dozwolonych kanałach;
- **social drive** narastający po okresie poza voice;
- historia pozytywnych/negatywnych relacji;
- **reward opportunity**;
- novelty i niepewność;
- learned episode predictions;
- remembered people/place/social scene/voice dynamics;
- personality, motivation, intent, goal, autobiographical recall.

**Konfiguracja dostrojona po Stage 35** (stan referencyjny):

~~~toml
social_drive_start_seconds = 30
social_drive_ramp_seconds = 180
social_drive_max_magnitude = 1.60
social_drive_stay_punish = 0.045
reward_opportunity_min_strength = 0.55
~~~

Gdy jest poza VC, potrzeba społeczna zaczyna narastać po 30 sekundach i zwiększa się w ciągu kolejnych 180 sekund. To daje sieci więcej sensorycznego wsparcia dla JOIN, ale nie nakazuje JOIN.

**join_threshold** dotyczy przede wszystkim starszej ścieżki threshold/legacy. Przy **connectome_voice_control_enabled = true** liczy się neuronowy winner, feasibility i pozostałe warunki.

### 9.5 Zmiana kanału i wychodzenie

Są osobne pojęcia:
- **minimum_dwell_seconds** — minimalny pobyt przed zwykłym MOVE/LEAVE;
- **maximum_dwell_seconds** — granica, po której może pojawić się presja overstay;
- **social_fatigue** — zmęczenie interakcjami na danym VC;
- **habituation** — przyzwyczajenie do sceny;
- **exploration_drive** — potrzeba nowości;
- **threat** — zagrożenie/Chaser lub inne szczególne warunki;
- **circadian fatigue** — ogólne zmęczenie Muchy, odrębne od social fatigue.

Nie należy mylić **social_fatigue_level** z globalnym **fatigue**.

### 9.6 Chaser

Mucha Chaser jest osobnym botem. Mucha może wykrywać go po ID/sygnałach, budować suspicion → confirmed encounter → panic, reagować na jego obecność, krzyczeć, opuścić/zmienić kanał i oznaczać niebezpieczne miejsca.

Kluczowe metody: **_register_chaser_encounter()**, **_schedule_chaser_escape()**, **_escape_from_chaser()**, **_chaser_scream_loop()**. Te ścieżki bezpieczeństwa mogą omijać zwykłą konkurencję motorową.

---

## 10. Tired, sleep pressure i sen — obowiązujący opis po poprawce

**To ważna korekta względem starszych fragmentów README:** wcześniejszy opis mówił, że bez epizodów replay sen nie wystartuje i zawsze wymaga 900 s ciszy. Kod po poprawce z 2 października działa inaczej.

### 10.1 Zmęczenie

W **brain.py** globalny drive **fatigue** narasta z czasem; aktywność i social contact mogą przyspieszać przyrost. **TIRED** jest diagnozowane od domyślnego **0.65**. Zmęczenie jest składnikiem wewnętrznego stanu i motywu REST.

### 10.2 Progi

~~~text
fatigue < 0.65
    normalny AWAKE
fatigue >= 0.65
    TIRED i rosnące sleep pressure
    sensory cue do satiety i voice_leave
    wyjście z VC może ominąć minimum dwell
fatigue >= 0.98
    krytyczne sleep preparation
    wymagane opuszczenie VC, nawet gdy zwykła akcja nie wygrała
po opuszczeniu VC i wymaganej ciszy
    SLEEP
~~~

Domyślnie:

~~~toml
[brain]
circadian_tired_threshold = 0.65
circadian_fatigue_per_minute = 0.006
circadian_activity_fatigue_per_minute = 0.003
circadian_sleep_recovery_per_cycle = 0.16

[voice]
sleep_enabled = true
sleep_idle_seconds = 900
sleep_tired_idle_seconds = 30
sleep_leave_signal_gain = 1.75
sleep_force_disconnect_fatigue = 0.98
sleep_cycle_interval_seconds = 15
sleep_max_cycles = 8
~~~

**_sleep_pressure_snapshot()** oblicza presję i flagę critical; **_voice_decision()** wprowadza sleep-prep sensory cues, umożliwia LEAVE przy TIRED i egzekwuje krytyczne opuszczenie VC. **_sleep_tick()** sprawdza możliwość wejścia w sen.

### 10.3 Warunki i przebieg snu

- Sleep musi być włączony.
- Mucha nie może pozostawać połączona z VC ani być w aktywnym Chaser panic.
- Musi upłynąć wymagana cisza: standardowo 900 s, przy TIRED 30 s.
- Zwykły sen wymaga dostępnych replay candidates; **przy TIRED może rozpocząć się także bez kandydatów**.
- Każdy cykl jest regeneracyjny, nawet jeżeli replay nie znalazł wspomnienia.
- Jeśli są epizody, wykonywany jest także offline replay + consolidation.

Po zakończeniu pełnej sesji następuje krótkie **POST-SLEEP**. Nowe zdarzenie Discord może obudzić Muchę; aktywność wstrzymuje cykle.

**Praktyczna diagnostyka, gdy Mucha utknie na VC przy 100%:** sprawdzić w dashboardzie **fatigue**, **circadian_state**, **sleep_pressure**, **sleep_critical**, **sleep_leave_cue**, **voice-connected** oraz potencjalny błąd Discord disconnect. Upewnić się, że uruchomiona jest wersja po poprawce i nie ma lokalnych override.

---

## 11. Stage 24–36: mapa rozwoju i zależności

Wczesne funkcje (1–23) były budowane etapowo; najdokładniejsza dostępna historia ich rozwoju jest w commitach Git i historycznych sekcjach **README.md**. Poniżej obecna architektura 24–36.

| Etap | Nazwa | Co wprowadził |
|---|---|---|
| **24A** | Homeostatic Internal Drives | rosnące i malejące potrzeby, ich sensoryczny wpływ |
| **24B** | Autonomous Action Candidates | dopuszczalne akcje i ich techniczna wykonalność |
| **24C** | Predicted Reward | prognoza nagrody z confidence, sensory evidence |
| **24D** | Autonomous Loop | ciągła autonomia z neuronowym winnerem |
| **24E** | Autonomy Dashboard | kandydaci, wyniki, wykonanie i telemetria |
| **25** | One Brain | wspólny arbiter TEXT/REACT/TTS/AUTONOMY |
| **26** | Circadian Fatigue / Sleep Pressure | fatigue, TIRED/SLEEP, post-sleep |
| **27** | Persistent Affective State | utrwalany stan afektywny |
| **28** | Durable Person Relationship Memory | trwałe modele poznanych osób |
| **29** | Durable Channel / Place Memory | trwałe modele miejsc/kanałów |
| **30** | Natural Drives and Motivation | SOCIAL/NOVELTY/SAFETY/REST, frustration/satiation |
| **31** | Internal Counterfactual Foresight | hipotetyczne następstwa kandydatów, bez mutowania live state |
| **32** | Persistent Intent / Goal Holding | krótkoterminowy zamiar neuronalnego winnera |
| **33** | Multi-step Motivational Goals | utrzymanie potrzeby-celu między kolejnymi decyzjami |
| **34** | Neuro-map 2.0 | głęboki debugger: neuron, region, przepływ, replay, atrybucja akcji |
| **35** | Emergent Personality | wolno uczony temperament pięciu cech |
| **36** | Autobiographical Memory | trwałe wydarzenia własnej historii i sensoryczny recall |

**Numeracja historyczna:** wcześniejsza roadmapa przewidywała osobny krok „Dlaczego to zrobiła?”, lecz został on zaimplementowany w rozbudowanym Neuro-map 2.0 (34). W rezultacie personality figuruje jako 35, a autobiografia jako 36. Nie należy na tej podstawie tworzyć drugiej funkcji explainability.

---

## 12. Stage 27–31: stany, motywacja i przewidywanie

### 12.1 Persistent Affective State (27)

To utrzymywany stan emocjonalno-afektywny, różny od bieżącego pojedynczego reward. Rozpoznasz go po **tick_affective_state()**, **affective_state_diagnostics()**, konfiguracji **affective_state_smoothing**, **affective_state_feedback_gain**, **affective_state_reward_gain**.

### 12.2 People, Places i Social Scenes (28–29 i późniejsze rozszerzenia)

Pamięci nie są tylko licznikiem:
- model osoby: familiarity, valence, confidence, historia zachowania i interakcji;
- model kanału/miejsca: wizyty, dynamika, feedback, preferencje;
- model sytuacji społecznej: kombinacja osób/sceny, warunki i outcome;
- model voice dynamics: kto mówi, overlap, turn-taking, kontakt/reakcja.

**episodic.py** przechowuje historię i wyprowadza profil; **discord_bot.py** buduje konkretny kontekst; **brain.py** koduje profil jako sensory cue. Stan nie powinien decydować przez ręczny ranking w miejscu finalnego winnera.

### 12.3 Natural Motivation (30)

Cztery główne motywy:

- **SOCIAL** — potrzeba kontaktu;
- **NOVELTY** — poznawanie nowych bodźców;
- **SAFETY** — unikanie zagrożeń;
- **REST** — odpoczynek.

Każdy ma **urgency**, **frustration** i **satiation**. Frustration zwiększa presję przy długo niezaspokojonej potrzebie, satiation ją chwilowo obniża po zaspokojeniu. To nie jest sztywna kolejność akcji.

Metody: **tick_motivation_state()**, **motivation_state_diagnostics()**, **_motivation_pressures_for_values()**.

### 12.4 Foresight (31)

**simulate_action_outcome()** i **simulate_candidate_outcomes()** hipotetycznie przewidują wpływ każdego kandydata na drive/motivation/reward/risk. Symulacja nie zmienia live state.

~~~text
candidate → predicted relief / risk / confidence
          → sensory cue
          → FAFB
          → final action competition
~~~

**Foresight winner** to *ranking modelu kontrfaktycznego*, nie finalna akcja. Te dwie wartości mogą się różnić — i jest to poprawne.

---

## 13. Stage 32–33: intent i cele

### 13.1 Persistent Intent (32)

Krótka pamięć ostatniego neuronalnego zamiaru. Nowy input może go wzmocnić, zmienić lub wygasić.

Domyślnie:

~~~toml
intention_enabled = true
intention_half_life_seconds = 90.0
intention_max_age_seconds = 300.0
intention_signal_gain = 0.45
intention_reinforcement_gain = 0.30
intention_switch_margin = 0.10
intention_min_evidence = 0.10
intention_outcome_gain = 0.75
~~~

Kluczowe: **intention_state_diagnostics()**, **_update_intention_from_decision()**, **register_intention_outcome()**.

Intencja **nie** wykonuje przyszłej akcji samodzielnie. Po NOOP może trwać i wrócić jako słabnące sensory evidence.

### 13.2 Multi-step Motivational Goals (33)

Goal reprezentuje **potrzebę, którą Mucha chce zmniejszyć**, a nie listę skryptowanych akcji.

Przykład:

~~~text
GOAL: SOCIAL
  • VOICE_JOIN
  • NOOP
  • SPEAK
  • VOICE_MOVE
  • goal complete / abandoned
~~~

Kolejne kroki mogą się zmieniać, bo każdy musi ponownie wygrać w FAFB. Progress wiąże się z redukcją urgency, a nie tylko liczbą wykonanych akcji.

~~~toml
goal_enabled = true
goal_signal_gain = 0.40
goal_min_relief = 0.01
goal_min_start_urgency = 0.20
goal_success_progress = 0.45
goal_max_age_seconds = 900.0
goal_max_steps = 12
goal_max_failed_steps = 3
~~~

Metody: **goal_state_diagnostics()**, **_update_goal_from_decision()**, **_goal_candidate_signal()**, **register_goal_step()**.

---

## 14. Stage 35: osobowość wyłaniająca się z doświadczeń

Pięć trwałych cech:

| Cecha | Skojarzone doświadczenia | Przykładowy sensory expression |
|---|---|---|
| **sociability** | SPEAK, REACT, VOICE_JOIN oraz outcome | SOCIAL_NEED / AROUSAL |
| **curiosity** | EXPLORE, VOICE_MOVE oraz outcome | CURIOSITY / AROUSAL |
| **caution** | przede wszystkim negatywne outcome | STRESS / AROUSAL |
| **persistence** | wykonanie i ukończenie/porzucenie goal | AROUSAL / CURIOSITY / SOCIAL_NEED |
| **expressiveness** | SPEAK, REACT oraz outcome | AROUSAL / SOCIAL_NEED |

Wszystkie startują neutralnie przy **0.50**. System uczy je powolnym EMA i mierzy confidence według liczby obserwacji. Sygnał neuronalny jest mały, ważony confidence i wprowadzany przez istniejące sensoryczne attractory, a nie poprzez bezpośrednie zwiększanie wyniku akcji.

~~~toml
personality_enabled = true
personality_learning_rate = 0.03
personality_signal_gain = 0.18
personality_min_observations = 6
~~~

Metody: **_observe_personality()**, **register_personality_action()**, **register_personality_outcome()**, **personality_state_diagnostics()**, **inject_personality_context()**.

Persistence: **brain_state.npz**, pola **personality_values** oraz **personality_observations**. Panel: **/autonomy**, sekcja Emergent Personality.

---

## 15. Stage 36: autobiografia — co konkretnie przeżyła Mucha

**Stan: wdrożone na main, commit 07d3f511ad, smoke test zakończony sukcesem.**

Do wcześniejszych pamięci statystycznych dochodzi **zdarzenie autobiograficzne**. To zapis konkretnego faktu z życia instancji.

Wspomnienie zawiera m.in.:

~~~text
timestamp / kind / decision_context
guild_id + guild_name
channel_id + channel_name
user_ids + user_names
action / executed? [external_effect] / success
detail
predicted_reward / actual_reward / prediction_error
salience
state snapshot:
  goal / intention / affective
  motivation / personality / circadian
~~~

**Zapis:** **mucha/episodic.py** — **record_autobiographical_event()** i tabela **autobiographical_events** w **state/voice_episodes.sqlite3**. Wywołanie w **discord_bot.py** — **_record_autobiographical_event()**. Neutralne, mało znaczące NOOP bez zewnętrznego efektu i bez silnych powodów mogą być pomijane.

### 15.1 Salience

Znaczenie wspomnienia zależy m.in. od:
- rzeczywistego efektu akcji;
- sukcesu;
- natężenia przewidywanej i faktycznej nagrody;
- prediction error;
- aktywnego goal i intent;
- intensywności affect.

Przykładowy wpis (schematyczny, nie z danych produkcyjnych):

~~~text
czas: 2026-10-02 13:20
guild: ASG
channel: Ogólny
osoby: [użytkownik 1, użytkownik 2]
action: VOICE_JOIN
goal: SOCIAL
intent: VOICE_JOIN
fatigue: 0.31
actual reward: +0.18
prediction error: +0.07
salience: 0.54
~~~

### 15.2 Recall podobnego kontekstu

**autobiographical_recall()** porównuje:
- rodzaj zdarzenia;
- serwer;
- kanał;
- nakładających się uczestników;
- zgodność celu i intencji;
- salience i recency.

Zwraca wspomnienia i ich agregowany **action_signals**. **_inject_autobiographical_recall()** przekształca te sygnały w **signed action-guided sensory input**, po czym zwykły connectome wykonuje propagację.

~~~text
teraz podobna sytuacja na VC
  → znalezione stare wydarzenia
  → similarity × recency × salience × outcome
  → słaby signed cue dla zapamiętanej akcji
  → FAFB
  → normalny winner
~~~

Konfiguracja:

~~~toml
autobiographical_memory_enabled = true
autobiographical_recall_magnitude = 0.55
autobiographical_min_salience = 0.10
autobiographical_recall_limit = 6
~~~

Panel: **/autonomy**, **Autobiographical Memory / Stage 36**. Pokazuje wspomnienia, ich kontekst, recall i wpływ na sensory.

**Istotne ograniczenie:** przypomnienie pozytywnego wydarzenia nie oznacza, że bot „wie”, co użytkownik czuje obecnie. To dopasowanie zapisanych danych i podpisanego outcome. Salience nie jest biologiczną miarą ważności, tylko heurystyką projektu.

---

## 16. Neuro-map 2.0 i diagnostyka „dlaczego”

**/neuromap** (Stage 34) to narzędzie badawcze, a nie animacja udająca biologiczną aktywność.

Dostępne m.in.:
- XY/XZ/YZ i oznaczone neuropile;
- aktywne neurony, nazwy/identyfikatory, rola, region;
- zaznaczenie neuronu i powiązań wejściowych/wyjściowych;
- strukturalne current IN/OUT connections;
- learned synapses, utrwalenie i attractory;
- signal flow po konkretnych klatkach i replay historycznych ramek;
- region inspector i przepływ IN/OUT/INTERNAL;
- supporting/opposing neurons dla winnera;
- winner, runner-up, margin;
- timeline i rekonstrukcja śladu decyzji.

**Ważne ograniczenie explainability:** signed neuron→action contribution w panelu to bieżący odczyt bezpośredniego presynaptycznego wpływu wybranych neuronów na output pool danej akcji. **Nie jest to dowód pełnej przyczynowości biologicznej**, ani rekonstrukcja każdego synaptycznego wkładu całego 139k connectome. Nie należy opisywać go jako „prawdziwych myśli Muchy”.

Kluczowe funkcje: **neuro_map_snapshot()**, **signal_flow_snapshot()**, **action_path_snapshot()**, **_capture_signal_flow_tick()**, **learned_synapses_snapshot()**. Rozmiar live snapshotu mapy wynosi do około **320 widocznych neuronów** w wywołaniu dashboardu, a nie wszystkie neurony na ekranie naraz.

---

## 17. Pamięć, reward, prediction error i replay — jak się łączą

### 17.1 Różne rodzaje pamięci

| Pamięć | Czas/skala | Gdzie |
|---|---|---|
| aktywacja sieci | krótkotrwała, ciągła | brain.state / brain_state.npz |
| eligibility trace | niedawna przyczynowość uczenia | brain |
| plastic bias / learned synapses | trwała neuronalna | brain_state.npz |
| action policy | wyuczona preferencja readoutu | brain_state.npz |
| attention / working memory | krótki kontekst rozmowy | discord_bot.py |
| język / słowa / affinity | długotrwała językowo-społeczna | language.sqlite3 |
| voice episode prediction | wynik dawnych scen | voice_episodes.sqlite3 |
| people/place/social scene/dynamics | uogólnienia doświadczeń | voice_episodes.sqlite3 |
| autobiography | konkretne ważne zdarzenia + snapshot stanu | autobiographical_events w voice_episodes.sqlite3 |
| intent | krótki zamiar | brain_state.npz |
| goal | wielotickowy cel motywacyjny | brain_state.npz |
| personality | powolne cechy | brain_state.npz |

### 17.2 Uczenie po zdarzeniu

~~~text
neural winner
   ↓
capture_learning_trace()
   ↓
Discord executor / feedback użytkownika / upływ czasu
   ↓
realny outcome ±
   ↓
reward() + attribution
   ↓
plasticity / policy / prediction error / model osoby/miejsca
   ↓
późniejsze memory replay / autobiographical recall
~~~

**Prediction error** to różnica między przewidywanym i obserwowanym wynikiem; pomaga poprawiać model, ale nie powinien automatycznie stawać się finalnym wyborem akcji.

**Replay** reaktywuje wybrane realnie zapisane doświadczenia. **Semantic rehearsal** utrwala uogólnienia bez udawania nowego kontaktu z człowiekiem.

---

## 18. Dashboard WWW i API — mapa widoków

Domyślny adres lokalny: **http://127.0.0.1:8765**.

| Ścieżka | Zawartość / zastosowanie |
|---|---|
| **/** | ogólny Control Center: system, proces, neuro/voice |
| **/details** | pełne diagnostyki, sleep, internal state, rewards |
| **/autonomy** | One Brain, Stage 24–36, candidates, intent, goal, personality, autobiography |
| **/connectome** | graf/connectome, neurons, action circuits |
| **/neuromap** | anatomiczna mapa i szczegółowy signal flow |
| **/associations** | Language Brain, powstawanie słów, ślady/association map |
| **/affinity** | relacje, social memory, osoby |
| **/config** | lokalny edytor konfiguracji |
| **/brain** | dodatkowy/legacy widok mózgu |
| **/login**, **/logout** | autoryzacja |
| **/public** | publiczny read-only overview |
| **/public/connectome** | publiczny read-only connectome |
| **/public/neuromap** | publiczny read-only Neuro-map |
| **/public/associations** | publiczny read-only Language Brain |
| **/health** | stan HTTP dashboardu |

**Endpointy prywatne** (wymagają sesji przy włączonym auth):
- **GET /api/state** — pełne bieżące dane instancji;
- **GET /api/overview** i **GET /api/system** — status systemu;
- **GET /api/connectome**, **/api/neuromap**, **/api/associations** — odpowiednie snapshoty;
- **GET /api/config** oraz **POST /api/config** — odczyt i edycja konfiguracji.

**Endpointy publiczne:**
- **/api/public/state**,
- **/api/public/connectome**,
- **/api/public/neuromap**,
- **/api/public/associations**.

Publiczny snapshot jest sanitizowany przez dashboard — nie zastępuje autoryzowanego API. Publiczne wystawienie powinno odbywać się wyłącznie świadomie.

### 18.1 Jak znaleźć stan, który widzisz w przeglądarce

~~~text
mucha/web_ui.py        HTML, CSS, JS, route i fetch
        ↓
WebDashboard._state / _neuromap_state / ...
        ↓
mucha/discord_bot.py   budowanie snapshotu
        ↓
brain / language / episodic / sensory
~~~

Jeśli w panelu pole jest puste, sprawdzaj **obie strony kontraktu**: pole JSON z bota oraz identyfikator i renderer JS w **web_ui.py**.

### 18.2 Konfiguracja z dashboardu

**/config** wystawia formy połączone z polami w **config.py**, obsługą i walidacją w **discord_bot.py**. Lokalny override zapisuje się do **config.local.toml**. Ręczna zmiana **config.toml** może nie odnieść skutku, jeśli dana wartość istnieje już w lokalnym override.

Dla nowego parametru zwykle zmieniaj równolegle:
1. domyślne pole w **config.py**;
2. wartość w **config.toml**;
3. listę edytowalnych pól i walidację w **discord_bot.py**;
4. opis, typ i zakres inputu w **web_ui.py**;
5. test w **tests/smoke_test.py**;
6. dokumentację.

---

## 19. Konfiguracja — przewodnik po sekcjach

**Podstawowa zasada:** **config.toml** = defaults; **config.local.toml** = lokalne nadpisania; **.env** = sekretne zmienne środowiskowe. Nie przepisuj tokenów do TOML.

### 19.1 [brain] — fizyka i wyuczone zachowanie mózgu

| Grupa pól | Za co odpowiada |
|---|---|
| **connectome_dir, state_file, seed** | źródło FAFB, plik wyuczonego stanu, seed |
| **leak, propagation_gain, noise, steps_per_event, idle_steps** | pamiętanie aktywności, propagacja, szum i liczba kroków |
| **plasticity_lr, plasticity_decay, max_bias** | neuronalny bias i uczenie |
| **synaptic_plasticity_*** | sparse learned synapses: włączenie, LR, maks. delta, limit edges |
| **consolidation_*, bias_forgetting_*, synaptic_forgetting_*, synaptic_prune_*** | wzmacnianie, ochrona i zapominanie |
| **neuromodulation_*, dopamine_*, serotonin_*, octopamine_*** | modulatory globalne |
| **internal_states_*, internal_state_*** | attractory i przełożenie stanów na sieć |
| **internal_drives_*, internal_drive_*** | homeostatyczne potrzeby |
| **circadian_*** | fatigue, próg TIRED, regeneracja, post-sleep |
| **affective_state_*** | utrzymywany afekt |
| **motivation_*** | SOCIAL/NOVELTY/SAFETY/REST, urgency/frustration/satiation |
| **foresight_*** | kontrfaktyczna symulacja i jej sensoryczny wpływ |
| **intention_*** | formowanie, utrwalanie i wygaszanie intent |
| **goal_*** | wieloetapowe motywacyjne cele |
| **personality_*** | tempo, confidence i wpływ temperamentu |
| **action_policy_*** | nauka bezpośredniego biasu readoutów |
| **one_brain_noop_reafference_*** | dodatkowy sensory pass przy NOOP z dostatecznym unmet need |
| **backend, gpu_device** | CPU/GPU |

**Nie zmieniaj losowo leak/propagation/noise i strength jednocześnie:** trudno wtedy odróżnić regresję od zmiany charakteru sieci.

### 19.2 [language]

| Grupa | Znaczenie |
|---|---|
| **database** | plik SQLite |
| **min_chars_before_speaking, min_unique_chars_before_speaking** | minimum nauczonego materiału |
| **max_generated_chars** | limit wypowiedzi |
| **spontaneous_text, reply_cooldown_seconds, spontaneous_cooldown_seconds** | autopisanie i pauzy |
| **learn_from_bots** | czy cudze boty uczą słownika |
| **hybrid_word_*, word_model_*, word_max_tokens** | mieszanie modeli char/word |
| **word_recent_*, word_frequency_*, word_reward_*** | częstotliwość, świeżość i nagradzanie |
| **char_frequency_*, char_arousal_*, word_arousal_*** | wpływ arousal na rozkłady |
| **connectome_word_control_*** | neuralny wybór słów |
| **connectome_word_feedback_*** | zwrotny bodziec wybranego słowa |

### 19.3 [voice] — parametry podgrup

| Parametry | Rola |
|---|---|
| **enabled, poll_seconds, connectome_voice_control_enabled** | uruchomienie voice i neuroarbiter |
| **voice_sensory_*** | live pomiary rozmowy, interwały, siła |
| **social_drive_*** | chęć kontaktu po okresie poza VC |
| **social_join_reward** | feedback za kontakt |
| **reward_opportunity_*** | niepewna szansa nagrody i kara za ignorowanie |
| **motivation_propagation_steps** | kroki sieci przy motywacyjnym kontekście voice |
| **homeostasis_*, social_fatigue_*** | zmęczenie towarzyskie podczas VC |
| **habituation_*** | przyzwyczajenie do sceny |
| **exploration_drive_*** | zainteresowanie nowymi kanałami |
| **episodic_prediction_*, prediction_*** | nauczone prognozy i błędy |
| **memory_replay_*** | konsolidacja w ciszy |
| **sleep_*** | warunki snu i parametry replay w czasie snu |
| **autobiographical_*** | zapis, salience/recall Stage 36 |
| **semantic_memory_*** | pamięć semantyczna z epizodów |
| **uncertainty_*, information_gain_*** | szukanie nieznanych sytuacji i reward za informację |
| **minimum_dwell_seconds, maximum_dwell_seconds** | twarde czasy pobytu |
| **overstay_*, threat_*, deadly_*** | niebezpieczeństwo i awaryjne zachowania |
| **exploration_memory_*, exploration_novelty_*, exploration_recent_*, exploration_temperature** | eksploracja kanałów |
| **random_audio_*** | rzadki klip audio |
| **ffmpeg_executable** | ścieżka do narzędzia audio |
| **tts_*** | model, silnik, tempo, głośność i limity syntezy |
| **stt_*** | Whisper i segmentacja rozpoznawania |
| **chaser_*** | identyfikacja predatora, panic, ucieczka, scream |
| **move_threshold, join_threshold, leave_threshold, move_margin** | legacy progi i marginesy |
| **blocked_voice_channel_ids, blocked_voice_guild_ids** | twarde blokady |
| **include_empty_channels, exclude_afk_channel** | filtrowanie listy kanałów |

### 19.4 [behavior]

| Grupa | Rola |
|---|---|
| **idle_tick_seconds** | takt autonomii |
| **attention_*, working_memory_*** | utrzymywanie niedawnego kontekstu |
| **speak_threshold, reaction_threshold, reaction_cooldown_seconds** | zachowania tekstowe i legacy progi |
| **connectome_behavior_competition_enabled** | neuralny arbiter zachowania |
| **one_brain_*** | centralny arbiter i sensoryczny predicted reward |
| **autonomous_loop_enabled, autonomous_*** | Stage 24D i zgodność ze starszą konfiguracją |
| **save_every_seconds** | częstotliwość zapisu brain state |
| **social_learning_*, neural_social_*** | uczenie relacji |
| **person_model_*, channel_model_*, social_scene_model_*, voice_dynamics_*** | modele kontekstu |
| **word_reuse_*, phrase_reuse_*, direct_reply_*, self_repeat_penalty** | językowo-społeczny feedback |
| **user_affinity_*, *_affinity_step** | zmiany relacji |
| **social_window_*, negative_streak_*, positive_contact_*** | seria i cooldown kontaktów |
| **familiar_affinity_threshold, user_avoid_threshold** | rozpoznanie znajomych / niechętnych |
| **ignore_disliked_users_text, avoid_disliked_users_on_voice** | reguły unikania |

### 19.5 [discord], [console_ui], [web_ui]

- **[discord] command_prefix**, **blocked_text_channel_ids** — komendy i blokady tekstu.
- **[console_ui] mode**, **refresh_seconds**, **top_neurons** — lokalny terminal.
- **[web_ui] enabled**, **host**, **port**, **refresh_ms**, **history_points** — serwer dashboardu.
- **auth_enabled**, **auth_username**, **auth_password_env**, **session_hours** — dostęp administratora.
- **public_readonly_enabled** — widoki publiczne.
- **chaser_status_file**, **service_unit** — telemetria/integracja z procesami i wdrożeniami.

**Pełna, zawsze aktualna lista pól i wartości bazowych znajduje się w pliku config.toml.** Ten plik jest referencją maszynową i częścią dokumentacji; nie dubluj go w kilku miejscach bez potrzeby.

---

## 20. Przewodnik po typowych problemach

### Mucha nie startuje

Sprawdź:
1. **.env** oraz zmienną **DISCORD_TOKEN**;
2. obecność **.venv/Scripts/python.exe**;
3. komplet **data/connectome/**, szczególnie **matrix.npz**, **root_ids.npy**, **pools.npz**, **manifest.json**;
4. działanie FFmpeg i wymaganych bibliotek;
5. komunikat błędu w konsoli.

### Dashboard nie odpowiada

~~~powershell
Invoke-WebRequest http://127.0.0.1:8765/health
~~~

Sprawdź, czy port 8765 nie jest zajęty i czy w **config.local.toml** nie zmieniono host/port. Trzymaj prywatny panel na **127.0.0.1**, nie przekierowuj portu routera bez autoryzacji.

### Mucha nie mówi

Sprawdź:
- language readiness i liczbę poznanych znaków/słów;
- **speak** w One Brain, feasibility i cooldown;
- zablokowane kanały i relacje;
- generację w **/associations**;
- dla VC: Piper, model głosu, FFmpeg, playback, STT/TTS status.

### Mucha nie wchodzi na VC

Sprawdź **/autonomy** i **/details**:
- czy **VOICE_JOIN** jest feasible;
- czy użytkownicy są na dozwolonym VC;
- **social_drive_level** i minuty poza VC;
- winner vs runner-up i margin;
- czy Chaser/panic, blokada guild albo social avoid nie wyłączają celu;
- czy **config.local.toml** nie nadpisuje defaultów.

**Nie podbijaj od razu trzech różnych wzmocnień.** Najpierw diagnostyka, później jedna zmiana na raz i ponowny smoke.

### Mucha nie opuszcza VC przy TIRED

Sprawdź kod po Stage 35 sleep fix, parametry **circadian_tired_threshold**, **sleep_leave_signal_gain**, **sleep_force_disconnect_fatigue**, w dashboardzie **sleep_critical** i komunikat disconnect. Pamiętaj: **TIRED = stan 65%**, a wymuszone krytyczne opuszczenie VC zaczyna się dopiero przy 98% (wartości bazowe).

### Mucha stale wybiera NOOP

Sprawdź:
- czy jakakolwiek alternatywa jest feasible;
- wynik i score akcji;
- reward prediction, foresight, intent, goal, personality i autobiographical cues;
- aktywność / zamieranie sieci i **backend**;
- ustawienia **one_brain_noop_reafference_***;
- trace w Neuro-map.

### Pamięć znika po restarcie

Sprawdź ścieżki **brain.state_file**, **language.database**, **voice.episodic_database**, zapis (logi), uprawnienia do folderu oraz **config.local.toml**. Zobacz czy nie startujesz w innej kopii repozytorium albo z pustym **state/**.

### GPU nie działa

Uruchom **tools/check_gpu.py**, sprawdź sterowniki NVIDIA, wersję CUDA/CuPy i **backend**. Bez GPU backend auto może przejść na CPU — to nie oznacza awarii connectome.

### W dashboardzie brakuje nowej funkcji

Zweryfikuj aktualny commit **git rev-parse --short HEAD**, restart procesu, ewentualny cache przeglądarki, pola JSON i renderer JS. Stary proces może nadal podawać poprzednią wersję stron.

---

## 21. Jak bezpiecznie rozwijać projekt

### 21.1 Zasada jednej odpowiedzialności

Nie wstawiaj całej logiki w jeden callback Discorda. Rozdziel:
- **obserwację** → sensory/context;
- **pamięć i przewidywanie** → episode/prediction;
- **neuralne wskazówki** → inject;
- **propagację i konkurencję** → FlyBrain;
- **wykonanie** → Discord executor;
- **outcome** → reward/memory;
- **diagnostykę** → snapshot/dashboard.

### 21.2 Checklista nowej funkcji

1. Nazwij problem i wskaż wejścia/wyjścia.
2. Sprawdź, czy istniejący moduł już nie robi podobnej rzeczy.
3. Zachowaj FAFB jako finalny arbiter wszędzie, gdzie mowa o swobodnym zachowaniu.
4. Dodaj sensory/context zamiast hard-coded winnerów.
5. Jeśli mechanizm się uczy — zaplanuj persistence i kompatybilność starego stanu.
6. Jeśli ma konfigurację — dodaj komplet pól do **config.py**, **config.toml**, edytora i walidatora.
7. Jeśli jest istotny dla zachowania — dodaj live telemetry z nazwą/metodą/wartością.
8. Dodaj smoke z przypadkiem normalnym, brzegowym i save/reload, jeśli dotyczy.
9. Zaktualizuj dokumentację i commituj zmianę.
10. Po stronie lokalnej wykonaj **git pull**, smoke i restart.

### 21.3 Jak opisywać commit

Przykładowe czytelne komunikaty:
- **Add …** — nowy mechanizm;
- **Fix …** — naprawa;
- **Cover …** — testy;
- **Expose …** — API/telemetria;
- **Document …** — dokumentacja.

### 21.4 Testy i ich granice

GitHub workflow **.github/workflows/smoke.yml**:
- checkout;
- Python 3.13 na Ubuntu;
- minimalne zależności smoke;
- **py_compile**;
- **python tests/smoke_test.py**.

To gwarantuje określony zakres lokalnych testów, ale **nie zastępuje testu na prawdziwym Discordzie, urządzeniu audio, dużym FAFB i Windowsie**.

---

## 22. Bezpieczeństwo i prywatność

- **DISCORD_TOKEN** i **MUCHA_DASHBOARD_PASSWORD** trzymać w **.env**, nigdy w Git.
- **config.local.toml** może zawierać prywatne dostosowania — jest ignorowany przez Git.
- **state/** przechowuje historię interakcji, identyfikatory użytkowników, relacje i kontekst autobiograficzny. **Traktuj backup jak prywatne dane**, nie publikuj go.
- **data/connectome/** to duże dane techniczne; warunki redystrybucji danych źródłowych sprawdzić osobno.
- Publiczny dashboard to **read-only**, ale nadal może eksponować wybrane informacje o działaniu instancji. Przed udostępnieniem sprawdź zawartość.
- Najbezpieczniej trzymać Web UI na **127.0.0.1:8765** i wystawiać wyłącznie kontrolowany tunel publiczny, gdy jest to potrzebne.
- **SHARE_DASHBOARD.bat** używa Cloudflare Quick Tunnel; jest przeznaczony przede wszystkim do testowego udostępnienia znajomym.
- Nie umieszczaj prywatnego backupu ani danych użytkowników w repozytorium.

---

## 23. Co można usprawnić w przyszłości (nie jest jeszcze automatycznie wdrożone)

To **kierunki rozwoju**, nie lista istniejących funkcji:

1. **Retrospektywna atrybucja 1:1** — zapisywać pełniejszy neuronalny ślad i signed contribution per historyczny frame, nie tylko bieżące przybliżenie.
2. **Autobiografia z lepszą konsolidacją** — odkrywać długie wzorce i ważne „okresy”, zamiast wyłącznie pojedynczych epizodów.
3. **Uczenie strategii osiągania celu** — wyciągać z dawnych zakończonych/niezakończonych goals słabe sensoryczne priory.
4. **Testy długotrwałe** — sen, wejście/wyjście VC, goal consistency, regeneracja i recall po wielu godzinach na pełnym FAFB.
5. **Zintegrowana telemetria przyczynowa** — zestawić w jednej osi czasu input, cel, intent, personality, recall, neural winner, wykonanie i outcome.
6. **Migracje schematu i wersji pamięci** — wersjonowanie stanu i skrypt kontroli zgodności przy aktualizacji.

Każdy następny stage należy zatwierdzać jako konkretną zmianę architektury; po zakończonym Stage 36 **nie przypisujemy z góry stałej nazwy Stage 37**, dopóki nie ustalimy zakresu.

---

## 24. Mini-słownik pojęć

| Pojęcie | Znaczenie w Musze |
|---|---|
| **FAFB v783** | źródłowy connectome i jego wersja |
| **connectome** | skierowana sieć połączeń neuronowych |
| **root_id** | identyfikator biologicznego neuronu z danych |
| **sensory cue** | bodziec wprowadzany do stanu sieci |
| **attractor** | grupa neuronów reprezentująca wewnętrzny stan |
| **readout** | odczyt zbioru neuronów przypisany do akcji |
| **winner** | akcja wybrana przez konkurencję |
| **NOOP / stay** | świadomie dopuszczony brak zewnętrznej akcji |
| **feasibility** | czy akcja jest technicznie możliwa teraz |
| **reward / punish** | dodatni / ujemny sygnał uczenia |
| **prediction error** | różnica między przewidywaniem a wynikiem |
| **eligibility trace** | ślad, które neurony można przypisać do feedbacku |
| **synaptic overlay** | wyuczone zmiany nakładane na bazowy FAFB |
| **consolidation** | utrwalanie śladu po nauce/replay |
| **working memory** | krótkotrwały kontekst ostatnich zdarzeń |
| **affect** | utrzymywany stan afektywny |
| **homeostasis** | rosnące/malejące potrzeby i zmęczenie |
| **motivation** | SOCIAL / NOVELTY / SAFETY / REST |
| **foresight** | hipotetyczne przewidywanie skutków akcji |
| **intent** | krótkotrwały zamiar poprzedniego winnera |
| **goal** | potrzeba do rozładowania przez wiele decyzji |
| **personality** | wolno uczony temperament z historii |
| **autobiography** | konkretne ważne własne wydarzenia |
| **salience** | wyliczona ważność zdarzenia |
| **recall** | odtworzenie pasujących wspomnień w nowym kontekście |
| **dwell** | czas przebywania na kanale VC |
| **STT** | speech-to-text, rozpoznanie wypowiedzi |
| **TTS** | text-to-speech, synteza głosu |
| **GPU backend** | wykonywanie obliczeń przez CuPy/CUDA |
| **smoke test** | szybki test najważniejszych kontraktów programu |

---

## 25. Skrócona ściąga na powrót do projektu

**Sprawdź kod:**

~~~powershell
cd F:\mucha
git status
git log -1 --oneline
git pull
~~~

**Sprawdź, czy nie stracisz pamięci:**

~~~text
state/brain_state.npz
state/language.sqlite3
state/voice_episodes.sqlite3
data/connectome/matrix.npz
data/connectome/root_ids.npy
data/connectome/pools.npz
data/connectome/manifest.json
~~~

**Test i uruchomienie:**

~~~powershell
.\.venv\Scripts\python.exe tests\smoke_test.py
.\START_MUCHA.bat
~~~

**Dashboard:**

~~~text
http://127.0.0.1:8765
http://127.0.0.1:8765/autonomy
http://127.0.0.1:8765/neuromap
http://127.0.0.1:8765/details
http://127.0.0.1:8765/config
~~~

**Najważniejsze pliki do zapamiętania:**

~~~text
mucha/brain.py        „mózg”
mucha/discord_bot.py  „ciało / środowisko / pętle / wykonanie”
mucha/language.py     „mowa”
mucha/episodic.py     „wspomnienia”
mucha/web_ui.py       „panel i telemetria”
config.toml           „domyślne ustawienia”
config.local.toml     „ustawienia lokalne”
state/                „wyuczone doświadczenie”
data/connectome/      „bazowa anatomia FAFB”
~~~

**Ważne dla przyszłego ChatGPT / programisty:** pracujemy nad **oryginalnym repozytorium Davidgracz/mucha**, aktualnym końcem roadmapy jest **Stage 36**, nie mylić z osobnym forkiem selfaware. Przed każdą zmianą sprawdzić najnowszy commit, konfigurację lokalną i działający kod. Nie kasować pamięci i nie zastępować pełnego FAFB demo.

---

## 26. Źródła i dokumentacja uzupełniająca

- Repozytorium: https://github.com/Davidgracz/mucha
- Historia zmian: https://github.com/Davidgracz/mucha/commits/main/
- Automatyczne testy: https://github.com/Davidgracz/mucha/actions
- Szczegółowe opisy etapów: [README.md](README.md)
- Lista źródeł badań / licencji: [SOURCES.md](SOURCES.md)
- Aktualne defaults: [config.toml](config.toml)
- Testy integracyjne: [tests/smoke_test.py](tests/smoke_test.py)
- FAFB/Codex: https://codex.flywire.ai/?dataset=fafb

**Utrzymanie dokumentacji:** po dodaniu istotnego stage aktualizować rozdziały 3 (pliki), 6 (pętle), 11 (etapy), 19 (konfiguracja), 20 (diagnostyka), 23 (roadmapa) i sekcję funkcjonalną nowego modułu. Trzymać wersję repozytorium i datę aktualizacji w nagłówku.
