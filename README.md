# 🪰 Mucha v0.1

Autonomiczny bot Discord sterowany stanem sieci o topologii **FlyWire FAFB v783**. Bot:

- utrzymuje ciągły stan neuronów zamiast resetować się po każdej wiadomości,
- odbiera wiadomości i zdarzenia voice jako bodźce,
- sam decyduje czy pisać,
- sam decyduje czy wejść, wyjść lub przeskoczyć na inny kanał voice,
- zaczyna z **pustym modelem języka** i uczy się wyłącznie z tekstu na serwerze,
- nie używa ChatGPT/LLM do generowania tekstu,
- zapisuje stan mózgu i model języka między restartami,
- reakcje pod wiadomościami Muchy mogą wzmacniać/osłabiać jej zachowanie i użyte przejścia znakowe.

## Co jest biologiczne, a co jest naszym interfejsem

Realne dane: neurony, kierunek i siła połączeń oraz klasy neuronów z FlyWire. Runtime v0.1 korzysta ze sparse matrix i ciągłej propagacji aktywności inspirowanej modelami firing-rate/LIF. Mapowanie tekstu/Discorda na neurony sensoryczne oraz neuronów wyjściowych na akcje Discorda jest sztucznym interfejsem, bo mucha nie ma biologicznych wejść „Discord” ani aparatu językowego.

**Nie należy interpretować tego jako emulacji świadomości lub dokładnego mózgu żywej muchy.** To eksperyment: prawdziwa topologia connectome + sztuczne wejścia/wyjścia.

## 1. Szybki test bez FlyWire

Na Windowsie najlepiej Python 3.12.

```bat
install_windows.bat
```

Skrypt utworzy środowisko i **demo connectome**. Demo służy tylko do sprawdzenia bota.

Skopiuj token bota do `.env`:

```env
DISCORD_TOKEN=...
```

Uruchom:

```bat
run_windows.bat
```

## 2. Bot Discord

W Discord Developer Portal utwórz Application → Bot.

Włącz privileged intents:

- Message Content Intent
- Server Members Intent

Bot potrzebuje co najmniej:

- View Channels
- Send Messages
- Read Message History
- Add Reactions
- Connect (voice)

Bot łączy się z voice jako `self_deaf=True`; v0.1 nie nagrywa i nie analizuje audio.

## 3. Pełny FlyWire FAFB v783

Codex wymaga zalogowania. Pobierz z portalu **Download Data** dla datasetu FAFB v783 co najmniej:

- `classification.csv.gz`
- `neurons.csv.gz`
- `connections_princeton.csv.gz` (wersja filtrowana, około 3.7 mln połączeń)

Można użyć `connections_princeton_no_threshold.csv.gz`, ale będzie większy i cięższy.

Umieść np. w:

```text
raw_flywire/
    classification.csv.gz
    neurons.csv.gz
    connections_princeton.csv.gz
    coordinates.csv.gz                # opcjonalne, realna Neuro-map
    consolidated_cell_types.csv.gz    # opcjonalne, nazwy typów neuronów
```

Następnie:

```bat
.venv\Scripts\activate
python tools\prepare_connectome.py --input raw_flywire --output data\connectome
```

Po zakończeniu `data/connectome/manifest.json` powinien pokazać około **139 255 neuronów** i około **3 732 460** połączeń dla filtrowanego exportu v783 (dokładna liczba może zależeć od aktualnego pliku eksportowego Codex).

### Neuro-map: współrzędne i biologiczne adnotacje

Runtime connectome może zostać wzbogacony bez przebudowy macierzy połączeń. Jeżeli w folderze z danymi Codex masz `classification.csv.gz`, `neurons.csv.gz`, opcjonalnie `coordinates.csv.gz` i `consolidated_cell_types.csv.gz`, uruchom:

```bash
python tools/augment_connectome_map.py \
  --raw raw_flywire \
  --connectome data/connectome
```

Skrypt tworzy `data/connectome/neuron_meta.npz`. Neuro-map wykorzystuje wtedy klasy biologiczne, stronę mózgu, predicted neurotransmitter i — gdy dostępny jest `coordinates.csv.gz` — rzeczywiste oznaczone współrzędne FlyWire. Brakujące pozycje mają jawnie oznaczony fallback i nie są prezentowane jako anatomia.

Potem po prostu:

```bat
python bot.py
```

## 4. Jak uczy się pisać

Model języka nadal nie korzysta z ChatGPT ani innego pretrenowanego LLM. Uczy się online wyłącznie z wiadomości Discord, transkrypcji STT i późniejszego feedbacku.

Aktualny generator jest hybrydowy:

- `char_unigram/bigram/trigram` uczą pisowni i pozwalają składać nowe formy znak po znaku,
- `word_unigram/bigram/trigram` uczą kolejności całych słów i dzięki temu szybciej tworzą sensowne krótkie wypowiedzi,
- `word_starts` zapamiętuje typowe początki,
- specjalny znacznik końca wypowiedzi uczy model, kiedy zakończyć zdanie zamiast doklejać losowe słowa,
- świeżo zaobserwowane przejścia słów dostają tymczasowy `recent boost`, który stopniowo zanika.

Każda wiadomość lub transkrypcja aktualizuje oba poziomy pamięci w `state/language.sqlite3` od razu. Domyślnie generator próbuje użyć modelu słów z prawdopodobieństwem 0.90, a model znakowy pozostaje fallbackiem i źródłem bardziej eksperymentalnych form.

Domyślna konfiguracja:

```toml
[language]
min_chars_before_speaking = 800
min_unique_chars_before_speaking = 16
max_generated_chars = 120
hybrid_word_enabled = true
word_model_probability = 0.90
word_max_tokens = 14
word_recent_window_seconds = 3600
word_recent_boost = 2.00
word_frequency_exponent = 0.95
word_arousal_flatten = 0.12
char_frequency_exponent = 0.90
char_arousal_flatten = 0.15
word_reward_scale = 0.12
connectome_word_control_enabled = true
connectome_word_control_min_vocab = 1500
connectome_word_control_strength = 0.35
connectome_word_control_candidates = 24
```

Connectome wpływa na wybór konkretnych słów dopiero po zbudowaniu słownika
o rozmiarze `connectome_word_control_min_vocab`. Do tego momentu Mucha normalnie
zbiera słownictwo i uczy unigramy/bigramy/trigramy, a stan mózgu wpływa na język
tak jak wcześniej przez decyzję o mówieniu i poziom eksploracji. Po przekroczeniu
progu każde poznane słowo ma stabilną reprezentację sensoryczną w connectomie,
a aktualny stan odpowiadających mu populacji może lekko podbić lub osłabić jego
szansę wśród sensownych kandydatów generatora. Connectome nie omija modelu
językowego i nie może wybierać słów spoza wyuczonego słownika.

Pozytywny/negatywny feedback wzmacnia lub osłabia zarówno wykorzystane przejścia znakowe, jak i znane przejścia słów. Bezpośredni reply człowieka do wypowiedzi Muchy daje również małe wzmocnienie tej konkretnej wypowiedzi językowej.

Stare tabele `unigram/bigram/trigram/starts`, jeśli istnieją w bazie z wcześniejszej wersji, są jednorazowo używane do zasilenia nowego modelu słów. Nie dostają jednak bonusu świeżości, więc nowe rozmowy mogą szybko zacząć wpływać na aktualne zachowanie.

W `/details` widać rozmiar słownika, liczbę word-bigramów i word-trigramów oraz czy ostatnia wypowiedź pochodziła z generatora `words` czy `characters`. W `/config` wszystkie parametry hybrydowego języka można zmieniać na żywo.

## 5. Voice

Co `voice.poll_seconds` sekund bot daje mózgowi snapshot kanałów voice i znajdujących się w nich użytkowników. Potem odczytuje populacje wyjściowe:

- `voice_join`
- `voice_move`
- `voice_leave`
- `stay`

Każdy kanał ma stabilną „sygnaturę” sensoryczną i wyjściową zależną od jego Discord ID. Dzięki temu wybór kanału zależy od aktualnego stanu connectome i wcześniejszych bodźców.

`minimum_dwell_seconds` zapobiega skakaniu kilka razy na sekundę i problemom z rate-limitami Discorda. Nie jest to reguła wyboru kanału — tylko okres refrakcji.

## 6. Admin/debug

Tylko administrator serwera:

```text
!mucha status
!mucha save
!mucha pause
!mucha resume
!mucha reward
!mucha punish
```

Normalnego zachowania nie kontrolujesz komendami. Komendy są tylko diagnostyczne/awaryjne.

## 7. Pliki stanu

```text
state/brain_state.npz     aktywność + plastyczny bias + eligibility trace
state/language.sqlite3   wyuczony język
```

Jeżeli przeniesiesz te dwa pliki razem z tym samym connectome, przenosisz „tę konkretną Muchę” w sensie tego projektu.

## 8. Test

Po instalacji zależności:

```bat
python tests\smoke_test.py
```

## Następny krok: v0.2

Najważniejsze rozszerzenia, które warto zrobić dalej:

1. pełny biologiczny LIF na parametrach Shiu et al. dla zdarzeń o znaczeniu behawioralnym,
2. synaptyczna plastyczność jako sparse overlay zamiast tylko persistent neuronal bias,
3. opcjonalna mała sieć char-GRU od zera jako kolejny etap bardziej długiego kontekstu,
4. dashboard WebSocket pokazujący aktywne regiony/neurony na żywo,
5. audio voice jako bodziec (VAD/energia/cechy akustyczne) bez rozpoznawania mowy albo opcjonalnie z transkrypcją.

## Status Discord zależny od connectome

Po połączeniu z Discordem Mucha zawsze ustawia dostępność na **online**. Co 30 sekund odczytuje aktualny stan connectome i aktualizuje widoczną aktywność, np.:

- 🧠 nasłuchuje kanałów,
- 🧠 uczy się rozmów,
- 🧠 eksploruje serwer,
- 🧠 obserwuje reakcje,
- 🧠 przetwarza bodźce.

Wybór stanu wynika z aktualnych readoutów connectome (`speak`, `explore`, `voice_join`, `voice_move`, `stay`, `react`). Status pokazuje też bieżącą średnią aktywność i liczbę neuronów z aktywacją powyżej 0.1.

**Uwaga:** ustawienie `online` nie utrzymuje bota przy życiu. Jeśli program `bot.py` przestanie działać albo komputer/hosting zostanie wyłączony, Discord pokaże Muchę jako offline.


## Konsolowy dashboard mózgu

Mucha ma teraz live UI w terminalu. Domyślnie działa tryb:

```toml
[console_ui]
mode = "dashboard"
refresh_seconds = 1.0
top_neurons = 8
```

Dostępne tryby:

- `dashboard` — pełny odświeżany panel z aktywnością mózgu, readoutami, językiem, voice i top neuronami,
- `simple` — jedna linia diagnostyczna co odświeżenie,
- `off` — wyłącza UI i zostawia zwykłe logi.

Dashboard pokazuje m.in. liczbę aktywnych neuronów, średnią i maksymalną aktywację, reward trace, tick symulacji, wszystkie readouty zachowania, ostatni bodziec, ostatnią akcję, stan języka i bieżący kanał voice.

Przy prawdziwym FlyWire tabela **Top aktywnych neuronów** pokazuje rzeczywiste `root_id` z datasetu FAFB. W demo-connectome są to tylko sztuczne identyfikatory.

Po `git pull` doinstaluj zależność UI:

```powershell
python -m pip install -r requirements.txt
```


## Graficzny Web UI

Po uruchomieniu Muchy lokalny panel otwiera się automatycznie pod:

```text
http://127.0.0.1:8765
```

Panel pokazuje na żywo:

- osobną zakładkę `/connectome` z funkcjonalnym grafem przepływu aktywności,
- osobną zakładkę `/neuromap` z projekcją XY/XZ/YZ mózgu, heatmapą aktywnych regionów i inspektorem neuronów,
- aktywne neurony, średnią i maksymalną aktywację,
- readouty `speak/react/voice_join/voice_move/voice_leave/explore/stay`,
- reward trace i tick runtime,
- stan języka i voice,
- ostatni bodziec oraz ostatnią akcję,
- top aktywnych neuronów z `root_id` FlyWire,
- aktywny backend obliczeń i nazwę urządzenia,
- historię aktywności na wykresie.

Konfiguracja:

```toml
[web_ui]
enabled = true
host = "127.0.0.1"
port = 8765
auto_open = true
refresh_ms = 500
history_points = 180
```

Domyślnie panel jest dostępny tylko lokalnie. Nie ustawiaj `host = "0.0.0.0"` na komputerze wystawionym do Internetu bez dodatkowego uwierzytelniania/firewalla.

## GPU / CUDA

Mucha potrafi używać NVIDIA CUDA do propagacji sparse connectome. W `config.toml`:

```toml
[brain]
backend = "auto"
gpu_device = 0
```

Tryby:

- `auto` — próbuje CUDA, a gdy CuPy/GPU jest niedostępne wraca do CPU,
- `cuda` — wymaga działającej CUDA; błąd przy starcie, jeśli GPU nie działa,
- `cpu` — wymusza NumPy/SciPy.

### Instalacja GPU na Windows

Najprościej:

```bat
install_gpu_windows.bat
```

albo ręcznie w aktywnym venv:

```powershell
python -m pip install -r requirements-gpu.txt
python tools\check_gpu.py
```

Jeśli test pokaże kartę NVIDIA i `CuPy test: OK`, uruchom:

```powershell
python bot.py
```

W GUI i konsoli powinieneś wtedy zobaczyć:

```text
backend: CUDA
device: NVIDIA ...
```

Stan `brain_state.npz` pozostaje przenośny między CPU i GPU, bo przy zapisie jest konwertowany do zwykłych tablic NumPy.


## Learning / Reaction telemetry

Web UI pokazuje teraz pełny ślad uczenia i autonomicznych reakcji:

- **Learning Debug** — ostatni reward, target action, liczba zmienionych neuronów, max/mean delta bias,
- **before → after** dla wszystkich readoutów zachowania,
- **Top changed neurons** z FlyWire `root_id`, delta bias i bieżącą aktywacją,
- **Plasticity** — średni/max bias, liczba dodatnich/ujemnych neuronów i histogram,
- **Reward timeline** — reward trace oraz znaczniki nagród/kar,
- **Action History** — wiadomości, reakcje, voice join/move/leave i reward,
- **Reaction Debug** — `react` vs próg, wybrane emoji, target, decyzja i cooldown,
- **Voice Debug** — progi, permissions, affinity kanałów oraz dokładny powód decyzji.

Reward dla wiadomości jest przypisywany do zapisanej aktywności neuronów z chwili wysłania wiadomości oraz do akcji `speak`. Manualne `!mucha reward` / `!mucha punish` wzmacniają lub osłabiają ostatnią akcję możliwą do nagrodzenia (np. voice join/move, react albo speak).

Mucha może autonomicznie dodawać reakcje do wiadomości. Zachowanie kontrolują:

```toml
[behavior]
reaction_threshold = 0.73
reaction_cooldown_seconds = 20
```

Do reakcji bot potrzebuje Discord permission **Add Reactions**.


## Voice overstay punishment

Mucha może dostać automatyczną karę za zbyt długie pozostawanie na jednym kanale voice:

```toml
[voice]
maximum_dwell_seconds = 300
overstay_punish_amount = 0.5
overstay_punish_interval_seconds = 60
```

Po przekroczeniu limitu reward `-overstay_punish_amount` jest przypisywany do akcji `stay`. Jeśli nadal pozostaje na tym samym kanale, kara może zostać powtórzona po skonfigurowanym interwale. Join, move i leave resetują licznik kar dla danego pobytu.

## Full reaction emoji pool

Autonomiczne reakcje korzystają z pełnej bazy Unicode emoji oraz custom emoji dostępnych na bieżącym serwerze. Dla wydajności Mucha losuje i ocenia connectome tylko podzbiór całej puli dla jednej wiadomości:

```toml
[behavior]
reaction_candidate_sample = 64
```

Każde emoji z pełnej puli może trafić do kolejnych próbek. Jeśli Discord odrzuci najwyżej ocenioną reakcję, bot próbuje następnych kandydatów. Reaction Debug pokazuje całkowity rozmiar puli, liczbę ocenionych kandydatów i top wyników.


## Per-server reinforcement context

Mucha nadal używa jednego wspólnego connectome na wszystkich serwerach, ale kontekst reward/punish jest rozdzielony per Discord guild.

- ostatnia nagradzalna akcja jest zapisywana osobno dla każdego serwera,
- `!mucha reward` i `!mucha punish` działają tylko na ostatnią akcję z bieżącego serwera,
- jeśli na bieżącym serwerze nie ma jeszcze akcji do nagrodzenia, komenda nie wykonuje globalnego rewardu,
- voice join/move/leave, speak, react i overstay `stay` aktualizują własny kontekst danego serwera,
- Web UI pokazuje sekcję **Server Learning Context** z ostatnią nagradzalną akcją każdego serwera.


## Voice threat / escape

Po przekroczeniu `voice.maximum_dwell_seconds` Mucha nie dostaje już tylko kary za `stay`. Connectome otrzymuje osobny bodziec sensoryczny zagrożenia związany z aktualnym guild i kanałem.

Zagrożenie zaczyna się od lekkiego poziomu i narasta do 100%. Wraz z nim:

- do connectome trafiają bodźce `internal:threat:voice-overstay`, guild threat i channel threat,
- `stay` nadal jest okresowo karane,
- rośnie efektywny impuls do `voice_move`,
- wymagane affinity nowego kanału jest stopniowo poluzowywane,
- podczas zagrożenia wybierany jest najlepszy **inny** kanał, nawet gdy aktualny kanał nadal ma najwyższe affinity,
- udana ucieczka wzmacnia `voice_move` małym dodatnim rewardem i resetuje czas pobytu.

Konfiguracja bazowa:

```toml
[voice]
maximum_dwell_seconds = 300
threat_ramp_seconds = 120
threat_magnitude = 1.2
threat_move_boost = 0.28
threat_affinity_relaxation = 0.18
threat_escape_reward = 0.35
threat_steps = 3
```

Voice Debug pokazuje poziom zagrożenia, magnitude bodźca, efektywny move score, efektywny margin oraz aktualny escape target.


## Blocked text channels / deadly voice memory

Mucha może obserwować i uczyć się z wiadomości na wybranych kanałach tekstowych, ale nie może tam wysyłać własnego tekstu. Dotyczy to odpowiedzi autonomicznych, wiadomości spontanicznych oraz tekstowych odpowiedzi komend diagnostycznych.

```toml
[discord]
blocked_text_channel_ids = [
  344519890083774475,
  506193122460434443,
  1049295352680947752,
]
```

Po udanej ucieczce spowodowanej voice threat kanał, z którego Mucha uciekła, zostaje oznaczony jako śmiertelnie niebezpieczny:

```toml
[voice]
deadly_channel_seconds = 600
deadly_threat_magnitude = 1.6
```

Przez ten czas kanał:
- nie może zostać wybrany do join,
- nie może zostać wybrany do zwykłego move,
- nie może być celem kolejnej ucieczki,
- emituje osobny bodziec `voice:deadly-channel:<guild>:<channel>` do connectome.

Voice Debug pokazuje taki kanał jako `☠ ŚMIERTELNE <sekundy>s`. Po wygaśnięciu pamięci automatycznie wraca do normalnej puli kanałów.


## Whole-server voice exploration

Wybór voice target nie jest już deterministycznym `max(affinity)`. Mucha pamięta ostatnie odwiedziny kanałów i łączy affinity connectome z bonusem nowości oraz karą za niedawne odwiedziny.

```toml
[voice]
exploration_memory_seconds = 1800
exploration_novelty_bonus = 0.32
exploration_recent_penalty = 0.38
exploration_temperature = 0.18
exploration_min_candidates = 3
```

Kanały nigdy nieodwiedzone dostają największy bonus, a świeżo odwiedzone są tymczasowo mniej atrakcyjne. Spośród wszystkich dostępnych kanałów cel jest losowany wagowo zamiast zawsze wybierać ten sam najwyższy wynik. Deadly channels nadal są całkowicie wykluczone.

Voice Debug pokazuje teraz `Affinity`, `Explore`, `Novelty` oraz `Last visit` dla każdego kanału.


## Rare random voice audio

Gdy Mucha jest podłączona do co najmniej jednego kanału voice, raz na sekundę wykonywany jest globalny los:

```text
1 / 10000
```

Po trafieniu Mucha wybiera jeden z aktualnie połączonych serwerów, na którym nic już nie gra, i odtwarza:

```text
assets/random_audio.mp3
```

Konfiguracja:

```toml
[voice]
random_audio_enabled = true
random_audio_file = "assets/random_audio.mp3"
random_audio_chance_denominator = 10000
random_audio_volume = 0.8
ffmpeg_executable = "ffmpeg"
```

Komputer uruchamiający bota musi mieć FFmpeg dostępny w PATH albo należy podać pełną ścieżkę w `ffmpeg_executable`.

Przy rare evencie connectome dostaje dodatkowe bodźce `internal:rare-audio` i `voice:rare-audio:guild:<id>`, a zdarzenie jest widoczne w Action History jako `rare_audio`.

Jeśli pliku audio nie ma, bot tylko zapisuje jednorazowe ostrzeżenie do logu i działa dalej normalnie.


## Autonomous TTS on voice

Gdy Mucha siedzi na voice, co 10 sekund dostaje okazję do powiedzenia czegoś. Nie mówi automatycznie co każde 10 sekund: connectome najpierw musi mieć `speak >= behavior.speak_threshold`.

Jeśli chce mówić:

1. connectome dostaje bodziec `voice:tts-opportunity:guild:<id>`,
2. ten sam hybrydowy generator słowo+znak, który tworzy wiadomości tekstowe, generuje treść,
3. ostatnia wiadomość tekstowa z tego samego serwera jest używana jako kontekst,
4. lokalny `pyttsx3` generuje WAV,
5. Discord odtwarza WAV przez FFmpeg na aktualnym voice.

Konfiguracja:

```toml
[voice]
tts_enabled = true
tts_interval_seconds = 10
tts_rate = 185
tts_volume = 0.9
tts_voice_name = ""
tts_max_chars = 180
```

`tts_voice_name = ""` oznacza domyślny głos systemowy. Można wpisać część nazwy zainstalowanego głosu Windows, np. `Paulina` albo `Zofia`, jeśli taki głos istnieje w systemie.

Wypowiedź TTS jest zapisywana w Action History jako `tts_speak` i staje się ostatnią akcją `speak` do ręcznego `!mucha reward` / `!mucha punish`.

TTS i rare audio nie grają jednocześnie: obie ścieżki sprawdzają, czy klient voice aktualnie coś odtwarza.


## Mucha Chaser predator response

Mucha ma osobne zachowanie dopasowane do projektu Mucha Chaser, który okresowo wchodzi na voice i przez około 30 sekund podąża za Muchą między kanałami.

Gdy obcy bot wejdzie dokładnie na kanał, na którym siedzi Mucha:

- pierwsze spotkanie uruchamia krótki odruch podejrzenia i natychmiastową próbę ucieczki,
- jeśli ten sam bot podąży za Muchą ponownie w krótkim oknie, zostaje rozpoznany jako Chaser,
- potwierdzony Chaser uruchamia około 35 sekund stanu PANIC / CHASE,
- zwykły `minimum_dwell_seconds` nie blokuje odruchowej ucieczki,
- Mucha wybiera inny dostępny kanał z użyciem bieżących affinity i mechanizmu eksploracji connectomu,
- kanał, na którym Chaser ją dopadł, jest tymczasowo oznaczany jako niebezpieczny,
- kanał zawierający rozpoznanego Chasera jest wykluczany z normalnych celów ruchu,
- każda kolejna pogoń Chasera na nowy kanał może wywołać kolejną szybką ucieczkę.

Connectome dostaje przy takim spotkaniu silne bodźce `internal:predator-chaser`, `voice:predator:<guild>:<bot>` i `voice:danger-channel:<guild>:<channel>`. Udana ucieczka może wzmacniać akcję `voice_move`.

Konfiguracja:

```toml
[voice]
chaser_enabled = true
chaser_bot_id = 0
chaser_name_hint = "chaser"
chaser_confirm_hits = 2
chaser_follow_window_seconds = 12.0
chaser_panic_seconds = 35.0
chaser_suspicion_seconds = 8.0
chaser_escape_delay_min_seconds = 0.15
chaser_escape_delay_max_seconds = 0.75
chaser_channel_avoid_seconds = 90.0
chaser_threat_magnitude = 2.2
chaser_escape_reward = 0.25
```

`chaser_bot_id = 0` oznacza automatyczne wykrywanie. Jeśli znasz ID bota Mucha Chaser, wpisanie go tutaj daje natychmiastowe rozpoznanie bez oczekiwania na drugi follow.


## Publiczny Control Center

Web UI ma teraz chroniony hasłem ekran główny łączący telemetrię Muchy, Chasera i VPS.

Domyślna konfiguracja:

```toml
[web_ui]
enabled = true
host = "0.0.0.0"
port = 8765
auto_open = false
auth_enabled = true
auth_username = "admin"
auth_password_env = "MUCHA_DASHBOARD_PASSWORD"
session_hours = 168
chaser_status_file = "/opt/mucha-chaser/state/chaser_status.json"
```

Hasła nie zapisuj w `config.toml`. Ustaw je w lokalnym `.env`:

```env
MUCHA_DASHBOARD_PASSWORD=TU_MOCNE_HASLO
```

Jeżeli hasło nie jest ustawione, a host to `0.0.0.0`, aplikacja celowo wraca do `127.0.0.1`, żeby nie wystawić niezabezpieczonego debug panelu do Internetu.

Ścieżki:

- `/` — Control Center: VPS, usługi, Chaser, voice, audio, connectome i logi,
- `/details` — pełny rozbudowany dashboard kafelkowy ze szczegółami,
- `/brain` — przekierowanie kompatybilności do `/details`,
- `/api/overview` — JSON Control Center,
- `/api/state` — pełny JSON Muchy,
- `/logout` — wylogowanie.

Mucha Chaser zapisuje live status do `state/chaser_status.json`. Dashboard odczytuje ten plik lokalnie i pokazuje bieżący pościg oraz czas następnej rundy.


## Social learning i relacje

Mucha utrzymuje trwały profil relacji z użytkownikami w `state/language.sqlite3`.

Affinity użytkownika ma zakres `-1.0 .. +1.0` i zmienia się głównie na podstawie reakcji dodawanych do wiadomości Muchy. Pozytywne reakcje zwiększają affinity, negatywne je zmniejszają. Domyślny próg unikania to `-0.35`.

Gdy affinity użytkownika spadnie poniżej progu:

- Mucha nie odpowiada na jego wiadomości,
- nie reaguje autonomicznie na jego wiadomości,
- nie generuje spontanicznej odpowiedzi ani TTS na podstawie jego ostatniej wiadomości,
- podczas normalnego zachowania voice nie wybiera kanałów z tym użytkownikiem,
- jeśli taki użytkownik wejdzie na aktualny kanał Muchy, Mucha próbuje przenieść się na inny kanał albo wychodzi z voice.

Unikanie użytkowników jest wyłączane podczas aktywnej ucieczki przed Mucha Chaser. Ucieczka przed Chaserem ma pierwszeństwo i może prowadzić przez kanał z nielubianym użytkownikiem.

Social learning dodatkowo wykrywa:

- bezpośrednie reply do wiadomości Muchy,
- powtórzenie słowa wygenerowanego przez Muchę,
- powtórzenie dwu- lub trzysłownej frazy,
- potwierdzenie tego samego słowa przez różnych użytkowników,
- silne samopowtórzenia Muchy, które dostają małą karę.

Dashboard `/details` pokazuje relacje, affinity, liczbę pozytywnych i negatywnych reakcji oraz najlepiej utrwalone słowa.

Dashboard `/config` pozwala bez ręcznej edycji TOML:

- wykluczać kanały tekstowe,
- wykluczać kanały voice,
- zmieniać progi join/move/leave,
- zmieniać dwell i voice poll,
- włączać/wyłączać TTS i rare audio,
- zmieniać progi i siłę social learning,
- ustawiać próg unikania użytkowników.

Zmiany z zakładki Konfiguracja są zapisywane do `config.toml` i dla obsługiwanych opcji stosowane od razu.


### Naturalne pozytywne sygnały

Affinity nie wymaga ręcznego klikania reakcji. Małe dodatnie zmiany powstają także podczas zwykłej interakcji:

- reply do wiadomości Muchy,
- mention `@Mucha`,
- kontynuowanie rozmowy w tym samym kanale krótko po wypowiedzi Muchy,
- powtórzenie słowa lub frazy użytej przez Muchę,
- wejście użytkownika na kanał voice, na którym jest Mucha,
- pozostanie z Muchą na voice przez określony czas,
- pozostanie na kanale po wypowiedzi TTS.

Te zdarzenia generują osobne bodźce connectome, między innymi:

```text
social:user-replied
social:user-mentioned-me
social:user-continued-conversation
social:user-reused-word
social:user-reused-phrase
social:user-joined-my-voice
social:user-stayed-with-me
social:user-stayed-after-tts
social:repeated-positive-contact
social:familiar-user
social:liked-user
```

Naturalne plusy mają cooldown, więc spamowanie jednego zachowania nie pozwala szybko nabić affinity. Długie wspólne siedzenie na voice jest dodatkowo ograniczone do maksymalnie jednego przyrostu co 10 minut, a pozostanie po TTS do jednego przyrostu co 5 minut.

Po kilku różnych pozytywnych kontaktach connectome dostaje `social:repeated-positive-contact`. Po przekroczeniu `familiar_affinity_threshold` dostaje również `social:familiar-user`, a przy wysokim affinity `social:liked-user`.


## Słuchanie użytkowników na voice

Mucha może lokalnie rozpoznawać mowę użytkowników z kanału Discord voice.

Pipeline:

```text
Discord PCM 48 kHz stereo
→ bufor per użytkownik
→ wykrycie ciszy / końca wypowiedzi
→ resampling do 16 kHz mono
→ faster-whisper
→ tekst
→ language.learn(...)
→ brain.inject_text(...)
→ bodźce voice/social connectome
```

Odbiór audio używa `discord-ext-voice-recv` i `VoiceRecvClient`. Transkrypcja używa lokalnego `faster-whisper`. Domyślnie:

```toml
stt_enabled = true
stt_model = "base"
stt_language = "pl"
stt_device = "cpu"
stt_compute_type = "int8"
stt_cpu_threads = 2
stt_download_root = "state/whisper"
stt_silence_seconds = 0.9
stt_min_segment_seconds = 0.7
stt_max_segment_seconds = 12.0
stt_min_chars = 2
stt_beam_size = 1
```

Model jest ładowany w tle po starcie. Przy pierwszym uruchomieniu może zostać pobrany do `state/whisper`.

Surowe audio użytkowników nie jest zapisywane jako pliki. PCM jest trzymane tymczasowo w RAM na czas krótkiego segmentu i usuwane po przekazaniu go do transkrypcji. Rozpoznany tekst trafia do tego samego hybrydowego uczenia słów i znaków, z którego Mucha korzysta dla wiadomości tekstowych.

Connectome dostaje m.in.:

```text
voice:speech-heard
voice:speech:user:<ID>
voice:spoken-rejection
voice:word-reused
voice:phrase-reused
social:user-mentioned-me
social:user-continued-conversation
```

Jeśli użytkownik powie do Muchy wulgarną frazę odrzucającą albo zrobi to do 20 sekund po jej TTS, może to dać ujemny reward do śladu neuronalnego ostatniego TTS oraz obniżyć affinity. Powtarzanie słów/fraz z TTS i normalne kontynuowanie rozmowy mogą dać dodatni feedback.

W `/details` jest kafel `Voice Recognition / STT` z ostatnią transkrypcją, użytkownikiem, językiem, pewnością i kolejką. Ustawienia STT są dostępne również w `/config`.


## Learning Since Startup

Zakładka `/details` zawiera kafel `Learning Since Startup`, który zeruje się przy każdym uruchomieniu procesu i pokazuje zmiany od tego momentu.

Kafel mierzy:

- nowe znaki nauczone przez model języka,
- nowe próbki wiadomości/wypowiedzi,
- nowe unikalne przejścia znakowe,
- liczbę udanych transkrypcji voice,
- liczbę dodatnich i ujemnych reward events,
- skumulowany dodatni i ujemny reward,
- liczbę unikalnych neuronów zmienionych przez `brain.reward()`,
- sumę operacji aktualizacji bias,
- średnie i maksymalne skumulowane `|Δ bias|`,
- neuron FlyWire `root_id`, który dostał największą skumulowaną zmianę od rewardów.

Metryki reward-learning są liczone bezpośrednio w `FlyBrain.reward()`. Są oddzielone od zwykłego `plasticity_decay`, dzięki czemu zanik wcześniej zapisanych biasów nie jest błędnie prezentowany jako nowa nauka.


## Affinity dashboard i naturalne minusy

Dashboard ma osobną zakładkę `/affinity`. Pokazuje na żywo:

- wszystkie dodatnie i ujemne zdarzenia wpływające na affinity,
- aktualne wartości `Δ affinity`,
- pozytywne i negatywne emoji,
- frazy odrzucające wraz z severity,
- progi `ZNAJOMY`, `LUBI` i `OMIJA`,
- aktualne relacje z użytkownikami,
- bieżący `negative streak` i jego mnożnik,
- ostatni sygnał społeczny.

Naturalne ujemne zachowania obejmują:

- wyjście lub przeniesienie się użytkownika krótko po wejściu Muchy na kanał; kara dotyczy tylko osób, które były na kanale przed jej wejściem,
- wyjście lub przeniesienie się krótko po TTS Muchy,
- powtarzające się uciekanie od Muchy po kolejnych jej wejściach,
- aktywne ignorowanie bezpośredniej odpowiedzi Muchy: użytkownik nie kontynuuje rozmowy w tym kanale, ale w tym czasie pisze gdzie indziej na serwerze.

Sam brak odpowiedzi lub AFK nie obniża affinity.

Kolejne negatywne sygnały w oknie `negative_streak_window_seconds` zwiększają mnożnik o `negative_streak_multiplier_step`, maksymalnie do `negative_streak_max_multiplier`. Pozytywne interakcje stopniowo wygaszają ten streak.

Podczas aktywnej ucieczki przed Chaserem naturalne kary za opuszczanie kanału po wejściu Muchy są pomijane, a social avoidance nie ogranicza kanału ucieczki.


## Lokalna konfiguracja VPS

`config.toml` jest bazową konfiguracją śledzoną przez Git. Zmiany wykonywane z dashboardu są zapisywane do `config.local.toml`, który jest ignorowany przez Git.

Przy starcie Mucha ładuje `config.toml`, a następnie nakłada na niego wartości z `config.local.toml`. Dzięki temu nowe opcje z repo są automatycznie dostępne, a ustawienia konkretnego VPS nie powodują konfliktów przy `git pull`.

Jednorazowa migracja ze starszej instalacji może polegać na skopiowaniu lokalnie zmodyfikowanego `config.toml` do `config.local.toml`, a następnie przywróceniu bazowego `config.toml` z repo. Stare klucze językowe `min_tokens_before_speaking`, `min_unique_tokens_before_speaking` i `max_generated_tokens` są ignorowane, gdy bazowy config zawiera nowe odpowiedniki znakowe.

## Hybrydowe uczenie języka

Model językowy uczy się jednocześnie przejść znakowych oraz słownych. Warstwa słów zapisuje lokalnie w SQLite:

- unigramy słów,
- bigramy słów,
- trigramy słów,
- początki wypowiedzi,
- reward i czas ostatniego wystąpienia.

Świeżo nauczone przejścia mogą dostać czasowy `word_recent_boost`, dzięki czemu nowe zwroty zaczynają wpływać na generację szybciej, bez usuwania starszej pamięci. Character model pozostaje fallbackiem i nadal odpowiada za bardziej swobodne składanie tekstu.

Domyślne ustawienia szybszego uczenia:

```toml
min_chars_before_speaking = 800
min_unique_chars_before_speaking = 16
max_generated_chars = 120
hybrid_word_enabled = true
word_model_probability = 0.90
word_max_tokens = 14
word_recent_window_seconds = 3600
word_recent_boost = 2.00
word_frequency_exponent = 0.95
word_arousal_flatten = 0.12
char_frequency_exponent = 0.90
char_arousal_flatten = 0.15
word_reward_scale = 0.12
```

Te parametry są dostępne także w `/config`.
