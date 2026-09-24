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
```

Następnie:

```bat
.venv\Scripts\activate
python tools\prepare_connectome.py --input raw_flywire --output data\connectome
```

Po zakończeniu `data/connectome/manifest.json` powinien pokazać około **139 255 neuronów** i około **3 732 460** połączeń dla filtrowanego exportu v783 (dokładna liczba może zależeć od aktualnego pliku eksportowego Codex).

Potem po prostu:

```bat
python bot.py
```

## 4. Jak uczy się pisać

Model języka jest pusty na pierwszym starcie. Nie ma słownika, listy gotowych słów, zdań ani pretreningu.

Każda zwykła wiadomość użytkownika aktualizuje online model przejść **znak po znaku** w `state/language.sqlite3`. Generator nie wybiera gotowych słów z tabeli — każdą wypowiedź składa litera po literze na podstawie wyuczonych przejść znakowych. Dlatego na początku wynik może przypominać bełkot, a wraz z obserwacją większej liczby rozmów powinny pojawiać się coraz bardziej naturalne fragmenty i słowa.

Stare tabele word-level (`unigram`, `bigram`, `trigram`) mogą pozostać w istniejącej bazie po aktualizacji, ale nowy generator ich nie używa. Nowe uczenie korzysta wyłącznie z tabel `char_*`, które przy pierwszym uruchomieniu są puste.

Domyślnie Mucha dopuści generowanie dopiero po:

- 1200 zaobserwowanych znakach,
- 18 różnych znakach.

Konfiguracja:

```toml
[language]
min_chars_before_speaking = 1200
min_unique_chars_before_speaking = 18
max_generated_chars = 220
```

Stary `config.toml` z polami `min_tokens_before_speaking`, `min_unique_tokens_before_speaking` i `max_generated_tokens` jest automatycznie mapowany przy starcie na ustawienia modelu znakowego.

Pozytywny lub negatywny feedback pod wypowiedzią Muchy wzmacnia albo osłabia konkretne trójki znaków użyte podczas generowania oraz odpowiedni ślad uczenia w connectome.

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
