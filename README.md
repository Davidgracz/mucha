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

To właśnie te pliki zawierają dużą część indywidualnego doświadczenia konkretnej instancji Muchy.

Dwie instancje uruchomione z osobnymi katalogami `state/` mogą z czasem wykształcić różne zachowania mimo identycznego kodu i bazowego connectome.

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

# Najbliższy kierunek rozwoju

Dalsze kierunki:

- bogatsze sensory voice i lepsze kodowanie dynamiki rozmowy,
- historia Decision Trace, aby można było cofać się po kilku ostatnich decyzjach zamiast oglądać wyłącznie najnowszą,
- dalsza redukcja ręcznie zakodowanych wyjątków na rzecz sygnałów sensorycznych i konkurencji readoutów connectomu,
- bogatsza długoterminowa reprezentacja ludzi, kanałów i powtarzalnych sytuacji.

---

## Założenie projektu

Najważniejsza zasada Muchy:

**jak najwięcej zachowania ma wynikać z connectomu, jego aktualnego stanu i doświadczenia, a jak najmniej z ręcznie zakodowanych decyzji.**
