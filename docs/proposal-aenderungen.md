# Proposal: Änderungsvorschläge

Arbeitsnotiz zu [proposal.typ](proposal.typ). Jeder Punkt enthält den aktuellen Stand, den Vorschlag und die Begründung.
Quellen: Proposal Guide (PG), Projektfolien (PF), Vorlesung 11 „Processing and Prediction Modes, Features“ (V11), Vorlesung 15 „FTI Pipeline Architecture“ (V15), Vorlesung 08 „Model Registry“ (V08), Vorlesung 04 „MLOps Introduction“ (V04).

---

## 0. Übersicht der Entscheidungen

| Thema | Bisher | Neu |
|---|---|---|
| Horizont | 16 Tage | **10 Tage** |
| Wetterdaten | Open-Meteo Forecast + Archiv (tatsächliches Wetter) | **Open-Meteo Single Runs API, ECMWF IFS HRES** (archivierte Prognosen, wie sie damals herausgegeben wurden) |
| Wetter-Ort | ein Punkt in der Regionsmitte | **Mittel über 3–4 Orte** (Luzern, Schwyz, Altdorf, Sarnen) |
| Training | 2019–2024, tatsächliches Wetter | **März 2024 bis Ende 2025, echte Prognosen** |
| Test | 2025 bis heute | **2026 (1.1. bis heute)** |
| Klimatologie-Baseline | nicht spezifiziert | aus Waagendaten **2019–2023** |
| Label | „one weight per day, subtract“ | **Veränderung von Mitternacht zu Mitternacht** (Details hängen vom Format der API ab, siehe §3.3) |
| Prediction Mode | „daily schedule or on demand“ | **Offline-/Batch-Prediction** |
| Registry | „registers the best version“ | **Champion/Challenger-Vergleich**, Git-Commit und Datenversion werden mitgespeichert |
| Retraining | „weekly, from scratch“ | **Automated stateless retraining**, nur auf Zeilen mit vollständigen Labels |

---

## 1. Problem statement

### 1.1 Horizont 16 → 10 Tage
- **Bisher:** „for each of the next 16 days (h = 1…16)“
- **Neu:** h = 1…10 Tage, täglich aktualisiert
- **Begründung:**
  - Für die Jahre vor heute sind keine echten 16-Tage-Prognosen archiviert. Die Open-Meteo Single Runs API liefert ECMWF IFS HRES erst ab 14.03.2024 und laut Doku mit **10 Tagen** Horizont. Alle anderen Modelle (z. B. GFS mit 16 Tagen) gibt es dort erst ab April 2026, also ohne eine volle Saison für das Training.
  - Mit 10 Tagen lässt sich jeder Horizont **ehrlich mit echten Prognosen** testen. Bei 16 Tagen wäre der Test für h 8–16 entweder unehrlich oder bräuchte eine Sonderregel.
  - Über ca. Tag 8–10 hinaus liegt jede Wetterprognose nahe an der Klimatologie. Imker planen ohnehin eine gute Woche im Voraus.

### 1.2 Testperiode und Metrik
- **Bisher:** „tested on unseen data from 2025 to today“, MAE ohne Aufschlüsselung
- **Neu:** Test auf **2026 (1.1. bis heute)**. Die MAE in kg/Tag wird **gesamt und pro Horizont h** angegeben.
- **Begründung:**
  - Echte Prognosen für das Training gibt es erst ab März 2024. Deshalb Training 2024–2025 und Test 2026. Die Saison 2026 ist bereits vorbei, der Test hat also eine volle Saison.
  - Die MAE pro h zeigt, wie stark der Fehler mit dem Horizont wächst. Genau das ist die Kernfrage des Projekts.
  - Ein Saisonfilter ist unnötig, weil die Testperiode nur Januar bis September umfasst und damit fast nur Saison ist.

### 1.3 Erfolgskriterium präzisieren
- **Bisher:** „beat persistence by ≥ 10 %, reach or beat seasonal climatology“
- **Neu:** bleibt, ergänzt um: *Klimatologie = mittlere Tagesänderung pro Tag im Jahr, berechnet aus den Waagendaten 2019–2023.*
- **Begründung:** PG verlangt ein „success criterion with a number and a baseline“, PF nennt „no baseline“ als häufigen Fehler. Wenn die Baseline genau definiert ist und nur Jahre vor dem Test nutzt, ist sie reproduzierbar und frei von Leakage.

**Textbaustein (EN):**
> The forecast covers the next 10 days (h = 1…10) and is refreshed daily. It is evaluated on the unseen 2026 season (1 Jan to today) with MAE in kg/day, reported overall and per horizon. Success means beating persistence by ≥ 10 % and matching or beating a seasonal climatology (mean daily change per day of year, computed from the 2019–2023 scale data).

---

## 2. Originality & motivation

### 2.1 Abgleich mit früheren Projekten benennen
- **Bisher:** „I have not come across … in earlier course projects or elsewhere“
- **Neu:** *„Checked against the HSLU projects on mlops-lab.ch and the KTH ID2223 project lists: no forecast based on hive-scale networks.“*
- **Begründung:** PG fragt ausdrücklich: „Checked against past HSLU projects (mlops-lab.ch) and the KTH ID2223 project lists?“ Das ist ein eigenes Bewertungskriterium, deshalb gehört der Abgleich wörtlich hinein. **Vorher selbst auf beiden Seiten kurz nachsehen.**

### 2.2 Kernfrage beibehalten
Der Absatz „whether a weather forecast lets it beat what a beekeeper already knows“ bleibt. Er ist das Alleinstellungsmerkmal und passt nach der Umstellung auf echte Prognosen noch besser.

---

## 3. Data source & features

### 3.1 Wetterquelle: Single Runs API statt Archiv
- **Bisher:** „Open-Meteo … 16-day forecast and a historical archive back to 2019“. Dazu die Aussage, die Features stammen „always from the forecast available at t₀, never from the weather that actually happened“.
- **Neu:** Open-Meteo **Single Runs API** mit `model = ecmwf_ifs`. Beim Backfill wird für jeden Tag ab März 2024 der 00-UTC-Lauf geladen, im Betrieb täglich der neueste Lauf.
- **Begründung:**
  - Die Archive-API liefert **gemessenes/reanalysiertes Wetter**, keine Prognosen. Der bisherige Satz ist damit technisch falsch. Training auf echtem Wetter und Betrieb mit Prognosen ergibt einen **Training-Serving-Skew**, und ein Test auf echtem Wetter wäre **Leakage** (Zukunftswissen). PG: „Leakage: … where future information could leak into a feature.“
  - Mit archivierten Prognosen sind Training, Test und Betrieb identisch. Das Modell lernt so auch, dass Tag 10 unsicherer ist als Tag 1 (MOS-Ansatz).
  - Backfill und täglicher Lauf verwenden **denselben Aufruf** mit anderem Datum. Die Feature-Pipeline ist damit eine einzige Funktion (V15: FTI, Backfill-Fähigkeit laut PF).
  - Ein festes Wettermodell statt `best_match` sorgt dafür, dass sich die Verteilung der Features zwischen Training und Betrieb nicht verschiebt.
- **Warum ECMWF:** Es ist das stärkste globale Mittelfristmodell und das einzige, das im Archiv 10 Tage **und** genug Vergangenheit bietet. Hochaufgelöste Modelle wie ICON-CH, ICON-D2 oder ICON-EU reichen nur 2–5 Tage.

### 3.2 Mehrere Orte statt einem Punkt
- **Neu:** Wetter an 3–4 Orten (Luzern, Schwyz, Altdorf, Sarnen) abrufen und mitteln
- **Begründung:** Das Label ist ein Regionsmittel über ~372 Waagen in bergigem Gelände, für das ein einzelner Punkt kaum repräsentativ ist. Open-Meteo nimmt mehrere Koordinaten pro Request an, der Aufwand bleibt gleich. (Für die Single Runs API noch zu prüfen.)

### 3.3 Label-Definition
- **Bisher:** „keep one weight per day and subtract consecutive days“
- **Neu:** Netto-Veränderung von **Mitternacht zu Mitternacht**. Die genaue Rechnung hängt davon ab, was `ext_weight` enthält (**noch offen**):
  - *absolutes Gewicht:* Wert um 00:00 (± 2 h) am Ende von Tag d − Wert um 00:00 am Ende des Vortags
  - *Änderung pro Stunde:* Summe der Stundenwerte von 00:00 bis 24:00. Fehlen mehr als 2 Stunden, wird der Tag als fehlend markiert.
- **Begründung:**
  - Nachts sind alle Bienen im Stock. Ein Tagesmittel oder -median würde Sammlerinnen, die gerade ausgeflogen sind (1–2 kg, wetterabhängig), mitmessen. Es würde außerdem die Tracht von zwei Tagen vermischen, weil die Differenz zweier Tagesmittel ungefähr von Mittag zu Mittag reicht.
  - Mitternacht zu Mitternacht ordnet den Netto-Eintrag eines Tages genau dem Wetter desselben Tages zu.
  - Fehlende Tage werden **nicht interpoliert**, damit kein künstliches Label entsteht.
  - Falls die API absolute Gewichte liefert: prüfen, ob das Regionsmittel springt, wenn Waagen dazukommen oder wegfallen. Deltas hätten dieses Problem nicht.
- **Kein Leakage:** Das Label wird nur aus Waagendaten berechnet, nie aus dem Wetter (PG: „must not be mechanically derivable from your features“).

### 3.4 Features mit Feature-Typ benennen (V11, Folien 8–12, 17)
| Feature | Beschreibung | Typ laut V11 |
|---|---|---|
| Wetterprognose pro Zieltag | Tmax, Tmin, Niederschlag, Strahlung, max. Wind (Mittel über die Orte) | Batch |
| Gewichts-Momentum | Summe der Labels der letzten 1/3/7/14 Tage | Batch (Aggregat über viele Zeilen, Folie 17) |
| Kalender | Tag im Jahr als sin/cos, Monat | Einzelzeilen-Feature (könnte RT sein), wird im Batch mitberechnet |
| Horizont h | 1…10 | Einzelzeilen-Feature |

- **Begründung:** PG verlangt „Named features, not various features“. Die Zuordnung zu den Feature-Typen zeigt, dass du den Stoff aus V11 verstanden hast, und ist ein naheliegendes Thema für die mündliche Prüfung.

### 3.5 Point-in-time-Korrektheit
- **Neu:** Wetterzeilen im Feature Store mit Schlüssel **(Ausgabedatum der Prognose, Zieltag)**
- **Begründung:** V04 (Challenges Streaming): „Time-Travel for Retraining“. So kann das Training für jedes t₀ genau die damals verfügbare Prognose laden. Das ist technisch der eigentliche Leakage-Schutz.

### 3.6 Split
- **Bisher:** Training 2019–2024, Test 2025 bis heute, 16 Tage Lücke
- **Neu:** Training **März 2024 bis Ende 2025**, Test **2026**, **10 Tage Lücke** zwischen den Label-Fenstern
- **Begründung:** Echte Prognosen gibt es erst ab März 2024. Die Lücke entspricht dem neuen Horizont, damit sich die Label-Fenster von Training und Test nicht überlappen.

### 3.7 Datenzugang und Fallback
- **Bisher:** „a fallback source (another open hive-scale network, or a climatology-only model)“
- **Neu:**
  - HiveWatch/BienenSchweiz um Erlaubnis anfragen und das im Proposal erwähnen. Der Zugriff läuft über einen fixen Request-Header, also eine inoffizielle API.
  - Konkreter Fallback: Waagen → die eigene, bereits gespeicherte Historie im Feature Store plus ein reines Klimatologie-Modell. Wetter → Previous Runs API (7 Tage Vorlauf).
- **Begründung:** PG: „If scraping: the terms of service allow it, and you name a fallback source.“ Eine vage Formulierung wie „another network“ wirkt nicht geprüft.

### 3.8 Datenmenge
- **Neu:** kurz beziffern: Waagen ab 2019 (~7 Jahre, stündlich, täglich +1 Tag), ECMWF-Läufe ab März 2024 (~900 Läufe, täglich +1)
- **Begründung:** PG: „How much data exists today and how fast it grows.“

**Textbaustein (EN, Wetter):**
> Weather features come from the Open-Meteo Single Runs API (model ECMWF IFS HRES, 10-day horizon, archived since March 2024), averaged over four points in the region. For every issue date t₀ the pipeline stores the forecast exactly as it was issued, keyed by (issue date, target date), so training, test and serving all see real forecasts and no observed future weather enters any feature.

---

## 4. System design

### 4.1 Begriffe aus der Vorlesung verwenden (V11)
- **Neu, ein Satz:** *„All three pipelines are batch jobs; the model is used in offline (batch) prediction mode with batch features only, and the Gradio UI serves the stored forecasts request-response.“*
- **Begründung:**
  - Processing Mode (V11, Folien 2–6): Die Daten kommen einmal pro Nacht. Streaming brächte keinen Nutzen (V13, Folie 10).
  - Prediction Mode (V11, Folie 14): Offline Prediction, „Demand Forecasting“ ist dort das Beispiel.
  - Das bisherige „daily schedule **or on demand**“ vermischt Batch- und Online-Prediction. Online Prediction mit Batch-Features würde laut Folie 15 einen Key-Value-Store verlangen, ohne Mehrwert.

### 4.2 Labels entstehen in der Feature-Pipeline (V15)
- **Neu:** Die Feature-Pipeline schreibt **Features und Labels** in den Store. Das Training liest nur noch.
- **Begründung:** V15, Folie 7: „Feature Pipeline: Input = Raw data, output = Features (and Labels)“.

### 4.3 Retraining
- **Bisher:** „retrains the model from scratch on a fixed schedule (for example weekly)“
- **Neu:** *„Automated stateless retraining once a week on natural labels; only rows whose full 10-day label window is complete are used.“*
- **Begründung:**
  - V11, Folie 21: automatisches Stateless-Retraining ist „possible with natural labels“, aber „complex when … labels arrive with significant time delay“. Bei dir stehen die Labels erst nach bis zu 10 Tagen fest, deshalb die Regel.
  - Der Begriff „Natural Labels“ (Folie 18) passt genau.

### 4.4 Champion/Challenger in der Registry
- **Bisher:** „compares it against the baselines and registers the best version“
- **Neu:** Jedes neu trainierte Modell wird als Challenger registriert. Nur bei besserer Test-MAE als der aktuelle Champion bekommt es das Alias `champion`. Die Inferenz lädt immer `@champion`.
- **Begründung:** V08, Folie 36 (Version Aliases champion/challenger); V11, Folie 23: „Is the new model actually better?“ Das verhindert, dass ein schlechteres Modell automatisch live geht. W&B unterstützt Aliase ebenfalls.

### 4.5 Reproduzierbarkeit
- **Neu:** Zu jedem Modell werden **Git-Commit** und **Revision des HF-Datasets** gespeichert.
- **Begründung:** V08, Folie 7: Die Registry soll „a reference to the training data and commit hashes“ enthalten. Der Aufwand ist gering.

### 4.6 Vorhersagen speichern und Monitoring-Job (optional)
- **Neu:** Die Inferenz speichert jede Tagesprognose. Ein kleiner **Monitoring-Job** vergleicht sie mit den später eintreffenden Labels und berechnet so die Live-MAE.
- **Begründung:** V15, Folie 11 (FTI(M)): „When Natural Labels are strongly delayed → does not work in I Pipeline“, also ein separater Monitoring-Job. Das deckt das Kriterium „Observability“ in MS4 ab (PF, Folie 15) und liefert später einen Trigger für Continual Learning.

### 4.7 Diagramm
- **Bisher:** `flowchart LR`, im PDF sehr klein und kaum lesbar. Die Beschriftung „Open-Meteo 16-day forecast + archive“ ist veraltet.
- **Neu:** `flowchart TB` oder zweizeiliges Layout. Beschriftungen: „Open-Meteo Single Runs (ECMWF, 10 d)“, „Features + Labels“, „@champion“. Optional ein Monitoring-Knoten, gestrichelt.
- **Begründung:** PG: „FTI architecture diagram embedded as an image“; PF: „diagram missing or not embedded“ gilt als häufiger Fehler. Ein Diagramm, das eingebettet, aber unlesbar ist, verschenkt Punkte.

### 4.8 Optional-Abschnitt aktualisieren
- **Neu:** Monitoring-Job und Drift-Check, Continual Learning (Retraining bei steigender Live-MAE), 16-Tage-Horizont ab Saison 2027 (GFS über Single Runs oder eigene Snapshots), Spiegelung nach GCP
- **Streichen oder behalten:** Prognosen pro Volk und Trachtlücken-Klassifikator können bleiben, sind aber nicht nötig
- **Begründung:** PG: „Stretch layers … explicitly marked optional; core FTI first.“

### 4.9 Tech-Stack
Bleibt unverändert (HF Hub als Parquet-Store, W&B, GitHub Actions, Gradio auf HF Spaces, Docker).
- **Begründung:** Er ist schlank, kostenlos, deckt alle vier verlangten Bausteine ab (PG: „feature store, experiment tracking, orchestration, serving“) und ist nicht „too ambitious for the core“. W&B ist in V08, Folie 43/44 als Alternative zu MLflow genannt.

---

## 5. Offene Prüfungen vor dem Umschreiben

1. **HiveWatch:** Ist `ext_weight` ein absolutes Gewicht oder eine Änderung pro Stunde? Wie lückenhaft sind die Werte um Mitternacht? Springt das Regionsmittel? → bestimmt §3.3
2. **Single Runs API:** Kommt ECMWF IFS für die Koordinaten der Nordalpen zurück, mit 10 Tagen und ab März 2024? Gehen mehrere Koordinaten pro Request? Wie schnell ist der neueste Lauf verfügbar?
3. **mlops-lab.ch / KTH ID2223:** selbst kurz durchsehen (§2.1)
4. **Umfang:** Nach dem Umschreiben muss das PDF weiterhin **≤ 2 Seiten** haben.

---

## 6. Nicht umsetzen (bewusst verworfen)

| Idee | Warum nicht |
|---|---|
| 16 Tage mit echtem Wetter als Trainingsdaten (Perfect Prog) | Training-Serving-Skew; h 8–16 nicht ehrlich testbar |
| Rohe Archive (ECMWF TIGGE, NOAA GFS auf AWS) | GRIB, große Downloads, Registrierung: „stack too ambitious“ |
| Stündliche statt tägliche Vorhersage | Stundenwerte zeigen vor allem Flugbetrieb und Verdunstung; 240 statt 10 Ausgaben; Imker planen tageweise |
| Tagesmittel/-median als Tagesgewicht | vermischt zwei Tage und misst ausgeflogene Sammlerinnen mit |
| Stream Processing / Online Prediction | Die Daten kommen einmal pro Nacht; kein Nutzen, nur Komplexität (V11, V13) |
