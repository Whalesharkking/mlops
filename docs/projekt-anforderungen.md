# MLOps HS26 — Projekt-Anforderungen (Arbeitsreferenz)

> **Single Source of Truth sind die PDFs** in `infos/`
> (`MLops_HS26_Project_Presentation_v2.pdf`, `MLOps_HS26_Proposal_Guide.pdf`).
> Dieses Dokument ist nur eine Zusammenfassung als Arbeitshilfe.

## Gesamtauftrag

Baue ein **"Live" ML-System**: eine automatisierte, cloud-taugliche Pipeline, die
dynamische Daten einliest, Features berechnet, ein Modell trainiert und Vorhersagen
ausliefert. **Einzelprojekt** (allein).

- **Daten:** Nur **dynamische** Quellen (Crypto, Wetter, SBB-Delays …). Kein statisches CSV.
- **Stack:** GitHub (public), Docker, MLflow/W&B, GitHub Actions oder Airflow.
- **Deployment:** **Cloud-Deployment ist Pflicht** (z. B. GCP / AWS / HuggingFace).
- **Bewertet wird die Pipeline, nicht die Accuracy.** Zitat: "A fully automated
  pipeline at 60% is an A; a manual notebook at 99% is a fail."

### FTI-Architektur (die drei Pipelines)

1. **Feature Pipeline** — läuft nach Schedule oder als Streaming. Holt Rohdaten,
   berechnet Features, speichert in **Feature Store**. Inkl. **Backfill**.
2. **Training Pipeline** — getriggert/manuell. Liest aus Feature Store, trainiert,
   evaluiert, registriert in **Model Registry**.
3. **Inference Pipeline** — on demand (UI) oder Schedule. Lädt bestes Modell aus
   Registry und liefert Vorhersagen.

## Meilensteine, Gewichtung & Deadlines

| Milestone | Gewicht | Own | Peer | Deadline |
|---|---|---|---|---|
| MS1: Project Proposal | 20% | 10% | 10% | Do **01.10.2026**, 23:59 |
| MS1 Peer Reviews | — | — | — | Do 08.10.2026 |
| MS2: Feature Pipeline | 20% | 10% | 10% | Do **05.11.2026** |
| MS2 Peer Reviews | — | — | — | Do 12.11.2026 |
| MS3: Training Pipeline | 10% | 5% | 5% | Do **03.12.2026** |
| MS3 Peer Reviews | — | — | — | Do 10.12.2026 |
| Oral Exam (Q&A) | 30% | 30% | — | Do 17.12.2026 |
| MS4: Live System (+ Video) | 20% | 20% | — | **10.01.2027** |

Peer Reviews = **25% der Note**, bewertet nach Qualität (generisches Lob = 0 Punkte).

## MS1 — Proposal (aktuelle Aufgabe)

- **Format:** max. **2 Seiten PDF**. Auf ILIAS "MS1 Submission" + Repo-URL.
- **Kopie im Repo:** `docs/proposal.pdf` (bei uns via Typst-Workflow aus `docs/proposal.typ` gerendert).
- **Genau vier Überschriften** — jede wird als eigenes Kriterium benotet.

### 1. Problem statement
- Was wird vorhergesagt und **für wen**?
- **Exakter Horizont** ("60 minutes ahead", nicht "soon") + Scope (welche Stadt/Assets/Zeitraum).
- **Erfolgskriterium mit Zahl UND Baseline.**
- Stark: *"Predict per-station bike availability 60 min ahead, MAE ≤ 2, beating persistence."*
- Schwach: *"Analyse bike-sharing data."*

### 2. Originality & motivation
- Warum dieses Problem, warum du?
- Gegen frühere HSLU-Projekte (**mlops-lab.ch**) und **KTH ID2223** geprüft?
- **Ein Satz**, was deins anders macht. Crypto-Direction, Air Quality, News-Classification
  sind oft gemacht → undifferenzierte Wiederholung punktet schlecht.

### 3. Data source & features
- **Live-Quelle + Provider**; API oder Scraping? Bei Scraping: ToS erlaubt es + Fallback-Quelle nennen.
- **Update-Frequenz** — dynamische Daten Pflicht, statisches CSV zählt nicht.
- Wie viel Daten existieren heute und wie schnell wachsen sie (sagen, falls Historie erst mit Poller startet).
- **Das Label:** was es ist und woher — **darf nicht mechanisch aus den Features ableitbar sein.**
- Seltene Positiv-Klasse: wie behandelt/evaluiert.
- **Benannte Features**, nicht "various features".
- **Leakage:** Out-of-time-Split, wo könnte Zukunftsinfo in ein Feature lecken.

### 4. System design
- **FTI-Architektur-Diagramm als Bild eingebettet** (Draft ok; **nicht** ASCII, nicht nur referenziert).
- **Tech-Stack:** feature store, experiment tracking, orchestration, serving — je eine Zeile Begründung.
- **Stretch-Layers** (heavy orchestration, IaC, Managed-Cloud-Extras, Extra-Monitoring)
  explizit als **optional** markieren; Core FTI zuerst.
- Repository ist **public** bestätigen.

### Häufige Fehler (laut Guide)
- keine Baseline · Diagramm fehlt/nicht eingebettet · Label per Regel aus Inputs abgeleitet
- recycletes Thema ohne Twist · Stack zu ambitioniert für den Core

## Regeln
- **AI-Tools erlaubt**, aber du musst **jede Zeile Code** und jede Entscheidung erklären
  können (Oral Exam fragt genau das).
- Individuelle Arbeit; Inspiration von mlops-lab.ch / KTH ok, **Code/Doku kopieren nicht** (Plagiatsprüfung).
- **Keine Secrets** im Repo oder in der History. Geleakte Keys müssen rotiert werden
  (Commit löschen reicht nicht). Keys in `.env` (gitignored) + GitHub Secrets.

## Support
- Coaching (freiwillig): SW11–SW13 (26.11 / 03.12 / 10.12).
- **USD 50 Google Cloud Credits** optional (gcp.secure.force.com, @stud.hslu.ch-Login).
