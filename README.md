# Yelp Mobility Data Analysis (FYP Demo)

A lightweight web dashboard for exploring **mobility patterns in Yelp users** using **DBSCAN-based spatial hubs**.  
Built as a Final Year Project demo to communicate key findings clearly with interactive, research-style UI.

![cover](./fyp_demo_cover.png)

## ✨ What this demo shows

### 1) Dataset overview (10K user sample)
- High-level snapshot cards (users, restaurants, reviews, tips, check-ins)
- Location distribution tables (top cities, coverage by state)

### 2) Mobility patterns (DBSCAN hubs)
We cluster **restaurant-only visits** per user using **DBSCAN with haversine distance** and classify users into:

- **One-area users**: primarily dine within a single neighborhood
- **Two-area commuters**: show a clear two-hub pattern (e.g., home + work)
- **Explorers**: wide-spread dining across many areas

The dashboard highlights:
- % breakdown by mobility type
- Average hub separation (commute-scale distance for two-hub users)
- Typical within-hub travel range (tight clustering around primary hub)

It also includes **Research Notes** to explain:
- why DBSCAN is used
- how `eps` is selected (k-distance analysis)
- sanity checks for whether the results “smell right”
- why centroid-based models can fail for multi-hub users

---

## 🧠 Key insight (research framing)

Most users are adequately represented by a **single centroid**, but a **small yet meaningful subset** exhibits multi-hub behavior.  
For these users, centroid-based location features can land in “in-between” areas they never visit, causing systematic recommendation errors.

---

## 🏗️ Tech stack

- React + TypeScript
- Tailwind CSS (custom design tokens)
- lucide-react (icons)
- shadcn/ui components (Switch, Label, etc.)
- CSV parsing via `papaparse`
- Local data loading via `/public/data/*`

---

## 📁 Data files

Place the following files inside:

```

public/data/
mobility_summary.json
user_mobility_table.csv

````

Example:
- `mobility_summary.json` contains the final aggregated metrics used by the summary cards.
- `user_mobility_table.csv` contains per-user mobility outputs (used for future interactivity / deep dives).

---

## 🚀 Getting started

### 1) Install dependencies
```bash
npm install
````

### 2) Run locally

```bash
npm run dev
```

Open the local URL printed in your terminal (usually `http://localhost:5173` if using Vite).

### 3) Build

```bash
npm run build
```

---

## 🧩 Project structure (important bits)

```
src/
  components/
    dashboard/
      MobilityTab.tsx
  hooks/
    useMobilityData.ts
  data/
    dashboardData.ts   # mock visualization (fallback / schematic)
public/
  data/
    mobility_summary.json
    user_mobility_table.csv
```

---

## 🔌 How the data is loaded

The UI pulls local files through the custom hook:

```ts
import { useMobilityData } from "@/hooks/useMobilityData";
```

* Loads JSON summary → maps into `MobilitySummary`
* Loads CSV → parses into rows (for deeper exploration)

---

## 👤 Author

**LIM JING JIE**
Final Year Project — Yelp Mobility Data Analysis

---

## 📄 Notes

This repository is a **demo UI** for presenting analysis outcomes.
Core clustering + preprocessing is performed offline and exported into JSON/CSV for visualization.

