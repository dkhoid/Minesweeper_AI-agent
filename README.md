This is a **Minesweeper application with AI** written in Python + Tkinter, featuring multiple strategies:

* **RandomStrategy** (random selection)
* **AutoOpenStrategy** (based on basic open/flag rules)
* **ProbabilisticStrategy** (estimating mine probabilities)
* **CSPStrategy** (Constraint Satisfaction Problem – constraint propagation + backtracking)
* **HybridStrategy** (hybrid: deterministic + CSP + probabilistic + pattern recognition)
* **BasicPatternStrategy** (based on recognizing common patterns: 1-2-1, 1-2-2-1, 1-2, 1-1)

The application also includes:

* **UI** with Tkinter (menu for difficulty selection, theme selection, viewing statistics).
* **Settings & Statistics** (saving configurations, recording results to JSON files).
* **Auto Play** (AI automatically plays using the selected strategy).

## Run the Application
To run the application, ensure you have Python installed and then execute the following command in your terminal:

```
python index.py
```
or run benchmark
```
python index.py --benchmark
```
