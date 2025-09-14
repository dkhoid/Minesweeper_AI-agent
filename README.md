This is a **Minesweeper application with AI** written in Python + Tkinter, featuring multiple strategies:

* **Basic Strategy** (based on basic open/flag rules + pattern recognition)
* **Probabilistic Strategy** (estimating mine probabilities)
* **CSP Strategy** (Constraint Satisfaction Problem – constraint propagation + backtracking)
* **Hybrid Strategy** (hybrid: deterministic + CSP + probabilistic + pattern recognition)
* **CNN Strategy** (Convolutional Neural Network-based pattern recognition)

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
