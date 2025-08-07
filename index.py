from tkinter import *
from tkinter import messagebox as tkMessageBox
from tkinter import ttk
from collections import deque
import random
import platform
import time
from datetime import datetime
import json
import os
from numpy import tile

# Difficulty settings
DIFFICULTIES = {
    "Beginner": {"size_x": 9, "size_y": 9, "mines": 10},
    "Intermediate": {"size_x": 16, "size_y": 16, "mines": 40},
    "Expert": {"size_x": 16, "size_y": 30, "mines": 99},
    "Custom": {"size_x": 10, "size_y": 10, "mines": 15}
}

# Game states
STATE_DEFAULT = 0
STATE_CLICKED = 1
STATE_FLAGGED = 2

# Button bindings
BTN_CLICK = "<Button-1>"
BTN_FLAG = "<Button-2>" if platform.system() == 'Darwin' else "<Button-3>"

# Themes
THEMES = {
    "Classic": {
        "bg": "#c0c0c0",
        "button_bg": "#c0c0c0",
        "text_color": "#000000",
        "mine_color": "#ff0000",
        "flag_color": "#ff0000"
    },
    "Dark": {
        "bg": "#2b2b2b",
        "button_bg": "#404040",
        "text_color": "#ffffff",
        "mine_color": "#ff4444",
        "flag_color": "#ffaa00"
    },
    "Blue": {
        "bg": "#e6f3ff",
        "button_bg": "#cce7ff",
        "text_color": "#003366",
        "mine_color": "#cc0000",
        "flag_color": "#0066cc"
    }
}

class Settings:
    def __init__(self):
        self.difficulty = "Beginner"
        self.theme = "Classic"
        self.safe_first_click = True
        self.auto_flag = False
        self.show_timer = True
        self.play_sounds = False
        self.save_stats = True
        self.load_settings()
    
    def save_settings(self):
        try:
            settings_data = {
                "difficulty": self.difficulty,
                "theme": self.theme,
                "safe_first_click": self.safe_first_click,
                "auto_flag": self.auto_flag,
                "show_timer": self.show_timer,
                "play_sounds": self.play_sounds,
                "save_stats": self.save_stats
            }
            with open("settings.json", "w") as f:
                json.dump(settings_data, f)
        except:
            pass
    
    def load_settings(self):
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r") as f:
                    settings_data = json.load(f)
                    self.difficulty = settings_data.get("difficulty", "Beginner")
                    self.theme = settings_data.get("theme", "Classic")
                    self.safe_first_click = settings_data.get("safe_first_click", True)
                    self.auto_flag = settings_data.get("auto_flag", False)
                    self.show_timer = settings_data.get("show_timer", True)
                    self.play_sounds = settings_data.get("play_sounds", False)
                    self.save_stats = settings_data.get("save_stats", True)
        except:
            pass

class Statistics:
    def __init__(self):
        self.stats = {
            "Beginner": {"games_played": 0, "games_won": 0, "best_time": None},
            "Intermediate": {"games_played": 0, "games_won": 0, "best_time": None},
            "Expert": {"games_played": 0, "games_won": 0, "best_time": None},
            "Custom": {"games_played": 0, "games_won": 0, "best_time": None}
        }
        self.load_stats()
    
    def save_stats(self):
        try:
            with open("stats.json", "w") as f:
                json.dump(self.stats, f)
        except:
            pass
    
    def load_stats(self):
        try:
            if os.path.exists("stats.json"):
                with open("stats.json", "r") as f:
                    self.stats = json.load(f)
        except:
            pass
    
    def add_game(self, difficulty, won, time_taken=None):
        self.stats[difficulty]["games_played"] += 1
        if won:
            self.stats[difficulty]["games_won"] += 1
            if time_taken and (not self.stats[difficulty]["best_time"] or time_taken < self.stats[difficulty]["best_time"]):
                self.stats[difficulty]["best_time"] = time_taken

class SettingsWindow:
    def __init__(self, parent, settings, callback):
        self.parent = parent
        self.settings = settings
        self.callback = callback
        self.window = Toplevel(parent)
        self.window.title("Settings")
        self.window.geometry("400x500")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()
        
        self.create_widgets()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_widgets(self):
        notebook = ttk.Notebook(self.window)
        notebook.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Game Settings Tab
        game_frame = ttk.Frame(notebook)
        notebook.add(game_frame, text="Game")
        
        # Difficulty
        ttk.Label(game_frame, text="Difficulty:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.difficulty_var = StringVar(value=self.settings.difficulty)
        difficulty_combo = ttk.Combobox(game_frame, textvariable=self.difficulty_var, 
                                       values=list(DIFFICULTIES.keys()), state="readonly")
        difficulty_combo.grid(row=0, column=1, sticky=EW, padx=5, pady=5)
        difficulty_combo.bind("<<ComboboxSelected>>", self.on_difficulty_change)
        
        # Custom difficulty settings
        self.custom_frame = ttk.LabelFrame(game_frame, text="Custom Settings")
        self.custom_frame.grid(row=1, column=0, columnspan=2, sticky=EW, padx=5, pady=5)
        
        ttk.Label(self.custom_frame, text="Width:").grid(row=0, column=0, padx=5, pady=2)
        self.custom_width = IntVar(value=DIFFICULTIES["Custom"]["size_y"])
        ttk.Spinbox(self.custom_frame, from_=5, to=50, textvariable=self.custom_width, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self.custom_frame, text="Height:").grid(row=1, column=0, padx=5, pady=2)
        self.custom_height = IntVar(value=DIFFICULTIES["Custom"]["size_x"])
        ttk.Spinbox(self.custom_frame, from_=5, to=30, textvariable=self.custom_height, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self.custom_frame, text="Mines:").grid(row=2, column=0, padx=5, pady=2)
        self.custom_mines = IntVar(value=DIFFICULTIES["Custom"]["mines"])
        ttk.Spinbox(self.custom_frame, from_=1, to=500, textvariable=self.custom_mines, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        # Game options
        self.safe_first_var = BooleanVar(value=self.settings.safe_first_click)
        ttk.Checkbutton(game_frame, text="Safe first click", variable=self.safe_first_var).grid(row=2, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        self.auto_flag_var = BooleanVar(value=self.settings.auto_flag)
        ttk.Checkbutton(game_frame, text="Auto flag when all mines found", variable=self.auto_flag_var).grid(row=3, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        # Appearance Tab
        appear_frame = ttk.Frame(notebook)
        notebook.add(appear_frame, text="Appearance")
        
        ttk.Label(appear_frame, text="Theme:").grid(row=0, column=0, sticky=W, padx=5, pady=5)
        self.theme_var = StringVar(value=self.settings.theme)
        theme_combo = ttk.Combobox(appear_frame, textvariable=self.theme_var, 
                                  values=list(THEMES.keys()), state="readonly")
        theme_combo.grid(row=0, column=1, sticky=EW, padx=5, pady=5)
        
        self.show_timer_var = BooleanVar(value=self.settings.show_timer)
        ttk.Checkbutton(appear_frame, text="Show timer", variable=self.show_timer_var).grid(row=1, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        # Other Tab
        other_frame = ttk.Frame(notebook)
        notebook.add(other_frame, text="Other")
        
        self.play_sounds_var = BooleanVar(value=self.settings.play_sounds)
        ttk.Checkbutton(other_frame, text="Play sounds", variable=self.play_sounds_var).grid(row=0, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        self.save_stats_var = BooleanVar(value=self.settings.save_stats)
        ttk.Checkbutton(other_frame, text="Save statistics", variable=self.save_stats_var).grid(row=1, column=0, columnspan=2, sticky=W, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=X, padx=10, pady=5)
        
        ttk.Button(button_frame, text="OK", command=self.on_ok).pack(side=RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.on_cancel).pack(side=RIGHT)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.on_reset).pack(side=LEFT)
        
        self.on_difficulty_change()
    
    def on_difficulty_change(self, event=None):
        is_custom = self.difficulty_var.get() == "Custom"
        for widget in self.custom_frame.winfo_children():
            widget.configure(state=NORMAL if is_custom else DISABLED)
    
    def on_ok(self):
        # Update custom difficulty if selected
        if self.difficulty_var.get() == "Custom":
            DIFFICULTIES["Custom"] = {
                "size_x": self.custom_height.get(),
                "size_y": self.custom_width.get(),
                "mines": self.custom_mines.get()
            }
        
        # Update settings
        self.settings.difficulty = self.difficulty_var.get()
        self.settings.theme = self.theme_var.get()
        self.settings.safe_first_click = self.safe_first_var.get()
        self.settings.auto_flag = self.auto_flag_var.get()
        self.settings.show_timer = self.show_timer_var.get()
        self.settings.play_sounds = self.play_sounds_var.get()
        self.settings.save_stats = self.save_stats_var.get()
        
        self.settings.save_settings()
        self.callback()
        self.window.destroy()
    
    def on_cancel(self):
        self.window.destroy()
    
    def on_reset(self):
        self.difficulty_var.set("Beginner")
        self.theme_var.set("Classic")
        self.safe_first_var.set(True)
        self.auto_flag_var.set(False)
        self.show_timer_var.set(True)
        self.play_sounds_var.set(False)
        self.save_stats_var.set(True)
    
    def on_close(self):
        self.window.destroy()

class StatsWindow:
    def __init__(self, parent, stats):
        self.window = Toplevel(parent)
        self.window.title("Statistics")
        self.window.geometry("400x300")
        self.window.resizable(False, False)
        self.window.transient(parent)
        self.window.grab_set()
        
        # Create treeview for stats
        columns = ("Difficulty", "Games Played", "Games Won", "Win Rate", "Best Time")
        tree = ttk.Treeview(self.window, columns=columns, show="headings", height=10)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=80, anchor=CENTER)
        
        # Populate with data
        for difficulty, data in stats.stats.items():
            games_played = data["games_played"]
            games_won = data["games_won"]
            win_rate = f"{(games_won/games_played*100):.1f}%" if games_played > 0 else "0%"
            best_time = f"{data['best_time']:.2f}s" if data["best_time"] else "N/A"
            
            tree.insert("", END, values=(difficulty, games_played, games_won, win_rate, best_time))
        
        tree.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Close button
        ttk.Button(self.window, text="Close", command=self.window.destroy).pack(pady=5)

class Minesweeper:
    def __init__(self, tk):
        self.tk = tk
        self.settings = Settings()
        self.stats = Statistics()
        self.ai = MinesweeperAI(self)

        # Initialize game variables
        self.size_x = DIFFICULTIES[self.settings.difficulty]["size_x"]
        self.size_y = DIFFICULTIES[self.settings.difficulty]["size_y"]
        self.total_mines = DIFFICULTIES[self.settings.difficulty]["mines"]
        
        self.create_menu()
        self.setup_ui()
        self.apply_theme()
        self.restart()
        self.update_timer()
    
    def create_menu(self):
        menubar = Menu(self.tk)
        self.tk.config(menu=menubar)
        
        # Game menu
        game_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game", command=self.restart, accelerator="F2")
        game_menu.add_separator()
        game_menu.add_command(label="Settings", command=self.open_settings, accelerator="F3")
        game_menu.add_command(label="Statistics", command=self.open_stats, accelerator="F4")
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.tk.quit)
        
        # Bind keyboard shortcuts
        self.tk.bind("<F2>", lambda e: self.restart())
        self.tk.bind("<F3>", lambda e: self.open_settings())
        self.tk.bind("<F4>", lambda e: self.open_stats())
    
    def setup_ui(self):
        # Main container frame
        self.main_frame = Frame(self.tk)
        self.main_frame.pack(padx=10, pady=10)
        
        # Top frame for status labels
        self.top_frame = Frame(self.main_frame)
        self.top_frame.pack(fill=X, pady=(0, 10))
        
        # Status labels
        self.labels = {
            "mines": Label(self.top_frame, text=f"Mines: {self.total_mines}", font=("Arial", 12, "bold")),
            "flags": Label(self.top_frame, text="Flags: 0", font=("Arial", 12, "bold")),
            "time": Label(self.top_frame, text="00:00:00", font=("Arial", 12, "bold"))
        }
        
        self.labels["mines"].pack(side=LEFT)
        self.labels["flags"].pack(side=LEFT, padx=(20, 0))
        self.labels["time"].pack(side=RIGHT)
        
        # Game frame for the minefield
        self.game_frame = Frame(self.main_frame)
        self.game_frame.pack()
        
        # Bottom frame for controls
        self.bottom_frame = Frame(self.main_frame)
        self.bottom_frame.pack(fill=X, pady=(10, 0))
        
        # Control buttons
        restart_btn = Button(self.bottom_frame, text="New Game (F2)", command=self.restart, 
                           font=("Arial", 10), padx=10)
        restart_btn.pack(side=LEFT)
        
        settings_btn = Button(self.bottom_frame, text="Settings (F3)", command=self.open_settings,
                            font=("Arial", 10), padx=10)
        settings_btn.pack(side=LEFT, padx=(10, 0))
        
        stats_btn = Button(self.bottom_frame, text="Stats (F4)", command=self.open_stats,
                         font=("Arial", 10), padx=10)
        stats_btn.pack(side=LEFT, padx=(10, 0))

        auto_play_btn = Button(self.bottom_frame, text="Auto Play", command=self.auto_play,
                               font=("Arial", 10), padx=10)
        auto_play_btn.pack(side=LEFT, padx=(10, 0))
        
        # Difficulty label
        difficulty_text = f"Difficulty: {self.settings.difficulty}"
        if self.settings.difficulty == "Custom":
            difficulty_text += f" ({self.size_y}x{self.size_x}, {self.total_mines} mines)"
        
        self.difficulty_label = Label(self.bottom_frame, text=difficulty_text, font=("Arial", 9))
        self.difficulty_label.pack(side=RIGHT)
    
    def auto_play(self):
        if self.game_over_flag:
            return
        if self.ai.play():
            self.tk.after(100, self.auto_play)

    def apply_theme(self):
        theme = THEMES[self.settings.theme]
        self.tk.configure(bg=theme["bg"])
        self.main_frame.configure(bg=theme["bg"])
        self.top_frame.configure(bg=theme["bg"])
        self.game_frame.configure(bg=theme["bg"])
        self.bottom_frame.configure(bg=theme["bg"])
        
        for label in self.labels.values():
            label.configure(bg=theme["bg"], fg=theme["text_color"])
        
        self.difficulty_label.configure(bg=theme["bg"], fg=theme["text_color"])
    
    def create_images(self):
        """Load images from the images folder, falling back to simple colors if files don't exist"""
        self.images = {
            "plain": None,
            "clicked": None,
            "mine": None,
            "flag": None,
            "wrong": None,
            "numbers": []
        }
        
        # Try to load actual image files first
        image_path = "images"
        image_files = {
            "plain": "tile_plain.gif",
            "clicked": "tile_clicked.gif",
            "mine": "tile_mine.gif",
            "flag": "tile_flag.gif",
            "wrong": "tile_wrong.gif"
        }
        
        # Load basic tiles
        for key, filename in image_files.items():
            try:
                filepath = os.path.join(image_path, filename)
                if os.path.exists(filepath):
                    self.images[key] = PhotoImage(file=filepath)
                else:
                    # Fallback to colored rectangles
                    self.images[key] = self.create_fallback_image(key)
            except:
                self.images[key] = self.create_fallback_image(key)
        
        # Load number tiles (1-8)
        for i in range(1, 9):
            try:
                filepath = os.path.join(image_path, f"tile_{i}.gif")
                if os.path.exists(filepath):
                    img = PhotoImage(file=filepath)
                else:
                    img = self.create_fallback_number_image(i)
                self.images["numbers"].append(img)
            except:
                self.images["numbers"].append(self.create_fallback_number_image(i))
    
    def create_fallback_image(self, image_type):
        """Create fallback colored rectangle images"""
        theme = THEMES[self.settings.theme]
        img = PhotoImage(width=20, height=20)
        
        if image_type == "plain":
            img.put(theme["button_bg"], (0, 0, 20, 20))
        elif image_type == "clicked":
            img.put("#ffffff", (0, 0, 20, 20))
        elif image_type == "mine":
            img.put(theme["mine_color"], (0, 0, 20, 20))
        elif image_type == "flag":
            img.put(theme["flag_color"], (0, 0, 20, 20))
        elif image_type == "wrong":
            img.put("#888888", (0, 0, 20, 20))
        
        return img
    
    def create_fallback_number_image(self, number):
        """Create fallback number images"""
        img = PhotoImage(width=20, height=20)
        img.put("#ffffff", (0, 0, 20, 20))
        return img
    
    def setup(self):
        # Clear existing game tiles
        for widget in self.game_frame.winfo_children():
            widget.destroy()
        
        # Initialize game state
        self.flag_count = 0
        self.correct_flag_count = 0
        self.clicked_count = 0
        self.start_time = None
        self.first_click = True
        self.game_over_flag = False
        
        self.create_images()
        
        # Create tiles using grid layout in the game_frame
        self.tiles = {}
        
        for x in range(self.size_x):
            self.tiles[x] = {}
            for y in range(self.size_y):
                tile = {
                    "id": f"{x}_{y}",
                    "is_mine": False,
                    "state": STATE_DEFAULT,
                    "coords": {"x": x, "y": y},
                    "mines": 0,
                    "button": Button(self.game_frame, image=self.images["plain"], 
                                   width=25, height=25, bd=1, relief="raised")
                }
                
                tile["button"].bind(BTN_CLICK, self.on_click_wrapper(x, y))
                tile["button"].bind(BTN_FLAG, self.on_right_click_wrapper(x, y))
                tile["button"].grid(row=x, column=y, padx=1, pady=1)
                
                self.tiles[x][y] = tile
        
        # Place mines (will be done on first click if safe_first_click is enabled)
        if not self.settings.safe_first_click:
            self.place_mines()
    
    def place_mines(self, avoid_x=None, avoid_y=None):
        mine_positions = set()
        avoid_positions = set()
        
        # If safe first click, avoid the clicked cell and its neighbors
        if avoid_x is not None and avoid_y is not None:
            avoid_positions.add((avoid_x, avoid_y))
            for neighbor in self.get_neighbors(avoid_x, avoid_y):
                avoid_positions.add((neighbor["coords"]["x"], neighbor["coords"]["y"]))
        
        # Place mines randomly
        while len(mine_positions) < self.total_mines:
            x = random.randint(0, self.size_x - 1)
            y = random.randint(0, self.size_y - 1)
            
            if (x, y) not in mine_positions and (x, y) not in avoid_positions:
                mine_positions.add((x, y))
                self.tiles[x][y]["is_mine"] = True
        
        # Calculate numbers for each tile
        for x in range(self.size_x):
            for y in range(self.size_y):
                if not self.tiles[x][y]["is_mine"]:
                    count = sum(1 for neighbor in self.get_neighbors(x, y) if neighbor["is_mine"])
                    self.tiles[x][y]["mines"] = count
    
    def restart(self):
        # Update difficulty settings
        difficulty_data = DIFFICULTIES[self.settings.difficulty]
        self.size_x = difficulty_data["size_x"]
        self.size_y = difficulty_data["size_y"]
        self.total_mines = difficulty_data["mines"]
        
        self.setup()
        self.refresh_labels()
        self.apply_theme()
        
        # Update difficulty label
        difficulty_text = f"Difficulty: {self.settings.difficulty}"
        if self.settings.difficulty == "Custom":
            difficulty_text += f" ({self.size_y}x{self.size_x}, {self.total_mines} mines)"
        self.difficulty_label.config(text=difficulty_text)
        
        # Adjust window size to fit the grid
        self.tk.update_idletasks()
        self.tk.geometry("")  # Let tkinter calculate the size
    
    def refresh_labels(self):
        self.labels["flags"].config(text=f"Flags: {self.flag_count}")
        self.labels["mines"].config(text=f"Mines: {self.total_mines}")
    
    def game_over(self, won):
        self.game_over_flag = True
        
        # Record statistics
        if self.settings.save_stats and self.start_time:
            time_taken = (datetime.now() - self.start_time).total_seconds()
            self.stats.add_game(self.settings.difficulty, won, time_taken if won else None)
            self.stats.save_stats()
        
        # Show mines and wrong flags
        for x in range(self.size_x):
            for y in range(self.size_y):
                tile = self.tiles[x][y]
                if tile["is_mine"] and tile["state"] != STATE_FLAGGED:
                    tile["button"].config(image=self.images["mine"])
                elif not tile["is_mine"] and tile["state"] == STATE_FLAGGED:
                    tile["button"].config(image=self.images["wrong"])
        
        self.tk.update()
        
        # Show result dialog
        msg = "Congratulations! You won!\n\nPlay again?" if won else "Game Over! You hit a mine!\n\nPlay again?"
        if tkMessageBox.askyesno("Game Over", msg):
            self.restart()
    
    def update_timer(self):
        if self.settings.show_timer:
            ts = "00:00:00"
            if self.start_time and not self.game_over_flag:
                delta = datetime.now() - self.start_time
                total_seconds = int(delta.total_seconds())
                hours = total_seconds // 3600
                minutes = (total_seconds % 3600) // 60
                seconds = total_seconds % 60
                ts = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.labels["time"].config(text=ts)
        else:
            self.labels["time"].config(text="")
        
        self.tk.after(100, self.update_timer)
    
    def get_neighbors(self, x, y):
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size_x and 0 <= ny < self.size_y:
                    neighbors.append(self.tiles[nx][ny])
        return neighbors
    
    def on_click_wrapper(self, x, y):
        return lambda event: self.on_click(self.tiles[x][y])
    
    def on_right_click_wrapper(self, x, y):
        return lambda event: self.on_right_click(self.tiles[x][y])
    
    def on_click(self, tile):
        if self.game_over_flag or tile["state"] != STATE_DEFAULT:
            return
        
        # Start timer on first click
        if self.start_time is None:
            self.start_time = datetime.now()
        
        # Handle safe first click
        if self.first_click and self.settings.safe_first_click:
            self.place_mines(tile["coords"]["x"], tile["coords"]["y"])
            self.first_click = False
        
        # Check if mine
        if tile["is_mine"]:
            self.game_over(False)
            return
        
        # Reveal tile
        if tile["mines"] == 0:
            tile["button"].config(image=self.images["clicked"], relief="sunken")
            self.clear_surrounding_tiles(tile["id"])
        else:
            tile["button"].config(image=self.images["numbers"][tile["mines"] - 1], relief="sunken")
            self.check_auto_open_neighbors(tile)
        
        if tile["state"] != STATE_CLICKED:
            tile["state"] = STATE_CLICKED
            self.clicked_count += 1
        
        # Check win condition
        if self.clicked_count == (self.size_x * self.size_y) - self.total_mines:
            self.game_over(True)

    def check_auto_open_neighbors(self, tile):
        x, y = tile["coords"]["x"], tile["coords"]["y"]
        neighbors = self.get_neighbors(x, y)
        flag_count = sum(1 for n in neighbors if n["state"] == STATE_FLAGGED)

        if flag_count == tile["mines"]:
            for n in neighbors:
                if n["state"] == STATE_DEFAULT:
                    self.on_click(n)

    
    def on_right_click(self, tile):
        if self.game_over_flag or tile["state"] == STATE_CLICKED:
            return
        
        # Start timer on first click
        if self.start_time is None:
            self.start_time = datetime.now()
        
        if tile["state"] == STATE_DEFAULT:
            # Flag tile
            tile["button"].config(image=self.images["flag"])
            tile["state"] = STATE_FLAGGED
            tile["button"].unbind(BTN_CLICK)
            self.flag_count += 1
            if tile["is_mine"]:
                self.correct_flag_count += 1
        else:  # STATE_FLAGGED
            # Unflag tile
            tile["button"].config(image=self.images["plain"])
            tile["state"] = STATE_DEFAULT
            tile["button"].bind(BTN_CLICK, self.on_click_wrapper(tile["coords"]["x"], tile["coords"]["y"]))
            self.flag_count -= 1
            if tile["is_mine"]:
                self.correct_flag_count -= 1
        
        self.refresh_labels()
    
    def clear_surrounding_tiles(self, tile_id):
        queue = deque([tile_id])
        
        while queue:
            current_id = queue.popleft()
            parts = current_id.split("_")
            x, y = int(parts[0]), int(parts[1])
            
            for neighbor in self.get_neighbors(x, y):
                self.clear_tile(neighbor, queue)
    
    def clear_tile(self, tile, queue):
        if tile["state"] != STATE_DEFAULT:
            return
        
        if tile["mines"] == 0:
            tile["button"].config(image=self.images["clicked"], relief="sunken")
            queue.append(tile["id"])
        else:
            tile["button"].config(image=self.images["numbers"][tile["mines"] - 1], relief="sunken")
        
        tile["state"] = STATE_CLICKED
        self.clicked_count += 1
    
    def open_settings(self):
        SettingsWindow(self.tk, self.settings, self.on_settings_changed)
    
    def on_settings_changed(self):
        self.apply_theme()
        self.restart()
    
    def open_stats(self):
        StatsWindow(self.tk, self.stats)


class MinesweeperAI:
    def __init__(self, game):
        self.game = game
        self.actions_taken = []

    def play(self):
        """AI tự động đưa ra hành động"""
        # Ví dụ đơn giản: chọn 1 ô chưa mở và chưa flag, rồi click
        for x in range(self.game.size_x):
            for y in range(self.game.size_y):
                tile = self.game.tiles[x][y]
                if tile["state"] == STATE_DEFAULT:
                    self.game.on_click(tile)
                    self.actions_taken.append(("click", x, y))
                    return  # chỉ thực hiện 1 hành động mỗi lần gọi

    def reset(self):
        self.actions_taken = []

def main():
    # Create main window
    window = Tk()
    window.title("Enhanced Minesweeper")
    window.resizable(False, False)
    
    # Center window on screen
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = (window.winfo_screenwidth() // 2) - (width // 2)
    y = (window.winfo_screenheight() // 2) - (height // 2)
    window.geometry(f"+{x}+{y}")
    
    # Create game instance
    minesweeper = Minesweeper(window)
    
    # Run event loop
    window.mainloop()

if __name__ == "__main__":
    main()