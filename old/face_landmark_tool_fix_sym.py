"""
face_landmark_tool.py
----------------------
Frontal-face landmark annotation tool.
- Multi-tab workspace (Google-Sheets-style tab bar at the bottom)
- Paste (Ctrl+V) or open an image per tab
- Sequential landmark prompts; bilateral points prompt L then R
- Click-and-drag to reposition placed markers
- Ctrl+scroll to zoom toward cursor; scroll to pan
- Export JSON/CSV + paired image copy; reload later

Dependencies:  pip install Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import json
import csv
import os
import math
from dataclasses import dataclass, field
from typing import Optional

try:
    from PIL import Image, ImageTk, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Landmark definitions
# ---------------------------------------------------------------------------

LANDMARK_DEFS: list[tuple[str, str, bool]] = [
    # ── Face outline ──────────────────────────────────────────────────
    ("face_outline_lip_crease",   "Side Face Outline - line up with lip crease",        True),
    ("face_under_ear",            "Side Face Under Ear - where ear and face meet",       True),
    ("face_above_ear",            "Side Face Above Ear - where ear and face meet",       True),
    ("cheekbone_outer",           "Cheekbone Outer - widest point of face at cheek",     True),   # NEW
    # ── Brows ─────────────────────────────────────────────────────────
    ("eyebrow_outside",           "Outside Eyebrow",                                     True),
    ("eyebrow_inside",            "Inside Eyebrow",                                      True),
    ("eyebrow_under_apex",        "Under Apex Eyebrow",                                  True),
    ("eyebrow_upper_apex",        "Upper Apex Eyebrow",                                  True),
    ("glabella",                  "Glabella - bridge of nose between brows (once)",      False),  # NEW
    # ── Eyes ──────────────────────────────────────────────────────────
    ("eye_upper_apex",            "Upper Apex Eye",                                      True),
    ("eye_upper_apex_crease",     "Upper Apex Eye Crease Line",                          True),
    ("eye_outside_corner",        "Outside Eye White Corner",                            True),
    ("eye_inside_corner",         "Inside Eye White Corner",                             True),
    ("eye_under_apex",            "Under Apex Eye",                                      True),
    # ── Nose ──────────────────────────────────────────────────────────
    ("nose_bottom_middle",        "Bottom Middle of Nose (once)",                        False),
    ("nose_nostril_outside",      "Side of Nose - Nostril Outside",                      True),
    ("alar_base",                 "Alar Base - where nose base meets face",               True),  # NEW
    # ── Chin / jaw ────────────────────────────────────────────────────
    ("chin_outer_side",           "Side of Outer Chin",                                  True),
    # ── Mouth / lips ──────────────────────────────────────────────────
    ("mouth_upper_apex_side",     "Upper Mouth Apex Side",                               True),
    ("philtrum_peak",             "Philtrum Peak - Cupid's Bow Peak",                    True),  # NEW
    ("mouth_upper_low_u",         "Upper Mouth Low U Apex (once)",                       False),
    ("lips_center_meet",          "Between Lips Where They Meet - Center (once)",        False),
    ("lips_outer_crease",         "Outer Crease of Lips",                                True),
    ("mouth_under_apex",          "Under Mouth Apex",                                    True),
    ("lip_bottom_center",         "Bottom of Lower Lip - Center (once)",                 False),  # NEW
    # ── Chin / neck ───────────────────────────────────────────────────
    ("chin_bottom_apex",          "Bottom of Chin Apex (once)",                          False),
    ("neck_face_corner",          "Neck Meets Face Corner",                              True),
]


def build_prompt_sequence() -> list[dict]:
    seq = []
    for key_base, label, bilateral in LANDMARK_DEFS:
        if bilateral:
            seq.append(dict(key=f"{key_base}_L", prompt_text=f"LEFT  .  {label}", side="L"))
            seq.append(dict(key=f"{key_base}_R", prompt_text=f"RIGHT .  {label}", side="R"))
        else:
            seq.append(dict(key=key_base, prompt_text=label, side=None))
    return seq


PROMPTS = build_prompt_sequence()
TOTAL   = len(PROMPTS)


# ---------------------------------------------------------------------------
# Colours / geometry
# ---------------------------------------------------------------------------

LEFT_COLOR      = "#00d4ff"
RIGHT_COLOR     = "#ff6b35"
SINGLE_COLOR    = "#a8ff3e"
ESTIMATED_COLOR = "#ffcc00"   # mirrored / auto-estimated point
SKIPPED_COLOR   = "#555566"   # point explicitly skipped
RADIUS          = 5
HIT_RADIUS      = 12

TAB_BG_ACTIVE   = "#1a1a2e"
TAB_BG_INACTIVE = "#0d0d1a"
TAB_FG_ACTIVE   = "#ffffff"
TAB_FG_INACTIVE = "#666688"
TAB_BAR_BG      = "#08080f"
TAB_BORDER      = "#2a2a4a"


# ---------------------------------------------------------------------------
# Workspace dataclass  (one per tab, holds all per-image state)
# ---------------------------------------------------------------------------

@dataclass
class Workspace:
    name:           str   = "Tab 1"
    original_image: object = None
    image_stem:     str   = "landmarks"
    landmarks:      dict  = field(default_factory=dict)   # key -> (x, y) — real + estimated
    skipped_keys:   set   = field(default_factory=set)    # keys user explicitly skipped
    estimated_keys: set   = field(default_factory=set)    # keys filled by mirror / fallback
    current_step:   int   = 0
    marking_mode:   bool  = False
    scale_factor:   float = 1.0
    zoom_str:       str   = "Fit"
    scroll_x:       float = 0.0
    scroll_y:       float = 0.0


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class FaceLandmarkApp(tk.Tk):

    APP_TITLE         = "Face Landmark Annotator"
    MIN_WIDTH         = 960
    MIN_HEIGHT        = 700
    PAN_SPEED         = 30
    ZOOM_STEP         = 1.15
    ZOOM_MIN          = 0.05
    ZOOM_MAX          = 8.0
    IMAGE_COPY_SUFFIX = "_image.png"

    def __init__(self):
        super().__init__()
        self.title(self.APP_TITLE)
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.geometry(f"{self.MIN_WIDTH}x{self.MIN_HEIGHT}")
        self.configure(bg="#1a1a2e")

        # Tab / workspace state
        self.workspaces:   list[Workspace] = [Workspace(name="Tab 1")]
        self.active_idx:   int             = 0

        # Display refs (rebuilt on every refresh, not stored per-workspace)
        self.display_image = None
        self.photo_image   = None

        # Drag state (transient)
        self._drag_key:    Optional[str]   = None
        self._drag_offset: tuple           = (0.0, 0.0)

        # Internal list of tab frame widgets (rebuilt by _refresh_tab_bar)
        self._tab_frames:  list[tk.Widget] = []

        self._build_styles()
        self._build_menu()
        self._build_ui()
        self._bind_shortcuts()
        self._refresh_tab_bar()

        if not PIL_AVAILABLE:
            messagebox.showwarning("Pillow not found",
                                   "Install Pillow:\n\n  pip install Pillow")

    # ── Quick accessor ────────────────────────────────────────────────
    @property
    def ws(self) -> Workspace:
        return self.workspaces[self.active_idx]

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("TFrame",         background="#1a1a2e")
        s.configure("TLabel",         background="#1a1a2e", foreground="#e0e0e0",
                    font=("Helvetica", 10))
        s.configure("TButton",        font=("Helvetica", 10), padding=(8, 4))
        s.configure("Action.TButton", font=("Helvetica", 11, "bold"))
        s.configure("TCombobox",      font=("Helvetica", 10))

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self):
        mb = tk.Menu(self, bg="#1a1a2e", fg="#e0e0e0", activebackground="#0f3460")

        fm = tk.Menu(mb, tearoff=False)
        fm.add_command(label="Open Image...",         accelerator="Ctrl+O", command=self.open_file)
        fm.add_command(label="Paste from Clipboard",  accelerator="Ctrl+V", command=self.paste_image)
        fm.add_command(label="Load Landmarks...",     accelerator="Ctrl+L", command=self.load_landmarks)
        fm.add_separator()
        fm.add_command(label="Export -> JSON",        command=self.export_json)
        fm.add_command(label="Export -> CSV",         command=self.export_csv)
        fm.add_separator()
        fm.add_command(label="Exit",                  command=self.quit)
        mb.add_cascade(label="File", menu=fm)

        em = tk.Menu(mb, tearoff=False)
        em.add_command(label="Start / Restart Marking", accelerator="F5",     command=self.start_marking)
        em.add_command(label="Undo Last Point",          accelerator="Ctrl+Z", command=self.undo_last)
        em.add_command(label="Clear All Landmarks",                            command=self.clear_landmarks)
        mb.add_cascade(label="Mark", menu=em)

        tm = tk.Menu(mb, tearoff=False)
        tm.add_command(label="New Tab",     accelerator="Ctrl+T", command=self.add_tab)
        tm.add_command(label="Rename Tab",                        command=self.rename_tab)
        tm.add_command(label="Close Tab",   accelerator="Ctrl+W", command=self.close_tab)
        mb.add_cascade(label="Tabs", menu=tm)

        hm = tk.Menu(mb, tearoff=False)
        hm.add_command(label="About", command=self.show_about)
        mb.add_cascade(label="Help", menu=hm)

        self.config(menu=mb)

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Toolbar ───────────────────────────────────────────────────
        toolbar = ttk.Frame(self, padding=(6, 3))
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="Open",   command=self.open_file      ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Paste",  command=self.paste_image    ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Load",   command=self.load_landmarks ).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="Start Marking",
                   style="Action.TButton", command=self.start_marking  ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Undo",   command=self.undo_last      ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Clear",  command=self.clear_landmarks).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="JSON",   command=self.export_json    ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="CSV",    command=self.export_csv     ).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="+ Tab",  command=self.add_tab        ).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.compare_btn = ttk.Button(toolbar, text="⚖ Compare",
                                      command=self.open_compare_dialog,
                                      style="Action.TButton")
        self.compare_btn.pack(side="left", padx=2)
        self.compare_btn.state(["disabled"])

        ttk.Label(toolbar, text="Zoom:").pack(side="right", padx=(4, 0))
        self.zoom_var = tk.StringVar(value="Fit")
        zoom_cb = ttk.Combobox(toolbar, textvariable=self.zoom_var,
                               values=["Fit","25%","50%","75%","100%","150%","200%","300%","400%"],
                               state="readonly", width=7)
        zoom_cb.pack(side="right", padx=4)
        zoom_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh_display())

        # ── Prompt banner ─────────────────────────────────────────────
        pb = tk.Frame(self, bg="#0f3460")
        pb.pack(side="top", fill="x")
        self.prompt_var = tk.StringVar(value="Open or paste an image, then press  Start Marking")
        tk.Label(pb, textvariable=self.prompt_var, bg="#0f3460", fg="#ffffff",
                 font=("Helvetica", 12, "bold"), pady=8, padx=14, anchor="w"
                 ).pack(side="left", fill="x", expand=True)
        self.step_var = tk.StringVar(value="")
        tk.Label(pb, textvariable=self.step_var, bg="#0f3460", fg="#aaaaaa",
                 font=("Helvetica", 10), padx=6).pack(side="right")
        self.skip_btn = tk.Button(pb, text="Skip  (S)", bg="#1a1a40", fg="#ffcc00",
                                  font=("Helvetica", 9, "bold"), relief="flat",
                                  padx=10, pady=6, cursor="hand2",
                                  command=self.skip_current_point,
                                  state="disabled")
        self.skip_btn.pack(side="right", padx=(0, 8))

        # ── Main area ─────────────────────────────────────────────────
        main_frame = ttk.Frame(self)
        main_frame.pack(side="top", fill="both", expand=True)

        # Canvas
        cf = ttk.Frame(main_frame)
        cf.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(cf, bg="#0d0d0d", cursor="crosshair",
                                highlightthickness=0,
                                xscrollincrement=self.PAN_SPEED,
                                yscrollincrement=self.PAN_SPEED)
        vsb = ttk.Scrollbar(cf, orient="vertical",   command=self.canvas.yview)
        hsb = ttk.Scrollbar(cf, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right",  fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind("<Button-1>",        self.on_canvas_click)
        self.canvas.bind("<B1-Motion>",       self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_release)
        self.canvas.bind("<Motion>",          self._on_mouse_move)
        self.canvas.bind("<Control-MouseWheel>",  self._on_ctrl_scroll)
        self.canvas.bind("<Control-Button-4>",    self._on_ctrl_scroll)
        self.canvas.bind("<Control-Button-5>",    self._on_ctrl_scroll)
        self.canvas.bind("<MouseWheel>",          self._on_pan_vertical)
        self.canvas.bind("<Button-4>",            self._on_pan_vertical)
        self.canvas.bind("<Button-5>",            self._on_pan_vertical)
        self.canvas.bind("<Shift-MouseWheel>",    self._on_pan_horizontal)
        self.canvas.bind("<Shift-Button-4>",      self._on_pan_horizontal)
        self.canvas.bind("<Shift-Button-5>",      self._on_pan_horizontal)

        # Side panel
        side = tk.Frame(main_frame, bg="#16213e", width=260)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        tk.Label(side, text="LANDMARKS", bg="#16213e", fg="#888888",
                 font=("Helvetica", 9, "bold"), pady=8).pack(anchor="w", padx=10)
        lf = tk.Frame(side, bg="#16213e")
        lf.pack(fill="both", expand=True, padx=4)
        ls = ttk.Scrollbar(lf, orient="vertical")
        self.landmark_listbox = tk.Listbox(
            lf, yscrollcommand=ls.set, bg="#16213e", fg="#cccccc",
            selectbackground="#0f3460", font=("Courier", 9),
            borderwidth=0, highlightthickness=0, activestyle="none")
        ls.config(command=self.landmark_listbox.yview)
        ls.pack(side="right", fill="y")
        self.landmark_listbox.pack(side="left", fill="both", expand=True)

        # ── Status bar ────────────────────────────────────────────────
        status = tk.Frame(self, bg="#0d0d0d", pady=3)
        status.pack(side="bottom", fill="x")
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(status, textvariable=self.status_var, bg="#0d0d0d", fg="#888888",
                 font=("Helvetica", 9), anchor="w", padx=8).pack(side="left")
        self.coord_var = tk.StringVar(value="")
        tk.Label(status, textvariable=self.coord_var, bg="#0d0d0d", fg="#555555",
                 font=("Courier", 9), anchor="e", padx=8).pack(side="right")

        # ── Tab bar (above status bar, below canvas) ──────────────────
        self.tab_bar = tk.Frame(self, bg=TAB_BAR_BG, height=34)
        self.tab_bar.pack(side="bottom", fill="x", before=status)
        self.tab_bar.pack_propagate(False)

        self._draw_placeholder()

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _bind_shortcuts(self):
        self.bind("<Control-v>", lambda _e: self.paste_image())
        self.bind("<Control-V>", lambda _e: self.paste_image())
        self.bind("<Control-o>", lambda _e: self.open_file())
        self.bind("<Control-l>", lambda _e: self.load_landmarks())
        self.bind("<Control-z>", lambda _e: self.undo_last())
        self.bind("<Control-t>", lambda _e: self.add_tab())
        self.bind("<Control-w>", lambda _e: self.close_tab())
        self.bind("<F5>",        lambda _e: self.start_marking())
        self.bind("<s>",         lambda _e: self.skip_current_point())
        self.bind("<S>",         lambda _e: self.skip_current_point())

    # ------------------------------------------------------------------
    # Tab bar
    # ------------------------------------------------------------------

    def _refresh_tab_bar(self):
        """Rebuild all tab widgets from scratch."""
        for w in self._tab_frames:
            w.destroy()
        self._tab_frames.clear()

        for idx, ws in enumerate(self.workspaces):
            is_active = (idx == self.active_idx)
            frame = self._make_tab_frame(idx, ws.name, is_active)
            self._tab_frames.append(frame)

        # (+) button at the far right
        add = tk.Label(self.tab_bar, text="  +  ", bg=TAB_BAR_BG, fg="#445566",
                       font=("Helvetica", 13, "bold"), padx=2, pady=5, cursor="hand2")
        add.pack(side="left")
        add.bind("<Button-1>", lambda _e: self.add_tab())
        add.bind("<Enter>",    lambda _e: add.config(fg="#88aacc"))
        add.bind("<Leave>",    lambda _e: add.config(fg="#445566"))
        self._tab_frames.append(add)

        # Enable Compare only when >= 2 tabs exist
        if hasattr(self, "compare_btn"):
            tabs_with_landmarks = [w for w in self.workspaces if w.landmarks]
            if len(self.workspaces) >= 2 and len(tabs_with_landmarks) >= 2:
                self.compare_btn.state(["!disabled"])
            else:
                self.compare_btn.state(["disabled"])

    def _make_tab_frame(self, idx: int, name: str, active: bool) -> tk.Frame:
        bg = TAB_BG_ACTIVE   if active else TAB_BG_INACTIVE
        fg = TAB_FG_ACTIVE   if active else TAB_FG_INACTIVE
        relief = "flat"

        outer = tk.Frame(self.tab_bar, bg=TAB_BORDER, padx=1, pady=1)
        outer.pack(side="left", padx=(2, 0), pady=(4, 0) if active else (6, 0))

        inner = tk.Frame(outer, bg=bg)
        inner.pack()

        lbl = tk.Label(inner, text=f"  {name}  ", bg=bg, fg=fg,
                       font=("Helvetica", 9, "bold" if active else "normal"),
                       pady=4, cursor="hand2")
        lbl.pack(side="left")

        close_col = "#555577" if active else "#333355"
        close = tk.Label(inner, text=" x ", bg=bg, fg=close_col,
                         font=("Helvetica", 9), pady=4, padx=2, cursor="hand2")
        close.pack(side="left")

        # Click to switch
        for w in (outer, inner, lbl):
            w.bind("<Button-1>",       lambda _e, i=idx: self.switch_tab(i))
            w.bind("<Double-Button-1>", lambda _e, i=idx: self.rename_tab(i))
        # Close
        close.bind("<Button-1>", lambda _e, i=idx: self.close_tab(i))
        close.bind("<Enter>",    lambda _e: close.config(fg="#ff4455"))
        close.bind("<Leave>",    lambda _e: close.config(fg=close_col))

        return outer

    # ------------------------------------------------------------------
    # Tab actions
    # ------------------------------------------------------------------

    def add_tab(self):
        n = len(self.workspaces) + 1
        self._snapshot_current()
        self.workspaces.append(Workspace(name=f"Tab {n}"))
        self.active_idx = len(self.workspaces) - 1
        self._restore_workspace(self.active_idx)

    def close_tab(self, idx: int = None):
        if idx is None:
            idx = self.active_idx
        if len(self.workspaces) == 1:
            messagebox.showinfo("Cannot close", "At least one tab must remain."); return
        self.workspaces.pop(idx)
        new_idx = min(idx, len(self.workspaces) - 1)
        self.active_idx = new_idx
        self._restore_workspace(new_idx)

    def rename_tab(self, idx: int = None):
        if idx is None:
            idx = self.active_idx
        name = simpledialog.askstring("Rename Tab", "New name:",
                                      initialvalue=self.workspaces[idx].name, parent=self)
        if name and name.strip():
            self.workspaces[idx].name = name.strip()
            self._refresh_tab_bar()

    def switch_tab(self, idx: int):
        if idx == self.active_idx: return
        self._snapshot_current()
        self.active_idx = idx
        self._restore_workspace(idx)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------

    def _snapshot_current(self):
        ws = self.ws
        ws.zoom_str = self.zoom_var.get()
        try:
            ws.scroll_x = self.canvas.xview()[0]
            ws.scroll_y = self.canvas.yview()[0]
        except Exception:
            pass

    def _restore_workspace(self, idx: int):
        ws = self.workspaces[idx]
        self.active_idx  = idx
        self._drag_key   = None
        self.display_image = None
        self.photo_image   = None

        self.zoom_var.set(ws.zoom_str)

        # Repopulate landmark list
        self.landmark_listbox.delete(0, tk.END)
        for k, (x, y) in ws.landmarks.items():
            self.landmark_listbox.insert(tk.END, f"{k:<28}  ({x:>4}, {y:>4})")
            side = k.split("_")[-1]
            c = LEFT_COLOR if side=="L" else RIGHT_COLOR if side=="R" else SINGLE_COLOR
            self.landmark_listbox.itemconfig(tk.END, fg=c)

        # Restore prompt banner
        if ws.original_image is None:
            self.prompt_var.set("Open or paste an image, then press  Start Marking")
            self.step_var.set("")
        elif ws.current_step >= TOTAL:
            self.prompt_var.set(f"  All {TOTAL} landmarks recorded.  Export with JSON or CSV")
            self.step_var.set(f"{TOTAL} / {TOTAL}  complete")
        elif ws.marking_mode:
            p = PROMPTS[ws.current_step]
            side_tag = {"L": "  [LEFT]", "R": "  [RIGHT]", None: ""}[p["side"]]
            self.prompt_var.set(f"  Click ->  {p['prompt_text']}{side_tag}")
            self.step_var.set(f"Step {ws.current_step + 1} / {TOTAL}")
        else:
            self.prompt_var.set("Image loaded.  Press  Start Marking  to begin.")
            self.step_var.set("")

        self.canvas.config(cursor="crosshair" if ws.marking_mode else "arrow")
        self._refresh_tab_bar()

        if ws.original_image is None:
            self._draw_placeholder()
        else:
            self.refresh_display()
            self.after(40, lambda: (
                self.canvas.xview_moveto(ws.scroll_x),
                self.canvas.yview_moveto(ws.scroll_y),
            ))

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def paste_image(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow required", "pip install Pillow"); return
        try:
            img = ImageGrab.grabclipboard()
        except Exception as exc:
            messagebox.showerror("Clipboard error", str(exc)); return
        if img is None:
            self.status_var.set("Nothing on clipboard."); return
        if isinstance(img, list):
            for p in img:
                try: img = Image.open(p); break
                except Exception: pass
            else:
                self.status_var.set("No image on clipboard."); return
        self._load_image(img, "Clipboard", stem="clipboard_image")

    def open_file(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow required", "pip install Pillow"); return
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image files","*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff"),("All","*.*")])
        if not path: return
        try:
            self._load_image(Image.open(path), os.path.basename(path),
                             stem=os.path.splitext(os.path.basename(path))[0])
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))

    def _load_image(self, img: "Image.Image", source: str = "", stem: str = "landmarks"):
        ws = self.ws
        ws.original_image = img.copy()
        ws.image_stem     = stem
        if ws.name.startswith("Tab "):           # auto-name tab from file stem
            ws.name = stem[:22]
        self.clear_landmarks(silent=True)
        ws.marking_mode = False
        ws.current_step = 0
        ws.zoom_str     = "Fit"
        self.zoom_var.set("Fit")
        self.prompt_var.set("Image loaded.  Press  Start Marking  to begin.")
        self.step_var.set("")
        self.status_var.set(f"Loaded: {source}  |  {img.size[0]} x {img.size[1]} px")
        self._refresh_tab_bar()
        self.refresh_display()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def refresh_display(self, new_scale: float = None):
        ws = self.ws
        if ws.original_image is None:
            self._draw_placeholder(); return

        iw, ih = ws.original_image.size

        if new_scale is not None:
            ws.scale_factor = new_scale
            self.zoom_var.set(f"{round(new_scale * 100)}%")
            ws.zoom_str = self.zoom_var.get()
        else:
            zoom_str = self.zoom_var.get()
            cw = max(self.canvas.winfo_width(),  self.MIN_WIDTH  - 260)
            ch = max(self.canvas.winfo_height(), self.MIN_HEIGHT - 100)
            if zoom_str == "Fit":
                ws.scale_factor = min(cw / iw, ch / ih)
            else:
                try:
                    ws.scale_factor = int(zoom_str.rstrip("%")) / 100
                except ValueError:
                    ws.scale_factor = 1.0

        sf = ws.scale_factor
        nw = max(1, int(iw * sf))
        nh = max(1, int(ih * sf))

        self.display_image = ws.original_image.resize((nw, nh), Image.LANCZOS)
        self.photo_image   = ImageTk.PhotoImage(self.display_image)

        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, nw, nh))
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image, tags="image")

        for key, (ox, oy) in ws.landmarks.items():
            cx    = int(ox * sf)
            cy    = int(oy * sf)
            is_est = key in ws.estimated_keys
            color  = ESTIMATED_COLOR if is_est else _marker_color(key)
            self._draw_marker(cx, cy, key, color, estimated=is_est)

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.MIN_WIDTH // 2, self.MIN_HEIGHT // 2,
            text="Press  Ctrl+V  to paste an image\nor use  Open",
            fill="#333355", font=("Helvetica", 16), justify="center")

    def _draw_marker(self, cx: int, cy: int, key: str, color: str,
                     highlight: bool = False, estimated: bool = False):
        r   = RADIUS
        tag = f"mk_{key}"
        self.canvas.delete(tag)
        if estimated:
            # hollow circle with dashed outline = estimated/mirrored point
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                     fill="", outline=color,
                                     width=2, dash=(3, 2),
                                     tags=("marker", "estimated", tag))
            self.canvas.create_text(cx + r + 4, cy, text=f"~{_short_label(key)}",
                                     fill=color, font=("Helvetica", 7), anchor="w",
                                     tags=("marker", "estimated", tag))
        else:
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                     fill=color,
                                     outline="#ffff00" if highlight else "#ffffff",
                                     width=2 if highlight else 1,
                                     tags=("marker", tag))
            self.canvas.create_text(cx + r + 4, cy, text=_short_label(key), fill=color,
                                     font=("Helvetica", 7), anchor="w",
                                     tags=("marker", tag))

    # ------------------------------------------------------------------
    # Scroll / zoom
    # ------------------------------------------------------------------

    def _on_pan_vertical(self, event):
        if   event.num == 4: self.canvas.yview_scroll(-1, "units")
        elif event.num == 5: self.canvas.yview_scroll(1,  "units")
        else:                self.canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _on_pan_horizontal(self, event):
        if   event.num == 4: self.canvas.xview_scroll(-1, "units")
        elif event.num == 5: self.canvas.xview_scroll(1,  "units")
        else:                self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")

    def _on_ctrl_scroll(self, event):
        if self.ws.original_image is None: return
        if   event.num == 4: direction = 1
        elif event.num == 5: direction = -1
        else:                direction = 1 if event.delta > 0 else -1

        old = self.ws.scale_factor
        new = max(self.ZOOM_MIN, min(self.ZOOM_MAX,
                  old * (self.ZOOM_STEP if direction > 0 else 1 / self.ZOOM_STEP)))
        if abs(new - old) < 1e-9: return

        mx = self.canvas.canvasx(event.x)
        my = self.canvas.canvasy(event.y)
        ix, iy = mx / old, my / old

        self.refresh_display(new_scale=new)

        iw, ih = self.ws.original_image.size
        ncw, nch = max(1, int(iw * new)), max(1, int(ih * new))
        self.canvas.xview_moveto(max(0.0, (ix * new - event.x) / ncw))
        self.canvas.yview_moveto(max(0.0, (iy * new - event.y) / nch))

    # ------------------------------------------------------------------
    # Marker hit-test
    # ------------------------------------------------------------------

    def _hit_test(self, cx: float, cy: float) -> Optional[str]:
        best_key, best_dist = None, HIT_RADIUS
        sf = self.ws.scale_factor
        for key, (ox, oy) in self.ws.landmarks.items():
            d = ((cx - ox*sf)**2 + (cy - oy*sf)**2)**0.5
            if d < best_dist:
                best_dist, best_key = d, key
        return best_key

    # ------------------------------------------------------------------
    # Canvas interaction
    # ------------------------------------------------------------------

    def on_canvas_click(self, event):
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # Drag existing marker?
        hit = self._hit_test(cx, cy)
        if hit is not None:
            self._drag_key    = hit
            ox, oy            = self.ws.landmarks[hit]
            self._drag_offset = (cx - ox*self.ws.scale_factor,
                                 cy - oy*self.ws.scale_factor)
            self._draw_marker(int(ox*self.ws.scale_factor), int(oy*self.ws.scale_factor),
                              hit, _marker_color(hit), highlight=True)
            self.canvas.config(cursor="fleur")
            return

        # Place new landmark
        self._drag_key = None
        if not self.ws.marking_mode: return
        if self.ws.current_step >= TOTAL: return

        ox = int(cx / self.ws.scale_factor)
        oy = int(cy / self.ws.scale_factor)
        p  = PROMPTS[self.ws.current_step]
        self.ws.landmarks[p["key"]] = (ox, oy)
        self._draw_marker(int(cx), int(cy), p["key"], _marker_color(p["key"]))
        self._update_list(p["key"], ox, oy, status="placed")
        self.status_var.set(f"  {p['prompt_text']}  ->  ({ox}, {oy})")
        self.ws.current_step += 1
        self._advance_prompt()

    def on_drag_motion(self, event):
        if self._drag_key is None: return
        key = self._drag_key
        cx  = self.canvas.canvasx(event.x) - self._drag_offset[0]
        cy  = self.canvas.canvasy(event.y) - self._drag_offset[1]
        if self.ws.original_image:
            iw, ih = self.ws.original_image.size
            sf = self.ws.scale_factor
            cx = max(0.0, min(cx, (iw-1)*sf))
            cy = max(0.0, min(cy, (ih-1)*sf))
        ox = int(cx / self.ws.scale_factor)
        oy = int(cy / self.ws.scale_factor)
        self.ws.landmarks[key] = (ox, oy)
        self._draw_marker(int(cx), int(cy), key, _marker_color(key), highlight=True)
        self._refresh_list_entry(key, ox, oy)
        self.coord_var.set(f"x={ox}  y={oy}")

    def on_drag_release(self, event):
        if self._drag_key is None: return
        key    = self._drag_key
        ox, oy = self.ws.landmarks[key]

        # If the user manually dragged an estimated point, promote it to placed
        was_estimated = key in self.ws.estimated_keys
        if was_estimated:
            self.ws.estimated_keys.discard(key)

        color = _marker_color(key)   # real colour now it's placed
        self._draw_marker(int(ox*self.ws.scale_factor), int(oy*self.ws.scale_factor),
                          key, color, highlight=False)
        self.status_var.set(f"Moved  {key}  ->  ({ox}, {oy})"
                            + ("  [promoted to placed]" if was_estimated else ""))
        self._drag_key = None
        self.canvas.config(cursor="crosshair" if self.ws.marking_mode else "arrow")

        # Re-run estimation: any estimated points that depend on the moved
        # source point (or the newly promoted point) need to be recomputed.
        if self.ws.skipped_keys or self.ws.estimated_keys:
            self._recompute_estimated()

    # ------------------------------------------------------------------
    # Live re-estimation after drag
    # ------------------------------------------------------------------

    def _recompute_estimated(self):
        """
        Clear all auto-estimated landmarks and re-derive them from the
        current placed points using the tilt-aware axis.
        Called automatically after any drag-release.
        """
        ws = self.ws

        # Remove previously estimated points from landmarks dict
        for k in list(ws.estimated_keys):
            ws.landmarks.pop(k, None)
        ws.estimated_keys.clear()

        # Re-run estimation (modifies ws.landmarks + ws.estimated_keys)
        self._compute_estimated_points()

        # Redraw the canvas so estimated markers reflect new positions
        self.refresh_display()

        # Sync side-panel list
        self._rebuild_list_panel()

    def _rebuild_list_panel(self):
        """Repopulate the landmark side panel from the current workspace state,
        in prompt order, respecting placed / skipped / estimated status."""
        ws = self.ws
        self.landmark_listbox.delete(0, tk.END)
        for p in PROMPTS:
            k = p["key"]
            if k in ws.skipped_keys and k not in ws.landmarks:
                self.landmark_listbox.insert(tk.END, f"{k:<28}  SKIPPED")
                self.landmark_listbox.itemconfig(tk.END, fg=SKIPPED_COLOR)
            elif k in ws.estimated_keys:
                x, y = ws.landmarks[k]
                self.landmark_listbox.insert(tk.END, f"{k:<28}  ~({x:>4}, {y:>4})")
                self.landmark_listbox.itemconfig(tk.END, fg=ESTIMATED_COLOR)
            elif k in ws.landmarks:
                x, y = ws.landmarks[k]
                self.landmark_listbox.insert(tk.END, f"{k:<28}  ({x:>4}, {y:>4})")
                side = k.split("_")[-1]
                c = LEFT_COLOR if side=="L" else RIGHT_COLOR if side=="R" else SINGLE_COLOR
                self.landmark_listbox.itemconfig(tk.END, fg=c)

    # ------------------------------------------------------------------
    # Marking mode
    # ------------------------------------------------------------------

    def start_marking(self):
        if self.ws.original_image is None:
            messagebox.showinfo("No image", "Load an image first."); return
        self.ws.marking_mode  = True
        self.ws.current_step  = 0
        self.ws.landmarks.clear()
        self.ws.skipped_keys.clear()
        self.ws.estimated_keys.clear()
        self.landmark_listbox.delete(0, tk.END)
        self.canvas.delete("marker")
        self._advance_prompt()
        self.canvas.config(cursor="crosshair")

    def _advance_prompt(self):
        if self.ws.current_step >= TOTAL:
            self._finish_marking(); return
        p        = PROMPTS[self.ws.current_step]
        side_tag = {"L": "  [LEFT]", "R": "  [RIGHT]", None: ""}[p["side"]]
        self.prompt_var.set(f"  Click ->  {p['prompt_text']}{side_tag}")
        self.step_var.set(f"Step {self.ws.current_step + 1} / {TOTAL}")
        self.skip_btn.config(state="normal")

    def skip_current_point(self):
        """Mark current prompt as skipped (point not visible) and advance."""
        if not self.ws.marking_mode: return
        if self.ws.current_step >= TOTAL: return
        p   = PROMPTS[self.ws.current_step]
        key = p["key"]
        self.ws.skipped_keys.add(key)
        # Add skipped entry to side-panel list
        self.landmark_listbox.insert(tk.END, f"{key:<28}  SKIPPED")
        self.landmark_listbox.itemconfig(tk.END, fg=SKIPPED_COLOR)
        self.landmark_listbox.see(tk.END)
        self.status_var.set(f"Skipped:  {key}  (will attempt mirror estimate)")
        self.ws.current_step += 1
        self._advance_prompt()

    def _finish_marking(self):
        self.ws.marking_mode = False
        self.skip_btn.config(state="disabled")
        # Attempt to fill skipped/missing points by mirroring + fallback
        n_estimated = self._compute_estimated_points()
        skipped = len(self.ws.skipped_keys)
        placed  = len([k for k in self.ws.landmarks
                       if k not in self.ws.estimated_keys])
        msg = f"Marking complete.\n\nPlaced: {placed}  |  Skipped: {skipped}"
        if n_estimated:
            msg += f"  |  Auto-estimated: {n_estimated}"
        self.prompt_var.set(
            f"  Marking complete — {placed} placed, {skipped} skipped, "
            f"{n_estimated} estimated.  Export with JSON or CSV")
        self.step_var.set(f"{TOTAL} / {TOTAL}  complete")
        self.canvas.config(cursor="arrow")
        self.refresh_display()
        messagebox.showinfo("Done", msg + "\n\nExport via the buttons or File menu.")

    def _update_list(self, key: str, ox: int, oy: int, status: str = "placed"):
        if status == "skipped":
            self.landmark_listbox.insert(tk.END, f"{key:<28}  SKIPPED")
            self.landmark_listbox.itemconfig(tk.END, fg=SKIPPED_COLOR)
        elif status == "estimated":
            self.landmark_listbox.insert(tk.END, f"{key:<28}  ~({ox:>4}, {oy:>4})")
            self.landmark_listbox.itemconfig(tk.END, fg=ESTIMATED_COLOR)
        else:
            self.landmark_listbox.insert(tk.END, f"{key:<28}  ({ox:>4}, {oy:>4})")
            side = key.split("_")[-1]
            c = LEFT_COLOR if side=="L" else RIGHT_COLOR if side=="R" else SINGLE_COLOR
            self.landmark_listbox.itemconfig(tk.END, fg=c)
        self.landmark_listbox.see(tk.END)

    def _refresh_list_entry(self, key: str, ox: int, oy: int):
        keys = [p["key"] for p in PROMPTS]
        if key not in keys: return
        idx = keys.index(key)
        if idx >= self.landmark_listbox.size(): return
        color = self.landmark_listbox.itemcget(idx, "fg")
        self.landmark_listbox.delete(idx)
        is_est = key in self.ws.estimated_keys
        txt = f"{key:<28}  ~({ox:>4}, {oy:>4})" if is_est else f"{key:<28}  ({ox:>4}, {oy:>4})"
        self.landmark_listbox.insert(idx, txt)
        self.landmark_listbox.itemconfig(idx, fg=color)

    # ------------------------------------------------------------------
    # Undo / clear
    # ------------------------------------------------------------------

    def undo_last(self):
        if self.ws.current_step == 0: return
        self.ws.current_step -= 1
        key = PROMPTS[self.ws.current_step]["key"]
        self.ws.landmarks.pop(key, None)
        self.ws.skipped_keys.discard(key)
        self.ws.estimated_keys.discard(key)
        if self.landmark_listbox.size() > 0:
            self.landmark_listbox.delete(tk.END)
        self.refresh_display()
        if not self.ws.marking_mode:
            self.ws.marking_mode = True
        self.skip_btn.config(state="normal")
        self._advance_prompt()
        self.status_var.set(f"Undid:  {key}")

    def clear_landmarks(self, silent: bool = False):
        self.ws.landmarks.clear()
        self.ws.skipped_keys.clear()
        self.ws.estimated_keys.clear()
        self.ws.current_step = 0
        self.ws.marking_mode = False
        self.landmark_listbox.delete(0, tk.END)
        self.canvas.delete("marker")
        self.skip_btn.config(state="disabled")
        self.prompt_var.set("Landmarks cleared.  Press  Start Marking  to begin.")
        self.step_var.set("")
        if not silent:
            self.status_var.set("All landmarks cleared.")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _save_image_copy(self, export_path: str) -> str:
        stem     = os.path.splitext(export_path)[0]
        img_path = stem + self.IMAGE_COPY_SUFFIX
        self.ws.original_image.save(img_path, format="PNG")
        return img_path

    def export_json(self):
        if not self.ws.landmarks and not self.ws.skipped_keys:
            messagebox.showinfo("Nothing to export", "No landmarks recorded yet."); return
        path = filedialog.asksaveasfilename(
            title="Save landmarks as JSON",
            initialfile=f"{self.ws.image_stem}.json",
            defaultextension=".json",
            filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return
        img_path = self._save_image_copy(path)
        payload  = {
            "total_landmarks": len(self.ws.landmarks),
            "image_size":      list(self.ws.original_image.size),
            "paired_image":    os.path.basename(img_path),
            "skipped_keys":    list(self.ws.skipped_keys),
            "estimated_keys":  list(self.ws.estimated_keys),
            "landmarks":       {k: {"x": v[0], "y": v[1],
                                    "status": ("estimated" if k in self.ws.estimated_keys
                                               else "placed")}
                                for k, v in self.ws.landmarks.items()},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        self.status_var.set(
            f"Exported {os.path.basename(path)}  +  {os.path.basename(img_path)}")

    def export_csv(self):
        if not self.ws.landmarks and not self.ws.skipped_keys:
            messagebox.showinfo("Nothing to export", "No landmarks recorded yet."); return
        path = filedialog.asksaveasfilename(
            title="Save landmarks as CSV",
            initialfile=f"{self.ws.image_stem}.csv",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        img_path = self._save_image_copy(path)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["# paired_image",  os.path.basename(img_path)])
            writer.writerow(["# skipped_keys",  "|".join(sorted(self.ws.skipped_keys))])
            writer.writerow(["# estimated_keys", "|".join(sorted(self.ws.estimated_keys))])
            writer.writerow(["key", "x", "y", "status"])
            for k, (x, y) in self.ws.landmarks.items():
                st = "estimated" if k in self.ws.estimated_keys else "placed"
                writer.writerow([k, x, y, st])
            for k in self.ws.skipped_keys:
                if k not in self.ws.landmarks:
                    writer.writerow([k, "", "", "skipped"])
        self.status_var.set(
            f"Exported {os.path.basename(path)}  +  {os.path.basename(img_path)}")

    # ------------------------------------------------------------------
    # Load landmarks
    # ------------------------------------------------------------------

    def load_landmarks(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow required", "pip install Pillow"); return
        path = filedialog.askopenfilename(
            title="Load Landmarks File",
            filetypes=[("Landmark files","*.json *.csv"),
                       ("JSON","*.json"),("CSV","*.csv"),("All","*.*")])
        if not path: return

        ext    = os.path.splitext(path)[1].lower()
        folder = os.path.dirname(path)
        try:
            if   ext == ".json": loaded, hint, skipped, estimated = self._parse_json(path)
            elif ext == ".csv":  loaded, hint, skipped, estimated = self._parse_csv(path)
            else:
                messagebox.showerror("Unknown format", "Open a .json or .csv file."); return
        except Exception as exc:
            messagebox.showerror("Parse error", str(exc)); return

        img_path = self._find_paired_image(folder, path, hint)
        if img_path is None:
            messagebox.showerror("Image not found",
                f"Could not find the paired image for:\n{os.path.basename(path)}\n\n"
                "Make sure the image copy is in the same folder.")
            return
        try:
            img = Image.open(img_path)
        except Exception as exc:
            messagebox.showerror("Image load failed", str(exc)); return

        stem = os.path.splitext(os.path.basename(path))[0]
        self._load_image(img, os.path.basename(img_path), stem=stem)
        self.ws.landmarks      = loaded
        self.ws.skipped_keys   = skipped
        self.ws.estimated_keys = estimated
        self.ws.current_step   = TOTAL  # treat loaded file as complete
        self.landmark_listbox.delete(0, tk.END)
        # Re-populate list panel in prompt order
        for p in PROMPTS:
            k = p["key"]
            if k in skipped:
                self.landmark_listbox.insert(tk.END, f"{k:<28}  SKIPPED")
                self.landmark_listbox.itemconfig(tk.END, fg=SKIPPED_COLOR)
            elif k in estimated:
                x, y = loaded[k]
                self.landmark_listbox.insert(tk.END, f"{k:<28}  ~({x:>4}, {y:>4})")
                self.landmark_listbox.itemconfig(tk.END, fg=ESTIMATED_COLOR)
            elif k in loaded:
                x, y = loaded[k]
                self.landmark_listbox.insert(tk.END, f"{k:<28}  ({x:>4}, {y:>4})")
                side = k.split("_")[-1]
                c = LEFT_COLOR if side=="L" else RIGHT_COLOR if side=="R" else SINGLE_COLOR
                self.landmark_listbox.itemconfig(tk.END, fg=c)
        self.refresh_display()
        placed_n = len([k for k in loaded if k not in estimated])
        est_n    = len(estimated)
        skip_n   = len(skipped)
        self.prompt_var.set(
            f"  Loaded — {placed_n} placed, {skip_n} skipped, {est_n} estimated.")
        self.step_var.set(f"{TOTAL} / {TOTAL}  complete")
        self.status_var.set(
            f"Loaded {os.path.basename(path)}  +  {os.path.basename(img_path)}")

    def _find_paired_image(self, folder, lm_path, hint):
        if hint:
            c = os.path.join(folder, hint)
            if os.path.isfile(c): return c
        stem = os.path.splitext(os.path.basename(lm_path))[0]
        for ext in (".png",".jpg",".jpeg",".bmp",".webp",".tiff"):
            c = os.path.join(folder, stem + ext)
            if os.path.isfile(c): return c
        c = os.path.join(folder, stem + self.IMAGE_COPY_SUFFIX)
        if os.path.isfile(c): return c
        return None

    def _parse_json(self, path):
        with open(path) as f:
            data = json.load(f)
        raw   = data.get("landmarks", {})
        loaded = {k: (v["x"], v["y"]) for k, v in raw.items()
                  if v.get("x") is not None}
        skipped   = set(data.get("skipped_keys",  []))
        estimated = set(k for k, v in raw.items()
                        if v.get("status") == "estimated")
        estimated |= set(data.get("estimated_keys", []))
        return loaded, data.get("paired_image"), skipped, estimated

    def _parse_csv(self, path):
        loaded, hint, skipped, estimated = {}, None, set(), set()
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row: continue
                if row[0].startswith("#"):
                    if len(row) >= 2:
                        if "paired_image"  in row[0]: hint      = row[1].strip()
                        if "skipped_keys"  in row[0]: skipped   = set(x for x in row[1].strip().split("|") if x)
                        if "estimated_keys" in row[0]: estimated = set(x for x in row[1].strip().split("|") if x)
                    continue
                if row[0] == "key": continue
                if len(row) >= 4 and row[3] == "skipped":
                    skipped.add(row[0]); continue
                if len(row) >= 2 and row[1]:
                    st = row[3] if len(row) >= 4 else "placed"
                    loaded[row[0]] = (int(row[1]), int(row[2]))
                    if st == "estimated":
                        estimated.add(row[0])
        return loaded, hint, skipped, estimated

    # ------------------------------------------------------------------
    # Mouse move
    # ------------------------------------------------------------------

    def _on_mouse_move(self, event):
        if self.ws.original_image is None: return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ox = int(cx / self.ws.scale_factor)
        oy = int(cy / self.ws.scale_factor)
        iw, ih = self.ws.original_image.size
        self.coord_var.set(f"x={ox}  y={oy}" if 0 <= ox < iw and 0 <= oy < ih else "")
        if self._drag_key is None:
            hit = self._hit_test(cx, cy)
            self.canvas.config(cursor="fleur" if hit
                               else "crosshair" if self.ws.marking_mode else "arrow")

    # ------------------------------------------------------------------
    # Estimation engine  (mirror + fallback for skipped points)
    # ------------------------------------------------------------------

    def _compute_estimated_points(self) -> int:
        """
        For each skipped/missing landmark:
          1. Bilateral: reflect opposite side across the face's PCA axis
             (accounts for head tilt — uses all placed points to fit the axis)
          2. Single: estimate from neighbouring available points
        Stores results in ws.landmarks + ws.estimated_keys.
        Returns number of new estimates added.
        """
        ws = self.ws
        n_before = len(ws.estimated_keys)

        axis = _face_axis(ws.landmarks)   # (cx, cy, dx, dy) or None

        # ── Pass 1: mirror bilateral skipped points across face axis ──
        for key in list(ws.skipped_keys):
            if key in ws.landmarks:
                continue
            mirror_key = _mirror_key(key)
            if (mirror_key and mirror_key in ws.landmarks
                    and mirror_key not in ws.skipped_keys
                    and axis is not None):
                ax, ay, dx, dy = axis
                ox, oy = ws.landmarks[mirror_key]
                ws.landmarks[key] = _mirror_across_axis(ox, oy, ax, ay, dx, dy)
                ws.estimated_keys.add(key)

        # ── Pass 2: estimate single (non-bilateral) skipped points ────
        _estimate_singles(ws.landmarks, ws.estimated_keys)

        return len(ws.estimated_keys) - n_before

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def show_about(self):
        messagebox.showinfo("About", (
            "Face Landmark Annotator\n\n"
            f"{TOTAL} frontal-face landmarks\n\n"
            "Ctrl+V / Open     load image into current tab\n"
            "Ctrl+L            load landmarks + paired image\n"
            "F5                start marking\n"
            "Ctrl+Z            undo last point\n"
            "Ctrl+T            new tab\n"
            "Ctrl+W            close tab\n"
            "Double-click tab  rename tab\n\n"
            "Scroll            pan up/down\n"
            "Shift+Scroll      pan left/right\n"
            "Ctrl+Scroll       zoom toward cursor\n\n"
            "Requires: Pillow  (pip install Pillow)"
        ))

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------

    def open_compare_dialog(self):
        """Open a small dialog asking the user to pick two tabs to compare."""
        tabs_with_lm = [(i, ws) for i, ws in enumerate(self.workspaces) if ws.landmarks]
        if len(tabs_with_lm) < 2:
            messagebox.showinfo("Compare", "You need at least two tabs with landmarks to compare.")
            return
        CompareSelectDialog(self, tabs_with_lm)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _marker_color(key: str) -> str:
    if key.endswith("_L"): return LEFT_COLOR
    if key.endswith("_R"): return RIGHT_COLOR
    return SINGLE_COLOR


def _short_label(key: str) -> str:
    parts = key.split("_")
    if parts[-1] in ("L", "R"):
        parts = parts[:-1]
    return "".join(p[0].upper() for p in parts[:4])



# ===========================================================================
# Landmark estimation helpers
# ===========================================================================

def _mirror_key(key: str) -> Optional[str]:
    """Return the bilateral opposite key, or None if not bilateral."""
    if key.endswith("_L"): return key[:-2] + "_R"
    if key.endswith("_R"): return key[:-2] + "_L"
    return None


def _face_axis(landmarks: dict):
    """
    Compute the face's central axis as (cx, cy, dx, dy):
      (cx, cy) — centroid of available centre-line points
      (dx, dy) — unit direction vector along the axis (pointing downward)

    Uses PCA on centre-line points so a tilted head is handled correctly.
    Falls back progressively to midpoints of bilateral pairs, then vertical.
    """
    centre_keys = ["glabella", "nose_bottom_middle", "mouth_upper_low_u",
                   "lips_center_meet", "chin_bottom_apex"]
    pts = [landmarks[k] for k in centre_keys if k in landmarks]

    # Supplement with midpoints of any bilateral pairs present
    bilateral_mids = []
    seen_bases = set()
    for k, v in landmarks.items():
        if k.endswith("_L"):
            base = k[:-2]
            rk   = base + "_R"
            if rk in landmarks and base not in seen_bases:
                seen_bases.add(base)
                lp, rp = landmarks[k], landmarks[rk]
                bilateral_mids.append(((lp[0]+rp[0])/2, (lp[1]+rp[1])/2))

    all_pts = pts + bilateral_mids
    if not all_pts:
        return None

    cx = sum(p[0] for p in all_pts) / len(all_pts)
    cy = sum(p[1] for p in all_pts) / len(all_pts)

    if len(all_pts) < 2:
        return (cx, cy, 0.0, 1.0)   # single point → assume vertical

    # PCA: eigenvector of the larger eigenvalue of the 2×2 covariance matrix
    sxx = sum((p[0]-cx)**2 for p in all_pts)
    syy = sum((p[1]-cy)**2 for p in all_pts)
    sxy = sum((p[0]-cx)*(p[1]-cy) for p in all_pts)

    diff = (sxx - syy) / 2.0
    hyp  = math.sqrt(diff**2 + sxy**2) if (diff**2 + sxy**2) > 0 else 0.0

    vx = diff + hyp
    vy = sxy
    mag = math.sqrt(vx**2 + vy**2)
    if mag < 1e-9:
        dx, dy = 0.0, 1.0
    else:
        dx, dy = vx / mag, vy / mag

    # Ensure direction points downward (screen y increases downward)
    if dy < 0:
        dx, dy = -dx, -dy

    return (cx, cy, dx, dy)


def _mirror_across_axis(px: float, py: float, ax: float, ay: float,
                        dx: float, dy: float) -> tuple:
    """
    Reflect point (px, py) across the line passing through (ax, ay)
    with unit direction (dx, dy).
    """
    vx = px - ax
    vy = py - ay
    dot = vx * dx + vy * dy          # projection length along axis
    # Foot of perpendicular from (px,py) onto axis
    fx = ax + dot * dx
    fy = ay + dot * dy
    # Reflection
    return (int(round(2 * fx - px)), int(round(2 * fy - py)))


def _estimate_singles(lm: dict, est: set):
    """Fill in missing single-centre landmarks from neighbours. Modifies lm and est."""
    def _mid(*keys):
        ps = [lm[k] for k in keys if k in lm]
        if not ps: return None
        return (int(sum(p[0] for p in ps)/len(ps)),
                int(sum(p[1] for p in ps)/len(ps)))

    if "glabella"          not in lm:
        p = _mid("eyebrow_inside_L", "eyebrow_inside_R")
        if p: lm["glabella"] = p; est.add("glabella")

    if "nose_bottom_middle" not in lm:
        p = _mid("alar_base_L", "alar_base_R",
                 "nose_nostril_outside_L", "nose_nostril_outside_R")
        if p: lm["nose_bottom_middle"] = p; est.add("nose_bottom_middle")

    if "mouth_upper_low_u"  not in lm:
        p = _mid("philtrum_peak_L", "philtrum_peak_R")
        if p: lm["mouth_upper_low_u"] = p; est.add("mouth_upper_low_u")

    if "lips_center_meet"   not in lm:
        p = _mid("lips_outer_crease_L", "lips_outer_crease_R",
                 "mouth_upper_low_u")
        if p: lm["lips_center_meet"] = p; est.add("lips_center_meet")

    if "lip_bottom_center"  not in lm:
        p = _mid("mouth_under_apex_L", "mouth_under_apex_R")
        if p: lm["lip_bottom_center"] = p; est.add("lip_bottom_center")

    if "chin_bottom_apex"   not in lm:
        p = _mid("chin_outer_side_L", "chin_outer_side_R")
        if p: lm["chin_bottom_apex"] = p; est.add("chin_bottom_apex")


def _effective_landmarks(ws) -> dict:
    """
    Return a fully-populated landmarks dict for metric computation.
    Runs tilt-aware estimation on a copy — does NOT modify the workspace.
    """
    lm  = dict(ws.landmarks)
    est = set(ws.estimated_keys)
    axis = _face_axis(lm)

    all_keys = set(p["key"] for p in PROMPTS)
    for key in all_keys:
        if key in lm: continue
        mk = _mirror_key(key)
        if mk and mk in lm and axis is not None:
            ax, ay, dx, dy = axis
            ox, oy = lm[mk]
            lm[key] = _mirror_across_axis(ox, oy, ax, ay, dx, dy)
            est.add(key)

    _estimate_singles(lm, est)
    return lm


# ===========================================================================
# Proportional comparison engine
# ===========================================================================

# ── Geometry helpers ─────────────────────────────────────────────────────────

def _pt(lm, key):
    return lm.get(key)

def _avg(*pts):
    v = [p for p in pts if p is not None]
    if not v: return None
    return (sum(p[0] for p in v)/len(v), sum(p[1] for p in v)/len(v))

def _hdist(a, b):
    if a is None or b is None: return None
    return abs(a[0] - b[0])

def _vdist(a, b):
    if a is None or b is None: return None
    return abs(a[1] - b[1])

def _edist(a, b):
    if a is None or b is None: return None
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def _angle_at(p1, apex, p2):
    """Interior angle in degrees at apex formed by rays apex->p1 and apex->p2."""
    if any(p is None for p in (p1, apex, p2)): return None
    v1 = (p1[0]-apex[0], p1[1]-apex[1])
    v2 = (p2[0]-apex[0], p2[1]-apex[1])
    m1 = (v1[0]**2+v1[1]**2)**0.5
    m2 = (v2[0]**2+v2[1]**2)**0.5
    if m1 < 1e-9 or m2 < 1e-9: return None
    cos_a = max(-1.0, min(1.0, (v1[0]*v2[0]+v1[1]*v2[1]) / (m1*m2)))
    return math.degrees(math.acos(cos_a))

def _line_angle(a, b):
    """Angle of segment a→b relative to horizontal, degrees. Positive = upward tilt."""
    if a is None or b is None: return None
    return math.degrees(math.atan2(-(b[1]-a[1]), b[0]-a[0]))  # negate y: screen coords


# ── Metric catalogue ─────────────────────────────────────────────────────────
# Each entry: (display_name, category, weight, fn(lm)->float|None)
# All *distance* metrics are returned as raw pixels – normalised later.
# *Ratio* metrics return dimensionless values (no normalisation needed).
# *Angle* metrics return degrees.

CAT_WIDTHS   = "Widths"
CAT_VERTICAL = "Vertical Spacing"
CAT_RATIO    = "Proportional Ratios"
CAT_SYMMETRY = "Symmetry (L vs R)"
CAT_ANGULAR  = "Angular"
CAT_THIRDS   = "Facial Thirds"

CATEGORY_META = {
    CAT_WIDTHS:   {"weight": 1.0, "color": "#4499ff"},
    CAT_VERTICAL: {"weight": 1.0, "color": "#44ddaa"},
    CAT_RATIO:    {"weight": 1.5, "color": "#ffbb44"},
    CAT_SYMMETRY: {"weight": 1.2, "color": "#dd55ff"},
    CAT_ANGULAR:  {"weight": 1.0, "color": "#ff7755"},
    CAT_THIRDS:   {"weight": 1.2, "color": "#55ddff"},
}

def _build_metric_defs():
    # ── WIDTHS ────────────────────────────────────────────────────────
    def face_width(lm):
        return _hdist(_pt(lm,"face_outline_lip_crease_L"), _pt(lm,"face_outline_lip_crease_R"))
    def cheekbone_width(lm):
        return _hdist(_pt(lm,"cheekbone_outer_L"), _pt(lm,"cheekbone_outer_R"))
    def jaw_width(lm):
        return _hdist(_pt(lm,"face_under_ear_L"), _pt(lm,"face_under_ear_R"))
    def eye_width_avg(lm):
        l = _hdist(_pt(lm,"eye_outside_corner_L"), _pt(lm,"eye_inside_corner_L"))
        r = _hdist(_pt(lm,"eye_outside_corner_R"), _pt(lm,"eye_inside_corner_R"))
        v = [x for x in (l,r) if x is not None]
        return sum(v)/len(v) if v else None
    def nose_width(lm):
        return _hdist(_pt(lm,"nose_nostril_outside_L"), _pt(lm,"nose_nostril_outside_R"))
    def alar_width(lm):
        return _hdist(_pt(lm,"alar_base_L"), _pt(lm,"alar_base_R"))
    def mouth_width(lm):
        return _hdist(_pt(lm,"lips_outer_crease_L"), _pt(lm,"lips_outer_crease_R"))
    def between_eyes(lm):
        return _hdist(_pt(lm,"eye_inside_corner_L"), _pt(lm,"eye_inside_corner_R"))
    def between_brows(lm):
        return _hdist(_pt(lm,"eyebrow_inside_L"), _pt(lm,"eyebrow_inside_R"))
    def philtrum_width(lm):
        return _hdist(_pt(lm,"philtrum_peak_L"), _pt(lm,"philtrum_peak_R"))

    # ── VERTICAL SPACING ──────────────────────────────────────────────
    def chin_to_brow(lm):
        brow = _avg(_pt(lm,"eyebrow_upper_apex_L"), _pt(lm,"eyebrow_upper_apex_R"))
        return _vdist(_pt(lm,"chin_bottom_apex"), brow)
    def eye_height_avg(lm):
        l = _vdist(_pt(lm,"eye_upper_apex_L"), _pt(lm,"eye_under_apex_L"))
        r = _vdist(_pt(lm,"eye_upper_apex_R"), _pt(lm,"eye_under_apex_R"))
        v = [x for x in (l,r) if x is not None]
        return sum(v)/len(v) if v else None
    def brow_to_eye_gap(lm):
        eye_t  = _avg(_pt(lm,"eye_upper_apex_L"), _pt(lm,"eye_upper_apex_R"))
        brow_b = _avg(_pt(lm,"eyebrow_under_apex_L"), _pt(lm,"eyebrow_under_apex_R"))
        return _vdist(eye_t, brow_b)
    def nose_to_mouth(lm):
        return _vdist(_pt(lm,"nose_bottom_middle"), _pt(lm,"mouth_upper_low_u"))
    def mouth_height(lm):
        under = _avg(_pt(lm,"mouth_under_apex_L"), _pt(lm,"mouth_under_apex_R"))
        return _vdist(_pt(lm,"mouth_upper_low_u"), under)
    def upper_lip_height(lm):
        return _vdist(_pt(lm,"mouth_upper_low_u"), _pt(lm,"lips_center_meet"))
    def lower_lip_height(lm):
        return _vdist(_pt(lm,"lips_center_meet"), _pt(lm,"lip_bottom_center"))
    def mouth_to_chin(lm):
        return _vdist(_pt(lm,"lips_center_meet"), _pt(lm,"chin_bottom_apex"))
    def nose_to_chin(lm):
        return _vdist(_pt(lm,"nose_bottom_middle"), _pt(lm,"chin_bottom_apex"))
    def mouth_to_eye_bottom(lm):
        eye_b = _avg(_pt(lm,"eye_under_apex_L"), _pt(lm,"eye_under_apex_R"))
        return _vdist(_pt(lm,"mouth_upper_low_u"), eye_b)
    def mouth_to_brow_bottom(lm):
        brow_b = _avg(_pt(lm,"eyebrow_under_apex_L"), _pt(lm,"eyebrow_under_apex_R"))
        return _vdist(_pt(lm,"mouth_upper_low_u"), brow_b)

    # ── FACIAL THIRDS ─────────────────────────────────────────────────
    def upper_third(lm):   # brow apex to nose bottom (proxy for mid-face)
        brow = _avg(_pt(lm,"eyebrow_upper_apex_L"), _pt(lm,"eyebrow_upper_apex_R"))
        return _vdist(brow, _pt(lm,"nose_bottom_middle"))
    def lower_third(lm):   # nose bottom to chin
        return _vdist(_pt(lm,"nose_bottom_middle"), _pt(lm,"chin_bottom_apex"))
    def mid_to_lower_ratio(lm):   # dimensionless
        u = upper_third(lm); lo = lower_third(lm)
        return u/lo if (u and lo and lo > 1e-9) else None
    def philtrum_to_lower(lm):    # philtrum / lower face
        ph = nose_to_mouth(lm); lo = lower_third(lm)
        return ph/lo if (ph and lo and lo > 1e-9) else None

    # ── PROPORTIONAL RATIOS (dimensionless) ───────────────────────────
    def nose_to_mouth_w_ratio(lm):
        nw = nose_width(lm); mw = mouth_width(lm)
        return nw/mw if (nw and mw and mw > 1e-9) else None
    def mouth_to_face_w_ratio(lm):
        mw = mouth_width(lm); fw = face_width(lm)
        return mw/fw if (mw and fw and fw > 1e-9) else None
    def nose_to_face_w_ratio(lm):
        nw = nose_width(lm); fw = face_width(lm)
        return nw/fw if (nw and fw and fw > 1e-9) else None
    def cheekbone_to_jaw_ratio(lm):
        cw = cheekbone_width(lm); jw = jaw_width(lm)
        return cw/jw if (cw and jw and jw > 1e-9) else None
    def intercanthal_ratio(lm):   # between-eyes / eye width avg
        be = between_eyes(lm); ew = eye_width_avg(lm)
        return be/ew if (be and ew and ew > 1e-9) else None
    def face_shape_index(lm):     # face width / face height
        fw = face_width(lm)
        brow = _avg(_pt(lm,"eyebrow_upper_apex_L"), _pt(lm,"eyebrow_upper_apex_R"))
        fh = _vdist(_pt(lm,"chin_bottom_apex"), brow)
        return fw/fh if (fw and fh and fh > 1e-9) else None
    def upper_lower_lip_ratio(lm):
        ul = upper_lip_height(lm); ll = lower_lip_height(lm)
        return ul/ll if (ul and ll and ll > 1e-9) else None
    def eye_aspect_ratio(lm):     # eye height / eye width avg
        eh = eye_height_avg(lm); ew = eye_width_avg(lm)
        return eh/ew if (eh and ew and ew > 1e-9) else None
    def eye_to_face_w_ratio(lm):
        ew = eye_width_avg(lm); fw = face_width(lm)
        return ew/fw if (ew and fw and fw > 1e-9) else None
    def mouth_to_nose_h_ratio(lm):   # philtrum / mouth height
        ph = nose_to_mouth(lm); mh = mouth_height(lm)
        return ph/mh if (ph and mh and mh > 1e-9) else None

    # ── SYMMETRY (L/R ratio within face, ideal = 1.0) ─────────────────
    def eye_width_sym(lm):
        l = _hdist(_pt(lm,"eye_outside_corner_L"), _pt(lm,"eye_inside_corner_L"))
        r = _hdist(_pt(lm,"eye_outside_corner_R"), _pt(lm,"eye_inside_corner_R"))
        return l/r if (l and r and r > 1e-9) else None
    def brow_height_sym(lm):
        l = _vdist(_pt(lm,"eyebrow_under_apex_L"), _pt(lm,"eyebrow_upper_apex_L"))
        r = _vdist(_pt(lm,"eyebrow_under_apex_R"), _pt(lm,"eyebrow_upper_apex_R"))
        return l/r if (l and r and r > 1e-9) else None
    def face_half_width_sym(lm):
        mid_x = None
        gl = _pt(lm,"glabella")
        nc = _pt(lm,"nose_bottom_middle")
        lc = _pt(lm,"lips_center_meet")
        ref_pts = [p for p in (gl, nc, lc) if p is not None]
        if ref_pts:
            mid_x = sum(p[0] for p in ref_pts) / len(ref_pts)
        if mid_x is None: return None
        lp = _pt(lm,"face_outline_lip_crease_L")
        rp = _pt(lm,"face_outline_lip_crease_R")
        if lp is None or rp is None: return None
        return abs(lp[0]-mid_x) / abs(rp[0]-mid_x) if abs(rp[0]-mid_x) > 1e-9 else None
    def eye_height_sym(lm):
        l = _vdist(_pt(lm,"eye_upper_apex_L"), _pt(lm,"eye_under_apex_L"))
        r = _vdist(_pt(lm,"eye_upper_apex_R"), _pt(lm,"eye_under_apex_R"))
        return l/r if (l and r and r > 1e-9) else None
    def brow_position_sym(lm):    # vertical brow apex L vs R (ideal = same height)
        bl = _pt(lm,"eyebrow_upper_apex_L"); br = _pt(lm,"eyebrow_upper_apex_R")
        if bl is None or br is None: return None
        # Use ratio of their y-distances to the eye apex on same side
        el = _pt(lm,"eye_upper_apex_L"); er = _pt(lm,"eye_upper_apex_R")
        dl = _vdist(bl, el); dr = _vdist(br, er)
        return dl/dr if (dl and dr and dr > 1e-9) else None

    # ── ANGULAR ───────────────────────────────────────────────────────
    def canthal_tilt_L(lm):
        return _line_angle(_pt(lm,"eye_inside_corner_L"), _pt(lm,"eye_outside_corner_L"))
    def canthal_tilt_R(lm):
        # Flip R so positive = upward tilt (mirrors L convention)
        a = _line_angle(_pt(lm,"eye_outside_corner_R"), _pt(lm,"eye_inside_corner_R"))
        return -a if a is not None else None
    def brow_arch_angle_L(lm):
        return _angle_at(_pt(lm,"eyebrow_inside_L"), _pt(lm,"eyebrow_upper_apex_L"),
                         _pt(lm,"eyebrow_outside_L"))
    def brow_arch_angle_R(lm):
        return _angle_at(_pt(lm,"eyebrow_inside_R"), _pt(lm,"eyebrow_upper_apex_R"),
                         _pt(lm,"eyebrow_outside_R"))
    def mouth_corner_tilt(lm):
        return _line_angle(_pt(lm,"lips_outer_crease_L"), _pt(lm,"lips_outer_crease_R"))
    def nose_tip_angle(lm):       # angle at nose tip: nostril_L – nose_bottom – nostril_R
        return _angle_at(_pt(lm,"nose_nostril_outside_L"), _pt(lm,"nose_bottom_middle"),
                         _pt(lm,"nose_nostril_outside_R"))
    def jaw_taper_angle_L(lm):    # angle at chin_outer_L: under_ear – chin_outer – chin_apex
        return _angle_at(_pt(lm,"face_under_ear_L"), _pt(lm,"chin_outer_side_L"),
                         _pt(lm,"chin_bottom_apex"))
    def jaw_taper_angle_R(lm):
        return _angle_at(_pt(lm,"face_under_ear_R"), _pt(lm,"chin_outer_side_R"),
                         _pt(lm,"chin_bottom_apex"))

    return [
        # (name, category, weight, fn)
        # ── Widths ────────────────────────────────────────────────────
        ("Face Width",              CAT_WIDTHS,   1.0, face_width),
        ("Cheekbone Width",         CAT_WIDTHS,   1.0, cheekbone_width),
        ("Jaw Width",               CAT_WIDTHS,   1.0, jaw_width),
        ("Eye Width (avg)",         CAT_WIDTHS,   1.0, eye_width_avg),
        ("Nose Width (nostrils)",   CAT_WIDTHS,   1.0, nose_width),
        ("Alar Base Width",         CAT_WIDTHS,   0.8, alar_width),
        ("Mouth Width",             CAT_WIDTHS,   1.0, mouth_width),
        ("Between Eyes Width",      CAT_WIDTHS,   1.0, between_eyes),
        ("Between Eyebrows Width",  CAT_WIDTHS,   0.8, between_brows),
        ("Philtrum Width",          CAT_WIDTHS,   0.8, philtrum_width),
        # ── Vertical Spacing ──────────────────────────────────────────
        ("Face Height (brow-chin)", CAT_VERTICAL, 1.0, chin_to_brow),
        ("Eye Height (avg)",        CAT_VERTICAL, 1.0, eye_height_avg),
        ("Brow to Eye Gap",         CAT_VERTICAL, 1.0, brow_to_eye_gap),
        ("Philtrum Length",         CAT_VERTICAL, 1.0, nose_to_mouth),
        ("Mouth Height",            CAT_VERTICAL, 1.0, mouth_height),
        ("Upper Lip Height",        CAT_VERTICAL, 0.9, upper_lip_height),
        ("Lower Lip Height",        CAT_VERTICAL, 0.9, lower_lip_height),
        ("Mouth to Chin",           CAT_VERTICAL, 1.0, mouth_to_chin),
        ("Nose to Chin",            CAT_VERTICAL, 1.0, nose_to_chin),
        ("Mouth to Eye Bottom",     CAT_VERTICAL, 0.9, mouth_to_eye_bottom),
        ("Mouth to Brow Bottom",    CAT_VERTICAL, 0.9, mouth_to_brow_bottom),
        # ── Facial Thirds ─────────────────────────────────────────────
        ("Mid-face Height (brow-nose)", CAT_THIRDS, 1.0, upper_third),
        ("Lower Face Height (nose-chin)",CAT_THIRDS,1.0, lower_third),
        ("Mid / Lower Face Ratio",  CAT_THIRDS,   1.2, mid_to_lower_ratio),
        ("Philtrum / Lower Face",   CAT_THIRDS,   1.0, philtrum_to_lower),
        # ── Proportional Ratios ───────────────────────────────────────
        ("Nose / Mouth Width",      CAT_RATIO,    1.5, nose_to_mouth_w_ratio),
        ("Mouth / Face Width",      CAT_RATIO,    1.5, mouth_to_face_w_ratio),
        ("Nose / Face Width",       CAT_RATIO,    1.5, nose_to_face_w_ratio),
        ("Cheekbone / Jaw Width",   CAT_RATIO,    1.3, cheekbone_to_jaw_ratio),
        ("Intercanthal Ratio",      CAT_RATIO,    1.3, intercanthal_ratio),
        ("Face Shape Index (W/H)",  CAT_RATIO,    1.5, face_shape_index),
        ("Upper / Lower Lip",       CAT_RATIO,    1.2, upper_lower_lip_ratio),
        ("Eye Aspect Ratio (H/W)",  CAT_RATIO,    1.2, eye_aspect_ratio),
        ("Eye / Face Width",        CAT_RATIO,    1.2, eye_to_face_w_ratio),
        ("Philtrum / Mouth Height", CAT_RATIO,    1.0, mouth_to_nose_h_ratio),
        # ── Symmetry ──────────────────────────────────────────────────
        ("Eye Width Symmetry L/R",  CAT_SYMMETRY, 1.2, eye_width_sym),
        ("Eye Height Symmetry L/R", CAT_SYMMETRY, 1.1, eye_height_sym),
        ("Brow Height Symmetry L/R",CAT_SYMMETRY, 1.1, brow_height_sym),
        ("Brow Position Sym L/R",   CAT_SYMMETRY, 1.0, brow_position_sym),
        ("Face Half-Width Sym L/R", CAT_SYMMETRY, 1.2, face_half_width_sym),
        # ── Angular ───────────────────────────────────────────────────
        ("Canthal Tilt Left (°)",   CAT_ANGULAR,  1.0, canthal_tilt_L),
        ("Canthal Tilt Right (°)",  CAT_ANGULAR,  1.0, canthal_tilt_R),
        ("Brow Arch Angle Left (°)",CAT_ANGULAR,  1.0, brow_arch_angle_L),
        ("Brow Arch Angle Right (°)",CAT_ANGULAR, 1.0, brow_arch_angle_R),
        ("Mouth Corner Tilt (°)",   CAT_ANGULAR,  1.0, mouth_corner_tilt),
        ("Nose Tip Angle (°)",      CAT_ANGULAR,  1.0, nose_tip_angle),
        ("Jaw Taper Angle Left (°)",CAT_ANGULAR,  0.9, jaw_taper_angle_L),
        ("Jaw Taper Angle Right (°)",CAT_ANGULAR, 0.9, jaw_taper_angle_R),
    ]

METRIC_DEFS = _build_metric_defs()


def _ref_length(landmarks: dict) -> Optional[float]:
    """Face height as normalisation reference. Falls back progressively."""
    brow = _avg(landmarks.get("eyebrow_upper_apex_L"), landmarks.get("eyebrow_upper_apex_R"))
    ref  = _vdist(landmarks.get("chin_bottom_apex"), brow)
    if ref and ref > 1: return ref
    ref = _hdist(landmarks.get("face_outline_lip_crease_L"),
                 landmarks.get("face_outline_lip_crease_R"))
    if ref and ref > 1: return ref
    ref = _hdist(landmarks.get("eye_outside_corner_L"),
                 landmarks.get("eye_outside_corner_R"))
    return ref if (ref and ref > 1) else None


def compute_proportions(landmarks: dict) -> dict[str, float]:
    """
    Return {metric_name: normalised_value}.
    Accepts a raw or pre-populated effective landmarks dict.
    Distance metrics → divided by face height.
    Ratio / angle metrics → returned as-is.
    """
    ref = _ref_length(landmarks)
    if ref is None: return {}

    result = {}
    for name, cat, _w, fn in METRIC_DEFS:
        val = fn(landmarks)
        if val is None: continue
        if cat in (CAT_RATIO, CAT_SYMMETRY, CAT_ANGULAR):
            result[name] = val          # already dimensionless or degrees
        else:
            if val >= 0:
                result[name] = val / ref
    return result


def run_comparison(ws_a, ws_b) -> dict:
    """
    Full comparison between two workspaces.
    Uses _effective_landmarks so skipped/missing points are filled in
    by mirroring + estimation before metrics are computed.
    Returns:
      metrics     : list of (name, cat, weight, val_a, val_b, pct_diff)
      cat_scores  : {category: score_0_to_100}
      score       : float 0-100 overall weighted score
      missing     : [name, ...]
    """
    lm_a = _effective_landmarks(ws_a)
    lm_b = _effective_landmarks(ws_b)
    pa   = compute_proportions(lm_a)
    pb   = compute_proportions(lm_b)

    rows    = []
    missing = []

    for name, cat, weight, fn in METRIC_DEFS:
        va = pa.get(name)
        vb = pb.get(name)
        if va is None or vb is None:
            missing.append(name)
            rows.append((name, cat, weight, va, vb, None))
            continue

        if cat == CAT_ANGULAR:
            # Angular: absolute difference in degrees (max ~180)
            diff_deg = abs(va - vb)
            pct = (diff_deg / 180.0) * 100
        elif cat == CAT_SYMMETRY:
            # Symmetry ratios: compare both faces' L/R ratios as %
            avg = (abs(va) + abs(vb)) / 2
            pct = abs(va - vb) / avg * 100 if avg > 1e-9 else 0.0
        else:
            avg = (va + vb) / 2
            pct = abs(va - vb) / avg * 100 if avg > 1e-9 else 0.0

        rows.append((name, cat, weight, va, vb, pct))

    # ── Category sub-scores ──────────────────────────────────────────
    cat_data: dict[str, list] = {c: [] for c in CATEGORY_META}
    for name, cat, weight, va, vb, pct in rows:
        if pct is not None:
            cat_data[cat].append(pct * weight)

    cat_scores = {}
    for cat, vals in cat_data.items():
        if vals:
            cat_scores[cat] = max(0.0, 100.0 - sum(vals)/len(vals))
        else:
            cat_scores[cat] = None

    # ── Overall weighted score ────────────────────────────────────────
    weighted_sum, weight_total = 0.0, 0.0
    for name, cat, weight, va, vb, pct in rows:
        if pct is not None:
            cat_w = CATEGORY_META[cat]["weight"]
            weighted_sum  += pct * weight * cat_w
            weight_total  += weight * cat_w

    overall = max(0.0, 100.0 - weighted_sum/weight_total) if weight_total > 0 else 0.0

    return {
        "metrics":    rows,
        "cat_scores": cat_scores,
        "score":      overall,
        "missing":    missing,
    }


# ===========================================================================
# Tab-selection dialog
# ===========================================================================

class CompareSelectDialog(tk.Toplevel):

    def __init__(self, master: FaceLandmarkApp, tabs: list):
        super().__init__(master)
        self.master_app = master
        self.tabs       = tabs
        self.title("Select Two Tabs to Compare")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")
        self.grab_set()

        tk.Label(self, text="Choose Face A:", bg="#1a1a2e", fg="#ffffff",
                 font=("Helvetica", 10)).grid(row=0, column=0, padx=14, pady=(16,4), sticky="w")
        tk.Label(self, text="Choose Face B:", bg="#1a1a2e", fg="#ffffff",
                 font=("Helvetica", 10)).grid(row=1, column=0, padx=14, pady=4, sticky="w")

        names = [ws.name for _, ws in tabs]
        self.var_a = tk.StringVar(value=names[0])
        self.var_b = tk.StringVar(value=names[1] if len(names) > 1 else names[0])

        ttk.Combobox(self, textvariable=self.var_a, values=names,
                     state="readonly", width=22).grid(row=0, column=1, padx=14, pady=(16,4))
        ttk.Combobox(self, textvariable=self.var_b, values=names,
                     state="readonly", width=22).grid(row=1, column=1, padx=14, pady=4)

        bf = tk.Frame(self, bg="#1a1a2e")
        bf.grid(row=2, column=0, columnspan=2, pady=14)
        ttk.Button(bf, text="Compare", command=self._go).pack(side="left", padx=6)
        ttk.Button(bf, text="Cancel",  command=self.destroy).pack(side="left", padx=6)

        self.update_idletasks()
        x = master.winfo_x() + master.winfo_width()  // 2 - self.winfo_width()  // 2
        y = master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{x}+{y}")

    def _go(self):
        na, nb = self.var_a.get(), self.var_b.get()
        if na == nb:
            messagebox.showwarning("Same Tab", "Select two different tabs.", parent=self); return
        ws_a = next(ws for _, ws in self.tabs if ws.name == na)
        ws_b = next(ws for _, ws in self.tabs if ws.name == nb)
        result = run_comparison(ws_a, ws_b)
        self.destroy()
        CompareResultsWindow(self.master_app, na, nb, result)


# ===========================================================================
# Results window
# ===========================================================================

class CompareResultsWindow(tk.Toplevel):

    @staticmethod
    def _score_color(score: float) -> str:
        """Green for high scores, red for low."""
        t = max(0.0, min(1.0, (100 - score) / 60.0))
        if t < 0.5:
            r, g = int(80 + 175*t*2), 220
        else:
            r, g = 220, int(220*(1-(t-0.5)*2))
        return f"#{r:02x}{g:02x}40"

    @staticmethod
    def _pct_color(pct: float) -> str:
        if pct < 5:   return "#44ff88"
        if pct < 10:  return "#aaee44"
        if pct < 20:  return "#ffdd44"
        if pct < 35:  return "#ff9944"
        return "#ff5544"

    def __init__(self, master, name_a: str, name_b: str, result: dict):
        super().__init__(master)
        self.title(f"Comparison  ·  {name_a}  vs  {name_b}")
        self.configure(bg="#0d0d1e")
        self.minsize(820, 600)
        self.resizable(True, True)

        score      = result["score"]
        cat_scores = result["cat_scores"]
        metrics    = result["metrics"]
        missing    = result["missing"]

        # ── Overall score header ──────────────────────────────────────
        hdr = tk.Frame(self, bg="#0f3460", pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="Overall Similarity Score", bg="#0f3460", fg="#8899bb",
                 font=("Helvetica", 10)).pack()
        tk.Label(hdr, text=f"{score:.1f} / 100",
                 bg="#0f3460", fg=self._score_color(score),
                 font=("Helvetica", 36, "bold")).pack()
        tk.Label(hdr, text=f"{name_a}   vs   {name_b}",
                 bg="#0f3460", fg="#778899", font=("Helvetica", 10)).pack()

        # ── Category sub-score bars ───────────────────────────────────
        cat_frame = tk.Frame(self, bg="#111130", pady=8)
        cat_frame.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(cat_frame, text="Category Scores", bg="#111130", fg="#667799",
                 font=("Helvetica", 9, "bold")).grid(row=0, column=0, columnspan=4,
                 sticky="w", padx=6, pady=(0, 4))

        for i, (cat, meta) in enumerate(CATEGORY_META.items()):
            cs = cat_scores.get(cat)
            col = i % 3
            row = 1 + i // 3
            cell = tk.Frame(cat_frame, bg="#111130")
            cell.grid(row=row, column=col, padx=8, pady=3, sticky="w")

            tk.Label(cell, text=cat, bg="#111130", fg=meta["color"],
                     font=("Helvetica", 8, "bold"), width=22, anchor="w").pack(side="left")

            bar_bg = tk.Frame(cell, bg="#1a1a33", width=100, height=10)
            bar_bg.pack(side="left", padx=(4, 6))
            bar_bg.pack_propagate(False)
            if cs is not None:
                fill_w = max(1, int(cs))
                tk.Frame(bar_bg, bg=self._score_color(cs),
                         width=fill_w, height=10).place(x=0, y=0)

            score_txt = f"{cs:.0f}" if cs is not None else "—"
            tk.Label(cell, text=score_txt, bg="#111130",
                     fg=self._score_color(cs) if cs is not None else "#555566",
                     font=("Helvetica", 8, "bold"), width=4).pack(side="left")

        # ── Column header bar ─────────────────────────────────────────
        col_bar = tk.Frame(self, bg="#12122a")
        col_bar.pack(fill="x", padx=12, pady=(8, 0))
        for txt, w, anc in [
            ("Metric",               28, "w"),
            (f"{name_a[:10]}",       11, "center"),
            (f"{name_b[:10]}",       11, "center"),
            ("% Diff",                7, "center"),
            ("Match",                14, "center"),
        ]:
            tk.Label(col_bar, text=txt, bg="#12122a", fg="#556688",
                     font=("Helvetica", 8, "bold"), width=w, anchor=anc,
                     pady=4).pack(side="left", padx=2)

        # ── Scrollable metric table ───────────────────────────────────
        table_outer = tk.Frame(self, bg="#0d0d1e")
        table_outer.pack(fill="both", expand=True, padx=12, pady=4)

        vsb = ttk.Scrollbar(table_outer, orient="vertical")
        vsb.pack(side="right", fill="y")
        tc = tk.Canvas(table_outer, bg="#0d0d1e", highlightthickness=0,
                       yscrollcommand=vsb.set)
        tc.pack(side="left", fill="both", expand=True)
        vsb.config(command=tc.yview)
        tc.bind("<MouseWheel>",  lambda e: tc.yview_scroll(-1 if e.delta>0 else 1, "units"))
        tc.bind("<Button-4>",    lambda e: tc.yview_scroll(-1, "units"))
        tc.bind("<Button-5>",    lambda e: tc.yview_scroll(1,  "units"))

        inner = tk.Frame(tc, bg="#0d0d1e")
        tc.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: tc.config(scrollregion=tc.bbox("all")))

        row_idx = 0
        current_cat = None
        for name, cat, weight, va, vb, pct in metrics:
            # Category section header
            if cat != current_cat:
                current_cat = cat
                cat_color = CATEGORY_META[cat]["color"]
                sec = tk.Frame(inner, bg="#0a0a20")
                sec.grid(row=row_idx, column=0, columnspan=5, sticky="ew", pady=(6, 0))
                cs = cat_scores.get(cat)
                hdr_txt = cat + (f"   —   {cs:.0f}/100" if cs is not None else "")
                tk.Label(sec, text=f"  {hdr_txt}", bg="#0a0a20", fg=cat_color,
                         font=("Helvetica", 9, "bold"), pady=3, anchor="w").pack(fill="x")
                row_idx += 1

            bg = "#0d0d1e" if row_idx % 2 == 0 else "#111128"

            tk.Label(inner, text=name, bg=bg, fg="#ccccdd",
                     font=("Helvetica", 8), width=28, anchor="w",
                     pady=4, padx=6).grid(row=row_idx, column=0, sticky="w")

            is_angle = cat == CAT_ANGULAR
            fmt = "{:.1f}°" if is_angle else "{:.4f}"

            for ci, val in enumerate((va, vb)):
                txt = fmt.format(val) if val is not None else "—"
                tk.Label(inner, text=txt, bg=bg, fg="#9999bb",
                         font=("Courier", 8), width=11,
                         anchor="center").grid(row=row_idx, column=ci+1)

            if pct is not None:
                pct_txt = f"{pct:.1f}%"
                pct_fg  = self._pct_color(pct)
            else:
                pct_txt, pct_fg = "N/A", "#444455"
            tk.Label(inner, text=pct_txt, bg=bg, fg=pct_fg,
                     font=("Helvetica", 8, "bold"), width=7,
                     anchor="center").grid(row=row_idx, column=3)

            # Match bar
            bar_f = tk.Frame(inner, bg=bg, width=110, height=10)
            bar_f.grid(row=row_idx, column=4, padx=6, pady=4)
            bar_f.pack_propagate(False)
            if pct is not None:
                fill_w = max(0, int(110 * max(0, 100 - pct*2) / 100))
                color  = self._pct_color(pct)
                tk.Frame(bar_f, bg=color,    width=fill_w,       height=10).place(x=0, y=0)
                tk.Frame(bar_f, bg="#1a1a33", width=110-fill_w,  height=10).place(x=fill_w, y=0)

            row_idx += 1

        # ── Missing note ──────────────────────────────────────────────
        if missing:
            note = tk.Frame(self, bg="#18100a", pady=5)
            note.pack(fill="x", padx=12)
            tk.Label(note, text=f"  {len(missing)} metric(s) skipped — landmark(s) not placed.",
                     bg="#18100a", fg="#886644", font=("Helvetica", 8),
                     anchor="w").pack(side="left")

        # ── Buttons ───────────────────────────────────────────────────
        btn_row = tk.Frame(self, bg="#0d0d1e", pady=8)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Export Report (CSV)",
                   command=lambda: self._export_csv(name_a, name_b, result)
                   ).pack(side="left", padx=12)
        ttk.Button(btn_row, text="Close", command=self.destroy).pack(side="right", padx=12)

        self.update_idletasks()
        mx = master.winfo_x() + master.winfo_width()  // 2 - self.winfo_width()  // 2
        my = master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{mx}+{my}")

    def _export_csv(self, name_a, name_b, result):
        path = filedialog.asksaveasfilename(
            title="Export comparison report",
            initialfile=f"compare_{name_a}_vs_{name_b}.csv".replace(" ", "_"),
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Category", "Metric", f"{name_a}", f"{name_b}", "% Difference"])
            prev_cat = None
            for name, cat, weight, va, vb, pct in result["metrics"]:
                if cat != prev_cat:
                    w.writerow([])
                    cs = result["cat_scores"].get(cat)
                    w.writerow([f"=== {cat}", "", "", "",
                                f"Category Score: {cs:.1f}/100" if cs else ""])
                    prev_cat = cat
                w.writerow([cat, name,
                            f"{va:.5f}" if va is not None else "",
                            f"{vb:.5f}" if vb is not None else "",
                            f"{pct:.2f}%" if pct is not None else "N/A"])
            w.writerow([])
            w.writerow(["", "OVERALL SIMILARITY SCORE", "", "", f"{result['score']:.2f}/100"])
        messagebox.showinfo("Exported", f"Saved to:\n{os.path.basename(path)}", parent=self)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = FaceLandmarkApp()
    app.mainloop()

