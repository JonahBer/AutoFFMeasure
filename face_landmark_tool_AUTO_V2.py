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

try:
    import numpy as np
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

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
# MediaPipe Face Mesh index map
# ---------------------------------------------------------------------------
# Maps our named landmarks to MediaPipe Face Mesh indices (0-477 with refine).
#
# IMPORTANT: This tool uses MONITOR-side L/R convention:
#   _L = appears on left side of the image  = subject's ANATOMICAL RIGHT
#   _R = appears on right side of the image = subject's ANATOMICAL LEFT
#
# MediaPipe uses subject-anatomical convention internally. Therefore every
# bilateral entry below is INVERTED relative to MediaPipe's own naming:
# our "_L" maps to MediaPipe's "right" indices and vice versa.
#
# Indices reference: https://github.com/google/mediapipe/blob/master/
#                    mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

MEDIAPIPE_INDEX_MAP = {
    # ── Face outline (jaw silhouette) ─────────────────────────────────
    # Monitor-left = subject's right side of jaw
    "face_outline_lip_crease_L": 215,   # subject-right cheek hollow at lip level
    "face_outline_lip_crease_R": 435,   # subject-left  cheek hollow at lip level
    "face_under_ear_L":          172,   # subject-right jaw under ear
    "face_under_ear_R":          397,   # subject-left  jaw under ear
    "face_above_ear_L":          127,   # subject-right temple (above ear)
    "face_above_ear_R":          356,   # subject-left  temple (above ear)
    "cheekbone_outer_L":         234,   # subject-right widest cheek point
    "cheekbone_outer_R":         454,   # subject-left  widest cheek point

    # ── Brows ─────────────────────────────────────────────────────────
    "eyebrow_outside_L":         70,    # subject-right brow outer end
    "eyebrow_outside_R":         300,   # subject-left  brow outer end
    "eyebrow_inside_L":          55,    # subject-right brow inner end
    "eyebrow_inside_R":          285,   # subject-left  brow inner end
    "eyebrow_under_apex_L":      52,    # subject-right brow lower-apex
    "eyebrow_under_apex_R":      282,   # subject-left  brow lower-apex
    "eyebrow_upper_apex_L":      105,   # subject-right brow upper-apex
    "eyebrow_upper_apex_R":      334,   # subject-left  brow upper-apex
    "glabella":                  9,     # midline between brows

    # ── Eyes ──────────────────────────────────────────────────────────
    "eye_upper_apex_L":          159,   # subject-right upper eyelid apex
    "eye_upper_apex_R":          386,   # subject-left  upper eyelid apex
    "eye_upper_apex_crease_L":   223,   # subject-right upper eyelid crease
    "eye_upper_apex_crease_R":   443,   # subject-left  upper eyelid crease
    "eye_outside_corner_L":      33,    # subject-right outer canthus
    "eye_outside_corner_R":      263,   # subject-left  outer canthus
    "eye_inside_corner_L":       133,   # subject-right inner canthus
    "eye_inside_corner_R":       362,   # subject-left  inner canthus
    "eye_under_apex_L":          145,   # subject-right lower eyelid apex
    "eye_under_apex_R":          374,   # subject-left  lower eyelid apex

    # ── Nose ──────────────────────────────────────────────────────────
    "nose_bottom_middle":        2,     # midline below nose tip (columella base)
    "nose_nostril_outside_L":    98,    # subject-right nostril outer rim
    "nose_nostril_outside_R":    327,   # subject-left  nostril outer rim
    "alar_base_L":               209,   # subject-right alar crease (where nose meets cheek)
    "alar_base_R":               429,   # subject-left  alar crease

    # ── Chin / jaw ────────────────────────────────────────────────────
    "chin_outer_side_L":         150,   # subject-right chin outer corner
    "chin_outer_side_R":         379,   # subject-left  chin outer corner
    "chin_bottom_apex":          152,   # midline chin tip

    # ── Mouth / lips ──────────────────────────────────────────────────
    "mouth_upper_apex_side_L":   37,    # subject-right upper lip peak
    "mouth_upper_apex_side_R":   267,   # subject-left  upper lip peak
    "philtrum_peak_L":           39,    # subject-right cupid's bow peak
    "philtrum_peak_R":           269,   # subject-left  cupid's bow peak
    "mouth_upper_low_u":         0,     # midline upper lip valley (cupid's bow center)
    "lips_center_meet":          13,    # midline where lips meet (inner upper lip)
    "lips_outer_crease_L":       61,    # subject-right mouth corner
    "lips_outer_crease_R":       291,   # subject-left  mouth corner
    "mouth_under_apex_L":        84,    # subject-right lower lip apex
    "mouth_under_apex_R":        314,   # subject-left  lower lip apex
    "lip_bottom_center":         17,    # midline bottom of lower lip

    # ── Neck ──────────────────────────────────────────────────────────
    # MediaPipe's mesh ends at the jawline; neck-face corner has no exact match.
    # Using the jaw point closest to where neck typically meets face as a fallback.
    "neck_face_corner_L":        58,    # subject-right jaw near neck
    "neck_face_corner_R":        288,   # subject-left  jaw near neck
}

# ---------------------------------------------------------------------------
# Colours / geometry
# ---------------------------------------------------------------------------

LEFT_COLOR      = "#00d4ff"
RIGHT_COLOR     = "#ff6b35"
SINGLE_COLOR    = "#a8ff3e"
ESTIMATED_COLOR = "#ffcc00"   # mirrored / auto-estimated point
SKIPPED_COLOR   = "#555566"   # point explicitly skipped
MAPPED_COLOR    = "#000000"   # 100%-match target position
MAPPED_ARROW    = "#555555"   # connecting arrow from real → mapped
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
    name:             str   = "Tab 1"
    original_image:   object = None
    image_stem:       str   = "landmarks"
    landmarks:        dict  = field(default_factory=dict)
    skipped_keys:     set   = field(default_factory=set)
    estimated_keys:   set   = field(default_factory=set)
    mapped_landmarks: dict  = field(default_factory=dict)  # key->(x,y) for 100% target overlay
    mapped_label:     str   = ""                            # e.g. "A→B"
    current_step:     int   = 0
    marking_mode:     bool  = False
    scale_factor:     float = 1.0
    zoom_str:         str   = "Fit"
    scroll_x:         float = 0.0
    scroll_y:         float = 0.0
    mesh_landmarks:   list  = field(default_factory=list)  # 478 (x,y) from MediaPipe; empty if manual-only


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
        fm.add_command(label="Export -> JSON", command=self.export_json)
        fm.add_command(label="Export -> CSV", command=self.export_csv)
        fm.add_separator()
        fm.add_command(label="Batch Auto-Detect Folder...",
                       command=self.batch_auto_detect_folder)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.quit)
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

        self.auto_btn = tk.Button(pb, text="Auto-Detect", bg="#1a1a40", fg="#44ddaa",
                                  font=("Helvetica", 9, "bold"), relief="flat",
                                  padx=10, pady=6, cursor="hand2",
                                  command=self.auto_detect_landmarks)
        self.auto_btn.pack(side="right", padx=(0, 4))

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

        # Repopulate landmark list (uses full status-aware rebuild)
        self.landmark_listbox.delete(0, tk.END)
        for p in PROMPTS:
            k = p["key"]
            if k in ws.skipped_keys and k not in ws.landmarks:
                self.landmark_listbox.insert(tk.END, f"{k:<28}  SKIPPED")
                self.landmark_listbox.itemconfig(tk.END, fg=SKIPPED_COLOR)
            elif k in ws.estimated_keys:
                if k in ws.landmarks:
                    x, y = ws.landmarks[k]
                    self.landmark_listbox.insert(tk.END, f"{k:<28}  ~({x:>4}, {y:>4})")
                    self.landmark_listbox.itemconfig(tk.END, fg=ESTIMATED_COLOR)
            elif k in ws.landmarks:
                x, y = ws.landmarks[k]
                # If mapping tab, also show mapped position
                if k in ws.mapped_landmarks:
                    mx, my = ws.mapped_landmarks[k]
                    self.landmark_listbox.insert(
                        tk.END, f"{k:<28}  ({x:>4},{y:>4}) → ({mx:>4},{my:>4})")
                else:
                    self.landmark_listbox.insert(tk.END, f"{k:<28}  ({x:>4}, {y:>4})")
                side = k.split("_")[-1]
                c = LEFT_COLOR if side=="L" else RIGHT_COLOR if side=="R" else SINGLE_COLOR
                self.landmark_listbox.itemconfig(tk.END, fg=c)

        # Restore prompt banner
        if ws.original_image is None:
            self.prompt_var.set("Open or paste an image, then press  Start Marking")
            self.step_var.set("")
        elif ws.mapped_landmarks:
            # Mapping view
            self.prompt_var.set(
                f"  Mapping view: {ws.mapped_label}  |  "
                f"Coloured = actual  ·  Black diamonds = 100% target  ·  "
                f"Arrows show required change")
            self.step_var.set(f"{len(ws.mapped_landmarks)} mapped")
            self.skip_btn.config(state="disabled")
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

        # Draw mapped (100%-target) markers and connecting arrows
        if ws.mapped_landmarks:
            for key, (mx, my) in ws.mapped_landmarks.items():
                cmx = int(mx * sf)
                cmy = int(my * sf)
                # Connecting arrow from real → mapped position
                if key in ws.landmarks:
                    ox, oy = ws.landmarks[key]
                    cox, coy = int(ox * sf), int(oy * sf)
                    if abs(cmx - cox) > 2 or abs(cmy - coy) > 2:
                        self.canvas.create_line(
                            cox, coy, cmx, cmy,
                            fill=MAPPED_ARROW, width=1, dash=(4, 3),
                            arrow=tk.LAST, arrowshape=(6, 8, 3),
                            tags=("mapped_marker",))
                self._draw_mapped_marker(cmx, cmy, key)

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

    def _draw_mapped_marker(self, cx: int, cy: int, key: str):
        """Black diamond marker for the 100%-similarity target position."""
        r   = RADIUS + 1
        tag = f"mp_{key}"
        self.canvas.delete(tag)
        # Diamond polygon
        self.canvas.create_polygon(
            cx, cy - r,
            cx + r, cy,
            cx, cy + r,
            cx - r, cy,
            fill=MAPPED_COLOR, outline="#ffffff", width=1,
            tags=("mapped_marker", tag))
        self.canvas.create_text(
            cx + r + 4, cy,
            text=f">{_short_label(key)}",
            fill=MAPPED_COLOR, font=("Helvetica", 7), anchor="w",
            tags=("mapped_marker", tag))

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

        # Re-run estimation ONLY for points that were skipped during manual
        # marking and got mirror-derived. Auto-detected workspaces (where every
        # point is "estimated" because it came from MediaPipe) must NOT be
        # re-derived — those positions came from the face mesh, not from
        # mirroring placed points, so wiping & recomputing would destroy them.
        if self.ws.skipped_keys:
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
            "mesh_landmarks": [list(p) for p in self.ws.mesh_landmarks],
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
            if ext == ".json":
                loaded, hint, skipped, estimated, mesh = self._parse_json(path)
            elif ext == ".csv":
                loaded, hint, skipped, estimated, mesh = self._parse_csv(path)
            else:
                messagebox.showerror("Unknown format", "Open a .json or .csv file.");
                return
        except Exception as exc:
            messagebox.showerror("Parse error", str(exc));
            return

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
        self.ws.mesh_landmarks = mesh
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
        mesh = [tuple(p) for p in data.get("mesh_landmarks", [])]
        return loaded, data.get("paired_image"), skipped, estimated, mesh

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
        return loaded, hint, skipped, estimated, []

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
        ws = self.ws
        n_before = len(ws.estimated_keys)

        axis = _face_axis(ws.landmarks)
        r    = _bilateral_half_ratio(ws.landmarks,
                                     axis[0] if axis else 0.0)
        r    = max(0.5, min(2.0, r))

        for key in list(ws.skipped_keys):
            if key in ws.landmarks:
                continue
            mirror_key = _mirror_key(key)
            if (mirror_key and mirror_key in ws.landmarks
                    and mirror_key not in ws.skipped_keys
                    and axis is not None):
                ax, ay, dx, dy = axis
                ox, oy = ws.landmarks[mirror_key]
                ex, ey = _mirror_across_axis(ox, oy, ax, ay, dx, dy)
                # Apply yaw foreshortening to the horizontal displacement
                vx = ox - ax
                vy = oy - ay
                perp_component = vx * dy + vy * (-dx)
                horiz_disp = ex - ax
                if perp_component > 0:
                    ex = int(round(ax + horiz_disp * r))
                else:
                    ex = int(round(ax + horiz_disp / r))
                ws.landmarks[key]    = (ex, ey)
                ws.estimated_keys.add(key)

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
    # MediaPipe auto-detection
    # ------------------------------------------------------------------

    def _get_face_mesh(self):
        """Lazily initialize the FaceMesh model. Returns None if unavailable."""
        if not MEDIAPIPE_AVAILABLE:
            return None
        if not hasattr(self, "_face_mesh") or self._face_mesh is None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        return self._face_mesh

    def _run_mesh_on_image(self, pil_image):
        """
        Run MediaPipe Face Mesh on a PIL image. Returns a tuple
        (mesh_landmarks, named_landmarks, estimated_keys) or None if
        no face was detected. Pure data — does not touch any UI.
        """
        face_mesh = self._get_face_mesh()
        if face_mesh is None:
            return None

        rgb = np.array(pil_image.convert("RGB"))
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        h, w = rgb.shape[:2]
        mesh = result.multi_face_landmarks[0].landmark
        n_mesh = len(mesh)

        mesh_landmarks = []
        for pt in mesh:
            mx = int(round(pt.x * w))
            my = int(round(pt.y * h))
            mx = max(0, min(mx, w - 1))
            my = max(0, min(my, h - 1))
            mesh_landmarks.append((mx, my))

        named = {}
        estimated = set()
        for our_key, mesh_idx in MEDIAPIPE_INDEX_MAP.items():
            if mesh_idx >= n_mesh:
                continue
            named[our_key] = mesh_landmarks[mesh_idx]
            estimated.add(our_key)

        return mesh_landmarks, named, estimated



    def auto_detect_landmarks(self):
        """Run MediaPipe Face Mesh on the current image and populate all
        landmarks as estimated. Wipes any existing landmarks first."""
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror(
                "MediaPipe not installed",
                "Auto-detect requires MediaPipe and NumPy.\n\n"
                "Install with:\n"
                "  pip install mediapipe numpy"
            )
            return

        ws = self.ws
        if ws.original_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return

        face_mesh = self._get_face_mesh()
        if face_mesh is None:
            messagebox.showerror("MediaPipe error", "Could not initialize Face Mesh.")
            return

        self.status_var.set("Running MediaPipe face detection...")
        self.update_idletasks()

        try:
            detection = self._run_mesh_on_image(ws.original_image)
        except Exception as exc:
            messagebox.showerror("Detection failed", f"MediaPipe error:\n{exc}")
            self.status_var.set("Auto-detect failed.")
            return

        if detection is None:
            messagebox.showwarning(
                "No face found",
                "MediaPipe could not detect a face in this image.\n"
                "Try a clearer frontal image, or place landmarks manually."
            )
            self.status_var.set("No face detected.")
            return

        mesh_landmarks, named, estimated_keys = detection

        # Wipe all existing state (per option A)
        ws.landmarks.clear()
        ws.skipped_keys.clear()
        ws.estimated_keys.clear()
        ws.marking_mode = False
        ws.current_step = TOTAL
        self.canvas.delete("marker")
        self.skip_btn.config(state="disabled")

        ws.mesh_landmarks = mesh_landmarks
        ws.landmarks = dict(named)
        ws.estimated_keys = set(estimated_keys)

        n_placed = len(named)
        n_missed = len(MEDIAPIPE_INDEX_MAP) - n_placed

        self.refresh_display()
        self._rebuild_list_panel()

        msg = (f"Auto-detected {n_placed} landmarks.\n\n"
               "All points are marked as estimated (yellow dashed). "
               "Drag any that look misplaced — dragging promotes them to placed.")
        if n_missed:
            msg += f"\n\n{n_missed} mesh indices were out of range (unexpected)."

        self.prompt_var.set(
            f"  Auto-detected — {n_placed} estimated.  "
            f"Drag to correct any misplaced points.")
        self.step_var.set(f"{n_placed} / {TOTAL}  estimated")
        self.status_var.set(f"Auto-detect complete: {n_placed} landmarks placed as estimated.")

        messagebox.showinfo("Auto-detect complete", msg)

    # ------------------------------------------------------------------
    # Batch auto-detection (folder of images → folder of JSON+image pairs)
    # ------------------------------------------------------------------

    BATCH_IMAGE_EXTS = (".png",)  # .png only, per spec
    BATCH_MAX_DEPTH = 2  # input folder + 1 level of subfolders

    def batch_auto_detect_folder(self):
        """Pick a folder of images, run Face Mesh on each, export
        {stem}.json + {stem}_image.png pairs into a sibling output folder."""
        if not MEDIAPIPE_AVAILABLE:
            messagebox.showerror(
                "MediaPipe not installed",
                "Batch auto-detect requires MediaPipe and NumPy.\n\n"
                "Install with:\n  pip install mediapipe numpy")
            return
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow required", "pip install Pillow")
            return

        in_folder = filedialog.askdirectory(
            title="Select folder of images to batch-process")
        if not in_folder:
            return
        in_folder = os.path.abspath(in_folder)

        # Collect images, depth-2 walk
        image_paths = self._scan_folder_for_images(in_folder,
                                                   self.BATCH_MAX_DEPTH,
                                                   self.BATCH_IMAGE_EXTS)
        if not image_paths:
            messagebox.showwarning(
                "No images found",
                f"No PNG files found in {os.path.basename(in_folder)} "
                f"(searched {self.BATCH_MAX_DEPTH} level(s) deep).")
            return

        # Resolve output folder name
        parent = os.path.dirname(in_folder)
        base = os.path.basename(in_folder)
        out_folder = os.path.join(parent, f"{base}_landmarks")
        n = 2
        while os.path.exists(out_folder):
            out_folder = os.path.join(parent, f"{base}_landmarks_{n}")
            n += 1
        try:
            os.makedirs(out_folder, exist_ok=False)
        except Exception as exc:
            messagebox.showerror("Cannot create output folder", str(exc))
            return

        # Confirm before starting
        if not messagebox.askyesno(
                "Begin batch auto-detect",
                f"Found {len(image_paths)} PNG file(s) in:\n  {in_folder}\n\n"
                f"Output will be written to:\n  {out_folder}\n\nProceed?"):
            try:
                os.rmdir(out_folder)
            except Exception:
                pass
            return

        BatchProgressDialog(self, image_paths, in_folder, out_folder)

    @staticmethod
    def _scan_folder_for_images(root_folder: str, max_depth: int,
                                exts: tuple) -> list:
        """Walk root_folder up to max_depth levels, return absolute paths
        to files matching exts. Same depth math as the JSON scanner."""
        found = []
        root_folder = os.path.abspath(root_folder)
        if not os.path.isdir(root_folder):
            return found
        exts_lower = tuple(e.lower() for e in exts)

        for current_dir, subdirs, files in os.walk(root_folder):
            rel = os.path.relpath(current_dir, root_folder)
            depth = 1 if rel == "." else 1 + rel.count(os.sep) + 1
            if depth > max_depth:
                subdirs[:] = []
                continue
            for fname in files:
                if fname.lower().endswith(exts_lower):
                    found.append(os.path.join(current_dir, fname))
        return sorted(found)

    def batch_export_one(self, image_path: str, out_folder: str) -> str:
        """
        Process a single image: run Face Mesh, write the JSON + paired
        image into out_folder. Returns a status string:
          "ok"          — exported successfully
          "no_face"     — MediaPipe found no face
          "load_failed" — couldn't open the image
          "error: ..."  — other failure (message follows)
        """
        try:
            img = Image.open(image_path)
            img.load()
        except Exception as exc:
            return f"load_failed: {exc}"

        try:
            detection = self._run_mesh_on_image(img)
        except Exception as exc:
            return f"error: {exc}"

        if detection is None:
            return "no_face"

        mesh_landmarks, named, estimated_keys = detection
        stem = os.path.splitext(os.path.basename(image_path))[0]

        # Sanitize stem to avoid collisions with weird filenames
        safe_stem = "".join(c if (c.isalnum() or c in "-_.") else "_"
                            for c in stem)
        if not safe_stem:
            safe_stem = "image"

        # Avoid overwriting if a sanitized collision occurs
        json_path = os.path.join(out_folder, f"{safe_stem}.json")
        img_path = os.path.join(out_folder, f"{safe_stem}{self.IMAGE_COPY_SUFFIX}")
        suffix = 2
        while os.path.exists(json_path) or os.path.exists(img_path):
            json_path = os.path.join(out_folder, f"{safe_stem}_{suffix}.json")
            img_path = os.path.join(out_folder,
                                    f"{safe_stem}_{suffix}{self.IMAGE_COPY_SUFFIX}")
            suffix += 1

        # Write paired image (always PNG, matches IMAGE_COPY_SUFFIX)
        try:
            img.save(img_path, format="PNG")
        except Exception as exc:
            return f"error: image save failed: {exc}"

        # Build payload matching export_json's format
        payload = {
            "total_landmarks": len(named),
            "image_size": list(img.size),
            "paired_image": os.path.basename(img_path),
            "skipped_keys": [],
            "estimated_keys": list(estimated_keys),
            "landmarks": {k: {"x": v[0], "y": v[1], "status": "estimated"}
                          for k, v in named.items()},
            "mesh_landmarks": [list(p) for p in mesh_landmarks],
        }
        try:
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            return f"error: json save failed: {exc}"

        return "ok"


    # ------------------------------------------------------------------
    # Mapping tab  (opens a new tab showing original + 100%-target overlay)
    # ------------------------------------------------------------------

    def open_mapping_tab(self, ws_source, ws_target, name_src: str, name_tgt: str):
        """
        Create a new tab containing ws_source's image with:
          • Coloured markers  — ws_source's actual landmark positions
          • Black diamonds    — where those landmarks would need to be
                                for 100% similarity with ws_target
          • Grey dashed arrows connecting each pair
        """
        if ws_source.original_image is None:
            messagebox.showerror("No image", f"'{name_src}' has no image."); return

        mapped = compute_mapped_landmarks(ws_source, ws_target)
        if not mapped:
            messagebox.showerror(
                "Cannot map",
                "Not enough landmarks to compute the mapping.\n"
                "Make sure both faces have at least brow and chin points placed."
            ); return

        # Snapshot the current tab before we switch away
        self._snapshot_current()

        # Build new workspace
        new_ws                  = Workspace()
        new_ws.name             = f"Map: {name_src} → {name_tgt}"
        new_ws.original_image   = ws_source.original_image.copy()
        new_ws.image_stem       = f"map_{name_src}_to_{name_tgt}".replace(" ", "_")
        new_ws.landmarks        = dict(ws_source.landmarks)
        new_ws.skipped_keys     = set(ws_source.skipped_keys)
        new_ws.estimated_keys   = set(ws_source.estimated_keys)
        new_ws.mapped_landmarks = mapped
        new_ws.mapped_label     = f"{name_src} → {name_tgt}"
        new_ws.zoom_str         = "Fit"
        new_ws.scale_factor     = 1.0
        new_ws.marking_mode     = False
        new_ws.current_step     = TOTAL  # read-only view

        self.workspaces.append(new_ws)
        self.active_idx = len(self.workspaces) - 1
        self._restore_workspace(self.active_idx)

    def open_mapping_tab_from_canonical(self, ws_solo, target_canonical: dict,
                                        source_name: str, target_name: str):
        """
        Create a mapping tab where the diamond targets are derived from
        averaged canonical-frame positions (used for group → solo mapping).
        ws_solo is the workspace whose image will be displayed.
        """
        if ws_solo.original_image is None:
            messagebox.showerror("No image", f"'{target_name}' has no image.")
            return

        mapped = compute_mapped_landmarks_from_proportions(ws_solo, None,
                                                           target_canonical)
        if not mapped:
            messagebox.showerror(
                "Cannot map",
                "Not enough landmarks to compute the mapping."
            )
            return

        self._snapshot_current()

        new_ws = Workspace()
        new_ws.name = f"Map: {source_name} → {target_name}"
        new_ws.original_image = ws_solo.original_image.copy()
        new_ws.image_stem = f"map_{source_name}_to_{target_name}".replace(" ", "_")
        new_ws.landmarks = dict(ws_solo.landmarks)
        new_ws.skipped_keys = set(ws_solo.skipped_keys)
        new_ws.estimated_keys = set(ws_solo.estimated_keys)
        new_ws.mapped_landmarks = mapped
        new_ws.mapped_label = f"{source_name} → {target_name}"
        new_ws.zoom_str = "Fit"
        new_ws.scale_factor = 1.0
        new_ws.marking_mode = False
        new_ws.current_step = TOTAL

        self.workspaces.append(new_ws)
        self.active_idx = len(self.workspaces) - 1
        self._restore_workspace(self.active_idx)

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


# Singleton midline anatomy keys (each is a single sample on the midline).
_MIDLINE_KEYS = [
    "glabella",
    "nose_bottom_middle",
    "mouth_upper_low_u",
    "lips_center_meet",
    "lip_bottom_center",
    "chin_bottom_apex",
]

# Every bilateral pair contributes its midpoint as a midline sample.
# The midpoint of (P_L, P_R) lies on the apparent symmetry line regardless
# of how wide the pair is, so wide pairs (jaw, cheeks) are NOT a problem
# — they are extra information, not a sideways pull. Furthermore, a
# bilateral midpoint averages out the user's per-side click error,
# so it has roughly half the variance of a singleton midline click.
_BILATERAL_PAIRS_FOR_AXIS = [
    ("face_outline_lip_crease_L", "face_outline_lip_crease_R"),
    ("face_under_ear_L",          "face_under_ear_R"),
    ("face_above_ear_L",          "face_above_ear_R"),
    ("cheekbone_outer_L",         "cheekbone_outer_R"),
    ("eyebrow_outside_L",         "eyebrow_outside_R"),
    ("eyebrow_inside_L",          "eyebrow_inside_R"),
    ("eyebrow_under_apex_L",      "eyebrow_under_apex_R"),
    ("eyebrow_upper_apex_L",      "eyebrow_upper_apex_R"),
    ("eye_upper_apex_L",          "eye_upper_apex_R"),
    ("eye_upper_apex_crease_L",   "eye_upper_apex_crease_R"),
    ("eye_outside_corner_L",      "eye_outside_corner_R"),
    ("eye_inside_corner_L",       "eye_inside_corner_R"),
    ("eye_under_apex_L",          "eye_under_apex_R"),
    ("nose_nostril_outside_L",    "nose_nostril_outside_R"),
    ("alar_base_L",               "alar_base_R"),
    ("chin_outer_side_L",         "chin_outer_side_R"),
    ("mouth_upper_apex_side_L",   "mouth_upper_apex_side_R"),
    ("philtrum_peak_L",           "philtrum_peak_R"),
    ("lips_outer_crease_L",       "lips_outer_crease_R"),
    ("mouth_under_apex_L",        "mouth_under_apex_R"),
    ("neck_face_corner_L",        "neck_face_corner_R"),
]


def _face_axis(landmarks: dict):
    """
    Compute the face's central axis as (cx, cy, dx, dy):
      (cx, cy) — origin on the apparent face symmetry line.
      (dx, dy) — unit vector along the axis (pointing toward the chin).

    Centroid is built from BOTH singleton midline anatomy AND the midpoints
    of every available bilateral pair. The bilateral midpoints dominate
    the count (roughly 21 vs 6 in a fully-marked face), and crucially the
    averaging of L/R clicks cancels per-side click bias — the single
    biggest source of the lateral skew the diamond markers used to show.

    PCA on the same point set gives the axis tilt; anatomy (chin vs brow)
    disambiguates the axis direction.
    """
    samples = []  # each entry is one sample on the midline

    # Singleton midline anatomy — each is one sample.
    for k in _MIDLINE_KEYS:
        if k in landmarks:
            samples.append(landmarks[k])

    # Every bilateral pair midpoint is one sample. Each is statistically
    # better than a singleton midline click (per-side click error cancels).
    for lk, rk in _BILATERAL_PAIRS_FOR_AXIS:
        if lk in landmarks and rk in landmarks:
            lp = landmarks[lk]
            rp = landmarks[rk]
            samples.append(((lp[0] + rp[0]) / 2.0, (lp[1] + rp[1]) / 2.0))

    if not samples:
        return None

    cx = sum(p[0] for p in samples) / len(samples)
    cy = sum(p[1] for p in samples) / len(samples)

    if len(samples) < 2:
        return (cx, cy, 0.0, 1.0)

    # ── Tilt (PCA) on the same combined sample set ────────────────────
    sxx = sum((p[0] - cx) ** 2 for p in samples)
    syy = sum((p[1] - cy) ** 2 for p in samples)
    sxy = sum((p[0] - cx) * (p[1] - cy) for p in samples)

    diff = (sxx - syy) / 2.0
    hyp  = math.sqrt(diff * diff + sxy * sxy)

    vx  = diff + hyp
    vy  = sxy
    mag = math.sqrt(vx * vx + vy * vy)
    if mag < 1e-9:
        dx, dy = 0.0, 1.0
    else:
        dx, dy = vx / mag, vy / mag

    # ── Resolve axis direction by anatomy: chin should sit at +along ──
    # Vote across as many along-axis anchors as we have, so a single
    # mis-clicked landmark cannot flip the entire frame.
    score = 0.0
    chin = landmarks.get("chin_bottom_apex")
    if chin:
        score += (chin[0] - cx) * dx + (chin[1] - cy) * dy
    # Brow midpoint should be at -along.
    bL = landmarks.get("eyebrow_upper_apex_L")
    bR = landmarks.get("eyebrow_upper_apex_R")
    if bL and bR:
        bx = (bL[0] + bR[0]) / 2.0
        by = (bL[1] + bR[1]) / 2.0
        score -= (bx - cx) * dx + (by - cy) * dy
    glab = landmarks.get("glabella")
    if glab:
        score -= (glab[0] - cx) * dx + (glab[1] - cy) * dy

    if score < 0:
        dx, dy = -dx, -dy
    elif score == 0 and dy < 0:
        # No anatomy info at all: default to "axis points down".
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
    Runs tilt + yaw-aware estimation on a copy — does NOT modify the workspace.
    """
    lm   = dict(ws.landmarks)
    est  = set(ws.estimated_keys)
    axis = _face_axis(lm)

    all_keys = set(p["key"] for p in PROMPTS)
    for key in all_keys:
        if key in lm:
            continue
        mk = _mirror_key(key)
        if mk and mk in lm and axis is not None:
            ax, ay, dx, dy = axis
            # Compute yaw ratio so the reflection accounts for foreshortening
            r = _bilateral_half_ratio(lm, ax)
            r = max(0.5, min(2.0, r))
            ox, oy = lm[mk]
            ex, ey = _mirror_across_axis(ox, oy, ax, ay, dx, dy)
            # Adjust the horizontal offset by the foreshortening ratio
            # The reflected point is on the opposite side: apply scale
            perp_dir_x = dy   # perpendicular to axis (not yet sign-checked, just for ratio)
            # Determine if source (mk) is on the _R (positive perp) side
            vx = ox - ax
            vy = oy - ay
            from math import sqrt as _sqrt
            perp_component = vx * dy + vy * (-dx)  # dot with (dy, -dx)
            if perp_component > 0:
                # mk is _R side (positive perp), reflected key is _L side
                # _R apparent width may differ from _L apparent width by ratio r
                # Scale the reflected horizontal displacement
                horiz_disp = ex - ax
                ex = int(round(ax + horiz_disp * r))
            else:
                # mk is _L side, reflected key is _R side
                horiz_disp = ex - ax
                ex = int(round(ax + horiz_disp / r))
            lm[key] = (ex, ey)
            est.add(key)

    _estimate_singles(lm, est)
    return lm


def _bilateral_half_ratio(landmarks: dict, cx: float) -> float:
    """
    Estimate the left/right apparent width ratio from bilateral landmark pairs.

    Assumes the face is anatomically symmetric. Any consistent asymmetry in
    how far _L vs _R landmarks sit from the midline is due to face yaw.

    Returns r = median(d_L / d_R) where:
      d_L = distance from centroid to each _L point (image-left side)
      d_R = distance from centroid to each _R point (image-right side)

    r = 1.0  → frontal / symmetric
    r > 1.0  → _L side appears wider → face rotated so _L side faces camera more
    r < 1.0  → _R side appears wider → face rotated toward _R
    """
    ratios = []
    seen   = set()
    for k in landmarks:
        if not k.endswith("_L"):
            continue
        base = k[:-2]
        if base in seen:
            continue
        rk = base + "_R"
        if rk not in landmarks:
            continue
        seen.add(base)
        d_L = cx - landmarks[k][0]    # how far _L point sits left of centroid
        d_R = landmarks[rk][0] - cx   # how far _R point sits right of centroid
        if d_L > 2 and d_R > 2:
            ratios.append(d_L / d_R)

    if not ratios:
        return 1.0
    ratios.sort()
    return ratios[len(ratios) // 2]   # median — robust against individual outliers


_RIGHT_PERP_ANCHORS = [
    "face_outline_lip_crease_R",
    "cheekbone_outer_R",
    "face_under_ear_R",
    "face_above_ear_R",
    "eye_outside_corner_R",
    "eyebrow_outside_R",
    "eyebrow_upper_apex_R",
    "lips_outer_crease_R",
    "chin_outer_side_R",
    "neck_face_corner_R",
    "eye_inside_corner_R",
    "eyebrow_inside_R",
    "mouth_upper_apex_side_R",
    "alar_base_R",
    "nose_nostril_outside_R",
    "philtrum_peak_R",
]


def _resolve_perp_sign(px: float, py: float, lm: dict,
                       cx: float, cy: float) -> tuple:
    """
    Decide whether perp = (dy, -dx) or its negation.

    Vote across EVERY available right-side anchor; do NOT break on the
    first one found. The first-anchor-wins approach used previously is
    fragile because a single anchor sitting near the centroid (e.g. a
    narrow Sims-4 nostril when the centroid happens to be biased a
    couple of pixels) can flip the perp axis catastrophically and
    every diamond ends up reflected to the wrong side.
    """
    score = 0.0
    n = 0
    for k in _RIGHT_PERP_ANCHORS:
        if k in lm:
            rx, ry = lm[k]
            score += (rx - cx) * px + (ry - cy) * py
            n += 1
    if n == 0:
        return (px, py)
    if score < 0:
        return (-px, -py)
    return (px, py)


def _shrink_yaw_ratio(r: float, n_pairs: int,
                      threshold: float = 0.10) -> float:
    """
    Shrink the bilateral half-width ratio toward 1.0 (no yaw) so that
    click jitter on a frontal face does not produce a fake yaw signal.

    On a perfectly frontal Sims-4 portrait, ordinary click noise
    routinely yields raw r values in roughly [0.93, 1.07]. Treating
    those as real foreshortening produces an asymmetric per-side scale
    factor (one side stretched, the other compressed) — which is
    exactly the "lateral skew" the diamond markers used to show.

    We map raw r into log space, subtract a `threshold` of magnitude
    `|log r| < threshold`, and fold what's left back into linear space.
    Effect:
      • Frontal-with-noise (|log r| ≤ threshold) → r' = 1.0  (no scale).
      • Real yaw of moderate size (|log r| > threshold) → still applied,
        but its magnitude is reduced by `threshold` units.
      • Very few pairs available → r' = 1.0 (the estimate is too noisy).
    """
    if n_pairs < 4:
        return 1.0
    lr = math.log(max(1e-6, r))
    if abs(lr) <= threshold:
        return 1.0
    return math.exp(math.copysign(abs(lr) - threshold, lr))


def _bilateral_half_ratio_with_count(landmarks: dict, cx: float) -> tuple:
    """Like _bilateral_half_ratio but also reports how many pairs voted."""
    ratios = []
    seen = set()
    for k in landmarks:
        if not k.endswith("_L"):
            continue
        base = k[:-2]
        if base in seen:
            continue
        rk = base + "_R"
        if rk not in landmarks:
            continue
        seen.add(base)
        d_L = cx - landmarks[k][0]
        d_R = landmarks[rk][0] - cx
        if d_L > 2 and d_R > 2:
            ratios.append(d_L / d_R)
    if not ratios:
        return 1.0, 0
    ratios.sort()
    return ratios[len(ratios) // 2], len(ratios)


def compute_mapped_landmarks(ws_source, ws_target) -> dict:
    """
    Compute where ws_source's landmarks would need to be positioned
    to achieve 100% proportional similarity with ws_target.

    Pipeline:
      1.  For each face build a coordinate frame:
            • origin = bilaterally-fair midline centroid (combines
              singleton midline points and the midpoints of every
              available bilateral pair, so per-side click error
              cancels)
            • axis   = PCA on the same point set, sign-resolved by
              the chin/brow vote
            • perp   = axis rotated 90° (image coords), sign-resolved
              by VOTING across every available _R anchor — robust to
              individual marginal anchors flipping the frame.
            • H      = face height reference for normalisation.
      2.  Express each TARGET landmark in normalised local coords
          (along, perp) divided by H.
      3.  Optionally apply a shrunk yaw correction (only when r_s and
          r_t deviate from 1.0 by more than the noise floor; otherwise
          treated as 1.0 to avoid amplifying click jitter into fake
          per-side stretching).
      4.  Reconstruct mapped position in source's frame:
              mapped = source_origin + along * H_s * source_axis
                                     + perp_corr * H_s * source_perp.
    """
    lm_s = _effective_landmarks(ws_source)
    lm_t = _effective_landmarks(ws_target)

    axis_s = _face_axis(lm_s)
    axis_t = _face_axis(lm_t)
    if axis_s is None or axis_t is None:
        return {}

    cs_x, cs_y, ds_x, ds_y = axis_s
    ct_x, ct_y, dt_x, dt_y = axis_t

    H_s = _ref_length(lm_s)
    H_t = _ref_length(lm_t)
    if not H_s or not H_t or H_s < 1 or H_t < 1:
        return {}

    # Perp candidates: rotate axis 90°. Sign resolved by voting across
    # ALL right-side anchors (not the first one found).
    ps_x, ps_y = _resolve_perp_sign(ds_y, -ds_x, lm_s, cs_x, cs_y)
    pt_x, pt_y = _resolve_perp_sign(dt_y, -dt_x, lm_t, ct_x, ct_y)

    # Yaw foreshortening — shrink toward 1.0 so click jitter on a frontal
    # face does not produce per-side scaling.
    r_s_raw, n_s = _bilateral_half_ratio_with_count(lm_s, cs_x)
    r_t_raw, n_t = _bilateral_half_ratio_with_count(lm_t, ct_x)
    r_s = _shrink_yaw_ratio(r_s_raw, n_s)
    r_t = _shrink_yaw_ratio(r_t_raw, n_t)
    # Hard clamp on extremes.
    r_s = max(0.5, min(2.0, r_s))
    r_t = max(0.5, min(2.0, r_t))

    eps = 1e-9
    scale_left  = math.sqrt(r_s / r_t) if r_t > eps else 1.0
    scale_right = math.sqrt(r_t / r_s) if r_s > eps else 1.0

    mapped = {}
    for key, (tx, ty) in lm_t.items():
        # Express target point in its own normalised local coords.
        vx = tx - ct_x
        vy = ty - ct_y
        along = (vx * dt_x + vy * dt_y) / H_t
        perp  = (vx * pt_x + vy * pt_y) / H_t

        # Yaw correction (no-op when shrinkage put both r's at 1.0).
        perp_corrected = perp * (scale_left if perp < 0 else scale_right)

        # Reconstruct in source's frame.
        mx = cs_x + along * H_s * ds_x + perp_corrected * H_s * ps_x
        my = cs_y + along * H_s * ds_y + perp_corrected * H_s * ps_y

        mapped[key] = (int(round(mx)), int(round(my)))

    return mapped


# ===========================================================================
# Dense mesh comparison  (auto-vs-auto only, all 478 points)
# ===========================================================================

# Empirical threshold: average per-point face-normalized distance at which
# two faces are considered "completely different." Below this maps linearly
# to a 0-100 score. Tune this if scores cluster too high or too low.
_MESH_DIFF_FLOOR = 0.05   # 5% of face height = score 0


def _normalize_mesh_to_canonical_frame(mesh: list, named_lm: dict):
    """
    Transform all mesh points into a canonical face-local frame:
      • origin = midline centroid (computed from named landmarks via _face_axis)
      • along-axis = face axis (chin direction = positive along)
      • perp-axis = perpendicular, sign-resolved by _resolve_perp_sign
      • unit = face height (from _ref_length)

    Returns:
      list of (along_norm, perp_norm) tuples in the canonical frame, or
      None if the face axis or reference length couldn't be computed.

    Note: uses the named landmarks (which already passed through MediaPipe's
    detection on auto-detected workspaces) to define the frame, then
    transforms ALL mesh points into that frame.
    """
    axis = _face_axis(named_lm)
    if axis is None:
        return None

    cx, cy, dx, dy = axis
    H = _ref_length(named_lm)
    if H is None or H < 1:
        return None

    # Resolve perp sign once, using the named-landmark _R anchors
    px, py = _resolve_perp_sign(dy, -dx, named_lm, cx, cy)

    normalized = []
    for (mx, my) in mesh:
        vx = mx - cx
        vy = my - cy
        along = (vx * dx + vy * dy) / H
        perp  = (vx * px + vy * py) / H
        normalized.append((along, perp))
    return normalized


def compute_mesh_comparison(ws_a, ws_b) -> Optional[dict]:
    """
    Dense point-by-point comparison of two faces using their full
    478-point MediaPipe meshes.

    Returns None if either workspace lacks mesh data, or if the
    canonical frame can't be built. Otherwise returns:
      {
        "score":         float (0-100, higher = more similar),
        "mean_dist":     float (mean per-point distance, face-height units),
        "median_dist":   float (median per-point distance),
        "max_dist":      float (worst single point),
        "n_points":      int (how many points compared),
      }
    """
    if not ws_a.mesh_landmarks or not ws_b.mesh_landmarks:
        return None
    if len(ws_a.mesh_landmarks) != len(ws_b.mesh_landmarks):
        # Shouldn't happen with same MediaPipe version, but guard anyway
        return None

    # Use the EFFECTIVE named landmarks to build each canonical frame.
    # This way, even partially-skipped manual workspaces could feed in,
    # but in practice this function is called only when both have mesh data.
    lm_a = _effective_landmarks(ws_a)
    lm_b = _effective_landmarks(ws_b)

    norm_a = _normalize_mesh_to_canonical_frame(ws_a.mesh_landmarks, lm_a)
    norm_b = _normalize_mesh_to_canonical_frame(ws_b.mesh_landmarks, lm_b)
    if norm_a is None or norm_b is None:
        return None

    # Per-point Euclidean distance in canonical (along, perp) units
    distances = []
    for (a_al, a_pe), (b_al, b_pe) in zip(norm_a, norm_b):
        d = math.sqrt((a_al - b_al) ** 2 + (a_pe - b_pe) ** 2)
        distances.append(d)

    if not distances:
        return None

    distances.sort()
    n = len(distances)
    mean_d   = sum(distances) / n
    median_d = distances[n // 2]
    max_d    = distances[-1]

    # Linear map: 0.0 → 100, _MESH_DIFF_FLOOR → 0
    score = max(0.0, min(100.0,
                         100.0 * (1.0 - mean_d / _MESH_DIFF_FLOOR)))

    return {
        "score":       score,
        "mean_dist":   mean_d,
        "median_dist": median_d,
        "max_dist":    max_d,
        "n_points":    n,
    }

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


def _average_proportions(ws_list: list) -> dict:
    """
    Average the 50-metric proportional values across a group of workspaces.
    For each metric, runs compute_proportions on each member's effective
    landmarks, then averages the metric across all members that produced
    a value (skipping members where the metric was None).

    Returns the same shape as compute_proportions: {metric_name: avg_value}.
    """
    per_member = []
    for ws in ws_list:
        lm = _effective_landmarks(ws)
        per_member.append(compute_proportions(lm))

    averaged = {}
    for name, _cat, _w, _fn in METRIC_DEFS:
        vals = [d[name] for d in per_member if name in d]
        if vals:
            averaged[name] = sum(vals) / len(vals)
    return averaged


def _average_mesh_canonical(ws_list: list):
    """
    Compute the per-point average of a group's meshes in canonical face frame.
    All-or-nothing: returns None if ANY workspace lacks mesh data.

    Each workspace's mesh is transformed into its own canonical frame, then
    we average position-by-position across all members. Result is a list of
    (along, perp) tuples in canonical units (face-height-normalized).
    """
    if not ws_list:
        return None
    # All-or-nothing check
    for ws in ws_list:
        if not ws.mesh_landmarks:
            return None

    # Verify all meshes have the same length
    n_points = len(ws_list[0].mesh_landmarks)
    for ws in ws_list[1:]:
        if len(ws.mesh_landmarks) != n_points:
            return None

    # Normalize each mesh to canonical frame
    normalized_per_member = []
    for ws in ws_list:
        lm = _effective_landmarks(ws)
        norm = _normalize_mesh_to_canonical_frame(ws.mesh_landmarks, lm)
        if norm is None:
            return None
        normalized_per_member.append(norm)

    # Per-point average
    n_members = len(normalized_per_member)
    averaged = []
    for i in range(n_points):
        sum_along = sum(normalized_per_member[m][i][0] for m in range(n_members))
        sum_perp  = sum(normalized_per_member[m][i][1] for m in range(n_members))
        averaged.append((sum_along / n_members, sum_perp / n_members))
    return averaged


def compute_mesh_comparison_canonical(canon_a: list, canon_b: list) -> Optional[dict]:
    """
    Dense mesh comparison given two ALREADY-canonicalized meshes (lists of
    (along, perp) tuples). Mirrors compute_mesh_comparison's output but
    skips the canonicalization step (already done upstream for groups).
    """
    if canon_a is None or canon_b is None:
        return None
    if len(canon_a) != len(canon_b):
        return None

    distances = []
    for (a_al, a_pe), (b_al, b_pe) in zip(canon_a, canon_b):
        d = math.sqrt((a_al - b_al) ** 2 + (a_pe - b_pe) ** 2)
        distances.append(d)

    if not distances:
        return None

    distances.sort()
    n = len(distances)
    mean_d   = sum(distances) / n
    median_d = distances[n // 2]
    max_d    = distances[-1]

    score = max(0.0, min(100.0, 100.0 * (1.0 - mean_d / _MESH_DIFF_FLOOR)))

    return {
        "score":       score,
        "mean_dist":   mean_d,
        "median_dist": median_d,
        "max_dist":    max_d,
        "n_points":    n,
    }


def run_group_comparison(ws_list_a: list, ws_list_b: list) -> dict:
    """
    Generalized comparison between two groups of workspaces.
    When both groups are length 1, behaves identically to a 1-vs-1 compare.
    When either is multi, that side's metrics are averaged across members.

    Returns the standard result dict plus:
      group_a_size, group_b_size — int counts
      group_a_names, group_b_names — list of tab names
      avg_proportions_a, avg_proportions_b — the averaged metric dicts
        (used by group→solo mapping)
    """
    # Per-side proportions (averaged if multi, single if length 1)
    pa = _average_proportions(ws_list_a)
    pb = _average_proportions(ws_list_b)

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
            diff_deg = abs(va - vb)
            pct = (diff_deg / 180.0) * 100
        elif cat == CAT_SYMMETRY:
            avg = (abs(va) + abs(vb)) / 2
            pct = abs(va - vb) / avg * 100 if avg > 1e-9 else 0.0
        else:
            avg = (va + vb) / 2
            pct = abs(va - vb) / avg * 100 if avg > 1e-9 else 0.0

        rows.append((name, cat, weight, va, vb, pct))

    # Category sub-scores
    cat_data: dict[str, list] = {c: [] for c in CATEGORY_META}
    for name, cat, weight, va, vb, pct in rows:
        if pct is not None:
            cat_data[cat].append(pct * weight)

    cat_scores = {}
    for cat, vals in cat_data.items():
        if vals:
            cat_scores[cat] = max(0.0, 100.0 - sum(vals) / len(vals))
        else:
            cat_scores[cat] = None

    # Overall weighted score
    weighted_sum, weight_total = 0.0, 0.0
    for name, cat, weight, va, vb, pct in rows:
        if pct is not None:
            cat_w = CATEGORY_META[cat]["weight"]
            weighted_sum += pct * weight * cat_w
            weight_total += weight * cat_w

    overall = max(0.0, 100.0 - weighted_sum / weight_total) if weight_total > 0 else 0.0

    # Dense mesh comparison — uses canonical-space averaging for groups.
    # All-or-nothing: if ANY tab in either group lacks mesh data, mesh is None.
    canon_a = _average_mesh_canonical(ws_list_a)
    canon_b = _average_mesh_canonical(ws_list_b)
    mesh_result = compute_mesh_comparison_canonical(canon_a, canon_b)

    return {
        "metrics":     rows,
        "cat_scores":  cat_scores,
        "score":       overall,
        "missing":     missing,
        "mesh":        mesh_result,
        "group_a_size":  len(ws_list_a),
        "group_b_size":  len(ws_list_b),
        "group_a_names": [ws.name for ws in ws_list_a],
        "group_b_names": [ws.name for ws in ws_list_b],
        "avg_proportions_a": pa,
        "avg_proportions_b": pb,
    }


def run_comparison(ws_a, ws_b) -> dict:
    """
    Backward-compatible 1-vs-1 wrapper. Identical output to the previous
    run_comparison; new code should use run_group_comparison directly.
    """
    return run_group_comparison([ws_a], [ws_b])


# ===========================================================================
# Group → solo mapping  (uses averaged proportions as the target)
# ===========================================================================

def compute_mapped_landmarks_from_proportions(ws_source, target_props: dict,
                                              target_ref_canonical: dict = None) -> dict:
    """
    Like compute_mapped_landmarks, but uses an averaged proportion vector
    instead of a target workspace. This is how group → solo mapping works:
    we don't have a "target image," we have target proportions averaged
    across the group.

    Approach:
      Compute where each named landmark "should be" on the source image
      such that source achieves the target proportions. We do this by
      using the existing canonical-frame trick: average the GROUP's
      named-landmark positions in canonical space, then project them
      back through the source's canonical frame.

      target_ref_canonical is required: a dict {key: (along, perp)} of
      named landmarks averaged across the group in canonical space.
    """
    if not target_ref_canonical:
        return {}

    lm_s = _effective_landmarks(ws_source)
    axis_s = _face_axis(lm_s)
    if axis_s is None:
        return {}
    cs_x, cs_y, ds_x, ds_y = axis_s

    H_s = _ref_length(lm_s)
    if not H_s or H_s < 1:
        return {}

    ps_x, ps_y = _resolve_perp_sign(ds_y, -ds_x, lm_s, cs_x, cs_y)

    mapped = {}
    for key, (along, perp) in target_ref_canonical.items():
        mx = cs_x + along * H_s * ds_x + perp * H_s * ps_x
        my = cs_y + along * H_s * ds_y + perp * H_s * ps_y
        mapped[key] = (int(round(mx)), int(round(my)))
    return mapped


def average_named_landmarks_canonical(ws_list: list) -> dict:
    """
    Average each named landmark's position across a group of workspaces,
    in canonical face-normalized coords. Returns {key: (along, perp)}.
    Skips members where a key has no value.
    """
    per_member = []
    for ws in ws_list:
        lm = _effective_landmarks(ws)
        axis = _face_axis(lm)
        if axis is None:
            continue
        cx, cy, dx, dy = axis
        H = _ref_length(lm)
        if H is None or H < 1:
            continue
        px, py = _resolve_perp_sign(dy, -dx, lm, cx, cy)

        member = {}
        for key, (kx, ky) in lm.items():
            vx = kx - cx
            vy = ky - cy
            along = (vx * dx + vy * dy) / H
            perp  = (vx * px + vy * py) / H
            member[key] = (along, perp)
        per_member.append(member)

    if not per_member:
        return {}

    averaged = {}
    all_keys = set()
    for m in per_member:
        all_keys.update(m.keys())
    for key in all_keys:
        vals = [m[key] for m in per_member if key in m]
        if vals:
            avg_along = sum(v[0] for v in vals) / len(vals)
            avg_perp  = sum(v[1] for v in vals) / len(vals)
            averaged[key] = (avg_along, avg_perp)
    return averaged

# ===========================================================================
# Folder-loaded comparison sources
# ===========================================================================

def _scan_folder_for_landmark_jsons(root_folder: str, max_depth: int = 2) -> list:
    """
    Walk root_folder up to max_depth levels deep (folder itself = depth 1,
    direct subfolders = depth 2) and return absolute paths to every .json
    file that parses as a valid landmark export.

    A "valid" landmark JSON has at minimum a 'landmarks' dict.
    """
    found = []
    root_folder = os.path.abspath(root_folder)
    if not os.path.isdir(root_folder):
        return found

    # Walk with depth tracking
    for current_dir, subdirs, files in os.walk(root_folder):
        rel = os.path.relpath(current_dir, root_folder)
        depth = 1 if rel == "." else 1 + rel.count(os.sep) + 1
        if depth > max_depth:
            # Prune deeper traversal
            subdirs[:] = []
            continue

        for fname in files:
            if not fname.lower().endswith(".json"):
                continue
            full = os.path.join(current_dir, fname)
            try:
                with open(full, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("landmarks"), dict):
                    found.append(full)
            except Exception:
                # Silently skip unreadable / non-landmark JSONs
                continue

    return found


def _build_workspace_from_json(json_path: str) -> Optional[Workspace]:
    """
    Build a lightweight throwaway Workspace from a landmark JSON file.
    No image is loaded — these workspaces exist only for comparison
    (proportions + mesh), never for visualization or mapping target.
    Returns None if the JSON can't be parsed as a landmark file.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    raw = data.get("landmarks")
    if not isinstance(raw, dict):
        return None

    landmarks = {k: (v["x"], v["y"]) for k, v in raw.items()
                 if isinstance(v, dict) and v.get("x") is not None}
    if not landmarks:
        return None

    skipped = set(data.get("skipped_keys", []))
    estimated = set(k for k, v in raw.items()
                    if isinstance(v, dict) and v.get("status") == "estimated")
    estimated |= set(data.get("estimated_keys", []))
    mesh = [tuple(p) for p in data.get("mesh_landmarks", [])]

    ws = Workspace()
    ws.name = os.path.splitext(os.path.basename(json_path))[0][:32]
    ws.original_image = None  # never displayed
    ws.image_stem = os.path.splitext(os.path.basename(json_path))[0]
    ws.landmarks = landmarks
    ws.skipped_keys = skipped
    ws.estimated_keys = estimated
    ws.mesh_landmarks = mesh
    ws.current_step = TOTAL
    return ws

# ===========================================================================
# Batch processing progress dialog
# ===========================================================================

class BatchProgressDialog(tk.Toplevel):

    def __init__(self, master: FaceLandmarkApp, image_paths: list,
                 in_folder: str, out_folder: str):
        super().__init__(master)
        self.master_app = master
        self.image_paths = image_paths
        self.in_folder = in_folder
        self.out_folder = out_folder
        self.cancel_requested = False
        self.done = False

        self.title("Batch Auto-Detect")
        self.configure(bg="#1a1a2e")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.grab_set()

        n = len(image_paths)

        tk.Label(self, text=f"Processing {n} image(s)...",
                 bg="#1a1a2e", fg="#e0e0e0",
                 font=("Helvetica", 11, "bold"), padx=20
                 ).pack(pady=(14, 4))

        tk.Label(self, text=f"Output: {out_folder}",
                 bg="#1a1a2e", fg="#888888", font=("Helvetica", 8),
                 wraplength=480, justify="left", padx=20
                 ).pack(pady=(0, 8))

        # Progress bar
        self.progress_var = tk.IntVar(value=0)
        self.progress = ttk.Progressbar(
            self, orient="horizontal", length=480, mode="determinate",
            variable=self.progress_var, maximum=n)
        self.progress.pack(padx=20, pady=4)

        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(self, textvariable=self.status_var,
                 bg="#1a1a2e", fg="#cccccc", font=("Courier", 9),
                 wraplength=480, justify="left", padx=20
                 ).pack(pady=4)

        self.counts_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.counts_var,
                 bg="#1a1a2e", fg="#888888", font=("Helvetica", 9), padx=20
                 ).pack(pady=(0, 8))

        bf = tk.Frame(self, bg="#1a1a2e")
        bf.pack(pady=(4, 14))
        self.cancel_button = ttk.Button(bf, text="Cancel",
                                        command=self._on_cancel)
        self.cancel_button.pack(side="left", padx=6)
        self.close_button = ttk.Button(bf, text="Close",
                                       command=self._on_close,
                                       state="disabled")
        self.close_button.pack(side="left", padx=6)

        # Counters
        self.n_ok = 0
        self.n_no_face = 0
        self.n_failed = 0
        self.failures = []   # list of (image_path, reason)
        self.idx = 0

        self.update_idletasks()
        x = master.winfo_x() + master.winfo_width()  // 2 - self.winfo_width()  // 2
        y = master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{x}+{y}")

        # Kick off processing — one image per Tk idle cycle, keeps UI responsive
        self.after(50, self._step)

    def _on_cancel(self):
        if self.done:
            return
        self.cancel_requested = True
        self.status_var.set("Cancelling — finishing current image...")

    def _on_close(self):
        if self.done:
            self.destroy()
        else:
            # Treat as cancel
            self._on_cancel()

    def _step(self):
        if self.cancel_requested or self.idx >= len(self.image_paths):
            self._finish()
            return

        path = self.image_paths[self.idx]
        rel = os.path.relpath(path, self.in_folder)
        self.status_var.set(f"[{self.idx + 1}/{len(self.image_paths)}]  {rel}")
        self.update_idletasks()

        result = self.master_app.batch_export_one(path, self.out_folder)

        if result == "ok":
            self.n_ok += 1
        elif result == "no_face":
            self.n_no_face += 1
            self.failures.append((rel, "no face detected"))
        else:
            self.n_failed += 1
            self.failures.append((rel, result))

        self.idx += 1
        self.progress_var.set(self.idx)
        self.counts_var.set(
            f"OK: {self.n_ok}   ·   No face: {self.n_no_face}   "
            f"·   Failed: {self.n_failed}")

        # Schedule next image
        self.after(1, self._step)

    def _finish(self):
        self.done = True
        self.cancel_button.config(state="disabled")
        self.close_button.config(state="normal")
        if self.cancel_requested:
            self.status_var.set(f"Cancelled after {self.idx} image(s).")
        else:
            self.status_var.set(f"Done. Processed {self.idx} image(s).")

        # Summary popup
        msg = (f"Batch complete.\n\n"
               f"  Successful:    {self.n_ok}\n"
               f"  No face found: {self.n_no_face}\n"
               f"  Failed:        {self.n_failed}\n\n"
               f"Output folder:\n{self.out_folder}")
        if self.failures and len(self.failures) <= 12:
            msg += "\n\nFailures:\n"
            msg += "\n".join(f"  • {r}: {reason}"
                             for r, reason in self.failures[:12])
        elif self.failures:
            msg += f"\n\n({len(self.failures)} failures — too many to list)"

        messagebox.showinfo("Batch Auto-Detect", msg, parent=self)

# ===========================================================================
# Tab-selection dialog
# ===========================================================================

class CompareSelectDialog(tk.Toplevel):

    def __init__(self, master: FaceLandmarkApp, tabs: list):
        super().__init__(master)
        self.master_app = master
        self.tabs = tabs   # list of (idx, ws) — workspace tabs
        self.title("Select Tabs to Compare")
        self.resizable(False, False)
        self.configure(bg="#1a1a2e")
        self.grab_set()

        # Folder-loaded entries: per side, list of dicts
        # Each dict: {"folder": path, "workspaces": [Workspace, ...], "var": BooleanVar}
        self.folders_a: list = []
        self.folders_b: list = []

        intro = tk.Label(
            self,
            text="Select tabs for each group, or load a folder of landmark JSONs.\n"
                 "When a side has multiple sources, its measurements are averaged.",
            bg="#1a1a2e", fg="#aaaaaa", font=("Helvetica", 9),
            justify="left", padx=14)
        intro.grid(row=0, column=0, columnspan=2, sticky="w", pady=(12, 8))

        # Group headers
        tk.Label(self, text="Group A", bg="#1a1a2e", fg="#00d4ff",
                 font=("Helvetica", 10, "bold")
                 ).grid(row=1, column=0, padx=14, sticky="w")
        tk.Label(self, text="Group B", bg="#1a1a2e", fg="#ff6b35",
                 font=("Helvetica", 10, "bold")
                 ).grid(row=1, column=1, padx=14, sticky="w")

        # Add Folder buttons
        ttk.Button(self, text="+ Add Folder...",
                   command=lambda: self._add_folder("A")
                   ).grid(row=2, column=0, padx=14, pady=(2, 4), sticky="w")
        ttk.Button(self, text="+ Add Folder...",
                   command=lambda: self._add_folder("B")
                   ).grid(row=2, column=1, padx=14, pady=(2, 4), sticky="w")

        # List frames (scrollable in case of many folders/tabs)
        list_a_outer = tk.Frame(self, bg="#0d0d1a", padx=4, pady=4,
                                width=320, height=280)
        list_a_outer.grid(row=3, column=0, padx=14, pady=4, sticky="nsew")
        list_a_outer.grid_propagate(False)
        list_b_outer = tk.Frame(self, bg="#0d0d1a", padx=4, pady=4,
                                width=320, height=280)
        list_b_outer.grid(row=3, column=1, padx=14, pady=4, sticky="nsew")
        list_b_outer.grid_propagate(False)

        # Inner content frames (rebuilt by _refresh_lists)
        self.list_a_frame = list_a_outer
        self.list_b_frame = list_b_outer
        self._a_widgets: list = []
        self._b_widgets: list = []

        # Per-tab variables (parallel to self.tabs)
        self.vars_a: list[tk.BooleanVar] = []
        self.vars_b: list[tk.BooleanVar] = []
        for _ in tabs:
            self.vars_a.append(tk.BooleanVar(value=False))
            self.vars_b.append(tk.BooleanVar(value=False))

        # Default: first tab in A, second in B (matches previous behavior)
        if len(self.vars_a) >= 1:
            self.vars_a[0].set(True)
        if len(self.vars_b) >= 2:
            self.vars_b[1].set(True)
        elif len(self.vars_b) >= 1:
            self.vars_b[0].set(True)

        # Status / hint line
        self.hint_var = tk.StringVar(value="")
        tk.Label(self, textvariable=self.hint_var, bg="#1a1a2e", fg="#888888",
                 font=("Helvetica", 9), padx=14
                 ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # Buttons
        bf = tk.Frame(self, bg="#1a1a2e")
        bf.grid(row=5, column=0, columnspan=2, pady=14)
        self.compare_button = ttk.Button(bf, text="Compare", command=self._go)
        self.compare_button.pack(side="left", padx=6)
        ttk.Button(bf, text="Cancel", command=self.destroy).pack(side="left", padx=6)

        self._refresh_lists()
        self._refresh_state()

        self.update_idletasks()
        x = master.winfo_x() + master.winfo_width()  // 2 - self.winfo_width()  // 2
        y = master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{x}+{y}")

    # -- folder loading -----------------------------------------------

    def _add_folder(self, side: str):
        folder = filedialog.askdirectory(
            title=f"Select folder of landmark JSONs for Group {side}",
            parent=self)
        if not folder:
            return

        # Avoid duplicate folder add
        existing = self.folders_a if side == "A" else self.folders_b
        for entry in existing:
            if os.path.normpath(entry["folder"]) == os.path.normpath(folder):
                messagebox.showinfo(
                    "Already added",
                    f"That folder is already added to Group {side}.",
                    parent=self)
                return

        json_paths = _scan_folder_for_landmark_jsons(folder, max_depth=2)
        if not json_paths:
            messagebox.showwarning(
                "No landmark JSONs found",
                f"Searched {os.path.basename(folder)} (depth 2) and found no\n"
                "valid landmark JSON files.",
                parent=self)
            return

        # Build throwaway workspaces (no images)
        loaded_ws = []
        bad = 0
        for p in json_paths:
            ws = _build_workspace_from_json(p)
            if ws is not None:
                loaded_ws.append(ws)
            else:
                bad += 1

        if not loaded_ws:
            messagebox.showwarning(
                "All files failed to parse",
                f"Found {len(json_paths)} JSON file(s) but none parsed as\n"
                "valid landmark exports.",
                parent=self)
            return

        entry = {
            "folder": folder,
            "workspaces": loaded_ws,
            "var": tk.BooleanVar(value=True),  # default on when first added
            "n_with_mesh": sum(1 for w in loaded_ws if w.mesh_landmarks),
        }
        if side == "A":
            self.folders_a.append(entry)
        else:
            self.folders_b.append(entry)

        self._refresh_lists()
        self._refresh_state()

    def _remove_folder(self, side: str, idx: int):
        target = self.folders_a if side == "A" else self.folders_b
        if 0 <= idx < len(target):
            target.pop(idx)
            self._refresh_lists()
            self._refresh_state()

    # -- list rendering ----------------------------------------------

    def _refresh_lists(self):
        # Clear existing widgets
        for w in self._a_widgets:
            w.destroy()
        for w in self._b_widgets:
            w.destroy()
        self._a_widgets.clear()
        self._b_widgets.clear()

        # Render side A
        self._render_side(
            self.list_a_frame, self._a_widgets,
            self.tabs, self.vars_a, self.folders_a, "A")
        # Render side B
        self._render_side(
            self.list_b_frame, self._b_widgets,
            self.tabs, self.vars_b, self.folders_b, "B")

    def _render_side(self, parent_frame, widget_list,
                     tabs, tab_vars, folders, side):
        # Tab checkboxes
        for i, (_, ws) in enumerate(tabs):
            cb = tk.Checkbutton(
                parent_frame, text=ws.name, variable=tab_vars[i],
                bg="#0d0d1a", fg="#cccccc", selectcolor="#0f3460",
                activebackground="#0d0d1a", activeforeground="#ffffff",
                font=("Helvetica", 9), anchor="w", padx=4, pady=2,
                command=lambda idx=i, s=side: self._on_tab_check(s, idx))
            cb.pack(fill="x", anchor="w")
            widget_list.append(cb)

        # Separator if both tabs and folders exist
        if tabs and folders:
            sep = tk.Frame(parent_frame, bg="#22335a", height=1)
            sep.pack(fill="x", padx=4, pady=4)
            widget_list.append(sep)

        # Folder entries — one row per folder, with a checkbox + remove (×)
        for fi, entry in enumerate(folders):
            row = tk.Frame(parent_frame, bg="#0d0d1a")
            row.pack(fill="x", anchor="w")

            n = len(entry["workspaces"])
            n_mesh = entry["n_with_mesh"]
            folder_name = os.path.basename(entry["folder"]) or entry["folder"]
            label = f"📁 {folder_name}  ({n} files"
            if n_mesh < n:
                label += f", {n_mesh} w/ mesh"
            label += ")"

            cb = tk.Checkbutton(
                row, text=label, variable=entry["var"],
                bg="#0d0d1a", fg="#a8ff3e", selectcolor="#0f3460",
                activebackground="#0d0d1a", activeforeground="#ffffff",
                font=("Helvetica", 9), anchor="w", padx=4, pady=2,
                command=self._refresh_state)
            cb.pack(side="left", fill="x", expand=True)

            close_btn = tk.Label(
                row, text=" × ", bg="#0d0d1a", fg="#666666",
                font=("Helvetica", 10, "bold"), cursor="hand2", padx=6)
            close_btn.pack(side="right")
            close_btn.bind(
                "<Button-1>",
                lambda _e, s=side, idx=fi: self._remove_folder(s, idx))
            close_btn.bind("<Enter>", lambda _e, b=close_btn: b.config(fg="#ff4455"))
            close_btn.bind("<Leave>", lambda _e, b=close_btn: b.config(fg="#666666"))

            widget_list.append(row)

    # -- check handling -----------------------------------------------

    def _on_tab_check(self, side: str, idx: int):
        # If a tab is checked on both sides, uncheck the other side (no double-counting)
        if side == "A" and self.vars_a[idx].get() and self.vars_b[idx].get():
            self.vars_b[idx].set(False)
        elif side == "B" and self.vars_b[idx].get() and self.vars_a[idx].get():
            self.vars_a[idx].set(False)
        self._refresh_state()

    def _refresh_state(self):
        ws_list_a = self._collect_side("A")
        ws_list_b = self._collect_side("B")
        n_a = len(ws_list_a)
        n_b = len(ws_list_b)
        valid = n_a >= 1 and n_b >= 1

        if not valid:
            self.hint_var.set("  Select at least one source on each side.")
        else:
            def desc(n, side):
                if n == 1:
                    return f"Group {side}: 1 source"
                return f"Group {side}: {n} sources (averaged)"
            self.hint_var.set("  " + desc(n_a, "A") + "   ·   " + desc(n_b, "B"))

        if valid:
            self.compare_button.state(["!disabled"])
        else:
            self.compare_button.state(["disabled"])

    # -- collection ---------------------------------------------------

    def _collect_side(self, side: str) -> list:
        result = []
        if side == "A":
            tab_vars = self.vars_a
            folders = self.folders_a
        else:
            tab_vars = self.vars_b
            folders = self.folders_b

        # Tab workspaces
        for i, v in enumerate(tab_vars):
            if v.get():
                result.append(self.tabs[i][1])

        # Folder workspaces (only if folder is checked)
        for entry in folders:
            if entry["var"].get():
                result.extend(entry["workspaces"])

        return result

    # -- go -----------------------------------------------------------

    def _go(self):
        ws_list_a = self._collect_side("A")
        ws_list_b = self._collect_side("B")
        if not ws_list_a or not ws_list_b:
            messagebox.showwarning(
                "Selection",
                "Each group needs at least one source.",
                parent=self)
            return

        result = run_group_comparison(ws_list_a, ws_list_b)
        self.destroy()
        CompareResultsWindow(self.master_app, ws_list_a, ws_list_b, result)

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

    def __init__(self, master, ws_list_a: list, ws_list_b: list, result: dict):
        super().__init__(master)
        self.master_app = master
        self.ws_list_a = ws_list_a
        self.ws_list_b = ws_list_b
        self.result = result

        # Build display names
        n_a = len(ws_list_a)
        n_b = len(ws_list_b)
        if n_a == 1:
            self.name_a = ws_list_a[0].name
            self.label_a_short = ws_list_a[0].name[:10]
            label_a_full = ws_list_a[0].name
        else:
            self.name_a = f"Group A ({n_a} tabs)"
            self.label_a_short = f"Grp A ({n_a})"
            label_a_full = self.name_a + ": " + ", ".join(ws.name for ws in ws_list_a)
        if n_b == 1:
            self.name_b = ws_list_b[0].name
            self.label_b_short = ws_list_b[0].name[:10]
            label_b_full = ws_list_b[0].name
        else:
            self.name_b = f"Group B ({n_b} tabs)"
            self.label_b_short = f"Grp B ({n_b})"
            label_b_full = self.name_b + ": " + ", ".join(ws.name for ws in ws_list_b)

        self.title(f"Comparison  ·  {self.name_a}  vs  {self.name_b}")
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

        mesh_data = result.get("mesh")
        if mesh_data is not None:
            cols = tk.Frame(hdr, bg="#0f3460")
            cols.pack()

            left = tk.Frame(cols, bg="#0f3460", padx=20)
            left.pack(side="left")
            tk.Label(left, text="50-Metric Score", bg="#0f3460", fg="#8899bb",
                     font=("Helvetica", 9)).pack()
            tk.Label(left, text=f"{score:.1f} / 100",
                     bg="#0f3460", fg=self._score_color(score),
                     font=("Helvetica", 28, "bold")).pack()
            tk.Label(left, text="proportional anatomy",
                     bg="#0f3460", fg="#667799", font=("Helvetica", 8)).pack()

            tk.Frame(cols, bg="#22335a", width=1, height=80).pack(side="left", fill="y")

            right = tk.Frame(cols, bg="#0f3460", padx=20)
            right.pack(side="left")
            mesh_score = mesh_data["score"]
            tk.Label(right, text="Dense Mesh Score", bg="#0f3460", fg="#8899bb",
                     font=("Helvetica", 9)).pack()
            tk.Label(right, text=f"{mesh_score:.1f} / 100",
                     bg="#0f3460", fg=self._score_color(mesh_score),
                     font=("Helvetica", 28, "bold")).pack()
            tk.Label(right,
                     text=f"{mesh_data['n_points']} points · "
                          f"avg {mesh_data['mean_dist'] * 100:.2f}% face-height",
                     bg="#0f3460", fg="#667799", font=("Helvetica", 8)).pack()
        else:
            tk.Label(hdr, text="Overall Similarity Score", bg="#0f3460", fg="#8899bb",
                     font=("Helvetica", 10)).pack()
            tk.Label(hdr, text=f"{score:.1f} / 100",
                     bg="#0f3460", fg=self._score_color(score),
                     font=("Helvetica", 36, "bold")).pack()
            # Why no mesh?
            reason = self._mesh_unavailable_reason()
            if reason:
                tk.Label(hdr, text=f"Dense mesh: {reason}",
                         bg="#0f3460", fg="#778899",
                         font=("Helvetica", 8, "italic")).pack(pady=(4, 0))

        tk.Label(hdr, text=f"{self.name_a}   vs   {self.name_b}",
                 bg="#0f3460", fg="#778899", font=("Helvetica", 10)
                 ).pack(pady=(8, 0))

        # Show full member lists when groups are multi
        if n_a > 1 or n_b > 1:
            tk.Label(hdr, text=f"A: {label_a_full}",
                     bg="#0f3460", fg="#5f6f8a",
                     font=("Helvetica", 8), wraplength=780, justify="left"
                     ).pack(pady=(2, 0), padx=12)
            tk.Label(hdr, text=f"B: {label_b_full}",
                     bg="#0f3460", fg="#5f6f8a",
                     font=("Helvetica", 8), wraplength=780, justify="left"
                     ).pack(pady=(0, 0), padx=12)

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
                     font=("Helvetica", 8, "bold"), width=22, anchor="w"
                     ).pack(side="left")

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
            ("Metric",                 28, "w"),
            (self.label_a_short,       11, "center"),
            (self.label_b_short,       11, "center"),
            ("% Diff",                  7, "center"),
            ("Match",                  14, "center"),
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
            if cat != current_cat:
                current_cat = cat
                cat_color = CATEGORY_META[cat]["color"]
                sec = tk.Frame(inner, bg="#0a0a20")
                sec.grid(row=row_idx, column=0, columnspan=5, sticky="ew", pady=(6, 0))
                cs = cat_scores.get(cat)
                hdr_txt = cat + (f"   —   {cs:.0f}/100" if cs is not None else "")
                tk.Label(sec, text=f"  {hdr_txt}", bg="#0a0a20", fg=cat_color,
                         font=("Helvetica", 9, "bold"), pady=3, anchor="w"
                         ).pack(fill="x")
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

            bar_f = tk.Frame(inner, bg=bg, width=110, height=10)
            bar_f.grid(row=row_idx, column=4, padx=6, pady=4)
            bar_f.pack_propagate(False)
            if pct is not None:
                fill_w = max(0, int(110 * max(0, 100 - pct*2) / 100))
                color  = self._pct_color(pct)
                tk.Frame(bar_f, bg=color,    width=fill_w,       height=10).place(x=0, y=0)
                tk.Frame(bar_f, bg="#1a1a33", width=110-fill_w,  height=10).place(x=fill_w, y=0)

            row_idx += 1

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
                   command=lambda: self._export_csv(self.name_a, self.name_b, result)
                   ).pack(side="left", padx=12)

        # Map buttons — ONLY when one side is exactly 1 source AND that source
        # has an image (folder-loaded sources have no image, so they can't be
        # the mapping target).
        solo_a = (n_a == 1) and (ws_list_a[0].original_image is not None)
        solo_b = (n_b == 1) and (ws_list_b[0].original_image is not None)

        if solo_a or solo_b:
            tk.Frame(btn_row, bg="#222244", width=1).pack(side="left", fill="y", padx=8)
            tk.Label(btn_row, text="Map onto new tab:",
                     bg="#0d0d1e", fg="#778899", font=("Helvetica", 9)
                     ).pack(side="left")

            # If A is solo, we can map B's averaged proportions onto A.
            if solo_a:
                ttk.Button(
                    btn_row,
                    text=f"  {self.name_b[:14]}  →  {self.name_a[:14]}  ",
                    command=lambda: self._open_map_group_to_solo("b_to_a")
                ).pack(side="left", padx=4)
            # If B is solo, we can map A's averaged proportions onto B.
            if solo_b:
                ttk.Button(
                    btn_row,
                    text=f"  {self.name_a[:14]}  →  {self.name_b[:14]}  ",
                    command=lambda: self._open_map_group_to_solo("a_to_b")
                ).pack(side="left", padx=4)

        ttk.Button(btn_row, text="Close", command=self.destroy
                   ).pack(side="right", padx=12)

        self.update_idletasks()
        mx = master.winfo_x() + master.winfo_width()  // 2 - self.winfo_width()  // 2
        my = master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{mx}+{my}")

    def _mesh_unavailable_reason(self) -> Optional[str]:
        """Explain why dense mesh score isn't shown."""
        a_lacking = [ws.name for ws in self.ws_list_a if not ws.mesh_landmarks]
        b_lacking = [ws.name for ws in self.ws_list_b if not ws.mesh_landmarks]
        if a_lacking and b_lacking:
            return "tabs in both groups lack mesh data (run Auto-Detect)"
        if a_lacking:
            return f"{len(a_lacking)} tab(s) in Group A lack mesh data"
        if b_lacking:
            return f"{len(b_lacking)} tab(s) in Group B lack mesh data"
        return None
    def _open_map_group_to_solo(self, direction: str):
        """
        Map averaged proportions from the source group onto the solo
        target tab. direction='a_to_b' means A (group or solo) maps onto B (solo);
        direction='b_to_a' means B maps onto A (solo).
        """
        if direction == "a_to_b":
            source_group = self.ws_list_a
            solo_target  = self.ws_list_b[0]
            source_name  = self.name_a
            target_name  = self.name_b
        else:
            source_group = self.ws_list_b
            solo_target  = self.ws_list_a[0]
            source_name  = self.name_b
            target_name  = self.name_a

        # Compute the source group's averaged named-landmark canonical positions
        target_canonical = average_named_landmarks_canonical(source_group)
        if not target_canonical:
            messagebox.showerror(
                "Cannot map",
                "Could not compute averaged proportions from the source group.",
                parent=self)
            return

        self.master_app.open_mapping_tab_from_canonical(
            solo_target, target_canonical, source_name, target_name)
        self.destroy()

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

