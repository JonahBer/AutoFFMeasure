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
    ("face_outline_lip_crease",   "Side Face Outline - line up with lip crease",       True),
    ("face_under_ear",            "Side Face Under Ear - where ear and face meet",      True),
    ("face_above_ear",            "Side Face Above Ear - where ear and face meet",      True),
    ("eyebrow_outside",           "Outside Eyebrow",                                    True),
    ("eyebrow_inside",            "Inside Eyebrow",                                     True),
    ("eyebrow_under_apex",        "Under Apex Eyebrow",                                 True),
    ("eyebrow_upper_apex",        "Upper Apex Eyebrow",                                 True),
    ("eye_upper_apex",            "Upper Apex Eye",                                     True),
    ("eye_upper_apex_crease",     "Upper Apex Eye Crease Line",                         True),
    ("eye_outside_corner",        "Outside Eye White Corner",                           True),
    ("eye_inside_corner",         "Inside Eye White Corner",                            True),
    ("eye_under_apex",            "Under Apex Eye",                                     True),
    ("nose_bottom_middle",        "Bottom Middle of Nose",                              False),
    ("nose_nostril_outside",      "Side of Nose - Nostril Outside",                     True),
    ("chin_outer_side",           "Side of Outer Chin",                                 True),
    ("mouth_upper_apex_side",     "Upper Mouth Apex Side",                              True),
    ("mouth_upper_low_u",         "Upper Mouth Low U Apex",                             False),
    ("lips_center_meet",          "Between Lips Where They Meet - Center",              False),
    ("lips_outer_crease",         "Outer Crease of Lips",                               True),
    ("mouth_under_apex",          "Under Mouth Apex",                                   True),
    ("chin_bottom_apex",          "Bottom of Chin Apex",                                False),
    ("neck_face_corner",          "Neck Meets Face Corner",                             True),
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

LEFT_COLOR   = "#00d4ff"
RIGHT_COLOR  = "#ff6b35"
SINGLE_COLOR = "#a8ff3e"
RADIUS       = 5
HIT_RADIUS   = 12

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
    original_image: object = None        # PIL Image or None
    image_stem:     str   = "landmarks"
    landmarks:      dict  = field(default_factory=dict)   # key -> (x, y) original-space
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
                 font=("Helvetica", 10), padx=14).pack(side="right")

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
            self._draw_marker(int(ox * sf), int(oy * sf), key, _marker_color(key))

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.MIN_WIDTH // 2, self.MIN_HEIGHT // 2,
            text="Press  Ctrl+V  to paste an image\nor use  Open",
            fill="#333355", font=("Helvetica", 16), justify="center")

    def _draw_marker(self, cx: int, cy: int, key: str, color: str, highlight: bool = False):
        r   = RADIUS
        tag = f"mk_{key}"
        self.canvas.delete(tag)
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                 fill=color,
                                 outline="#ffff00" if highlight else "#ffffff",
                                 width=2 if highlight else 1,
                                 tags=("marker", tag))
        self.canvas.create_text(cx + r + 4, cy, text=_short_label(key), fill=color,
                                 font=("Helvetica", 7), anchor="w", tags=("marker", tag))

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
        self._update_list(p["key"], ox, oy)
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
        key = self._drag_key
        ox, oy = self.ws.landmarks[key]
        self._draw_marker(int(ox*self.ws.scale_factor), int(oy*self.ws.scale_factor),
                          key, _marker_color(key), highlight=False)
        self.status_var.set(f"Moved  {key}  ->  ({ox}, {oy})")
        self._drag_key = None
        self.canvas.config(cursor="crosshair" if self.ws.marking_mode else "arrow")

    # ------------------------------------------------------------------
    # Marking mode
    # ------------------------------------------------------------------

    def start_marking(self):
        if self.ws.original_image is None:
            messagebox.showinfo("No image", "Load an image first."); return
        self.ws.marking_mode = True
        self.ws.current_step = 0
        self.ws.landmarks.clear()
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

    def _finish_marking(self):
        self.ws.marking_mode = False
        self.prompt_var.set(f"  All {TOTAL} landmarks recorded.  Export with JSON or CSV")
        self.step_var.set(f"{TOTAL} / {TOTAL}  complete")
        self.canvas.config(cursor="arrow")
        messagebox.showinfo("Done", f"All {TOTAL} landmarks recorded!\nExport via the buttons or File menu.")

    def _update_list(self, key: str, ox: int, oy: int):
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
        self.landmark_listbox.insert(idx, f"{key:<28}  ({ox:>4}, {oy:>4})")
        self.landmark_listbox.itemconfig(idx, fg=color)

    # ------------------------------------------------------------------
    # Undo / clear
    # ------------------------------------------------------------------

    def undo_last(self):
        if self.ws.current_step == 0: return
        self.ws.current_step -= 1
        key = PROMPTS[self.ws.current_step]["key"]
        self.ws.landmarks.pop(key, None)
        if self.landmark_listbox.size() > 0:
            self.landmark_listbox.delete(tk.END)
        self.refresh_display()
        if not self.ws.marking_mode:
            self.ws.marking_mode = True
        self._advance_prompt()
        self.status_var.set(f"Undid:  {key}")

    def clear_landmarks(self, silent: bool = False):
        self.ws.landmarks.clear()
        self.ws.current_step = 0
        self.ws.marking_mode = False
        self.landmark_listbox.delete(0, tk.END)
        self.canvas.delete("marker")
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
        if not self.ws.landmarks:
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
            "landmarks":       {k: {"x": v[0], "y": v[1]}
                                for k, v in self.ws.landmarks.items()},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        self.status_var.set(
            f"Exported {os.path.basename(path)}  +  {os.path.basename(img_path)}")

    def export_csv(self):
        if not self.ws.landmarks:
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
            writer.writerow(["# paired_image", os.path.basename(img_path)])
            writer.writerow(["key", "x", "y"])
            for k, (x, y) in self.ws.landmarks.items():
                writer.writerow([k, x, y])
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
            if   ext == ".json": loaded, hint = self._parse_json(path)
            elif ext == ".csv":  loaded, hint = self._parse_csv(path)
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
        self.ws.landmarks    = loaded
        self.ws.current_step = len(loaded)
        self.landmark_listbox.delete(0, tk.END)
        for k, (x, y) in loaded.items():
            self.landmark_listbox.insert(tk.END, f"{k:<28}  ({x:>4}, {y:>4})")
            side = k.split("_")[-1]
            c = LEFT_COLOR if side=="L" else RIGHT_COLOR if side=="R" else SINGLE_COLOR
            self.landmark_listbox.itemconfig(tk.END, fg=c)
        self.refresh_display()
        if len(loaded) >= TOTAL:
            self.prompt_var.set(f"  All {TOTAL} landmarks loaded from file.")
            self.step_var.set(f"{TOTAL} / {TOTAL}  complete")
        else:
            self.ws.marking_mode = True
            self._advance_prompt()
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
        loaded = {k: (v["x"], v["y"]) for k, v in data.get("landmarks", {}).items()}
        return loaded, data.get("paired_image")

    def _parse_csv(self, path):
        loaded, hint = {}, None
        with open(path, newline="") as f:
            for row in csv.reader(f):
                if not row: continue
                if row[0].startswith("#"):
                    if len(row) >= 2 and "paired_image" in row[0]:
                        hint = row[1].strip()
                    continue
                if row[0] == "key": continue
                loaded[row[0]] = (int(row[1]), int(row[2]))
        return loaded, hint

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = FaceLandmarkApp()
    app.mainloop()


# ===========================================================================
# Proportional comparison engine
# ===========================================================================

def _pt(lm: dict, key: str):
    """Return landmark point or None."""
    return lm.get(key)

def _avg(*pts):
    valid = [p for p in pts if p is not None]
    if not valid: return None
    return (sum(p[0] for p in valid)/len(valid), sum(p[1] for p in valid)/len(valid))

def _hdist(a, b):
    if a is None or b is None: return None
    return abs(a[0] - b[0])

def _vdist(a, b):
    if a is None or b is None: return None
    return abs(a[1] - b[1])

def _edist(a, b):
    if a is None or b is None: return None
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5


# Each metric: (display_name, compute_fn(lm) -> float|None)
def _build_metrics() -> list[tuple[str, object]]:
    def face_width(lm):
        return _hdist(_pt(lm,"face_outline_lip_crease_L"), _pt(lm,"face_outline_lip_crease_R"))

    def eye_width(lm):
        l = _hdist(_pt(lm,"eye_outside_corner_L"), _pt(lm,"eye_inside_corner_L"))
        r = _hdist(_pt(lm,"eye_outside_corner_R"), _pt(lm,"eye_inside_corner_R"))
        valid = [v for v in (l, r) if v is not None]
        return sum(valid)/len(valid) if valid else None

    def nose_width(lm):
        return _hdist(_pt(lm,"nose_nostril_outside_L"), _pt(lm,"nose_nostril_outside_R"))

    def mouth_width(lm):
        return _hdist(_pt(lm,"lips_outer_crease_L"), _pt(lm,"lips_outer_crease_R"))

    def between_eyes(lm):
        return _hdist(_pt(lm,"eye_inside_corner_L"), _pt(lm,"eye_inside_corner_R"))

    def between_brows(lm):
        return _hdist(_pt(lm,"eyebrow_inside_L"), _pt(lm,"eyebrow_inside_R"))

    def chin_to_brow(lm):
        brow = _avg(_pt(lm,"eyebrow_upper_apex_L"), _pt(lm,"eyebrow_upper_apex_R"))
        return _vdist(_pt(lm,"chin_bottom_apex"), brow)

    def mouth_height(lm):
        under = _avg(_pt(lm,"mouth_under_apex_L"), _pt(lm,"mouth_under_apex_R"))
        return _vdist(_pt(lm,"mouth_upper_low_u"), under)

    def eye_height(lm):
        l = _vdist(_pt(lm,"eye_upper_apex_L"), _pt(lm,"eye_under_apex_L"))
        r = _vdist(_pt(lm,"eye_upper_apex_R"), _pt(lm,"eye_under_apex_R"))
        valid = [v for v in (l, r) if v is not None]
        return sum(valid)/len(valid) if valid else None

    def mouth_to_chin(lm):
        return _vdist(_pt(lm,"mouth_upper_low_u"), _pt(lm,"chin_bottom_apex"))

    def nose_to_chin(lm):
        return _vdist(_pt(lm,"nose_bottom_middle"), _pt(lm,"chin_bottom_apex"))

    def nose_to_mouth(lm):
        return _vdist(_pt(lm,"nose_bottom_middle"), _pt(lm,"mouth_upper_low_u"))

    def mouth_to_eye_bottom(lm):
        eye_b = _avg(_pt(lm,"eye_under_apex_L"), _pt(lm,"eye_under_apex_R"))
        return _vdist(_pt(lm,"mouth_upper_low_u"), eye_b)

    def mouth_to_brow_bottom(lm):
        brow_b = _avg(_pt(lm,"eyebrow_under_apex_L"), _pt(lm,"eyebrow_under_apex_R"))
        return _vdist(_pt(lm,"mouth_upper_low_u"), brow_b)

    def eye_top_to_brow_bottom(lm):
        eye_t  = _avg(_pt(lm,"eye_upper_apex_L"), _pt(lm,"eye_upper_apex_R"))
        brow_b = _avg(_pt(lm,"eyebrow_under_apex_L"), _pt(lm,"eyebrow_under_apex_R"))
        return _vdist(eye_t, brow_b)

    return [
        ("Face Width",                    face_width),
        ("Eye Width (avg)",               eye_width),
        ("Nose Width",                    nose_width),
        ("Mouth Width",                   mouth_width),
        ("Between Eyes Width",            between_eyes),
        ("Between Eyebrows Width",        between_brows),
        ("Chin to Eyebrow Height",        chin_to_brow),
        ("Mouth Height",                  mouth_height),
        ("Eye Height (avg)",              eye_height),
        ("Top Mouth to Chin",             mouth_to_chin),
        ("Bottom Nose to Chin",           nose_to_chin),
        ("Nose to Top of Mouth",          nose_to_mouth),
        ("Top of Mouth to Bottom of Eye", mouth_to_eye_bottom),
        ("Top of Mouth to Bottom of Eyebrow", mouth_to_brow_bottom),
        ("Top of Eye to Bottom of Eyebrow",   eye_top_to_brow_bottom),
    ]

METRICS = _build_metrics()


def compute_proportions(landmarks: dict) -> dict[str, float]:
    """
    Compute each metric as a ratio normalised by face height
    (chin_bottom_apex -> midpoint of eyebrow_upper_apex L/R).
    Falls back to face_width if face height is unavailable.
    Returns {metric_name: normalised_ratio} for all computable metrics.
    """
    # Reference: face height
    brow_mid = _avg(landmarks.get("eyebrow_upper_apex_L"), landmarks.get("eyebrow_upper_apex_R"))
    ref = _vdist(landmarks.get("chin_bottom_apex"), brow_mid)

    # Fallback: face width
    if ref is None or ref < 1:
        ref = _hdist(landmarks.get("face_outline_lip_crease_L"),
                     landmarks.get("face_outline_lip_crease_R"))

    # Second fallback: outer eye corners
    if ref is None or ref < 1:
        ref = _hdist(landmarks.get("eye_outside_corner_L"),
                     landmarks.get("eye_outside_corner_R"))

    if ref is None or ref < 1:
        return {}   # can't normalise at all

    result = {}
    for name, fn in METRICS:
        val = fn(landmarks)
        if val is not None and val >= 0:
            result[name] = val / ref
    return result


def run_comparison(ws_a: Workspace, ws_b: Workspace) -> dict:
    """
    Compare proportions of two workspaces.
    Returns a dict with keys:
      'metrics': list of (name, ratio_a, ratio_b, pct_diff)
      'score':   float 0-100
      'missing': list of metric names where one/both faces lacked data
    """
    pa = compute_proportions(ws_a.landmarks)
    pb = compute_proportions(ws_b.landmarks)

    rows   = []
    diffs  = []
    missing = []

    for name, _ in METRICS:
        va = pa.get(name)
        vb = pb.get(name)
        if va is None or vb is None:
            missing.append(name)
            rows.append((name, va, vb, None))
            continue
        avg = (va + vb) / 2
        pct = abs(va - vb) / avg * 100 if avg > 1e-9 else 0.0
        diffs.append(pct)
        rows.append((name, va, vb, pct))

    score = max(0.0, 100.0 - (sum(diffs) / len(diffs))) if diffs else 0.0
    return {"metrics": rows, "score": score, "missing": missing}


# ===========================================================================
# Tab-selection dialog
# ===========================================================================

class CompareSelectDialog(tk.Toplevel):
    """Small dialog to pick two tabs for comparison."""

    def __init__(self, master: FaceLandmarkApp, tabs: list[tuple[int, Workspace]]):
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

        cb_a = ttk.Combobox(self, textvariable=self.var_a, values=names,
                             state="readonly", width=22)
        cb_b = ttk.Combobox(self, textvariable=self.var_b, values=names,
                             state="readonly", width=22)
        cb_a.grid(row=0, column=1, padx=14, pady=(16,4))
        cb_b.grid(row=1, column=1, padx=14, pady=4)

        btn_frame = tk.Frame(self, bg="#1a1a2e")
        btn_frame.grid(row=2, column=0, columnspan=2, pady=14)
        ttk.Button(btn_frame, text="Compare", command=self._go).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Cancel",  command=self.destroy).pack(side="left", padx=6)

        self.update_idletasks()
        x = master.winfo_x() + master.winfo_width()  // 2 - self.winfo_width()  // 2
        y = master.winfo_y() + master.winfo_height() // 2 - self.winfo_height() // 2
        self.geometry(f"+{x}+{y}")

    def _go(self):
        name_a = self.var_a.get()
        name_b = self.var_b.get()
        if name_a == name_b:
            messagebox.showwarning("Same Tab", "Please select two different tabs.", parent=self)
            return
        ws_a = next(ws for _, ws in self.tabs if ws.name == name_a)
        ws_b = next(ws for _, ws in self.tabs if ws.name == name_b)
        result = run_comparison(ws_a, ws_b)
        self.destroy()
        CompareResultsWindow(self.master_app, name_a, name_b, result)


# ===========================================================================
# Results window
# ===========================================================================

class CompareResultsWindow(tk.Toplevel):
    """Displays the proportional comparison results in a scrollable table."""

    # Colour scale: green (0% diff) -> yellow (25%) -> red (50%+)
    @staticmethod
    def _diff_color(pct: float) -> str:
        if pct is None: return "#444466"
        t = min(pct / 50.0, 1.0)          # 0 = perfect, 1 = 50%+ diff
        if t < 0.5:
            # green -> yellow
            r = int(255 * t * 2)
            g = 210
        else:
            # yellow -> red
            r = 220
            g = int(210 * (1 - (t - 0.5) * 2))
        return f"#{r:02x}{g:02x}30"

    def __init__(self, master, name_a: str, name_b: str, result: dict):
        super().__init__(master)
        self.title(f"Comparison: {name_a}  vs  {name_b}")
        self.configure(bg="#0d0d1e")
        self.minsize(720, 520)
        self.grab_set()

        score   = result["score"]
        metrics = result["metrics"]
        missing = result["missing"]

        # ── Score header ──────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#0f3460", pady=10)
        hdr.pack(fill="x")
        score_color = self._diff_color(100 - score)
        tk.Label(hdr, text=f"Similarity Score", bg="#0f3460", fg="#aaaacc",
                 font=("Helvetica", 11)).pack()
        tk.Label(hdr, text=f"{score:.1f} / 100", bg="#0f3460",
                 fg=score_color, font=("Helvetica", 32, "bold")).pack()
        tk.Label(hdr, text=f"{name_a}   vs   {name_b}",
                 bg="#0f3460", fg="#888899", font=("Helvetica", 10)).pack()

        # ── Column headers ────────────────────────────────────────────
        col_frame = tk.Frame(self, bg="#12122a")
        col_frame.pack(fill="x", padx=10, pady=(10, 0))
        for col, width, anchor in [
            ("Metric",          30, "w"),
            (f"{name_a[:12]} ratio", 12, "center"),
            (f"{name_b[:12]} ratio", 12, "center"),
            ("% Diff",          8,  "center"),
            ("",                16, "center"),   # bar
        ]:
            tk.Label(col_frame, text=col, bg="#12122a", fg="#888899",
                     font=("Helvetica", 9, "bold"),
                     width=width, anchor=anchor).pack(side="left", padx=2)

        # ── Scrollable rows ───────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg="#0d0d1e")
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=4)

        scroll = ttk.Scrollbar(canvas_frame, orient="vertical")
        scroll.pack(side="right", fill="y")

        rows_canvas = tk.Canvas(canvas_frame, bg="#0d0d1e", highlightthickness=0,
                                yscrollcommand=scroll.set)
        rows_canvas.pack(side="left", fill="both", expand=True)
        scroll.config(command=rows_canvas.yview)

        rows_inner = tk.Frame(rows_canvas, bg="#0d0d1e")
        rows_canvas.create_window((0, 0), window=rows_inner, anchor="nw")
        rows_inner.bind("<Configure>",
                        lambda e: rows_canvas.config(scrollregion=rows_canvas.bbox("all")))

        for i, (name, va, vb, pct) in enumerate(metrics):
            row_bg = "#0d0d1e" if i % 2 == 0 else "#111128"

            tk.Label(rows_inner, text=name, bg=row_bg, fg="#ccccdd",
                     font=("Helvetica", 9), width=30, anchor="w",
                     pady=5, padx=6).grid(row=i, column=0, sticky="w")

            for col_idx, val in enumerate((va, vb)):
                txt = f"{val:.4f}" if val is not None else "—"
                tk.Label(rows_inner, text=txt, bg=row_bg, fg="#aaaacc",
                         font=("Courier", 9), width=12,
                         anchor="center").grid(row=i, column=col_idx+1)

            # % diff cell
            if pct is not None:
                pct_txt = f"{pct:.1f}%"
                pct_fg  = "#44ff88" if pct < 5 else "#ffdd44" if pct < 15 else "#ff6655"
            else:
                pct_txt, pct_fg = "N/A", "#555577"
            tk.Label(rows_inner, text=pct_txt, bg=row_bg, fg=pct_fg,
                     font=("Helvetica", 9, "bold"), width=8,
                     anchor="center").grid(row=i, column=3)

            # Visual similarity bar
            bar_outer = tk.Frame(rows_inner, bg=row_bg, width=140, height=14)
            bar_outer.grid(row=i, column=4, padx=8, pady=3)
            bar_outer.pack_propagate(False)
            if pct is not None:
                fill_pct = max(0, 100 - pct * 2)   # 0% diff = 100% bar
                bar_bg = self._diff_color(pct)
                bar_fill = tk.Frame(bar_outer, bg=bar_bg,
                                    width=int(140 * fill_pct / 100), height=14)
                bar_fill.place(x=0, y=0)
                tk.Frame(bar_outer, bg="#222233",
                         width=int(140 * (100 - fill_pct) / 100), height=14
                         ).place(x=int(140 * fill_pct / 100), y=0)

        # ── Missing landmarks note ────────────────────────────────────
        if missing:
            note = tk.Frame(self, bg="#1a1010", pady=6)
            note.pack(fill="x", padx=10, pady=(0, 4))
            tk.Label(note, text=f"  Note: {len(missing)} metric(s) skipped due to missing landmarks.",
                     bg="#1a1010", fg="#886655", font=("Helvetica", 9),
                     anchor="w").pack(side="left")

        # ── Export / close buttons ────────────────────────────────────
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
            w.writerow(["Metric", f"{name_a} ratio", f"{name_b} ratio", "% Difference"])
            for name, va, vb, pct in result["metrics"]:
                w.writerow([name,
                            f"{va:.6f}" if va is not None else "",
                            f"{vb:.6f}" if vb is not None else "",
                            f"{pct:.2f}" if pct is not None else "N/A"])
            w.writerow([])
            w.writerow(["Overall Similarity Score", f"{result['score']:.2f}"])
        messagebox.showinfo("Exported", f"Report saved to:\n{os.path.basename(path)}", parent=self)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = FaceLandmarkApp()
    app.mainloop()

