"""
face_landmark_tool.py
----------------------
Frontal-face landmark annotation tool.
- Paste (Ctrl+V) or open an image
- Enter "Mark Landmarks" mode
- Click each point in order as prompted; both-sided points prompt L then R
- Markers + labels are drawn live on the canvas
- Export coordinates to JSON or CSV

Dependencies:  pip install Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import csv
import io
import os
from dataclasses import dataclass, field
from typing import Optional

try:
    from PIL import Image, ImageTk, ImageGrab, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Landmark definitions
# ---------------------------------------------------------------------------
# Each entry: (key_base, label, bilateral)
# bilateral=True  → prompts "Left {label}" then "Right {label}", keys become key_base_L / key_base_R
# bilateral=False → prompts once, key stays key_base

LANDMARK_DEFS: list[tuple[str, str, bool]] = [
    ("face_outline_lip_crease",   "Side Face Outline – line up with lip crease",       True),
    ("face_under_ear",            "Side Face Under Ear – where ear and face meet",      True),
    ("face_above_ear",            "Side Face Above Ear – where ear and face meet",      True),
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
    ("nose_nostril_outside",      "Side of Nose – Nostril Outside",                     True),
    ("chin_outer_side",           "Side of Outer Chin",                                 True),
    ("mouth_upper_apex_side",     "Upper Mouth Apex Side",                              True),
    ("mouth_upper_low_u",         "Upper Mouth Low U Apex",                             False),
    ("lips_center_meet",          "Between Lips Where They Meet – Center",              False),
    ("lips_outer_crease",         "Outer Crease of Lips",                               True),
    ("mouth_under_apex",          "Under Mouth Apex",                                   True),
    ("chin_bottom_apex",          "Bottom of Chin Apex",                                False),
    ("neck_face_corner",          "Neck Meets Face Corner",                             True),
]


def build_prompt_sequence() -> list[dict]:
    """
    Expand LANDMARK_DEFS into an ordered list of prompt dicts:
        { key, label_short, prompt_text, side }   (side: 'L' | 'R' | None)
    Bilateral landmarks emit LEFT first, then RIGHT.
    """
    seq = []
    for key_base, label, bilateral in LANDMARK_DEFS:
        if bilateral:
            seq.append(dict(key=f"{key_base}_L", label_short=label, prompt_text=f"LEFT  ·  {label}", side="L"))
            seq.append(dict(key=f"{key_base}_R", label_short=label, prompt_text=f"RIGHT ·  {label}", side="R"))
        else:
            seq.append(dict(key=key_base, label_short=label, prompt_text=f"{label}", side=None))
    return seq


PROMPTS = build_prompt_sequence()          # 38 total steps
TOTAL   = len(PROMPTS)                     # 38


# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------
LEFT_COLOR   = "#00d4ff"   # cyan
RIGHT_COLOR  = "#ff6b35"   # orange
SINGLE_COLOR = "#a8ff3e"   # lime-green
RADIUS       = 5           # dot radius on canvas
HIT_RADIUS   = 12          # px — how close you must click to grab a marker


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class FaceLandmarkApp(tk.Tk):

    APP_TITLE  = "Face Landmark Annotator"
    MIN_WIDTH  = 960
    MIN_HEIGHT = 700

    def __init__(self):
        super().__init__()
        self.title(self.APP_TITLE)
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.geometry(f"{self.MIN_WIDTH}x{self.MIN_HEIGHT}")
        self.configure(bg="#1a1a2e")

        # ── State ──────────────────────────────────────────────────────────
        self.original_image: Optional[Image.Image]    = None   # pristine source
        self.display_image:  Optional[Image.Image]    = None   # scaled version
        self.photo_image:    Optional[ImageTk.PhotoImage] = None
        self.scale_factor:   float                    = 1.0    # display / original
        self.image_stem:     str                      = "landmarks"  # filename base for export

        self.marking_mode:   bool                     = False
        self.current_step:   int                      = 0      # index into PROMPTS
        self.landmarks:      dict[str, tuple[int,int]] = {}    # key → original-space (x,y)
        self.canvas_markers: list[int]                = []     # canvas item ids

        # drag state
        self._drag_key:      Optional[str]            = None   # landmark being dragged
        self._drag_offset:   tuple[float,float]       = (0.0, 0.0)  # click offset within dot

        self._build_styles()
        self._build_menu()
        self._build_ui()
        self._bind_shortcuts()

        if not PIL_AVAILABLE:
            messagebox.showwarning("Pillow not found",
                                   "Install Pillow:\n\n  pip install Pillow")

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _build_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",      background="#1a1a2e")
        style.configure("TLabel",      background="#1a1a2e", foreground="#e0e0e0", font=("Helvetica", 10))
        style.configure("Header.TLabel", background="#1a1a2e", foreground="#ffffff", font=("Helvetica", 13, "bold"))
        style.configure("Prompt.TLabel", background="#0f3460",  foreground="#ffffff", font=("Helvetica", 12, "bold"),
                         padding=(12, 8), relief="flat")
        style.configure("Step.TLabel",   background="#1a1a2e", foreground="#888888", font=("Helvetica", 10))
        style.configure("TButton",       font=("Helvetica", 10), padding=(8, 4))
        style.configure("Action.TButton", font=("Helvetica", 11, "bold"))
        style.configure("TCombobox",     font=("Helvetica", 10))

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _build_menu(self):
        mb = tk.Menu(self, bg="#1a1a2e", fg="#e0e0e0", activebackground="#0f3460")

        fm = tk.Menu(mb, tearoff=False)
        fm.add_command(label="Open Image…",          accelerator="Ctrl+O", command=self.open_file)
        fm.add_command(label="Paste from Clipboard", accelerator="Ctrl+V", command=self.paste_image)
        fm.add_command(label="Load Landmarks…",      accelerator="Ctrl+L", command=self.load_landmarks)
        fm.add_separator()
        fm.add_command(label="Export → JSON",        command=self.export_json)
        fm.add_command(label="Export → CSV",         command=self.export_csv)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.quit)
        mb.add_cascade(label="File", menu=fm)

        em = tk.Menu(mb, tearoff=False)
        em.add_command(label="Start / Restart Marking", accelerator="F5", command=self.start_marking)
        em.add_command(label="Undo Last Point",          accelerator="Ctrl+Z", command=self.undo_last)
        em.add_command(label="Clear All Landmarks",      command=self.clear_landmarks)
        mb.add_cascade(label="Mark", menu=em)

        hm = tk.Menu(mb, tearoff=False)
        hm.add_command(label="About", command=self.show_about)
        mb.add_cascade(label="Help", menu=hm)

        self.config(menu=mb)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Toolbar ───────────────────────────────────────────────────
        toolbar = ttk.Frame(self, padding=(6, 3))
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="📂 Open",    command=self.open_file      ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="📋 Paste",   command=self.paste_image    ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="📥 Load",    command=self.load_landmarks ).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        self.btn_start = ttk.Button(toolbar, text="▶  Start Marking",
                                    style="Action.TButton", command=self.start_marking)
        self.btn_start.pack(side="left", padx=2)
        ttk.Button(toolbar, text="↩ Undo",    command=self.undo_last   ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="🗑 Clear",   command=self.clear_landmarks).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(toolbar, text="💾 JSON",    command=self.export_json ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="💾 CSV",     command=self.export_csv  ).pack(side="left", padx=2)

        # Zoom
        ttk.Label(toolbar, text="Zoom:").pack(side="right", padx=(4, 0))
        self.zoom_var = tk.StringVar(value="Fit")
        zoom_cb = ttk.Combobox(toolbar, textvariable=self.zoom_var,
                               values=["Fit","50%","75%","100%","150%","200%"],
                               state="readonly", width=6)
        zoom_cb.pack(side="right", padx=4)
        zoom_cb.bind("<<ComboboxSelected>>", lambda _e: self.refresh_display())

        # ── Prompt banner ─────────────────────────────────────────────
        prompt_frame = tk.Frame(self, bg="#0f3460", pady=0)
        prompt_frame.pack(side="top", fill="x")

        self.prompt_var = tk.StringVar(value="Open or paste an image, then press  ▶ Start Marking")
        prompt_lbl = tk.Label(prompt_frame, textvariable=self.prompt_var,
                              bg="#0f3460", fg="#ffffff",
                              font=("Helvetica", 12, "bold"), pady=8, padx=14, anchor="w")
        prompt_lbl.pack(side="left", fill="x", expand=True)

        self.step_var = tk.StringVar(value="")
        step_lbl = tk.Label(prompt_frame, textvariable=self.step_var,
                            bg="#0f3460", fg="#aaaaaa",
                            font=("Helvetica", 10), padx=14)
        step_lbl.pack(side="right")

        # ── Main area (canvas + side panel) ───────────────────────────
        main_frame = ttk.Frame(self)
        main_frame.pack(side="top", fill="both", expand=True)

        # Scrollable canvas
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#0d0d0d", cursor="crosshair",
                                highlightthickness=0)
        vscroll = ttk.Scrollbar(canvas_frame, orient="vertical",   command=self.canvas.yview)
        hscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vscroll.set, xscrollcommand=hscroll.set)

        hscroll.pack(side="bottom", fill="x")
        vscroll.pack(side="right",  fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>",        self.on_canvas_click)
        self.canvas.bind("<B1-Motion>",       self.on_drag_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_drag_release)

        # Side panel – landmark list
        side = tk.Frame(main_frame, bg="#16213e", width=260)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="LANDMARKS", bg="#16213e", fg="#888888",
                 font=("Helvetica", 9, "bold"), pady=8).pack(anchor="w", padx=10)

        list_frame = tk.Frame(side, bg="#16213e")
        list_frame.pack(fill="both", expand=True, padx=4)

        list_scroll = ttk.Scrollbar(list_frame, orient="vertical")
        self.landmark_listbox = tk.Listbox(
            list_frame, yscrollcommand=list_scroll.set,
            bg="#16213e", fg="#cccccc", selectbackground="#0f3460",
            font=("Courier", 9), borderwidth=0, highlightthickness=0,
            activestyle="none",
        )
        list_scroll.config(command=self.landmark_listbox.yview)
        list_scroll.pack(side="right", fill="y")
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

        self.canvas.bind("<Motion>", self._on_mouse_move)
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
        self.bind("<F5>",        lambda _e: self.start_marking())

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

    def _load_image(self, img: Image.Image, source: str = "", stem: str = "landmarks"):
        self.original_image = img.copy()
        self.image_stem     = stem
        self.clear_landmarks(silent=True)
        self.marking_mode  = False
        self.current_step  = 0
        self.prompt_var.set("Image loaded.  Press  ▶ Start Marking  to begin.")
        self.step_var.set("")
        self.status_var.set(f"Loaded: {source}  |  {img.size[0]}×{img.size[1]} px")
        self.refresh_display()

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def refresh_display(self):
        if self.original_image is None:
            self._draw_placeholder(); return

        zoom_str = self.zoom_var.get()
        cw = max(self.canvas.winfo_width(),  self.MIN_WIDTH  - 260)
        ch = max(self.canvas.winfo_height(), self.MIN_HEIGHT - 100)
        iw, ih = self.original_image.size

        if zoom_str == "Fit":
            self.scale_factor = min(cw / iw, ch / ih)
        else:
            self.scale_factor = int(zoom_str.rstrip("%")) / 100

        nw = max(1, int(iw * self.scale_factor))
        nh = max(1, int(ih * self.scale_factor))

        self.display_image = self.original_image.resize((nw, nh), Image.LANCZOS)
        self.photo_image   = ImageTk.PhotoImage(self.display_image)

        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, nw, nh))
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo_image, tags="image")

        # Re-draw all existing markers
        for key, (ox, oy) in self.landmarks.items():
            cx = int(ox * self.scale_factor)
            cy = int(oy * self.scale_factor)
            color = _marker_color(key)
            self._draw_marker(cx, cy, key, color)

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.MIN_WIDTH // 2, self.MIN_HEIGHT // 2,
            text="Press  Ctrl+V  to paste an image\nor use  File → Open",
            fill="#333355", font=("Helvetica", 16), justify="center")

    def _draw_marker(self, cx: int, cy: int, key: str, color: str, highlight: bool = False):
        r      = RADIUS
        outline = "#ffffff" if not highlight else "#ffff00"
        width   = 1         if not highlight else 2
        # Tag each marker's items with both "marker" (group) and the key (individual)
        item_tag = f"mk_{key}"
        self.canvas.delete(item_tag)   # remove stale items for this key before redrawing
        self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                 fill=color, outline=outline, width=width,
                                 tags=("marker", item_tag))
        short = _short_label(key)
        self.canvas.create_text(cx + r + 4, cy, text=short, fill=color,
                                 font=("Helvetica", 7), anchor="w",
                                 tags=("marker", item_tag))

    def _hit_test(self, cx: float, cy: float) -> Optional[str]:
        """Return the key of the nearest landmark within HIT_RADIUS canvas-pixels, or None."""
        best_key  = None
        best_dist = HIT_RADIUS
        for key, (ox, oy) in self.landmarks.items():
            kx = ox * self.scale_factor
            ky = oy * self.scale_factor
            dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_key  = key
        return best_key

    # ------------------------------------------------------------------
    # Marking mode
    # ------------------------------------------------------------------

    def start_marking(self):
        if self.original_image is None:
            messagebox.showinfo("No image", "Load an image first."); return
        self.marking_mode = True
        self.current_step = 0
        self.landmarks.clear()
        self.landmark_listbox.delete(0, tk.END)
        self.canvas.delete("marker")
        self._advance_prompt()
        self.canvas.config(cursor="crosshair")

    def _advance_prompt(self):
        if self.current_step >= TOTAL:
            self._finish_marking(); return
        p = PROMPTS[self.current_step]
        side_tag = {"L": "  [LEFT]", "R": "  [RIGHT]", None: ""}[p["side"]]
        self.prompt_var.set(f"  Click →  {p['prompt_text']}{side_tag}")
        self.step_var.set(f"Step {self.current_step + 1} / {TOTAL}")

    def on_canvas_click(self, event):
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # ── Drag initiation: always check for a nearby existing marker first ──
        hit = self._hit_test(cx, cy)
        if hit is not None:
            self._drag_key    = hit
            ox, oy            = self.landmarks[hit]
            # store fractional offset so the dot doesn't jump
            self._drag_offset = (cx - ox * self.scale_factor,
                                 cy - oy * self.scale_factor)
            color = _marker_color(hit)
            self._draw_marker(int(ox * self.scale_factor),
                              int(oy * self.scale_factor), hit, color, highlight=True)
            self.canvas.config(cursor="fleur")
            return   # don't place a new landmark

        # ── Otherwise: place next landmark if in marking mode ────────────────
        self._drag_key = None
        if not self.marking_mode: return
        if self.current_step >= TOTAL: return

        ox = int(cx / self.scale_factor)
        oy = int(cy / self.scale_factor)

        p     = PROMPTS[self.current_step]
        key   = p["key"]
        color = _marker_color(key)

        self.landmarks[key] = (ox, oy)
        self._draw_marker(int(cx), int(cy), key, color)
        self._update_list(key, ox, oy)

        self.status_var.set(f"✓  {p['prompt_text']}  →  ({ox}, {oy})")
        self.current_step += 1
        self._advance_prompt()

    def on_drag_motion(self, event):
        if self._drag_key is None: return
        key = self._drag_key

        cx = self.canvas.canvasx(event.x) - self._drag_offset[0]
        cy = self.canvas.canvasy(event.y) - self._drag_offset[1]

        # Clamp to image bounds
        if self.original_image:
            iw, ih = self.original_image.size
            cx = max(0.0, min(cx, (iw - 1) * self.scale_factor))
            cy = max(0.0, min(cy, (ih - 1) * self.scale_factor))

        ox = int(cx / self.scale_factor)
        oy = int(cy / self.scale_factor)

        self.landmarks[key] = (ox, oy)
        color = _marker_color(key)
        self._draw_marker(int(cx), int(cy), key, color, highlight=True)
        self._refresh_list_entry(key, ox, oy)
        self.coord_var.set(f"x={ox}  y={oy}")

    def on_drag_release(self, event):
        if self._drag_key is None: return
        key = self._drag_key
        ox, oy = self.landmarks[key]
        color  = _marker_color(key)
        # Redraw without highlight
        self._draw_marker(int(ox * self.scale_factor),
                          int(oy * self.scale_factor), key, color, highlight=False)
        self.status_var.set(f"↔  Moved  {key}  →  ({ox}, {oy})")
        self._drag_key = None
        # Restore cursor
        self.canvas.config(cursor="crosshair" if self.marking_mode else "arrow")

    def _update_list(self, key: str, ox: int, oy: int):
        self.landmark_listbox.insert(tk.END, f"{key:<28}  ({ox:>4}, {oy:>4})")
        side = key.split("_")[-1]
        if side == "L":
            self.landmark_listbox.itemconfig(tk.END, fg=LEFT_COLOR)
        elif side == "R":
            self.landmark_listbox.itemconfig(tk.END, fg=RIGHT_COLOR)
        else:
            self.landmark_listbox.itemconfig(tk.END, fg=SINGLE_COLOR)
        self.landmark_listbox.see(tk.END)

    def _refresh_list_entry(self, key: str, ox: int, oy: int):
        """Update the coordinates shown for an existing key in the side panel."""
        keys = [p["key"] for p in PROMPTS]
        if key not in keys: return
        idx = keys.index(key)
        # idx may exceed listbox size if not all landmarks placed yet
        size = self.landmark_listbox.size()
        if idx >= size: return
        color = self.landmark_listbox.itemcget(idx, "fg")
        self.landmark_listbox.delete(idx)
        self.landmark_listbox.insert(idx, f"{key:<28}  ({ox:>4}, {oy:>4})")
        self.landmark_listbox.itemconfig(idx, fg=color)

    def _finish_marking(self):
        self.marking_mode = False
        self.prompt_var.set(f"  ✅  All {TOTAL} landmarks recorded.  Export with  💾 JSON  or  💾 CSV")
        self.step_var.set(f"{TOTAL} / {TOTAL}  complete")
        self.canvas.config(cursor="arrow")
        messagebox.showinfo("Done", f"All {TOTAL} landmarks recorded!\nExport with File → Export.")

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def undo_last(self):
        if not self.marking_mode and self.current_step == 0: return
        if self.current_step == 0: return

        self.current_step -= 1
        key = PROMPTS[self.current_step]["key"]
        self.landmarks.pop(key, None)

        # Remove last listbox entry
        if self.landmark_listbox.size() > 0:
            self.landmark_listbox.delete(tk.END)

        # Redraw markers (easier than tracking individual ids)
        self.refresh_display()

        if not self.marking_mode:
            self.marking_mode = True
        self._advance_prompt()
        self.status_var.set(f"Undid:  {key}")

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear_landmarks(self, silent: bool = False):
        self.landmarks.clear()
        self.current_step = 0
        self.marking_mode = False
        self.landmark_listbox.delete(0, tk.END)
        self.canvas.delete("marker")
        self.prompt_var.set("Landmarks cleared.  Press  ▶ Start Marking  to begin.")
        self.step_var.set("")
        if not silent:
            self.status_var.set("All landmarks cleared.")

    # ------------------------------------------------------------------
    # Export  (JSON / CSV — always saves a paired image copy alongside)
    # ------------------------------------------------------------------

    IMAGE_COPY_SUFFIX = "_image.png"   # appended to stem for the paired photo

    def _save_image_copy(self, export_path: str) -> str:
        """Save a PNG copy of the original image next to the export file.
        Returns the path of the saved image copy."""
        stem    = os.path.splitext(export_path)[0]
        img_path = stem + self.IMAGE_COPY_SUFFIX
        self.original_image.save(img_path, format="PNG")
        return img_path

    def export_json(self):
        if not self.landmarks:
            messagebox.showinfo("Nothing to export", "No landmarks recorded yet."); return
        path = filedialog.asksaveasfilename(
            title="Save landmarks as JSON",
            initialfile=f"{self.image_stem}.json",
            defaultextension=".json",
            filetypes=[("JSON","*.json"),("All","*.*")])
        if not path: return

        img_path = self._save_image_copy(path)
        payload = {
            "total_landmarks": len(self.landmarks),
            "image_size": list(self.original_image.size) if self.original_image else None,
            "paired_image": os.path.basename(img_path),
            "landmarks": {k: {"x": v[0], "y": v[1]} for k, v in self.landmarks.items()},
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        self.status_var.set(
            f"Exported JSON → {os.path.basename(path)}  +  image copy → {os.path.basename(img_path)}")

    def export_csv(self):
        if not self.landmarks:
            messagebox.showinfo("Nothing to export", "No landmarks recorded yet."); return
        path = filedialog.asksaveasfilename(
            title="Save landmarks as CSV",
            initialfile=f"{self.image_stem}.csv",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if not path: return

        img_path = self._save_image_copy(path)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            # Header row includes paired image name so loader can find it
            writer.writerow(["# paired_image", os.path.basename(img_path)])
            writer.writerow(["key", "x", "y"])
            for k, (x, y) in self.landmarks.items():
                writer.writerow([k, x, y])
        self.status_var.set(
            f"Exported CSV → {os.path.basename(path)}  +  image copy → {os.path.basename(img_path)}")

    # ------------------------------------------------------------------
    # Load landmarks (JSON or CSV) + paired image
    # ------------------------------------------------------------------

    def load_landmarks(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow required", "pip install Pillow"); return

        path = filedialog.askopenfilename(
            title="Load Landmarks File",
            filetypes=[
                ("Landmark files", "*.json *.csv"),
                ("JSON", "*.json"),
                ("CSV",  "*.csv"),
                ("All",  "*.*"),
            ])
        if not path: return

        ext = os.path.splitext(path)[1].lower()
        folder = os.path.dirname(path)

        try:
            if ext == ".json":
                loaded, img_basename = self._parse_json(path)
            elif ext == ".csv":
                loaded, img_basename = self._parse_csv(path)
            else:
                messagebox.showerror("Unknown format", "Please open a .json or .csv file."); return
        except Exception as exc:
            messagebox.showerror("Parse error", str(exc)); return

        # ── Locate paired image ──────────────────────────────────────
        img_path = None

        # 1. Use the embedded filename hint if present
        if img_basename:
            candidate = os.path.join(folder, img_basename)
            if os.path.isfile(candidate):
                img_path = candidate

        # 2. Fallback: look for any image with the same stem
        if img_path is None:
            stem = os.path.splitext(os.path.basename(path))[0]
            for ext_try in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"):
                candidate = os.path.join(folder, stem + ext_try)
                if os.path.isfile(candidate):
                    img_path = candidate
                    break
            # Also try stem + IMAGE_COPY_SUFFIX
            if img_path is None:
                candidate = os.path.join(folder, stem + self.IMAGE_COPY_SUFFIX)
                if os.path.isfile(candidate):
                    img_path = candidate

        if img_path is None:
            messagebox.showerror(
                "Image not found",
                f"Could not find the paired image for:\n{os.path.basename(path)}\n\n"
                "Make sure the image copy is in the same folder as the landmark file.")
            return

        # ── Load image & restore landmarks ───────────────────────────
        try:
            img = Image.open(img_path)
        except Exception as exc:
            messagebox.showerror("Image load failed", str(exc)); return

        stem_name = os.path.splitext(os.path.basename(path))[0]
        self._load_image(img, os.path.basename(img_path), stem=stem_name)

        # Restore landmark dict and side-panel list
        self.landmarks = loaded
        self.current_step = len(loaded)   # resume from where it left off
        self.landmark_listbox.delete(0, tk.END)
        for k, (x, y) in loaded.items():
            self.landmark_listbox.insert(tk.END, f"{k:<28}  ({x:>4}, {y:>4})")
            side = k.split("_")[-1]
            color = LEFT_COLOR if side == "L" else RIGHT_COLOR if side == "R" else SINGLE_COLOR
            self.landmark_listbox.itemconfig(tk.END, fg=color)

        self.refresh_display()   # re-draws markers via refresh_display loop

        total_done = len(loaded)
        if total_done >= TOTAL:
            self.prompt_var.set(f"  ✅  All {TOTAL} landmarks loaded from file.")
            self.step_var.set(f"{TOTAL} / {TOTAL}  complete")
        else:
            remaining = TOTAL - total_done
            self.marking_mode = True
            self._advance_prompt()
            self.prompt_var.set(
                f"  Loaded {total_done} landmarks.  {remaining} remaining — click to continue.")

        self.status_var.set(
            f"Loaded landmarks from {os.path.basename(path)}  +  image {os.path.basename(img_path)}")

    def _parse_json(self, path: str) -> tuple[dict, Optional[str]]:
        with open(path) as f:
            data = json.load(f)
        raw = data.get("landmarks", {})
        loaded = {k: (v["x"], v["y"]) for k, v in raw.items()}
        img_basename = data.get("paired_image")
        return loaded, img_basename

    def _parse_csv(self, path: str) -> tuple[dict, Optional[str]]:
        loaded: dict[str, tuple[int,int]] = {}
        img_basename: Optional[str] = None
        with open(path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                if row[0].startswith("#"):
                    # metadata comment row: # paired_image, filename.png
                    if len(row) >= 2 and "paired_image" in row[0]:
                        img_basename = row[1].strip()
                    continue
                if row[0] == "key": continue   # header
                key, x, y = row[0], int(row[1]), int(row[2])
                loaded[key] = (x, y)
        return loaded, img_basename

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _on_mouse_move(self, event):
        if self.original_image is None: return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ox = int(cx / self.scale_factor)
        oy = int(cy / self.scale_factor)
        iw, ih = self.original_image.size
        if 0 <= ox < iw and 0 <= oy < ih:
            self.coord_var.set(f"x={ox}  y={oy}")
        else:
            self.coord_var.set("")

        # Show move cursor when hovering over an existing marker (if not mid-drag)
        if self._drag_key is None:
            hit = self._hit_test(cx, cy)
            if hit:
                self.canvas.config(cursor="fleur")
            else:
                self.canvas.config(cursor="crosshair" if self.marking_mode else "arrow")

    def show_about(self):
        messagebox.showinfo("About", (
            "Face Landmark Annotator\n\n"
            f"{TOTAL} frontal-face landmarks (bilateral + single)\n\n"
            "Ctrl+V  → paste image\n"
            "Ctrl+O  → open image\n"
            "Ctrl+L  → load landmarks (JSON/CSV) + paired image\n"
            "F5      → start marking\n"
            "Ctrl+Z  → undo last point\n\n"
            "Exporting saves a paired image copy alongside\n"
            "the JSON/CSV so it can be reloaded later.\n\n"
            "Requires: Pillow (pip install Pillow)"
        ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _marker_color(key: str) -> str:
    if key.endswith("_L"): return LEFT_COLOR
    if key.endswith("_R"): return RIGHT_COLOR
    return SINGLE_COLOR


def _short_label(key: str) -> str:
    """Very short abbreviation for the canvas dot label."""
    parts = key.split("_")
    # Drop trailing L/R — shown by colour
    if parts[-1] in ("L", "R"):
        parts = parts[:-1]
    # Take initials
    return "".join(p[0].upper() for p in parts[:4])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = FaceLandmarkApp()
    app.mainloop()
