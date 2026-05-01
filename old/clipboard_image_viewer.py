"""
clipboard_image_viewer.py
--------------------------
Boilerplate tkinter app with Ctrl+V clipboard image paste support.
Dependencies (stdlib only): tkinter, tkinter.ttk, tkinter.filedialog, tkinter.messagebox
Optional (for richer clipboard support): pip install Pillow
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import io
import os

# Pillow is required for clipboard image handling and image manipulation
try:
    from PIL import Image, ImageTk, ImageGrab
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------

class App(tk.Tk):
    """Root window — holds the menu bar and the main frame."""

    APP_TITLE   = "Clipboard Image Viewer"
    MIN_WIDTH   = 800
    MIN_HEIGHT  = 600

    def __init__(self):
        super().__init__()
        self.title(self.APP_TITLE)
        self.minsize(self.MIN_WIDTH, self.MIN_HEIGHT)
        self.geometry(f"{self.MIN_WIDTH}x{self.MIN_HEIGHT}")

        # Application state
        self.current_image: Image.Image | None = None   # PIL Image
        self.photo_image:   ImageTk.PhotoImage | None = None  # tk-compatible ref

        self._build_menu()
        self._build_ui()
        self._bind_shortcuts()

        # Check PIL at startup
        if not PIL_AVAILABLE:
            messagebox.showwarning(
                "Pillow not found",
                "Install Pillow for clipboard paste support:\n\n  pip install Pillow",
            )

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self):
        menu_bar = tk.Menu(self)

        # File menu
        file_menu = tk.Menu(menu_bar, tearoff=False)
        file_menu.add_command(label="Open Image…",      accelerator="Ctrl+O", command=self.open_file)
        file_menu.add_command(label="Paste from Clipboard", accelerator="Ctrl+V", command=self.paste_image)
        file_menu.add_separator()
        file_menu.add_command(label="Save As…",         accelerator="Ctrl+S", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit",             accelerator="Alt+F4",  command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menu_bar, tearoff=False)
        edit_menu.add_command(label="Clear Canvas",     command=self.clear_canvas)
        edit_menu.add_command(label="Copy to Clipboard",accelerator="Ctrl+C", command=self.copy_to_clipboard)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        # Help menu
        help_menu = tk.Menu(menu_bar, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menu_bar)

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        # ── Top toolbar ──────────────────────────────────────────────
        toolbar = ttk.Frame(self, relief="raised", padding=(4, 2))
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="📂 Open",  command=self.open_file   ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="📋 Paste", command=self.paste_image ).pack(side="left", padx=2)
        ttk.Button(toolbar, text="💾 Save",  command=self.save_file   ).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="🗑 Clear",  command=self.clear_canvas).pack(side="left", padx=2)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)

        # Zoom controls
        ttk.Label(toolbar, text="Zoom:").pack(side="left")
        self.zoom_var = tk.StringVar(value="Fit")
        zoom_combo = ttk.Combobox(
            toolbar, textvariable=self.zoom_var,
            values=["Fit", "25%", "50%", "75%", "100%", "150%", "200%"],
            state="readonly", width=6,
        )
        zoom_combo.pack(side="left", padx=4)
        zoom_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_display())

        # ── Canvas area (scrollable) ─────────────────────────────────
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(side="top", fill="both", expand=True)

        self.canvas = tk.Canvas(canvas_frame, bg="#2b2b2b", cursor="crosshair")
        v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical",   command=self.canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        h_scroll.pack(side="bottom", fill="x")
        v_scroll.pack(side="right",  fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Placeholder text on empty canvas
        self._draw_placeholder()

        # ── Status bar ───────────────────────────────────────────────
        status_bar = ttk.Frame(self, relief="sunken")
        status_bar.pack(side="bottom", fill="x")

        self.status_var = tk.StringVar(value="Ready  |  Ctrl+V to paste an image")
        ttk.Label(status_bar, textvariable=self.status_var, anchor="w").pack(side="left", padx=6)

        self.size_var = tk.StringVar(value="")
        ttk.Label(status_bar, textvariable=self.size_var, anchor="e").pack(side="right", padx=6)

    # ------------------------------------------------------------------
    # Keyboard shortcuts
    # ------------------------------------------------------------------

    def _bind_shortcuts(self):
        self.bind("<Control-v>", lambda _e: self.paste_image())
        self.bind("<Control-V>", lambda _e: self.paste_image())   # caps-lock
        self.bind("<Control-o>", lambda _e: self.open_file())
        self.bind("<Control-s>", lambda _e: self.save_file())
        self.bind("<Control-c>", lambda _e: self.copy_to_clipboard())

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def paste_image(self):
        """Grab whatever image is on the system clipboard and display it."""
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow required", "pip install Pillow")
            return
        try:
            img = ImageGrab.grabclipboard()
        except Exception as exc:
            messagebox.showerror("Clipboard error", str(exc))
            return

        if img is None:
            self.status_var.set("Nothing found on clipboard.")
            return

        if isinstance(img, list):
            # Windows can return a list of file paths; open the first image
            for path in img:
                try:
                    img = Image.open(path)
                    break
                except Exception:
                    pass
            else:
                self.status_var.set("Clipboard contains no image data.")
                return

        self._load_image(img, source="Clipboard")

    def open_file(self):
        """Open an image file via the file dialog."""
        if not PIL_AVAILABLE:
            messagebox.showerror("Pillow required", "pip install Pillow")
            return
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff"),
                ("All files",   "*.*"),
            ],
        )
        if not path:
            return
        try:
            img = Image.open(path)
            self._load_image(img, source=os.path.basename(path))
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))

    def save_file(self):
        """Save the currently displayed image."""
        if self.current_image is None:
            messagebox.showinfo("No image", "Nothing to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Image As",
            defaultextension=".png",
            filetypes=[
                ("PNG",  "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP",  "*.bmp"),
                ("All",  "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.current_image.save(path)
            self.status_var.set(f"Saved → {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def copy_to_clipboard(self):
        """Copy current image back to the system clipboard (Windows/macOS)."""
        if self.current_image is None:
            return
        if not PIL_AVAILABLE:
            return
        try:
            # Convert to BMP bytes for clipboard (cross-platform approach)
            output = io.BytesIO()
            self.current_image.convert("RGB").save(output, "BMP")
            data = output.getvalue()[14:]   # strip BMP file header
            output.close()
            self.clipboard_clear()
            self.clipboard_append(data)
            self.status_var.set("Image copied to clipboard.")
        except Exception as exc:
            messagebox.showerror("Copy failed", str(exc))

    def clear_canvas(self):
        self.current_image = None
        self.photo_image   = None
        self.canvas.delete("all")
        self._draw_placeholder()
        self.status_var.set("Canvas cleared.")
        self.size_var.set("")

    def show_about(self):
        messagebox.showinfo(
            "About",
            f"{self.APP_TITLE}\n\nPaste images with Ctrl+V or use File → Open.\n\nRequires: Pillow (pip install Pillow)",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, img: "Image.Image", source: str = ""):
        self.current_image = img.copy()
        self.refresh_display()
        w, h = img.size
        self.status_var.set(f"Loaded: {source}")
        self.size_var.set(f"{w} × {h} px  |  {img.mode}")

    def refresh_display(self):
        """Re-render the image on the canvas according to the current zoom level."""
        if self.current_image is None:
            return

        zoom_str = self.zoom_var.get()
        cw = self.canvas.winfo_width()  or self.MIN_WIDTH
        ch = self.canvas.winfo_height() or self.MIN_HEIGHT
        iw, ih = self.current_image.size

        if zoom_str == "Fit":
            scale = min(cw / iw, ch / ih, 1.0)   # never upscale beyond 100 % in Fit
            nw, nh = int(iw * scale), int(ih * scale)
        else:
            pct = int(zoom_str.rstrip("%")) / 100
            nw, nh = int(iw * pct), int(ih * pct)

        display = self.current_image.resize((nw, nh), Image.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(display)

        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, nw, nh))
        self.canvas.create_image(nw // 2, nh // 2, anchor="center", image=self.photo_image)

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            self.MIN_WIDTH  // 2,
            self.MIN_HEIGHT // 2,
            text="Press  Ctrl+V  to paste an image\nor use  File → Open",
            fill="#555555",
            font=("Helvetica", 16),
            justify="center",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = App()
    app.mainloop()
