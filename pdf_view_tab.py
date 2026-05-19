from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import tkinter as tk


@dataclass
class PdfOpenResult:
    ok: bool
    title: str = ""
    error: str = ""


class PdfTab(ctk.CTkFrame):
    """
    In-app PDF viewer:
    - Renders current page as an image (PyMuPDF)
    - Click-drag rectangle selection extracts text using word boxes
    - Selection persists while user types the question
    """

    def __init__(self, master, *, colors: dict, font_family: str, font_size: int, on_toggle_collapse=None):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self._colors = colors
        self._font_family = font_family
        self._font_size = font_size

        self._pdf_path: Optional[Path] = None
        self._doc = None
        self._page_index = 0
        self._page_count = 0
        self._zoom = 1.45

        self._page_imgs = []
        # Stored in reading order with canvas bboxes precomputed.
        # Each item: dict with keys: cx0, cy0, cx1, cy1, text, page, block, line, word_no
        self._words = []
        self._page_tops = []

        self._sel_start = None
        self._sel_end = None
        self._highlight_ids = []
        self._selected_text = ""
        self._on_toggle_collapse = on_toggle_collapse
        self._sel_start_idx: Optional[int] = None
        self._sel_end_idx: Optional[int] = None
        self._sel_drag_start_xy: Optional[tuple[float, float]] = None
        self._sel_drag_end_xy: Optional[tuple[float, float]] = None
        self._sel_pdf_bounds_by_page: dict[int, tuple[float, float, float, float]] = {}

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        top = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=40, pady=(10, 6))
        top.grid_columnconfigure(1, weight=1)

        self._title = ctk.CTkLabel(
            top,
            text="PDF",
            text_color=self._colors["text"],
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self._title.grid(row=0, column=0, sticky="w")

        nav = ctk.CTkFrame(top, corner_radius=0, fg_color="transparent")
        nav.grid(row=0, column=2, sticky="e")

        self._collapse_btn = ctk.CTkButton(
            nav,
            text="⇤",
            width=34,
            height=30,
            corner_radius=8,
            command=(self._on_toggle_collapse if self._on_toggle_collapse is not None else (lambda: None)),
        )
        self._collapse_btn.grid(row=0, column=0, padx=(0, 8))
        self._page_lbl = ctk.CTkLabel(nav, text="", text_color=self._colors["muted"])
        self._page_lbl.grid(row=0, column=1, padx=(0, 0))

        self._wrap = ctk.CTkFrame(
            self,
            corner_radius=12,
            fg_color=self._colors["chat_panel"],
            border_width=1,
            border_color=self._colors["border"],
        )
        self._wrap.grid(row=2, column=0, sticky="nsew", padx=16, pady=(10, 12))
        self._wrap.grid_rowconfigure(0, weight=1)
        self._wrap.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            self._wrap,
            bg=self._colors["chat_panel"],
            highlightthickness=0,
            bd=0,
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self._vscroll = tk.Scrollbar(self._wrap, command=self._canvas.yview)
        self._vscroll.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=self._vscroll.set)

        self._canvas.bind("<Button-1>", self._on_mouse_down)
        self._canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_mouse_up)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._canvas.bind("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Shift-MouseWheel>", self._on_mousewheel)
        self._canvas.bind("<Button-4>", self._on_mousewheel_linux)  # linux scroll up
        self._canvas.bind("<Button-5>", self._on_mousewheel_linux)  # linux scroll down

        self._sel_info = ctk.CTkLabel(self, text="", text_color=self._colors["muted"], font=ctk.CTkFont(size=12))
        self._sel_info.grid(row=3, column=0, sticky="w", padx=16, pady=(0, 10))

        self._set_placeholder()

    def _set_placeholder(self):
        self._title.configure(text="PDF")
        self._page_lbl.configure(text="")
        self._canvas.delete("all")
        self._canvas.create_text(
            20,
            20,
            anchor="nw",
            fill=self._colors["muted"],
            text="Open a PDF from the Uploaded Documents list to view it here.",
            font=(self._font_family, self._font_size),
        )
        self._canvas.configure(scrollregion=self._canvas.bbox("all") or (0, 0, 1, 1))
        self.clear_selection()

    def set_font(self, *, font_family: str, font_size: int):
        self._font_family = font_family
        self._font_size = font_size

    def show_error(self, *, title: str, error: str):
        self._title.configure(text=title or "PDF")
        self._canvas.delete("all")
        self._canvas.create_text(
            20,
            20,
            anchor="nw",
            fill=self._colors["muted"],
            text=f"Could not open PDF.\n\n{error}",
            font=(self._font_family, self._font_size),
        )
        self._canvas.configure(scrollregion=self._canvas.bbox("all") or (0, 0, 1, 1))
        self.clear_selection()

    def open_pdf(self, *, path: str, title: str = "") -> PdfOpenResult:
        try:
            import fitz  # PyMuPDF

            self._pdf_path = Path(path)
            if not self._pdf_path.exists():
                raise RuntimeError("File not found.")
            self._doc = fitz.open(str(self._pdf_path))
            self._page_count = len(self._doc)
            self._title.configure(text=title or self._pdf_path.name)
            self._fit_to_width()
            self._render_all_pages(clear_selection=True)
            return PdfOpenResult(ok=True, title=title or self._pdf_path.name)
        except Exception as e:
            self.show_error(title=title or "PDF", error=str(e))
            return PdfOpenResult(ok=False, title=title or "PDF", error=str(e))

    def clear_selection(self):
        self._selected_text = ""
        self._sel_info.configure(text="")
        self._sel_start = None
        self._sel_end = None
        self._sel_start_idx = None
        self._sel_end_idx = None
        self._sel_drag_start_xy = None
        self._sel_drag_end_xy = None
        self._sel_pdf_bounds_by_page = {}
        self._clear_highlights()

    def get_selected_text(self) -> str:
        return (self._selected_text or "").strip()

    def _render_all_pages(self, *, clear_selection: bool = False):
        if clear_selection:
            self.clear_selection()
        if not self._doc or self._page_count <= 0:
            self._set_placeholder()
            return

        try:
            import fitz  # PyMuPDF

            self._page_imgs = []
            self._words = []
            self._page_tops = []

            self._canvas.delete("all")
            y = 0
            gutter = 18
            for pi in range(self._page_count):
                page = self._doc.load_page(pi)
                mat = fitz.Matrix(self._zoom, self._zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                ppm = pix.tobytes("ppm")
                img = tk.PhotoImage(data=ppm)
                self._page_imgs.append(img)

                self._page_tops.append(y)
                self._canvas.create_image(0, y, anchor="nw", image=img)

                # collect words with y offset and zoom applied at draw-time
                words = page.get_text("words") or []
                for w in words:
                    wx0, wy0, wx1, wy1, txt, block, line, wno = w
                    cx0 = wx0 * self._zoom
                    cy0 = wy0 * self._zoom + y
                    cx1 = wx1 * self._zoom
                    cy1 = wy1 * self._zoom + y
                    self._words.append(
                        {
                            "cx0": cx0,
                            "cy0": cy0,
                            "cx1": cx1,
                            "cy1": cy1,
                            "text": txt,
                            "page": pi,
                            "block": block,
                            "line": line,
                            "word_no": wno,
                            "page_top": y,
                        }
                    )

                y += img.height() + gutter

            max_w = max((im.width() for im in self._page_imgs), default=1)
            self._canvas.configure(scrollregion=(0, 0, max_w, max(1, y)))
            self._page_lbl.configure(text=f"{self._page_count} page(s)")
            # Ensure stable reading order.
            self._words.sort(key=lambda ww: (ww["page"], ww["block"], ww["line"], ww["word_no"]))
            # Canvas was cleared; repaint selection highlight if it exists.
            if self._sel_pdf_bounds_by_page:
                self._update_selection_from_pdf_bounds()
        except Exception as e:
            self.show_error(title=self._title.cget("text"), error=str(e))

    def _fit_to_width(self):
        if not self._doc:
            return
        try:
            # Fit first page width into current canvas width.
            page0 = self._doc.load_page(0)
            page_w = float(page0.rect.width) if page0 else 612.0
            canvas_w = int(self._canvas.winfo_width() or 0)
            if canvas_w <= 50:
                return
            target_w = max(220, canvas_w - 18)  # some breathing room
            z = target_w / max(1.0, page_w)
            self._zoom = max(0.65, min(2.25, z))
        except Exception:
            pass

    def _on_canvas_configure(self, _e=None):
        # Re-fit to width when the pane is resized.
        if not self._doc:
            return
        try:
            old = self._zoom
            self._fit_to_width()
            if abs(self._zoom - old) > 0.05:
                self._render_all_pages(clear_selection=False)
        except Exception:
            pass

    def _on_mouse_down(self, e):
        if not self._doc:
            return
        x = self._canvas.canvasx(e.x)
        y = self._canvas.canvasy(e.y)
        self._sel_drag_start_xy = (x, y)
        self._sel_drag_end_xy = (x, y)
        self._update_selection_from_drag_rect()

    def _on_mouse_drag(self, e):
        if not self._doc or self._sel_drag_start_xy is None:
            return
        x = self._canvas.canvasx(e.x)
        y = self._canvas.canvasy(e.y)
        self._sel_drag_end_xy = (x, y)
        self._update_selection_from_drag_rect()

    def _on_mouse_up(self, e):
        if not self._doc:
            return
        self._update_selection_from_drag_rect(final=True)
        self._sel_drag_start_xy = None
        self._sel_drag_end_xy = None

    def _clear_highlights(self):
        for hid in self._highlight_ids:
            try:
                self._canvas.delete(hid)
            except Exception:
                pass
        self._highlight_ids = []

    def _word_index_at(self, x: float, y: float) -> Optional[int]:
        # First try exact hit.
        for i, w in enumerate(self._words):
            if w["cx0"] <= x <= w["cx1"] and w["cy0"] <= y <= w["cy1"]:
                return i

        # Snap to nearest word if user starts on whitespace.
        best_i = None
        best_d2 = None
        r = 14.0
        r2 = r * r
        for i, w in enumerate(self._words):
            cx = (w["cx0"] + w["cx1"]) / 2.0
            cy = (w["cy0"] + w["cy1"]) / 2.0
            dx = cx - x
            dy = cy - y
            d2 = dx * dx + dy * dy
            if d2 > r2:
                continue
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _update_selection_from_drag_rect(self, *, final: bool = False):
        self._clear_highlights()
        self._sel_pdf_bounds_by_page = {}
        if not self._sel_drag_start_xy or not self._sel_drag_end_xy:
            return

        x0, y0 = self._sel_drag_start_xy
        x1, y1 = self._sel_drag_end_xy
        left, right = (x0, x1) if x0 <= x1 else (x1, x0)
        top, bottom = (y0, y1) if y0 <= y1 else (y1, y0)

        # Click (no drag): select a single nearest word.
        if abs(right - left) < 3 and abs(bottom - top) < 3:
            idx = self._word_index_at(left, top)
            if idx is None:
                if final:
                    self._selected_text = ""
                    self._sel_info.configure(text="")
                return
            w = self._words[idx]
            left, right = w["cx0"], w["cx1"]
            top, bottom = w["cy0"], w["cy1"]

        # Convert the drag rectangle into per-page PDF-coordinate rectangles.
        # This avoids the "half width" bug in multi-column layouts where a reading-order range
        # can accidentally clamp to one column.
        if not self._page_imgs:
            return

        any_page = False
        for pi, img in enumerate(self._page_imgs):
            page_top = float(self._page_tops[pi]) if pi < len(self._page_tops) else 0.0
            page_bottom = page_top + float(img.height())
            page_right = float(img.width())

            # Intersect drag rect with this page's visible area.
            lcx = max(0.0, left)
            rcx = min(page_right, right)
            tcy = max(page_top, top)
            bcy = min(page_bottom, bottom)
            if rcx <= lcx or bcy <= tcy:
                continue

            # Convert to PDF coords (relative to page top).
            px0 = lcx / max(1e-6, self._zoom)
            px1 = rcx / max(1e-6, self._zoom)
            py0 = (tcy - page_top) / max(1e-6, self._zoom)
            py1 = (bcy - page_top) / max(1e-6, self._zoom)
            self._sel_pdf_bounds_by_page[pi] = (px0, py0, px1, py1)
            any_page = True

        if not any_page:
            if final:
                self._selected_text = ""
                self._sel_info.configure(text="")
            return

        self._update_selection_from_pdf_bounds()

    def _update_selection_from_pdf_bounds(self):
        self._clear_highlights()
        if not self._sel_pdf_bounds_by_page:
            return

        chosen: list[dict] = []
        for w in self._words:
            pi = int(w["page"])
            if pi not in self._sel_pdf_bounds_by_page:
                continue
            l, t, r, b = self._sel_pdf_bounds_by_page[pi]
            page_top = float(w.get("page_top", 0.0))
            # Use word-box intersection (not center-point) so partial overlaps still select.
            wx0 = float(w["cx0"]) / max(1e-6, self._zoom)
            wx1 = float(w["cx1"]) / max(1e-6, self._zoom)
            wy0 = (float(w["cy0"]) - page_top) / max(1e-6, self._zoom)
            wy1 = (float(w["cy1"]) - page_top) / max(1e-6, self._zoom)
            if wx1 < l or wx0 > r or wy1 < t or wy0 > b:
                continue
            chosen.append(w)

        if not chosen:
            return

        # Draw highlight
        for w in chosen:
            hid = self._canvas.create_rectangle(
                w["cx0"],
                w["cy0"],
                w["cx1"],
                w["cy1"],
                outline="",
                fill="#3B82F6",
                stipple="gray25",
            )
            self._highlight_ids.append(hid)

        # Sort visually for better multi-column output: page then y then x.
        chosen.sort(key=lambda ww: (ww["page"], ww["cy0"], ww["cx0"], ww["block"], ww["line"], ww["word_no"]))

        parts = []
        last_line = None
        for w in chosen:
            line_key = (w["page"], w["block"], w["line"])
            if last_line is not None and line_key != last_line:
                parts.append("\n")
            parts.append(w["text"])
            parts.append(" ")
            last_line = line_key
        text = "".join(parts).strip()
        self._selected_text = text
        self._sel_info.configure(text=f"Selected {len(text)} characters.")

    def _on_mousewheel(self, e):
        # Windows/Mac wheel
        try:
            delta = int(-1 * (e.delta / 120))
        except Exception:
            delta = 0
        if delta != 0:
            self._canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, e):
        # Linux wheel events
        if e.num == 4:
            self._canvas.yview_scroll(-3, "units")
        elif e.num == 5:
            self._canvas.yview_scroll(3, "units")

