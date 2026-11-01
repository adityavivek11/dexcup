"""Printable AprilTags (with proper white margins) and a calibration checkerboard, as exact-size PDFs.

WHY generate these instead of using any tag image off the web?
  1. The white margin is what the detector uses to find the tag (it looks for a dark square on a light
     background first, and only then reads the pattern). The first test clips failed because the tags
     were trimmed to the black edge. These sheets print a 2-cell margin plus a dashed cut line OUTSIDE it.
  2. The pose maths needs the true tag size. Printing at 300 dpi with "Actual size" (not "Fit to page")
     gives the stated size; you still measure with a ruler, because printers rescale silently.

Run:  python make_printables.py   ->  printables/tags.pdf, printables/checkerboard.pdf
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "printables"
DPI = 300
A4 = (2480, 3508)                     # A4 portrait at 300 dpi
MM = DPI / 25.4                       # pixels per mm

# tag36h11: 8x8 cells incl. the 1-cell black border. "Tag size" = edge of the black square = 8 cells.
# Coaster tag is big on purpose: a small error in its tilt gets multiplied by the wrist-coaster distance
# (lever arm). Synthetic test: 50 -> 70 mm cut wrist p95 error from 4.2 to 3.5 mm.
TAGS = {1: ("COASTER", 70.0), 0: ("WRIST", 35.0)}   # id: (role, black-square edge in mm)
MARGIN_CELLS = 2                                     # white quiet zone, in cells, on every side

# Checkerboard: 10x7 squares -> 9x6 inner corners (what OpenCV detects). 25 mm squares fit A4 landscape.
CB_SQUARES = (10, 7)
CB_SQUARE_MM = 25.0


def font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def tag_image(tag_id: int, edge_mm: float) -> Image.Image:
    """Tag + white margin, rendered so the black square is exactly edge_mm at DPI."""
    cell_px = round(edge_mm * MM / 8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(d, tag_id, 8 * cell_px)       # exact multiple: crisp cells
    m = MARGIN_CELLS * cell_px
    marker = cv2.copyMakeBorder(marker, m, m, m, m, cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(marker)


def dashed_rect(draw, box, dash=25, color=(150, 150, 150)):
    x0, y0, x1, y1 = box
    for x in range(x0, x1, 2 * dash):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=color, width=3)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=color, width=3)
    for y in range(y0, y1, 2 * dash):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=color, width=3)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=color, width=3)


def tags_sheet() -> Image.Image:
    page = Image.new("RGB", A4, "white")
    dr = ImageDraw.Draw(page)
    dr.text((150, 120), "dexcup tags: tag36h11. Print at ACTUAL SIZE / 100% (not 'fit to page').",
            fill="black", font=font(50))
    dr.text((150, 190), "Cut on the dashed line. Never cut into the white margin. Glue onto white card.",
            fill="black", font=font(42))
    y = 400
    for tag_id, (role, edge) in TAGS.items():
        img = tag_image(tag_id, edge)
        x = (A4[0] - img.width) // 2
        page.paste(img, (x, y))
        pad = 30
        dashed_rect(dr, (x - pad, y - pad, x + img.width + pad, y + img.height + pad))
        dr.text((150, y + img.height + 70),
                f"ID {tag_id}  {role}: black square should measure {edge:.0f} mm. Measure it and report the real value.",
                fill="black", font=font(40))
        y += img.height + 400
    # 50 mm scale bar: if this prints at a different length, the printer is rescaling
    bar = round(50 * MM)
    dr.rectangle([150, A4[1] - 300, 150 + bar, A4[1] - 270], fill="black")
    dr.text((150, A4[1] - 250), "scale check: this bar = 50 mm", fill="black", font=font(40))
    return page


def checkerboard_sheet() -> Image.Image:
    W, H = A4[1], A4[0]                                      # landscape
    page = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(page)
    sq = round(CB_SQUARE_MM * MM)
    nx, ny = CB_SQUARES
    x0, y0 = (W - nx * sq) // 2, (H - ny * sq) // 2 + 60
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                dr.rectangle([x0 + i * sq, y0 + j * sq, x0 + (i + 1) * sq - 1, y0 + (j + 1) * sq - 1],
                             fill="black")
    dr.text((x0, 100), f"Calibration board: {nx}x{ny} squares ({nx-1}x{ny-1} inner corners), "
                       f"{CB_SQUARE_MM:.0f} mm squares. Print at 100%. Tape FLAT on stiff board.",
            fill="black", font=font(42))
    return page


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tags_sheet().save(OUT / "tags.pdf", resolution=DPI)
    checkerboard_sheet().save(OUT / "checkerboard.pdf", resolution=DPI)
    # Self-check: the generated tags must decode, with the IDs we expect.
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    g = cv2.cvtColor(np.array(tags_sheet()), cv2.COLOR_RGB2GRAY)
    _, ids, _ = det.detectMarkers(cv2.resize(g, None, fx=0.3, fy=0.3))
    print("wrote", OUT / "tags.pdf", OUT / "checkerboard.pdf", "| decoded ids:", sorted(np.ravel(ids).tolist()))
