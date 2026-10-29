#!/usr/bin/env python3
"""Contact sheet of crop jpgs (for eye review): crop_sheet.py out.jpg files... (12 per row, 96 px cells, name under)."""
import sys
import cv2, numpy as np
out, files = sys.argv[1], sys.argv[2:]
C, R = 96, 12
rows = []
cells = []
for f in files:
    im = cv2.imread(f); cell = np.zeros((C + 14, C, 3), np.uint8)
    if im is not None:
        h, w = im.shape[:2]; s = C / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        cell[:im.shape[0], :im.shape[1]] = im
    cv2.putText(cell, f.rsplit("/", 1)[-1][:-4][-14:], (1, C + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
    cells.append(cell)
while len(cells) % R:
    cells.append(np.zeros_like(cells[0]))
sheet = np.vstack([np.hstack(cells[i:i + R]) for i in range(0, len(cells), R)])
cv2.imwrite(out, sheet)
