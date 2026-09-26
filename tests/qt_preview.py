"""Generate review screenshots using synthetic images; no inference dependencies."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QFontDatabase, QFont
from PySide6.QtWidgets import QApplication
from integra_pose.plugins.plugin_assisted_pose_curation.qt_editor import Editor

out = Path(".qt_preview")
out.mkdir(exist_ok=True)
image = QImage(1280, 800, QImage.Format.Format_RGB32)
image.fill(QColor("#899494"))
painter = QPainter(image)
painter.setRenderHint(QPainter.RenderHint.Antialiasing)
painter.setPen(QColor("#b6c1be"))
for x in range(0, 1280, 80):
    painter.drawLine(x, 0, x, 800)
for y in range(0, 800, 80):
    painter.drawLine(0, y, 1280, y)
painter.setBrush(QColor("#424747"))
painter.setPen(Qt.PenStyle.NoPen)
painter.drawEllipse(470, 300, 270, 160)
painter.drawEllipse(700, 322, 95, 93)
painter.end()
path = out / "synthetic.png"
image.save(str(path))
app = QApplication([])
app.setStyle("Fusion")
QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
app.setFont(QFont("Segoe UI", 10))
editor = Editor(lambda message: None)
state = dict(path=str(path.resolve()), frames=[str(path.resolve())], index=0, instances=[dict(class_id=0, pose=[dict(name=n, x=x, y=y, v=2, conf=0) for n, x, y in [("nose", 780, 365), ("left_ear", 726, 330), ("right_ear", 732, 398), ("spine", 620, 373), ("tail_base", 483, 381)]])], assist=[], active_instance=0, active_point=0, classes=["mouse"], keypoints=["nose", "left_ear", "right_ear", "spine", "tail_base"], skeleton=[[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]], dirty=False, settings={}, review="Manual — pending", counts="Reviewed 0 / 1", task="Idle — synthetic preview, no model loaded", log="Synthetic preview. No model execution.", export="")
editor.show()
editor.apply_state(state)
for width, height in ((1440, 900), (1024, 700), (720, 500)):
    editor.resize(width, height)
    app.processEvents()
    editor.grab().save(str(out / f"review_{width}.png"))
editor.resize(1440, 900)
editor._pin()
app.processEvents()
editor.grab().save(str(out / "comparison.png"))
editor._allow_close = True
editor.close()
print(out.resolve())
