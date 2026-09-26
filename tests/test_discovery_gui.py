import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from matplotlib.backend_bases import MouseEvent

from integra_pose.discovery.app import Explorer
from integra_pose.discovery.store import Workspace


def test_linked_video_map_review_and_reopen(tmp_path):
    import cv2
    video=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(80,60))
    if not writer.isOpened():
        pytest.skip('MJPG encoder unavailable')
    for i in range(20):
        writer.write(np.full((60,80,3),i*10,np.uint8))
    writer.release()
    rows=[dict(row_id=f's:t:{i}',source='s',track='0',frame=i,class_id=0,behavior='animal',
               cluster_label='-1' if i==0 else '0:0',cluster_status='noise' if i==0 else 'clustered',
               features=[float(i),float(i%3)],embedding=[float(i),float(i%3),float(i%7)],keypoints={'point':[.2,.3,1.]}) for i in range(20)]
    payload=dict(run_id='r',rows=rows,sources={'s':dict(video=str(video),fps=10,frames=20,width=80,height=60,group='Group',directory=str(tmp_path))},
                 keypoints=['point'],feature_catalog=[dict(name='speed',units='frame units'),dict(name='position',units='relative')],params={})
    store=Workspace(tmp_path/'discovery.sqlite'); store.add_run(payload)
    app=QApplication.instance() or QApplication([])
    window=Explorer(store.path)
    try:
        window.show(); app.processEvents()
        assert window.video.pixmap() is not None
        window.tabs.setCurrentIndex(1); window.draw_map(); window.map.canvas.draw()
        assert len(window.map.ax.collections)==1
        artist=window.map.ax.collections[0]
        assert len(artist.get_offsets())==20
        point=window.map.ax.transData.transform(artist.get_offsets()[8])
        event=MouseEvent('button_press_event',window.map.canvas,*point,button=1)
        artist.pick(event)
        assert window.selected
        assert window.map_preview.pixmap() is not None
        window.timer.stop()
        window.start_frame.setValue(4); window.end_frame.setValue(6)
        window.reviewer.setText('AB'); window.annotation.setText('reviewed interval')
        window.annotate_interval()
        assert sum(r['review_label']=='reviewed interval' for r in window.rows)==3
        window.undo(); assert all(r['review_label']!='reviewed interval' for r in window.rows)
        window.seek(8); window.checkpoint()
        for mode in ('Distribution','Source-video time','Interval-relative time','Bout progress (%)','Trajectory','Dwell heatmap','Occupancy'):
            window.mode.setCurrentText(mode); window.draw_features(); window.graph.canvas.draw()
        window.graph.export(tmp_path/'figure.png',window.view_state())
        assert (tmp_path/'figure.csv').is_file() and (tmp_path/'figure.json').is_file()
    finally:
        window.close(); app.processEvents()
    reopened=Explorer(store.path)
    try:
        assert reopened.frame==8
        assert reopened.reviewer.text()=='AB'
    finally:
        reopened.close()
