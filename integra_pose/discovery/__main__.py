import sys

from integra_pose.utils.qt_runtime import prepare_qt_runtime
prepare_qt_runtime()

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtNetwork import QLocalServer, QLocalSocket
import hashlib

from .app import Explorer


def main():
    app = QApplication(sys.argv)
    from pathlib import Path
    path = str(Path(sys.argv[1]).resolve())
    key = 'integrapose-discovery-' + hashlib.sha256(path.casefold().encode()).hexdigest()[:24]
    client = QLocalSocket()
    client.connectToServer(key)
    if client.waitForConnected(300):
        return 0
    QLocalServer.removeServer(key)
    server = QLocalServer()
    if not server.listen(key):
        raise RuntimeError('Another editor owns this workspace.')
    window = Explorer(path)
    window.showMaximized()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
