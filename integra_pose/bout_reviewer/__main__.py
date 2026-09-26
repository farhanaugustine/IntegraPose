from integra_pose.utils.qt_runtime import prepare_qt_runtime
prepare_qt_runtime()

from .app import main


if __name__ == "__main__":
    raise SystemExit(main())
