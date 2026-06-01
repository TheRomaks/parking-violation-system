import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from detect_and_read_plate import PlateReader
from detect_and_track_cars import CarTracker
from parking_vision.common.video import open_writer
from sign_detect import SignDetector
from violation_pipeline import ViolationPipeline


def frame_to_qimage(frame: Any) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    return QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()


def format_timestamp(timestamp_ms: float | None) -> str:
    if timestamp_ms is None:
        return "--:--.--"
    total_seconds = max(0.0, timestamp_ms / 1000.0)
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


class ClickableVideoLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    violation_ready = Signal(dict)
    status_changed = Signal(str)
    error = Signal(str)
    finished_processing = Signal()

    def __init__(
        self,
        video_path: str,
        algorithm: str,
        parking_time_limit_s: float,
        violation_debug: bool,
        save_results: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self.algorithm = algorithm
        self.parking_time_limit_s = parking_time_limit_s
        self.violation_debug = violation_debug
        self.save_results = save_results
        self._running = True
        self._violation_keys: set[tuple[Any, ...]] = set()

    def stop(self) -> None:
        self._running = False

    def _build_algorithm(self):
        if self.algorithm == "cars":
            return CarTracker(model_path="models/cars.pt")
        if self.algorithm == "signs":
            sign_model = "models/signs.pt"
            return SignDetector(model_path=sign_model)
        if self.algorithm == "plates":
            return PlateReader(model_path="models/plates.pt")
        return ViolationPipeline(
            car_model_path="models/cars.pt",
            sign_model_path="models/signs.pt",
            plate_model_path="models/plates.pt",
            parking_time_limit_s=self.parking_time_limit_s,
            draw_zone_debug=self.violation_debug,
        )

    @staticmethod
    def _annotate_plates(frame: Any, plates: list[dict[str, Any]]) -> Any:
        annotated = frame.copy()
        for plate in plates:
            x1, y1, x2, y2 = map(int, plate["bbox"])
            color = (0, 255, 0) if plate.get("valid") else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label_text = plate.get("text") or "plate"
            label = f"{label_text} ({plate.get('ocr_conf', 0.0):.2f})"
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _build_output_dir(self) -> Path:
        source_path = Path(self.video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / "gui" / f"{source_path.stem}_{self.algorithm}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @staticmethod
    def _write_violation_csv_row(
        writer: csv.writer,
        frame_index: int,
        timestamp_ms: float | None,
        violation: Any,
    ) -> None:
        writer.writerow(
            [
                frame_index,
                "" if timestamp_ms is None else f"{timestamp_ms:.2f}",
                violation.track_id,
                violation.plate,
                violation.sign_id,
                violation.sign_label,
                violation.status,
                f"{violation.time_in_zone_s:.2f}",
                f"{violation.stopped_duration_s:.2f}",
                int(violation.bbox.x1),
                int(violation.bbox.y1),
                int(violation.bbox.x2),
                int(violation.bbox.y2),
            ]
        )

    def _emit_unique_violations(self, violations: list[Any], timestamp_ms: float | None) -> list[Any]:
        emitted: list[Any] = []
        for violation in violations:
            key = (violation.track_id, violation.sign_id)
            if key in self._violation_keys:
                continue
            self._violation_keys.add(key)
            emitted.append(violation)
            self.violation_ready.emit(
                {
                    "timestamp": format_timestamp(timestamp_ms),
                    "duration": f"{violation.stopped_duration_s:.1f} c",
                    "status": "Зафиксировано",
                    "type": f"Знак {violation.sign_label}",
                    "plate": violation.plate or "unknown",
                }
            )
        return emitted

    def run(self) -> None:
        capture = cv2.VideoCapture(self.video_path)
        if not capture.isOpened():
            self.error.emit(f"Не удалось открыть видео: {self.video_path}")
            self.finished_processing.emit()
            return

        try:
            algorithm = self._build_algorithm()
        except Exception as exc:
            capture.release()
            self.error.emit(str(exc))
            self.finished_processing.emit()
            return

        frame_index = 0
        output_dir: Path | None = None
        writer = None
        jsonl_file = None
        violation_csv_file = None
        violation_csv_writer = None

        if self.save_results:
            output_dir = self._build_output_dir()
            jsonl_file = (output_dir / "results.jsonl").open("w", encoding="utf-8")
            if self.algorithm == "violations":
                violation_csv_file = (output_dir / "violations.csv").open("w", newline="", encoding="utf-8")
                violation_csv_writer = csv.writer(violation_csv_file)
                violation_csv_writer.writerow(
                    [
                        "frame_index",
                        "timestamp_ms",
                        "track_id",
                        "plate",
                        "sign_id",
                        "sign_label",
                        "status",
                        "time_in_zone_s",
                        "stopped_duration_s",
                        "x1",
                        "y1",
                        "x2",
                        "y2",
                    ]
                )

        self.status_changed.emit("Обработка запущена")
        try:
            while self._running:
                ok, frame = capture.read()
                if not ok:
                    break

                timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                timestamp_value = None if timestamp_ms < 0 else float(timestamp_ms)

                if self.algorithm == "cars":
                    frame_result = algorithm.process_frame(frame, frame_index, timestamp_value)
                    annotated = algorithm.annotate_frame(frame, frame_result)
                    result_payload = frame_result.to_dict()
                    violations = []
                    violations_for_csv = []
                elif self.algorithm == "signs":
                    frame_result = algorithm.process_frame(frame, frame_index, timestamp_value)
                    annotated = algorithm.annotate_frame(frame, frame_result)
                    result_payload = frame_result.to_dict()
                    violations = []
                    violations_for_csv = []
                elif self.algorithm == "plates":
                    plate_result = algorithm.process_frame(frame)
                    annotated = self._annotate_plates(frame, plate_result)
                    result_payload = {
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp_value,
                        "plates": plate_result,
                    }
                    violations = []
                    violations_for_csv = []
                else:
                    annotated, violations, frame_result = algorithm.process_frame(frame, frame_index, timestamp_value)
                    result_payload = frame_result.to_dict()
                    violations_for_csv = self._emit_unique_violations(violations, timestamp_value)

                if self.save_results and output_dir is not None and jsonl_file is not None:
                    if writer is None:
                        writer = open_writer(output_dir / "annotated.mp4", capture, annotated)
                    writer.write(annotated)
                    jsonl_file.write(json.dumps(result_payload, ensure_ascii=False) + "\n")

                    if violation_csv_writer is not None:
                        for violation in violations_for_csv:
                            self._write_violation_csv_row(
                                violation_csv_writer,
                                frame_index,
                                timestamp_value,
                                violation,
                            )

                self.frame_ready.emit(frame_to_qimage(annotated))
                frame_index += 1
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if writer is not None:
                writer.release()
            if jsonl_file is not None:
                jsonl_file.close()
            if violation_csv_file is not None:
                violation_csv_file.close()
            capture.release()
            if self.save_results and output_dir is not None:
                self.status_changed.emit(f"Обработка завершена. Сохранено: {output_dir}")
            else:
                self.status_changed.emit("Обработка завершена")
            self.finished_processing.emit()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Parking Vision UI")
        self.resize(1280, 820)

        self.video_path: str | None = None
        self.worker: VideoWorker | None = None
        self._violation_rows: set[tuple[str, str, str]] = set()
        self._last_frame_image: QImage | None = None

        self.video_label = ClickableVideoLabel("Нажмите для загрузки видео")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(760, 430)
        self.video_label.setStyleSheet(
            "QLabel { background: #e9e9e9; border: 2px dashed #bdbdbd; font-size: 22px; color: #555; }"
        )
        self.video_label.clicked.connect(self.open_video)

        self.path_label = QLabel("Видео не выбрано")
        self.path_label.setWordWrap(True)

        self.load_button = QPushButton("Загрузить видео")
        self.load_button.clicked.connect(self.open_video)

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("Детекция и трекинг авто", "cars")
        self.algorithm_combo.addItem("Детекция знаков", "signs")
        self.algorithm_combo.addItem("Распознавание номеров", "plates")
        self.algorithm_combo.addItem("Violation Pipeline", "violations")
        self.algorithm_combo.currentIndexChanged.connect(self._update_threshold_visibility)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(1.0, 3600.0)
        self.threshold_spin.setDecimals(1)
        self.threshold_spin.setSingleStep(10.0)
        self.threshold_spin.setValue(300.0)
        self.threshold_spin.setSuffix(" c")

        self.threshold_row = QWidget()
        threshold_layout = QFormLayout(self.threshold_row)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addRow("Порог времени (3.28-3.30)", self.threshold_spin)

        self.violation_debug_checkbox = QCheckBox("Debug: polygon and projection line")
        self.violation_debug_checkbox.setChecked(False)

        self.save_results_checkbox = QCheckBox("Сохранять результаты")
        self.save_results_checkbox.setChecked(False)

        self.start_button = QPushButton("Запустить")
        self.start_button.setMinimumHeight(52)
        self.start_button.clicked.connect(self.start_processing)

        self.stop_button = QPushButton("Остановить")
        self.stop_button.setMinimumHeight(52)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_processing)

        self.status_label = QLabel("Готово")

        self.violations_table = QTableWidget(0, 5)
        self.violations_table.setHorizontalHeaderLabels(
            ["Время", "Длительность", "Статус", "Тип нарушения", "Госномер"]
        )
        self.violations_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.violations_table.verticalHeader().setVisible(False)
        self.violations_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.violations_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        controls_group = QGroupBox("Управление")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.addWidget(self.load_button)
        controls_layout.addWidget(self.path_label)
        controls_layout.addWidget(QLabel("Алгоритм"))
        controls_layout.addWidget(self.algorithm_combo)
        controls_layout.addWidget(self.threshold_row)
        controls_layout.addWidget(self.violation_debug_checkbox)
        controls_layout.addWidget(self.save_results_checkbox)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.status_label)

        self.violations_group = QGroupBox("Нарушения")
        violations_layout = QVBoxLayout(self.violations_group)
        violations_layout.addWidget(self.violations_table)

        video_title = QLabel("Видеопоток")
        video_title.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)

        video_container = QWidget()
        video_container_layout = QVBoxLayout(video_container)
        video_container_layout.setContentsMargins(0, 0, 0, 0)
        video_container_layout.addStretch(1)
        video_container_layout.addWidget(self.video_label, 0, Qt.AlignmentFlag.AlignCenter)
        video_container_layout.addStretch(1)

        left_layout = QVBoxLayout()
        left_layout.addWidget(video_title)
        left_layout.addWidget(video_container, 1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(controls_group)
        right_layout.addWidget(self.violations_group, 1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 2)

        central = QWidget()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        self._update_threshold_visibility()

    def _update_threshold_visibility(self) -> None:
        is_violation_mode = self.algorithm_combo.currentData() == "violations"
        self.threshold_row.setVisible(is_violation_mode)
        self.violation_debug_checkbox.setVisible(is_violation_mode)
        self.violations_group.setVisible(is_violation_mode)

    def open_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбрать видео",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)",
        )
        if not file_path:
            return
        self.video_path = file_path
        self.path_label.setText(file_path)
        self._show_first_frame(file_path)

    def _show_first_frame(self, file_path: str) -> None:
        capture = cv2.VideoCapture(file_path)
        ok, frame = capture.read()
        capture.release()
        if not ok:
            QMessageBox.warning(self, "Ошибка", "Не удалось прочитать первый кадр видео.")
            return
        self._set_video_frame(frame_to_qimage(frame))

    def _set_video_frame(self, image: QImage) -> None:
        self._last_frame_image = image
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._last_frame_image is not None:
            self._set_video_frame(self._last_frame_image)

    def start_processing(self) -> None:
        if not self.video_path:
            QMessageBox.information(self, "Видео", "Сначала выберите видео.")
            return
        if self.worker is not None and self.worker.isRunning():
            return

        self.violations_table.setRowCount(0)
        self._violation_rows.clear()

        self.worker = VideoWorker(
            video_path=self.video_path,
            algorithm=self.algorithm_combo.currentData(),
            parking_time_limit_s=self.threshold_spin.value(),
            violation_debug=self.violation_debug_checkbox.isChecked(),
            save_results=self.save_results_checkbox.isChecked(),
            parent=self,
        )
        self.worker.frame_ready.connect(self._set_video_frame)
        self.worker.violation_ready.connect(self._append_violation_row)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.error.connect(self._show_error)
        self.worker.finished_processing.connect(self._on_processing_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()

    def stop_processing(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.status_label.setText("Остановка...")

    def _append_violation_row(self, item: dict[str, str]) -> None:
        key = (item["timestamp"], item["type"], item["plate"])
        if key in self._violation_rows:
            return
        self._violation_rows.add(key)

        row = self.violations_table.rowCount()
        self.violations_table.insertRow(row)
        self.violations_table.setItem(row, 0, QTableWidgetItem(item["timestamp"]))
        self.violations_table.setItem(row, 1, QTableWidgetItem(item["duration"]))
        self.violations_table.setItem(row, 2, QTableWidgetItem(item["status"]))
        self.violations_table.setItem(row, 3, QTableWidgetItem(item["type"]))
        self.violations_table.setItem(row, 4, QTableWidgetItem(item["plate"]))

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", message)

    def _on_processing_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
