"""Gravação de amostras rotuladas para o dataset de treinamento.

Usado por scripts/record_dataset.py. Reaproveita FrameSource e PoseEstimator
via injeção de dependência — a gravação usa exatamente o mesmo pipeline de
captura/pose do modo "live", o que evita divergência entre dataset e
inferência.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from eldorado_fighter.capture.base import FrameSource
from eldorado_fighter.dataset.schema import KeypointSample, save_sample
from eldorado_fighter.pose.base import PoseEstimator
from eldorado_fighter.pose.visualization import draw_pose

_PREVIEW_WINDOW = "Eldorado Fighter - gravacao"
_SPACE_KEY = 32


class DatasetRecorder:
    def __init__(
        self,
        frame_source: FrameSource,
        pose_estimator: PoseEstimator,
        keypoints_dir: Path,
        raw_videos_dir: Path | None = None,
        fps: int = 30,
        show_preview: bool = True,
    ) -> None:
        self._frame_source = frame_source
        self._pose_estimator = pose_estimator
        self._keypoints_dir = Path(keypoints_dir)
        self._raw_videos_dir = Path(raw_videos_dir) if raw_videos_dir else None
        self._fps = fps
        self._show_preview = show_preview

    def __enter__(self) -> "DatasetRecorder":
        self._frame_source.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._frame_source.stop()
        if self._show_preview:
            cv2.destroyWindow(_PREVIEW_WINDOW)

    def wait_for_space(self, message: str) -> None:
        """Mostra o preview ao vivo (se habilitado) e bloqueia até o usuário
        apertar espaço com a janela em foco. Sem preview, aguarda ENTER no
        terminal, já que não há janela para capturar a tecla.
        """
        if not self._show_preview:
            input(f"{message} [ENTER para iniciar] ")
            return

        while True:
            frame = self._frame_source.read()
            if frame is None:
                continue

            pose = self._pose_estimator.estimate(frame)
            annotated = draw_pose(frame, pose)
            cv2.putText(
                annotated, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
            )
            cv2.imshow(_PREVIEW_WINDOW, annotated)
            if cv2.waitKey(1) & 0xFF == _SPACE_KEY:
                break

    def record_clip(self, movement_label: str, action_label: str, duration_seconds: float) -> Path:
        """Grava um clipe de `duration_seconds` rotulado com uma classe por
        eixo (`movement_label`, `action_label` — ver classification.movements).
        Um clipe de um eixo só (ex.: só gravando ação) deve passar "idle"
        como label do outro eixo.

        Retorna o caminho do arquivo .npz salvo. Frames sem pose detectada
        são descartados na extração de keypoints (ex.: pessoa momentaneamente
        fora do quadro), mas o vídeo bruto é preservado integralmente — ele é
        a fonte de verdade, caso o estimador de pose seja trocado/melhorado
        no futuro (ver scripts/extract_keypoints.py).
        """
        keypoints_frames: list[np.ndarray] = []
        visibility_frames: list[np.ndarray] = []

        combo_label = f"{movement_label}__{action_label}"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        video_writer = self._open_video_writer(combo_label, timestamp)

        frame_interval = 1.0 / self._fps
        deadline = time.monotonic() + duration_seconds

        try:
            while time.monotonic() < deadline:
                frame_start = time.monotonic()
                frame = self._frame_source.read()
                if frame is None:
                    break

                if video_writer is not None:
                    video_writer.write(frame)

                pose = self._pose_estimator.estimate(frame)
                if pose is not None:
                    keypoints_frames.append(pose.landmarks)
                    visibility_frames.append(pose.visibility)

                if self._show_preview:
                    annotated = draw_pose(frame, pose)
                    cv2.imshow(_PREVIEW_WINDOW, annotated)
                    cv2.waitKey(1)

                elapsed = time.monotonic() - frame_start
                time.sleep(max(0.0, frame_interval - elapsed))
        finally:
            if video_writer is not None:
                video_writer.release()

        sample = KeypointSample(
            keypoints=np.stack(keypoints_frames, axis=0),
            visibility=np.stack(visibility_frames, axis=0),
            movement_label=movement_label,
            action_label=action_label,
            fps=self._fps,
        )

        output_path = self._keypoints_dir / combo_label / f"{timestamp}.npz"
        save_sample(output_path, sample)
        return output_path

    def _open_video_writer(self, combo_label: str, timestamp: str) -> cv2.VideoWriter | None:
        if self._raw_videos_dir is None:
            return None
        video_dir = self._raw_videos_dir / combo_label
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(str(video_path), fourcc, self._fps, (self._frame_source.width, self._frame_source.height))
