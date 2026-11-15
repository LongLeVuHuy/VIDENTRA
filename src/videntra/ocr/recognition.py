"""Text recognition models with real OCR backends and deterministic mock for tests."""
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
from PIL import Image

from videntra.evidence.normalize import normalize_text, has_vietnamese_diacritics


@dataclass(frozen=True)
class TextRecognitionResult:
    text: str
    confidence: float
    model_id: str
    script: str = "latin"


class OCRRecognizer:
    """Base text recognizer interface."""

    def __init__(self, model_id: str = "general_ocr"):
        self.model_id = model_id

    def recognize_crop(self, crop_image: np.ndarray) -> TextRecognitionResult:
        raise NotImplementedError


class MockOCRRecognizer(OCRRecognizer):
    """Deterministic mock recognizer for unit tests and synthetic pipelines."""

    def __init__(self, mock_text: str = "SALE", confidence: float = 0.95, model_id: str = "mock_ocr"):
        super().__init__(model_id=model_id)
        self.mock_text = mock_text
        self.confidence = confidence

    def recognize_crop(self, crop_image: np.ndarray) -> TextRecognitionResult:
        clean_text = normalize_text(self.mock_text)
        script = "vietnamese" if has_vietnamese_diacritics(clean_text) else "latin"
        return TextRecognitionResult(
            text=clean_text,
            confidence=self.confidence,
            model_id=self.model_id,
            script=script,
        )


class GeneralOCRRecognizer(OCRRecognizer):
    """General Latin/alphanumeric text recognizer using PaddleOCR (en) or RapidOCR."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_id: str = "paddleocr_en",
        lang: str = "en",
        use_rapidocr: bool = False,
        allow_rapidocr_fallback: bool = True,
        device: str = "cpu",
    ):
        super().__init__(model_id=model_id)
        self.model_path = Path(model_path) if model_path else None
        self.lang = lang
        self.use_rapidocr = use_rapidocr
        self.allow_rapidocr_fallback = allow_rapidocr_fallback
        self.device = device
        self._engine = None
        self._backend_kind: str | None = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return

        errors: list[str] = []
        if not self.use_rapidocr:
            try:
                from paddleocr import PaddleOCR

                use_gpu = str(self.device).startswith("cuda")
                self._engine = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=use_gpu,
                    show_log=False,
                )
                self._backend_kind = "paddleocr"
                return
            except Exception as exc:
                errors.append(f"PaddleOCR unavailable: {exc}")
                if not self.allow_rapidocr_fallback:
                    raise RuntimeError("; ".join(errors)) from exc

        if self.use_rapidocr or self.allow_rapidocr_fallback:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._engine = RapidOCR()
                self._backend_kind = "rapidocr"
                if self.model_id == "paddleocr_en":
                    self.model_id = "RapidOCR"
                return
            except Exception as exc:
                errors.append(f"RapidOCR unavailable: {exc}")

        raise RuntimeError(
            "Neither PaddleOCR nor RapidOCR is available for GeneralOCRRecognizer: "
            + "; ".join(errors)
        )

    def recognize_crop(self, crop_image: np.ndarray) -> TextRecognitionResult:
        if crop_image is None or crop_image.size == 0:
            return TextRecognitionResult(text="", confidence=0.0, model_id=self.model_id, script="latin")

        self._ensure_loaded()

        if self._backend_kind == "rapidocr":
            res, _ = self._engine(crop_image)
            if not res:
                return TextRecognitionResult(text="", confidence=0.0, model_id=self.model_id, script="latin")
            texts: list[str] = []
            confs: list[float] = []
            for item in res:
                _, txt, conf = item
                t = str(txt).strip()
                if t:
                    texts.append(t)
                    confs.append(float(conf))
            clean_text = normalize_text(" ".join(texts))
            avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
            script = "vietnamese" if has_vietnamese_diacritics(clean_text) else "latin"
            return TextRecognitionResult(
                text=clean_text,
                confidence=avg_conf,
                model_id=self.model_id,
                script=script,
            )

        # PaddleOCR backend
        result = self._engine.ocr(crop_image, cls=True)
        texts = []
        confs = []
        if result:
            for line_group in result:
                if not line_group:
                    continue
                for item in line_group:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        rec = item[1]
                        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
                            t = str(rec[0]).strip()
                            if t:
                                texts.append(t)
                                confs.append(float(rec[1]))
        clean_text = normalize_text(" ".join(texts))
        avg_conf = float(sum(confs) / len(confs)) if confs else 0.0
        script = "vietnamese" if has_vietnamese_diacritics(clean_text) else "latin"
        return TextRecognitionResult(
            text=clean_text,
            confidence=avg_conf,
            model_id=self.model_id,
            script=script,
        )


class VietOCRRecognizer(OCRRecognizer):
    """Vietnamese specialist recognizer preserving diacritics and tones via VietOCR."""

    DEFAULT_WEIGHTS_PATH = Path("models/ocr/vietocr/vgg_transformer.pth")

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_id: str = "vietocr_vgg",
        device: str = "cpu",
        config_name: str = "vgg_transformer",
    ):
        super().__init__(model_id=model_id)
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.config_name = config_name
        self._predictor = None

    def _resolve_weights_path(self) -> Path:
        env_w = os.environ.get("VIDENTRA_VIETOCR_WEIGHTS")
        candidates = [
            self.model_path,
            Path(env_w) if env_w else None,
            self.DEFAULT_WEIGHTS_PATH,
        ]
        for cand in candidates:
            if cand is not None and cand.is_file():
                return cand
        raise RuntimeError(
            f"VietOCR weights not found. Checked: {[str(c) for c in candidates if c is not None]}"
        )

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except Exception as exc:
            raise RuntimeError(f"vietocr library is not installed or failed to import: {exc}") from exc

        weights_path = self._resolve_weights_path()
        config = Cfg.load_config_from_name(self.config_name)
        config["weights"] = str(weights_path)
        config["device"] = self.device
        config.setdefault("cnn", {})["pretrained"] = False
        config.setdefault("predictor", {})["beamsearch"] = False
        self._predictor = Predictor(config)

    def recognize_crop(self, crop_image: np.ndarray) -> TextRecognitionResult:
        if crop_image is None or crop_image.size == 0:
            return TextRecognitionResult(text="", confidence=0.0, model_id=self.model_id, script="vietnamese")

        self._ensure_loaded()
        img_arr = crop_image
        if img_arr.dtype != np.uint8:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_arr)

        try:
            pred = self._predictor.predict(pil_img, return_prob=True)
        except TypeError:
            pred = self._predictor.predict(pil_img)

        if isinstance(pred, tuple) and len(pred) >= 2:
            raw_text, conf = str(pred[0]), float(pred[1])
        else:
            raw_text, conf = str(pred), 0.9

        clean_text = normalize_text(raw_text)
        script = "vietnamese" if has_vietnamese_diacritics(clean_text) else "latin"
        return TextRecognitionResult(
            text=clean_text,
            confidence=conf,
            model_id=self.model_id,
            script=script,
        )
