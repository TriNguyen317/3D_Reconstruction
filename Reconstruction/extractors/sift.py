import cv2
import numpy as np

from .extractor_base import ExtractorBase, FeaturesDict


class SIFTExtractor(ExtractorBase):
    default_conf = {
        "name:": "sift",
        "n_features": 8000,
        "nOctaveLayers": 3,
        "contrastThreshold": 0.04,
        "edgeThreshold": 10,
        "sigma": 1.6,
    }
    required_inputs = []
    grayscale = True
    as_float = False
    descriptor_size = 256

    def __init__(self, config: dict):
        # Init the base class
        super().__init__(config)

        # Load extractor
        cfg = self._config.get("extractor")
        self._extractor = cv2.SIFT_create(
            nfeatures=cfg["n_features"],
            nOctaveLayers=cfg["nOctaveLayers"],
            contrastThreshold=cfg["contrastThreshold"],
            edgeThreshold=cfg["edgeThreshold"],
            sigma=cfg["sigma"],
        )

    def _extract(self, image: np.ndarray) -> np.ndarray:
        kp, des = self._extractor.detectAndCompute(image, None)
        if kp:
            kpts = cv2.KeyPoint_convert(kp)
            des = des.astype(float).T
        else:
            kpts = np.array([], dtype=np.float32).reshape(0, 2)
            des = np.array([], dtype=np.float32).reshape(
                self.descriptor_size,
                0,
            )

        # Convert tensors to numpy arrays
        feats = FeaturesDict(keypoints=kpts, descriptors=des)

        return feats

    def _frame2tensor(self, image: np.ndarray, device: str = "cuda"):
        pass



