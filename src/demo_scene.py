import cv2
import numpy as np


class SyntheticScene:
    """Generate a reproducible moving-target demo with distractors and occlusions."""

    def __init__(self, width=960, height=540, total_frames=600):
        self.width = width
        self.height = height
        self.total_frames = total_frames

    def frame(self, index):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        gradient = np.linspace(20, 70, self.width, dtype=np.uint8)
        image[:, :, 0] = gradient
        image[:, :, 1] = gradient // 2
        image[:, :, 2] = 30

        t = index / max(self.total_frames - 1, 1)
        x = int(80 + t * (self.width - 160))
        y = int(self.height * 0.5 + 120 * np.sin(index * 0.045))

        occluded = 210 <= index <= 270 or 430 <= index <= 455
        if not occluded:
            cv2.circle(image, (x, y), 24, (0, 220, 0), -1)
            cv2.circle(image, (x, y), 24, (255, 255, 255), 2)

        cv2.circle(image, (180, 110), 18, (0, 0, 180), -1)
        cv2.rectangle(image, (700, 80), (755, 135), (180, 80, 0), -1)

        if occluded:
            cv2.rectangle(
                image,
                (x - 38, y - 45),
                (x + 38, y + 45),
                (110, 110, 110),
                -1,
            )

        noise = np.random.normal(0, 5, image.shape).astype(np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255)
        return image.astype(np.uint8)
