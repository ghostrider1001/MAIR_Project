"""
Spatial Integrity Guard — Post-Check Size Preservation
=======================================================
MAIR+ Contribution C5.

Ensures expert output maintains the same spatial dimensions as the input.
Applied as a post-check after expert_fn() returns, NOT as a context manager
(SR expert runs as subprocess via os.system — can't wrap it).

Usage in scheduler.py:
    output_path = expert_fn(current_path)
    if output_path and expert_entry.get("preserves_size", True):
        output_path = SpatialGuard(current_path).check_and_fix(output_path)
"""

import cv2
import os


class SpatialGuard:
    """
    Checks that an expert's output has the same dimensions as its input.
    If dimensions differ and the expert is supposed to preserve size,
    rescales the output using Lanczos interpolation.
    """

    def __init__(self, input_path: str):
        img = cv2.imread(input_path)
        if img is not None:
            self.input_h, self.input_w = img.shape[:2]
        else:
            self.input_h, self.input_w = None, None

    def check_and_fix(self, output_path: str) -> str:
        """
        Compare output dimensions to input. Rescale if mismatched.

        Returns:
            The (possibly corrected) output path.
        """
        if self.input_h is None or self.input_w is None:
            return output_path

        output = cv2.imread(output_path)
        if output is None:
            return output_path

        out_h, out_w = output.shape[:2]

        if out_h == self.input_h and out_w == self.input_w:
            return output_path  # dimensions match — no action needed

        # Mismatch detected — rescale
        print(
            f"[SpatialGuard] Size mismatch: input=({self.input_w}x{self.input_h}) "
            f"output=({out_w}x{out_h}) — rescaling to input dimensions"
        )
        corrected = cv2.resize(
            output,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_LANCZOS4,
        )
        cv2.imwrite(output_path, corrected)
        return output_path
