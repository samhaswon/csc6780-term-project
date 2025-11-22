from typing import Tuple, Union, Optional

import cv2
import numpy as np
from PIL.Image import Image as PILImage  # For typing
import torch
import torch.nn.functional as F
from .u2net import U2NETP


class PatchTorchSession:
    """
    Session for Torch inference with post-processing for the refiner U2NetP model
    """

    def __init__(self, model_path: str, input_size: Optional[Tuple[int, int]] = None, device: str = None) -> None:
        self.net = U2NETP(4, 1)
        self.input_size = input_size if input_size is not None else [512, 512]
        if device is not None:
            self.device = device
        else:
            self.device = "cpu"
        if torch.cuda.is_available() and self.device == "cuda":
            self.net.load_state_dict(
                torch.load(model_path, weights_only=False)
            )
            self.net.cuda()
        else:
            self.net.load_state_dict(
                torch.load(
                    model_path,
                    map_location=torch.device(self.device),
                    weights_only=False
                )
            )
        self.net.eval()
        self.net = torch.compile(self.net)
        self.half_precision = False

    @staticmethod
    def post_process(mask: np.ndarray) -> np.ndarray:
        """
        Morphs and blurs the mask to make it a bit better (generally speaking).
        :param mask: The mask to post-process.
        :return: The post-processed mask.
        """
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)
        return mask

    def remove(
            self,
            img: PILImage,
            size: Union[Tuple[int, int], None] = None,
            mask_only: bool = False
    ) -> np.ndarray:
        """
        Runs inferencing with post-processing.
        :param img: The image to be processed.
        :param size: Unused, for compatibility with other onnx code.
        :param mask_only: If True, it returns only the mask.
        :return: Either the mask (L or A) or the original image with the
        alpha channel applied (RGBA).
        """
        image_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)
        image_tensor = F.interpolate(
            torch.unsqueeze(image_tensor, 0), self.input_size, mode="bilinear").type(torch.float32)
        image_tensor = torch.divide(image_tensor, torch.max(image_tensor)).type(torch.float32)
        image_tensor = image_tensor.to(self.device)
        with (torch.no_grad(),
              torch.autocast(
                  device_type=self.device,
                  dtype=torch.float16,
                  enabled=self.half_precision)
              ):
            img_result = self.net(image_tensor)
        img_result = img_result[0][:, 0, :, :]
        result_array = img_result.cpu().data.numpy()

        # Norm the prediction
        re_max = np.max(result_array)
        re_min = np.min(result_array)
        if (re_max != re_min and (re_max != 1.0 and re_min != 0.0)) and re_min < 0.98:
            result_array = (result_array - re_min) / (re_max - re_min + 1E-8)
        result_array = np.squeeze(result_array)

        alpha_channel = np.uint8(result_array * 255)
        alpha_channel = cv2.resize(alpha_channel, img.size, interpolation=cv2.INTER_LANCZOS4)

        alpha_channel = self.post_process(alpha_channel)
        if mask_only:
            return alpha_channel
        return np.dstack((np.array(img), alpha_channel))

    def predict(
            self,
            img: PILImage,
            size: Union[Tuple[int, int], None] = None,
            do_sigmoid: bool = False,
            simple_norm: bool = True,
            convert_to: str = "RGB",
    ) -> np.ndarray:
        """
        Runs inferencing with post-processing.
        This is an alias of `remove` for compatibility with onnx code.
        :param img: The image to be processed.
        :param size: Unused, for compatibility with other onnx code.
        :param mask_only: If True, it returns only the mask.
        :return: Either the mask (L or A) or the original image with the
        alpha channel applied (RGBA).
        """
        return self.remove(
            img,
            size,
            mask_only = True
        )
