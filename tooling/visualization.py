import argparse
import logging
import random
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import distinctipy
import matplotlib.pyplot as plt
import numpy as np
from natsort import os_sorted
from tqdm import tqdm
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))

from utils.image_utils import load_image_array_from_path, save_image_array_to_path
from utils.input_utils import SUPPORTED_IMAGE_FORMATS, get_file_paths
from utils.logging_utils import get_logger_name

logger = logging.getLogger(get_logger_name())


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualization of prediction/GT of model")

    yolo_args = parser.add_argument_group("YOLO")
    yolo_args.add_argument("--yolo_model", type=str, required=True, help="Path to YOLO model")

    io_args = parser.add_argument_group("IO")
    # io_args.add_argument("-t", "--train", help="Train input folder/file",
    #                         nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, default=None)
    io_args.add_argument("-o", "--output", help="Output folder", type=str)

    parser.add_argument("--sorted", action="store_true", help="Sorted iteration")
    parser.add_argument("--save", nargs="?", const="all", default=None, help="Save images instead of displaying")

    args = parser.parse_args()

    return args


_keypress_result = None


def keypress(event):
    global _keypress_result
    # print('press', event.key)
    if event.key in ["q", "escape"]:
        sys.exit()
    if event.key in [" ", "right"]:
        _keypress_result = "forward"
        return
    if event.key in ["backspace", "left"]:
        _keypress_result = "back"
        return
    if event.key in ["e", "delete"]:
        _keypress_result = "delete"
        return
    if event.key in ["w"]:
        _keypress_result = "bad"
        return


def on_close(event):
    sys.exit()


def main(args) -> None:
    """
    Currently running the validation set and showing the ground truth and the prediction side by side

    Args:
        args (argparse.Namespace): arguments for where to find the images
    """
    if args.save and not args.output:
        raise ValueError("Cannot run saving when there is not save location given (--output)")

    image_paths = get_file_paths(args.input, SUPPORTED_IMAGE_FORMATS)

    model = YOLO(args.yolo_model)

    @lru_cache(maxsize=10)
    def load_image(path):
        data = load_image_array_from_path(path, mode="color")
        if data is None:
            raise TypeError(f"Image {path} is None, loading failed")

        image = data["image"]
        dpi = data["dpi"]
        return image, dpi

    @lru_cache(maxsize=10)
    def create_gt_visualization(image_path):
        image, dpi = load_image(image_path)

        return

    @lru_cache(maxsize=10)
    def create_pred_visualization(image_path):
        image, dpi = load_image(image_path)

        height, width = image.shape[:2]

        output = model(image)

        yolo_output = output[0]

        names = yolo_output.names

        distinct_colors = distinctipy.get_colors(len(names), rng=0)  # no rng should give the same colors
        distinct_colors = [tuple(int(channel * 255) for channel in color) for color in distinct_colors]

        if yolo_output.masks is not None:
            relative_contours = [yolo_output.masks.xyn[i].cpu().numpy() for i in range(yolo_output.masks.shape[0])]
        else:
            relative_bboxes = [yolo_output.boxes.xyxyn[i].cpu().numpy() for i in range(yolo_output.boxes.shape[0])]
            relative_contours = [
                np.array(
                    [
                        [relative_bbox[0], relative_bbox[1]],
                        [relative_bbox[2], relative_bbox[1]],
                        [relative_bbox[2], relative_bbox[3]],
                        [relative_bbox[0], relative_bbox[3]],
                    ]
                )
                for relative_bbox in relative_bboxes
            ]

        for i in range(len(relative_contours)):
            absolute_contour = relative_contours[i] * np.asarray([width, height])

            class_id = int(yolo_output.boxes.cls[i].cpu().numpy())

            color = distinct_colors[class_id]

            cv2.polylines(
                image,
                [absolute_contour.astype(np.int32)],
                isClosed=True,
                color=color,
                thickness=1,
            )

        return image

    # for i, inputs in enumerate(np.random.choice(val_loader, 3)):
    if args.sorted:
        loader = os_sorted(image_paths)
    else:
        loader = image_paths
        random.shuffle(image_paths)

    bad_results = np.zeros(len(loader), dtype=bool)
    delete_results = np.zeros(len(loader), dtype=bool)

    if args.save:
        for image_path in tqdm(image_paths, desc="Saving Images"):
            vis_gt = None
            vis_pred = None
            if args.save not in ["all", "both", "pred", "gt"]:
                raise ValueError(f"{args.save} is not a valid save mode")
            if args.save != "pred":
                vis_gt = create_gt_visualization(image_path)
            if args.save != "gt":
                vis_pred = create_pred_visualization(image_path)

            output_dir = Path(args.output)
            if not output_dir.is_dir():
                logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
                output_dir.mkdir(parents=True)

            if args.save in ["all", "both"]:
                save_path = output_dir.joinpath(image_path.stem + "_both.jpg")
                if vis_gt is not None and vis_pred is not None:
                    vis_gt = cv2.resize(vis_gt, (vis_pred.shape[1], vis_pred.shape[0]), interpolation=cv2.INTER_CUBIC)
                    save_image_array_to_path(save_path, np.hstack((vis_pred, vis_gt)))
            if args.save in ["all", "pred"]:
                if vis_pred is not None:
                    save_path = output_dir.joinpath(image_path.stem + "_pred.jpg")
                    save_image_array_to_path(save_path, vis_pred)
            if args.save in ["all", "gt"]:
                if vis_gt is not None:
                    save_path = output_dir.joinpath(image_path.stem + "_gt.jpg")
                    save_image_array_to_path(save_path, vis_gt)

    else:
        fig, axes = plt.subplots(1, 2)
        fig.tight_layout()
        fig.canvas.mpl_connect("key_press_event", keypress)
        fig.canvas.mpl_connect("close_event", on_close)
        axes[0].axis("off")
        axes[1].axis("off")
        fig_manager = plt.get_current_fig_manager()
        if fig_manager is None:
            raise ValueError("Could not find figure manager")
        fig_manager.window.showMaximized()

        i = 0
        while 0 <= i < len(loader):
            image_path = loader[i]

            vis_gt = create_gt_visualization(image_path)
            vis_pred = create_pred_visualization(image_path)

            # pano_gt = torch.IntTensor(rgb2id(cv2.imread(inputs["pan_seg_file_name"], cv2.IMREAD_COLOR)))
            # print(inputs["segments_info"])

            # vis_im = vis_im.draw_panoptic_seg(outputs["panoptic_seg"][0], outputs["panoptic_seg"][1])
            # vis_im_gt = vis_im_gt.draw_panoptic_seg(pano_gt, [item | {"isthing": True} for item in inputs["segments_info"]])

            fig_manager.window.setWindowTitle(str(image_path))

            # HACK Just remove the previous axes, I can't find how to resize the image otherwise
            axes[0].clear()
            axes[1].clear()
            axes[0].axis("off")
            axes[1].axis("off")

            if vis_pred is not None:
                axes[0].imshow(vis_pred)
            if vis_gt is not None:
                axes[1].imshow(vis_gt)

            if delete_results[i]:
                fig.suptitle("Delete")
            elif bad_results[i]:
                fig.suptitle("Bad")
            else:
                fig.suptitle("")
            # f.title(inputs["file_name"])
            global _keypress_result
            _keypress_result = None
            fig.canvas.draw()
            while _keypress_result is None:
                plt.waitforbuttonpress()
            if _keypress_result == "delete":
                # print(i+1, f"{inputs['original_file_name']}: DELETE")
                delete_results[i] = not delete_results[i]
                bad_results[i] = False
            elif _keypress_result == "bad":
                # print(i+1, f"{inputs['original_file_name']}: BAD")
                bad_results[i] = not bad_results[i]
                delete_results[i] = False
            elif _keypress_result == "forward":
                # print(i+1, f"{inputs['original_file_name']}")
                i += 1
            elif _keypress_result == "back":
                # print(i+1, f"{inputs['original_file_name']}: DELETE")
                i -= 1

    if args.output and (delete_results.any() or bad_results.any()):
        output_dir = Path(args.output)
        if not output_dir.is_dir():
            logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)
        if delete_results.any():
            output_delete = output_dir.joinpath("delete.txt")
            with output_delete.open(mode="w") as f:
                for i in delete_results.nonzero()[0]:
                    path = Path(loader[i])
                    line = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve()
                    f.write(f"{line}\n")
        if bad_results.any():
            output_bad = output_dir.joinpath("bad.txt")
            with output_bad.open(mode="w") as f:
                for i in bad_results.nonzero()[0]:
                    path = Path(loader[i])
                    line = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve()
                    f.write(f"{line}\n")

        remaining_results = np.logical_not(np.logical_or(bad_results, delete_results))
        if remaining_results.any():
            output_remaining = output_dir.joinpath("correct.txt")
            with output_remaining.open(mode="w") as f:
                for i in remaining_results.nonzero()[0]:
                    path = Path(loader[i])
                    line = path.relative_to(output_dir) if path.is_relative_to(output_dir) else path.resolve()
                    f.write(f"{line}\n")


if __name__ == "__main__":
    args = get_arguments()
    main(args)
