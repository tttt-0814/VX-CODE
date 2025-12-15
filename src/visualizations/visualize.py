from typing import List, Tuple
from pathlib import Path
import matplotlib.axes
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms.functional as F

DPI = 300


def voc_colormap(labels: int) -> np.array:
    """Color map used in PASCAL VOC
    Args:
        labels (iterable of ints): Class ids.
    Returns:
        numpy.ndarray: Colors in RGB order. The shape is :math:`(N, 3)`,
        where :math:`N` is the size of :obj:`labels`. The range of the values
        is :math:`[0, 255]`.
    """
    colors = []
    for label in labels:
        r, g, b = 0, 0, 0
        i = label
        for j in range(8):
            if i & (1 << 0):
                r |= 1 << (7 - j)
            if i & (1 << 1):
                g |= 1 << (7 - j)
            if i & (1 << 2):
                b |= 1 << (7 - j)
            i >>= 3
        colors.append((r, g, b))

    return np.array(colors, dtype=np.float32)


def vis_image(
    img: np.array, ax: matplotlib.axes.Axes = None, bgr2rgb: bool = True
) -> matplotlib.axes.Axes:
    """Visualize a color image.
    Args:
        img (~numpy.ndarray): See the table below.
            If this is :obj:`None`, no image is displayed.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    .. csv-table::
        :header: name, shape, dtype, format
        :obj:`img`, ":math:`(3, H, W)`", :obj:`float32`, \
        "RGB, :math:`[0, 255]`"
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    """
    from matplotlib import pyplot as plot

    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    if img is not None:
        # CHW -> HWC
        img = img.transpose((1, 2, 0))
        if bgr2rgb:
            img = img[:, :, [2, 1, 0]]
        ax.imshow(img.astype(np.uint8))

    return ax


def vis_bbox(
    img: np.array,
    bbox: np.array,
    label: np.array = None,
    score: np.array = None,
    label_names: List[str] = None,
    instance_colors: np.array = None,
    alpha: float = 1.0,
    linewidth: float = 3.0,
    sort_by_score: bool = True,
    ax: matplotlib.axes.Axes = None,
    bgr2rgb: bool = True,
) -> matplotlib.axes.Axes:
    from matplotlib import pyplot as plt

    if label is not None and not len(bbox) == len(label):
        raise ValueError("The length of label must be same as that of bbox")
    if score is not None and not len(bbox) == len(score):
        raise ValueError("The length of score must be same as that of bbox")

    if sort_by_score and score is not None:
        order = np.argsort(score)
        bbox = bbox[order]
        score = score[order]
        if label is not None:
            label = label[order]
        if instance_colors is not None:
            instance_colors = np.array(instance_colors)[order]

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax, bgr2rgb=bgr2rgb)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    if instance_colors is None:
        # Red
        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
        instance_colors[:, 0] = 255
    instance_colors = np.array(instance_colors)
    for i, bb in enumerate(bbox):
        xy = (bb[0], bb[1])
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
        color = instance_colors[i % len(instance_colors)] / 255
        ax.add_patch(
            plt.Rectangle(
                xy,
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                alpha=alpha,
            )
        )

        caption = []

        if label is not None and label_names is not None:
            lb = label[i]
            if not (0 <= lb < len(label_names)):
                raise ValueError("No corresponding name is given")
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append("{:.2f}".format(sc))

        if len(caption) > 0:
            ax.text(
                bb[0],
                bb[1],
                ": ".join(caption),
                style="italic",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
                fontsize=5,
            )

    return ax


def vis_cam(
    img: np.array,
    cam: np.array,
    ax: matplotlib.axes.Axes = None,
    alpha: float = 0.4,
    bgr2rgb: bool = True,
) -> matplotlib.axes.Axes:
    ax = vis_image(img, ax, bgr2rgb=bgr2rgb)
    cam = cam * 255 / cam.max()
    ax.imshow(cam, cmap="jet", alpha=alpha)

    return ax


def vis_vxcode_results(
    img: np.array,
    pred_boxes: np.array,
    results_eval_heatmap: List[torch.Tensor],
    results_patch_size: List[Tuple[int, int]],
    save_dir: Path,
    file_id: str,
    bgr2rgb: bool,
) -> None:
    save_prefix = Path(file_id).stem if file_id else "image"
    for i in range(len(pred_boxes)):
        pred_box_i = pred_boxes[i]
        result_eval_heatmap_i = results_eval_heatmap[i]
        patch_size_h_i, patch_size_w_i = results_patch_size[i]

        # Visualize detection result
        save_dir_detection = save_dir.joinpath("detection_results")
        save_dir_detection.mkdir(exist_ok=True, parents=True)
        save_name_detection = save_dir_detection.joinpath(
            f"{save_prefix}_{i}.png"
        )
        ax = vis_bbox(img, pred_boxes[i][None, ...], bgr2rgb=bgr2rgb)
        ax.axis("off")
        plt.savefig(save_name_detection, bbox_inches="tight", dpi=DPI)
        plt.close()

        # Visualize heat map
        save_dir_vis_heatmap = save_dir.joinpath("vis_heatmaps")
        save_dir_vis_heatmap.mkdir(exist_ok=True, parents=True)
        save_name_vis_heatmap = save_dir_vis_heatmap.joinpath(
            f"{save_prefix}_{i}.png"
        )
        vis_heatmap_i = F.resize(
            result_eval_heatmap_i[None, ...], size=(img.shape[1], img.shape[2])
        ).squeeze()

        kernel_w = patch_size_w_i + (1 - patch_size_w_i % 2)
        kernel_h = patch_size_h_i + (1 - patch_size_h_i % 2)
        blur = torchvision.transforms.GaussianBlur(
            (kernel_w, kernel_h), sigma=max(kernel_h, kernel_w) / 5
        )
        vis_heatmap_i = blur(vis_heatmap_i[None, ...]).numpy().squeeze()
        ax = vis_bbox(img, pred_box_i[None, ...], bgr2rgb=bgr2rgb)
        ax = vis_cam(img, vis_heatmap_i, ax=ax, bgr2rgb=bgr2rgb)
        ax.axis("off")
        plt.savefig(save_name_vis_heatmap, bbox_inches="tight", dpi=DPI)
        plt.close()
