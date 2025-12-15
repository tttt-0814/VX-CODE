import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import torch

ROOT = Path(__file__).resolve().parent
DETECTRON2_ROOT = ROOT / "third_party" / "detectron2"
if DETECTRON2_ROOT.exists():
    sys.path.insert(0, str(DETECTRON2_ROOT))
sys.path.insert(0, str(ROOT))

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402  # type: ignore
from detectron2.config import CfgNode, get_cfg  # noqa: E402  # type: ignore
from detectron2.data import build_detection_test_loader  # noqa: E402  # type: ignore
from detectron2.modeling import build_model  # noqa: E402  # type: ignore
from detectron2.utils.logger import setup_logger  # noqa: E402  # type: ignore

from src.models.detr.d2.detr.config import add_detr_config  # noqa: E402
from src.models.detr.d2.detr.trainer import TrainerDETR  # noqa: E402
from src.utils.utils import get_target_instances_from_preds  # noqa: E402
from src.explanations.vxcode import VXCODE  # noqa: E402
from src.visualizations.visualize import vis_vxcode_results  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VXCODE on COCO val.")
    parser.add_argument(
        "--save_dir",
        default="results/vxcode",
        help="Save directory.",
    )
    parser.add_argument(
        "--config",
        default="configs/dev_faster_rcnn_R_50_FPN_1x.yaml",
        help="Config file path.",
    )
    parser.add_argument(
        "--weights",
        default="weights/faster_rcnn/model_final_b275ba.pkl",
        help="Model weights.",
    )
    parser.add_argument(
        "--model_name",
        default="faster_r_cnn",
        choices=["faster_r_cnn", "detr"],
        help="Model type to build.",
    )
    parser.add_argument(
        "--dataset",
        default="coco_2017_val",
        help="Dataset name registered in Detectron2.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Max batches to run for a quick check.",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.95,
        help="Score threshold to filter detections for reporting.",
    )
    parser.add_argument(
        "--vxcode_mode",
        type=str,
        default="del",
        choices=["del", "ins"],
        help="Mode to run VXCODE (del: deletion, ins: insertion).",
    )
    parser.add_argument(
        "--num_patches_per_step",
        type=int,
        default=1,
        help="Number of selected patches per step.",
    )
    return parser.parse_args()


def setup(args: argparse.Namespace) -> CfgNode:
    cfg = get_cfg()
    if args.model_name == "detr":
        add_detr_config(cfg)
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DATASETS.TEST = (args.dataset,)
    cfg.DATALOADER.NUM_WORKERS = max(cfg.DATALOADER.NUM_WORKERS, 2)
    cfg.freeze()
    return cfg


def build_model_for_cfg(cfg: CfgNode, model_name: str):
    if model_name == "detr":
        return TrainerDETR.build_model(cfg)
    return build_model(cfg)


def compute_heatmaps(
    cfg: CfgNode,
    model,
    max_batches: int,
    score_thresh: float,
    model_name: str,
    vxcode_mode: str,
    num_patches_per_step: int,
    save_dir: str,
) -> None:
    vxcode = VXCODE(
        model, model_name, mode=vxcode_mode, num_patches_per_step=num_patches_per_step
    )
    for dataset_name in cfg.DATASETS.TEST:
        loader = build_detection_test_loader(cfg, dataset_name)

        for idx, inputs in enumerate(tqdm(loader, total=max_batches)):
            if idx >= max_batches:
                break

            _, target_results, target_flag = get_target_instances_from_preds(
                model,
                inputs,
                score_thresh=score_thresh,
                model_name=model_name,
            )
            if target_flag.sum() == 0:
                continue

            num_detected = int(target_flag.sum().item())
            raw_file_name = inputs[0].get("file_name") or inputs[0].get("image_id")
            if isinstance(raw_file_name, str):
                file_name = Path(raw_file_name).name
            elif raw_file_name is not None:
                file_name = str(raw_file_name)
            else:
                file_name = f"idx{idx}"
            save_id = Path(file_name).stem
            print(f"Detected {num_detected} instances in {file_name}")

            explanation_results = vxcode(inputs, target_results, target_flag)
            (
                target_results,
                list_results_eval_heatmap,
                list_results_patch_id,
                list_results_rewards,
                list_results_patch_imgs,
                list_results_patch_size,
                list_results_padding_len,
            ) = explanation_results

            vis_vxcode_results(
                inputs[0]["image"].numpy(),
                target_results[0].pred_boxes.tensor.detach().to("cpu").numpy(),
                list_results_eval_heatmap,
                list_results_patch_size,
                Path(save_dir),
                save_id,
                bgr2rgb=False if model_name == "detr" else True,
            )


def main() -> None:
    args = parse_args()
    setup_logger()

    cfg = setup(args)
    model = build_model_for_cfg(cfg, args.model_name)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()

    compute_heatmaps(
        cfg,
        model,
        args.max_batches,
        args.score_thresh,
        args.model_name,
        args.vxcode_mode,
        args.num_patches_per_step,
        args.save_dir,
    )


if __name__ == "__main__":
    main()
