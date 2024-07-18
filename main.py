import os
import random
import time
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm

from src.datasets import DatasetProvider, train_collate
from src.models.evflownet import EVFlowNet


class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    """
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    """
    epe = torch.mean(
        torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0
    )
    return epe


def compute_epe_error_from_flow_dict(
    pred_flow_dict: Dict[str, torch.Tensor], gt_flow: torch.Tensor
):
    """
    end-point-error (ground truthと予測値の二乗誤差)を計算
    pred_flow_dict: Dict[str, torch.Tensor] => 予測したオプティカルフローデータの辞書
    gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
    """
    epe = 0
    for key, pred_flow in pred_flow_dict.items():
        resized_gt_flow = torch.nn.functional.interpolate(
            gt_flow, size=pred_flow.shape[-2:], mode="bilinear", align_corners=False
        )
        epe += compute_epe_error(pred_flow, resized_gt_flow)
    return epe / len(pred_flow_dict)


def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    """
    optical flowをnpyファイルに保存
    flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
    file_name: str => ファイル名
    """
    np.save(f"{file_name}.npy", flow.cpu().numpy())


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
        ディレクトリ構造:

        data
        ├─test
        |  ├─test_city
        |  |    ├─events_left
        |  |    |   ├─events.h5
        |  |    |   └─rectify_map.h5
        |  |    └─forward_timestamps.txt
        └─train
            ├─zurich_city_11_a
            |    ├─events_left
            |    |       ├─ events.h5
            |    |       └─ rectify_map.h5
            |    ├─ flow_forward
            |    |       ├─ 000134.png
            |    |       |.....
            |    └─ forward_timestamps.txt
            ├─zurich_city_11_b
            └─zurich_city_11_c
        """

    # ------------------
    #    Dataloader
    # ------------------
    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4,
    )

    # train_val_set = loader.get_train_dataset()
    # train_index, valid_index = train_test_split(
    #     range(len(train_val_set)), test_size=0.3, random_state=42
    # )
    # train_set = Subset(train_val_set, train_index)
    # valid_set = Subset(train_val_set, valid_index)
    # test_set = loader.get_test_dataset()
    # collate_fn = train_collate
    # train_data = DataLoader(
    #     train_set,
    #     batch_size=args.data_loader.train.batch_size,
    #     shuffle=args.data_loader.train.shuffle,
    #     collate_fn=collate_fn,
    #     drop_last=False,
    # )
    # valid_data = DataLoader(
    #     valid_set,
    #     batch_size=args.data_loader.test.batch_size,
    #     shuffle=args.data_loader.test.shuffle,
    #     collate_fn=collate_fn,
    #     drop_last=False,
    # )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()
    collate_fn = train_collate
    train_data = DataLoader(
        train_set,
        batch_size=args.data_loader.train.batch_size,
        shuffle=args.data_loader.train.shuffle,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_data = DataLoader(
        test_set,
        batch_size=args.data_loader.test.batch_size,
        shuffle=args.data_loader.test.shuffle,
        collate_fn=collate_fn,
        drop_last=False,
    )

    """
    train data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
        Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
        Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない

    test data:
        Type of batch: Dict
        Key: seq_name, Type: list
        Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
    """
    # ------------------
    #       Model
    # ------------------
    model = EVFlowNet(args.train).to(device)

    # ------------------
    #   optimizer
    # ------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.train.initial_learning_rate,
        weight_decay=args.train.weight_decay,
    )
    # ------------------
    #   Start training
    # ------------------
    model.train()
    current_time = time.strftime("%Y%m%d%H%M%S")
    val_interval = 1
    for epoch in range(args.train.epochs):
        total_loss = 0
        print("on epoch: {}".format(epoch + 1))
        for i, batch in enumerate(tqdm(train_data)):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)  # [B, 4, 480, 640]
            ground_truth_flow = batch["flow_gt"].to(device)  # [B, 2, 480, 640]
            # flow = model(event_image)  # [B, 2, 480, 640]
            # loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
            flow_dict = model(event_image)
            loss: torch.Tensor = compute_epe_error_from_flow_dict(
                flow_dict, ground_truth_flow
            )
            print(f"batch {i} loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_data)}")

        # test
        model.eval()
        flow: torch.Tensor = torch.tensor([]).to(device)

        with torch.no_grad():
            print("start test")
            for batch in tqdm(test_data):
                batch: Dict[str, Any]
                event_image = batch["event_volume"].to(device)
                batch_flow = model(event_image)
                flow = torch.cat((flow, batch_flow), dim=0)
            print("test done")
        file_name = f"submission_{current_time}_epoch_{epoch+1}"
        save_optical_flow_to_npy(flow, file_name)

        # if (epoch + 1) % val_interval == 0:
        #     model.eval()
        #     total_loss = 0
        #     for i, batch in enumerate(tqdm(valid_data)):
        #         batch: Dict[str, Any]
        #         event_image = batch["event_volume"].to(device)
        #         ground_truth_flow = batch["flow_gt"].to(device)
        #         flow = model(event_image)
        #         loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
        #         total_loss += loss.item()
        #     model.train()
        #     print(f"Validation Loss: {total_loss / len(valid_data)}")

    # Create the directory if it doesn't exist
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # ------------------
    #   Start predicting
    # ------------------
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = batch["event_volume"].to(device)
            batch_flow = model(event_image)  # [1, 2, 480, 640]
            flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
        print("test done")
    # ------------------
    #  save submission
    # ------------------
    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)


if __name__ == "__main__":
    main()
