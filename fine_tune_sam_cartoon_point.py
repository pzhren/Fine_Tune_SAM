import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from transformers import pipeline
from tqdm import tqdm
from transformers import SamModel, SamConfig
import os
import shutil
from torch.utils.data import DataLoader
from transformers import SamProcessor
import cv2
import os
import numpy as np
from transformers import SamModel 
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
import logging
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
import argparse

# 配置参数
num_epochs = 50
num_report = 1
n_points = 1
bench_size = 6
test_batch_size = 4
lr = 1e-5
min_mask_ratio = 0.07
# dataset_version = "v1"
dataset_version = "v2"
data_path = "/home/daxin/renpengzhen/data"
dataset_root = f"{data_path}/datasets/hundunjianshen"
save_weight = f"{data_path}/train/fine_tune_sam/weights"
save_log = f"{data_path}/train/fine_tune_sam/logs"
masks_dir = f"{dataset_root}/{dataset_version}/mask"
images_dir = f"{dataset_root}/{dataset_version}/image"
model_type = f"{data_path}/weights/sam-vit-huge"

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=bench_size)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

def setup_distributed():
    """初始化分布式训练环境"""
    try:
        # 检查是否在分布式环境中运行
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            
            # 设置当前 GPU 设备
            torch.cuda.set_device(local_rank)
            
            # 初始化分布式进程组
            dist.init_process_group(
                backend="nccl",  # 使用 NCCL 后端（适用于多 GPU）
                init_method="env://",  # 从环境变量初始化
                world_size=world_size,
                rank=rank
            )
        else:
            # 单机单卡模式
            rank = 0
            world_size = 1
            local_rank = 0
            torch.cuda.set_device(local_rank)
        
        # 打印分布式信息
        print(f"Initialized distributed environment: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank
    
    except Exception as e:
        print(f"Failed to initialize distributed environment: {str(e)}")
        raise  # 重新抛出异常以便调试

def setup_logger(log_dir, rank):
    """设置日志记录器"""
    if rank != 0:
        return None
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_model_hf_format(model, save_weight, rank):
    """保存模型（只在主进程中执行）"""
    if rank != 0:
        return
        
    os.makedirs(save_weight, exist_ok=True)
    model.save_pretrained(save_weight)
    shutil.copy(f"{model_type}/preprocessor_config.json", 
                f"{save_weight}/preprocessor_config.json")
    print(f"Model saved to {save_weight}")

# Your existing dataset and utility functions here...
# (保持 SAMDataset, load_masks, _calculate_iou, points_based_calculate_average_iou, 
# mask_based_calculate_average_iou, evalution 等函数不变)

class SAMDataset(Dataset):
    def __init__(self, dataset, processor, is_train=True):
        self.dataset = dataset
        self.processor = processor
        self.is_train = is_train
        # 如果是测试模式，为每个样本预先生成固定的随机点
        if not self.is_train:
            self.fixed_points = {}
            for idx in range(len(dataset)):
                mask = dataset[idx]["gt_mask"]
                y_indices, x_indices = np.where(mask)
                if len(y_indices) >= n_points:
                    # 使用固定的随机种子
                    rng = np.random.RandomState(42 + idx)  # 每个样本使用不同但固定的种子
                    sampled_indices = rng.choice(len(y_indices), n_points, replace=False)
                    self.fixed_points[idx] = np.stack(
                        [[x_indices[i], y_indices[i]] for i in sampled_indices]
                    )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        mask = item["gt_mask"]
        
        # 根据模式选择采样点的方式
        if self.is_train:
            # 训练模式：随机采样点
            y_indices, x_indices = np.where(mask)
            sampled_indices = np.random.choice(len(y_indices), n_points, replace=False)
            prompt = np.stack([[x_indices[i], y_indices[i]] for i in sampled_indices])
        else:
            # 测试模式：使用预先生成的固定点
            prompt = self.fixed_points[idx]

        ground_truth_mask = np.array(item["gt_mask"].astype(np.int8))
        image = np.array(Image.open(item['image']).convert("RGB"))
        
        # 准备模型输入
        inputs = self.processor(image, input_points=[[prompt]], return_tensors="pt")
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        inputs["ground_truth_mask"] = cv2.resize(ground_truth_mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        
        #####################可视化测试训练集数据可视化
        # 可视化采样点和掩码
        if self.is_train:
            save_dir = f"{save_weight}/visualization/train"
        else:
            save_dir = f"{save_weight}/visualization/test"
        os.makedirs(save_dir, exist_ok=True)

        # 绘制采样点
        image_with_points = image.copy()
        for point in prompt:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image_with_points, (x, y), radius=5, color=(0, 255, 0), thickness=-1)  # 绿色点

        # 保存图像和掩码
        base_name = os.path.basename(item['image']).split('.')[0]
        cv2.imwrite(f"{save_dir}/{base_name}_points.png", cv2.cvtColor(image_with_points, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{save_dir}/{base_name}_mask.png", ground_truth_mask * 255)  # 将掩码保存为二值图像

        return inputs

def load_masks(im_path):
    # 加载mask图像输出[h,w,n_mask]
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # NOTE: 若背景为黑色，修改为白色
    im[np.all(im == [0, 0, 0], axis=-1)] = [255, 255, 255]

    # 获取唯一的颜色（掩膜）
    unique_colors = np.unique(im.reshape(-1, im.shape[2]), axis=0)
    # print(unique_colors)
    # 创建一个字典来存储每个颜色的掩膜
    masks = []
    # 遍历每个唯一颜色
    for color in unique_colors:
        # 创建一个掩膜，只有当前颜色的区域为 True
        if not np.array_equal(color, [255, 255, 255]):
            mask = np.all(im == color, axis=-1)
            masks.append(mask.astype(bool))
    masks = np.dstack(masks)
    return masks

def _calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0

def points_based_calculate_average_iou(pred_masks, gt_masks, n_samples=10):
  # 随机采点计算masks的平均iou精度:在每一个GT mask上进行采点计算对应的iou，最后求平均
    num_masks = gt_masks.shape[2] 
    average_iou = 0
    for i in range(num_masks):
        gt_mask = gt_masks[:, :, i]
        # 随机采样 n 个点
        y_indices, x_indices = np.where(gt_mask)

        np.random.seed(0)
        n_samples = min(n_samples, len(y_indices))
        sampled_indices = np.random.choice(len(y_indices), n_samples, replace=False)

        for idx in sampled_indices:
            y, x = y_indices[idx], x_indices[idx]
            # 在预测掩膜中找到对应位置的掩膜
            _find_mask = np.where(pred_masks[y, x, :])
            _find_mask = np.squeeze(_find_mask)
            # 计算 IoU
            if _find_mask.size==1:
                pred_mask_at_xy = pred_masks[:, :, _find_mask]
                iou = _calculate_iou(
                    gt_mask, pred_mask_at_xy
                ) 
            elif _find_mask.size>1:
                iou = 0
                for idx_j in _find_mask:
                    pred_mask_at_xy = pred_masks[:, :, idx_j]
                    iou += _calculate_iou(
                        gt_mask, pred_mask_at_xy
                    ) 
                iou = iou/_find_mask.size
            else:  # 无预测mask
                iou = 0
            average_iou += iou

    average_iou /= num_masks * n_samples  # 计算平均 IoU
    return average_iou


def mask_based_calculate_average_iou(prediction_masks, ground_truth_masks):
    """
    计算预测掩膜和真实掩膜的平均 IoU
    :param prediction_masks: 预测掩膜数组
    :param ground_truth_masks: 真实掩膜数组
    :return: 平均 IoU
    """
    num_pred_masks = prediction_masks.shape[-1]  # 预测掩膜数量
    num_gt_masks = ground_truth_masks.shape[-1]  # 真实掩膜数量

    # 创建一个矩阵保存每个预测掩膜与真实掩膜的 IoU
    iou_matrix = np.zeros((num_pred_masks, num_gt_masks))

    # 计算每对掩膜的 IoU
    for i in range(num_pred_masks):
        for j in range(num_gt_masks):
            iou_matrix[i, j] = _calculate_iou(prediction_masks[:, :, i], ground_truth_masks[:, :, j])

    # 找到最佳匹配（最大值匹配）
    matched_ious = []
    for _ in range(min(num_pred_masks, num_gt_masks)):
        max_iou_index = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)  # 找到 IoU 最大值的位置
        matched_ious.append(iou_matrix[max_iou_index])  # 保存最大 IoU
        iou_matrix[max_iou_index[0], :] = -1  # 将已匹配的行设为 -1（防止重复匹配）
        iou_matrix[:, max_iou_index[1]] = -1  # 将已匹配的列设为 -1

    # 计算平均 IoU
    average_iou = np.mean(matched_ious) if matched_ious else 0
    return average_iou

def preprocess_dataset(rank, dataset_split, save_dir, logger, min_mask_ratio=0.05):
    # 根据 rank 划分数据
    world_size = dist.get_world_size()
    per_rank_size = len(dataset_split) // world_size
    start_idx = rank * per_rank_size
    end_idx = start_idx + per_rank_size if rank != world_size - 1 else len(dataset_split)
    rank_split = dataset_split[start_idx:end_idx]

    # 每个进程处理自己的部分数据
    _dataset = []
    total_samples = 0  # 总样本数
    filtered_samples = 0  # 过滤掉的样本数
    remaining_samples = 0  # 剩余的样本数

    if rank == 0:
        logger.info(f"Building {save_dir} dataset across {world_size} GPUs...")
    
    for path in tqdm(rank_split, disable=rank != 0):
        try:
            masks = load_masks(path.replace('image', 'mask'))
            n_masks = masks.shape[2]
            total_samples += n_masks  # 统计总样本数

            for i in range(n_masks):
                mask = masks[:, :, i]
                y_indices, _ = np.where(mask)
                
                # 计算 mask 的面积占原图面积的比例
                mask_area = len(y_indices)  # mask 中非零像素的数量
                image_area = mask.shape[0] * mask.shape[1]  # 原图的面积
                mask_ratio = mask_area / image_area  # mask 面积占比
                
                # 设置阈值，过滤掉较小的 mask
                if len(y_indices) < n_points or mask_ratio < min_mask_ratio:
                    filtered_samples += 1  # 统计过滤掉的样本数
                    continue
                
                item = {
                    'image': path,
                    'gt_mask': mask,
                }
                _dataset.append(item)
                remaining_samples += 1  # 统计剩余的样本数

        except Exception as e:
            if rank == 0:
                logger.error(f"Error processing {path}: {str(e)}")
            continue

    # 收集所有进程的统计信息
    all_total_samples = [0] * world_size
    all_filtered_samples = [0] * world_size
    all_remaining_samples = [0] * world_size
    dist.all_gather_object(all_total_samples, total_samples)
    dist.all_gather_object(all_filtered_samples, filtered_samples)
    dist.all_gather_object(all_remaining_samples, remaining_samples)

    # 收集所有进程的数据
    all_datasets = [None] * world_size
    dist.all_gather_object(all_datasets, _dataset)
    
    # 只在主进程中合并和保存数据
    if rank == 0:
        combined_dataset = []
        for dataset in all_datasets:
            if isinstance(dataset, list):
                combined_dataset.extend(dataset)
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(combined_dataset, save_dir)

        # 计算所有卡上的总数
        total_samples_all_ranks = sum(all_total_samples)
        filtered_samples_all_ranks = sum(all_filtered_samples)
        remaining_samples_all_ranks = sum(all_remaining_samples)

        # 打印统计信息
        logger.info(f"Total samples (all ranks): {total_samples_all_ranks}")
        logger.info(f"Filtered samples (all ranks): {filtered_samples_all_ranks}")
        logger.info(f"Remaining samples (all ranks): {remaining_samples_all_ranks}")
        logger.info(f"Saved {len(combined_dataset)} samples to {save_dir}")
    
    # 等待主进程保存完成
    dist.barrier()
    return save_dir


def evalution_everything(test_dataset, weight_dir, rank, logger):
    # 只在主进程上运行评估
    if rank != 0:
        return
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    
    generator = pipeline(model=weight_dir, task="mask-generation", device=0)  # 明确指定设备
    points_miou = []
    masks_miou = []
    
    for image_path in tqdm(test_dataset):
        try:
            pred_masks = generator(image_path, points_per_batch=32)['masks']  # 减小 batch size
            if len(pred_masks)>=1:
                pred_masks = np.transpose(np.stack(pred_masks), (1, 2, 0))
            else:
                continue
            gt_masks = load_masks(image_path.replace('image','mask'))
            points_miou.append(points_based_calculate_average_iou(pred_masks, gt_masks))
            masks_miou.append(mask_based_calculate_average_iou(pred_masks, gt_masks))
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            continue
        
    logger.info(f"points_miou:{np.mean(points_miou)}")
    logger.info(f"masks_miou:{np.mean(masks_miou)}")



def evalution_points(test_datasets, weight_dir, rank, world_size, logger, batch_size=32):
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    # 初始化模型和处理器
    processor = SamProcessor.from_pretrained(weight_dir)
    model = SamModel.from_pretrained(weight_dir)
    model.to(rank)  # 将模型放到当前 GPU
    model = DDP(model, device_ids=[rank])  # 使用 DDP 包装模型
    model.eval()
    
    # 创建分布式采样器
    test_sampler = DistributedSampler(test_datasets, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_datasets,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True,
        shuffle=False
    )
    
    points_miou = []
    with torch.no_grad():
        for batch in tqdm(test_loader, disable=rank != 0):  # 只在主进程显示进度条
            try:
                # 将输入移到当前 GPU
                inputs = {k: v.to(rank) for k, v in batch.items()}
                
                # 使用模型进行预测
                outputs = model(**inputs, multimask_output=False)
                
                # 处理预测结果
                pred_masks = processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )
                
                # 批量计算 IoU
                batch_ious = []
                for i in range(len(pred_masks)):
                    # 获取预测 mask 并转换为 numpy
                    pred_mask = pred_masks[i][0].squeeze().cpu().numpy() > 0.5
                    
                    # 将 ground truth mask 还原到原始大小
                    original_h, original_w = inputs["original_sizes"][i].tolist()
                    gt_mask = batch["ground_truth_mask"][i].numpy()
                    gt_mask = cv2.resize(
                        gt_mask.astype(np.uint8), 
                        (original_w, original_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    # 确保维度匹配
                    assert pred_mask.shape == gt_mask.shape, \
                        f"Shape mismatch: pred_mask {pred_mask.shape} vs gt_mask {gt_mask.shape}"
                    
                    iou = _calculate_iou(pred_mask, gt_mask)
                    batch_ious.append(iou)
                
                points_miou.extend(batch_ious)
                
            except Exception as e:
                logger.error(f"Error in evaluation: {str(e)}")
                continue
    
    # 收集所有 GPU 的结果
    all_points_miou = [None] * world_size
    dist.all_gather_object(all_points_miou, points_miou)
    
    # 只在主进程计算平均 IoU
    if rank == 0:
        # 合并所有 GPU 的结果
        combined_miou = []
        for miou in all_points_miou:
            combined_miou.extend(miou)
        
        avg_points_miou = np.mean(combined_miou) if combined_miou else 0
        logger.info(f"points_miou:{avg_points_miou}")
        return avg_points_miou
    else:
        return 0
    

def train_before_loss(test_datasets, model, rank, world_size, logger, seg_loss, test_batch_size):
    """在训练前计算初始损失（支持多卡）"""
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    num_samples = 0

    # 创建分布式采样器
    test_sampler = DistributedSampler(test_datasets, num_replicas=world_size, rank=rank, shuffle=False)

    # 创建数据加载器
    test_loader = DataLoader(
        test_datasets,
        batch_size=test_batch_size,  # 可以根据需要调整
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )

    with torch.no_grad():
        for batch in test_loader:
            # 将数据移动到当前 GPU
            inputs = {k: v.to(rank) for k, v in batch.items()}

            # 模型预测
            outputs = model(
                pixel_values=inputs["pixel_values"],
                input_points=inputs["input_points"],
                multimask_output=False
            )

            # 计算损失
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = inputs["ground_truth_mask"].float()
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # 累加损失
            total_loss += loss.item() * len(inputs["pixel_values"])
            num_samples += len(inputs["pixel_values"])

    # 汇总所有 GPU 的损失
    total_loss_tensor = torch.tensor(total_loss).to(rank)
    num_samples_tensor = torch.tensor(num_samples).to(rank)

    # 使用 dist.all_reduce 汇总损失和样本数
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)

    # 计算全局平均损失
    avg_loss = total_loss_tensor.item() / num_samples_tensor.item() if num_samples_tensor.item() > 0 else 0

    # 记录日志
    if rank == 0:
        logger.info(f"Initial loss before training (global): {avg_loss:.4f}")

    return avg_loss
    


def train_epoch(model, train_loader, optimizer, seg_loss, device, epoch, logger, rank):
    """训练一个 epoch"""
    model.train()
    epoch_losses = []
    
    if rank == 0:
        train_loader = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch in train_loader:
        outputs = model(
            pixel_values=batch["pixel_values"].to(device),
            input_points=batch["input_points"].to(device),
            multimask_output=False
        )
        
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())
    
    # 计算所有进程的平均损失
    avg_loss = torch.tensor(mean(epoch_losses)).to(device)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
    avg_loss = avg_loss.item() / dist.get_world_size()
    
    if rank == 0 and logger:
        logger.info(f'Epoch {epoch}, Mean loss: {avg_loss:.4f}')
    
    return avg_loss

def main():
    args = setup_args()
    rank, world_size, local_rank = setup_distributed()
    logger = setup_logger(save_log, rank)
    
    if rank == 0:  # 只在主进程中记录
        logger.info(f"Parameters:")
        logger.info(f"num_epochs: {num_epochs}")
        logger.info(f"bench_size: {bench_size}")
        logger.info(f"dataset_version: {dataset_version}")
        logger.info(f"lr: {lr}")
        logger.info(f"min_mask_ratio: {min_mask_ratio}")

    # 准备数据集
    all_img = [os.path.join(images_dir, name) for name in sorted(os.listdir(images_dir))]
    train_img_split = []
    test_img_split = []
    for img in all_img:
        # if "dierji_1_" in img or "dierji_2_" in img:
        if "dierji_1_" in img:
            test_img_split.append(img)
        else:
            train_img_split.append(img)

    # 构建训练数据集
    train_dataset_path = f'{save_weight}/train_dataset.pt'
    test_dataset_path = f'{save_weight}/test_dataset.pt'
    # if not os.path.exists(train_dataset_path):
    train_dataset_path = preprocess_dataset(rank, train_img_split, train_dataset_path, logger, min_mask_ratio=min_mask_ratio)
    # if not os.path.exists(test_dataset_path):
    test_dataset_path = preprocess_dataset(rank, test_img_split, test_dataset_path, logger, min_mask_ratio=min_mask_ratio)
    train_dataset = torch.load(train_dataset_path)
    test_dataset = torch.load(test_dataset_path)


    # 记录数据集大小
    if rank == 0:  # 只在主进程中记录
        logger.info(f"Training dataset size: {len(train_dataset)} samples")
        logger.info(f"Testing dataset size: {len(test_dataset)} samples")

    # 创建数据加载器
    processor = SamProcessor.from_pretrained(model_type)
    train_datasets = SAMDataset(dataset=train_dataset, processor=processor, is_train=True)
    test_datasets = SAMDataset(dataset=test_dataset, processor=processor, is_train=False)
    sampler = DistributedSampler(train_datasets, 
                                num_replicas=world_size,
                                rank=rank,
                                shuffle=True)
    
    train_loader = DataLoader(
        train_datasets,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,  # 保持worker进程存活
        prefetch_factor=2,       # 预加载因子
        drop_last=True           # 丢弃不完整的最后一个batch
    )
    
    # 创建模型
    model = SamModel.from_pretrained(model_type)
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    optimizer = Adam(model.module.mask_decoder.parameters(), lr=lr, weight_decay=0)
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    # 替换原来的损失函数
    # seg_loss = IoULoss(smooth=1e-6)
    
    # 训练前评估
    if rank == 0:
        logger.info("Evaluating before training...")
    dist.barrier()  # 确保所有进程同步
    evalution_points(test_datasets, model_type, rank, world_size, logger, batch_size=test_batch_size)
    dist.barrier()  # 确保所有进程同步
    train_before_loss(test_datasets, model, rank, world_size, logger, seg_loss, test_batch_size)
    

    if rank == 0:
        logger.info(f"Starting training with {world_size} GPUs")
    # 训练循环
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        train_epoch(
            model, train_loader, optimizer, seg_loss, 
            local_rank, epoch, logger, rank
        )
        
        # 定期评估
        if (epoch+1) % num_report == 0 or epoch == (num_epochs-1):
            weight_dir = f"{save_weight}/epoch_{epoch}"
            if rank == 0:
                save_model_hf_format(model.module, weight_dir, rank)
            # 确保所有进程等待模型保存完成
            dist.barrier()
            evalution_points(test_datasets, weight_dir, rank, world_size, logger, batch_size=test_batch_size)
            dist.barrier()
    
    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()


"""
torchrun --nproc_per_node=8 fine_tune_sam_cartoon.py
"""