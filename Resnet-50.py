import numpy as np
import torch
import nrrd
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models
from torchvision.transforms import Compose, Resize ,CenterCrop,ToTensor ,Normalize
import SimpleITK as sitk
import tqdm

device = 'cuda' if torch.cuda.is_available else 'cpu'
print(device)

# 主文件夹路径（包含776个子文件夹）

input_folder =r"Inputdata" # orginal data

processed_folder = r"Processed_output"  # processed data

output_file = r"results.json"  # results


# 函数：加载并转换nrrd文件（将4D转换为3D）

def load_and_convert_nrrd(file_path):
    try:
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)  # 获取为numpy数组

        # 4维转换为3维, remove the last dimension
        if len(image_array.shape) == 4:
            image_array = image_array[ :, :, :, 0]
            print(f"Converted 4D to 3D for {file_path}")

        # 将nan或inf值替换为0
        image_array = np.nan_to_num(image_array)

        # 归一化图像
        scaler = MinMaxScaler()
        # 3D按每个切片进行归一化
        if len(image_array.shape) == 3:
            H, W, D = image_array.shape
            image_array_reshaped = image_array.reshape(-1, D)
            image_array_scaled = scaler.fit_transform(image_array_reshaped)
            image_array = image_array_scaled.reshape(H, W, D)
        elif len(image_array.shape) == 2:
            # 2D直接归一化
            image_array = scaler.fit_transform(image_array)

        return image_array, image  # 返回处理后的数组和原始图像
    except Exception as e:
        print(f"Error reading or processing {file_path}: {e}")
        return None, None


def save_nrrd(image_array, output_path, original_image):
    try:
        # 将处理后的图像数据转换回nrrd格式
        if len(image_array.shape) == 3:
            # SimpleITK expects [D, H, W]
            image_to_save = sitk.GetImageFromArray(image_array)
        elif len(image_array.shape) == 2:
            image_to_save = sitk.GetImageFromArray(image_array)
        else:
            print(f"Unsupported image shape {image_array.shape} for saving: {output_path}")
            return

        image_to_save.CopyInformation(original_image)

        sitk.WriteImage(image_to_save, output_path)
        print(f"Saved processed file to {output_path}")
    except Exception as e:
        print(f"Error saving {output_path}: {e}")


class ProcessedNRRDDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            # 读取nrrd文件
            image = sitk.ReadImage(file_path)
            image_array = sitk.GetArrayFromImage(image)

            # 处理缺失数据: 将nan或inf值替换为0
            image_array = np.nan_to_num(image_array)

            # 如果是2D图像，扩展为3通道
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            # 如果是3D图像，但没有3个通道，则扩展通道维度
            elif len(image_array.shape) == 3 and image_array.shape[0] != 3:
                # 假设需要3个通道，复制第一个维度
                image_array = np.stack([image_array] * 3, axis=0)

            # 转换为PyTorch张量
            image_tensor = torch.tensor(image_array, dtype=torch.float32)
            image_tensor = image_tensor.permute(2, 0, 1)  # 重新排列为(C, H, W)

            if self.transform:
                image_tensor = self.transform(image_tensor)

            return image_tensor, file_path
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, file_path


# DataLoader
def create_dataloader(file_paths, batch_size=16, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为ResNet输入的224x224
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # ImageNet的均值和标准差归一化
        transforms.ToTensor()
    ])

    dataset = ProcessedNRRDDataset(file_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader


# laod the model resnet50
def load_pretrained_resnet50():
    model = resnet50(pretrained=True)
    model.eval()
    return model

def inference(model, dataloader, device):
    model = model.to(device)
    results = []

    with torch.no_grad():
        for images, file_paths in tqdm(dataloader, desc="Running Inference"):
            if images is None:
                continue

            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            results.extend(zip(file_paths, preds.cpu().numpy()))

    return results


# save the results to a json file
def save_results(results, output_file):
    results_dict = {file_path: int(pred) for file_path, pred in results}
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {output_file}")

# 预处理所有nrrd文件
def preprocess(main_input_folder, processed_output_folder):
    # 获取所有子文件夹
    # join 拼接路径
    subfolders = [f for f in os.listdir(main_input_folder) if os.path.isdir(os.path.join(main_input_folder, f))]

    # 创建输出文件夹
    if not os.path.exists(processed_output_folder):
        os.makedirs(processed_output_folder)
        print(f"Created output folder: {processed_output_folder}")

    # 处理所有子文件夹中的nrrd
    print("Starting processing of nrrd files...")
    for subfolder in tqdm(subfolders, desc="Processing Subfolders"):
        input_subfolder_path = os.path.join(main_input_folder, subfolder)
        nrrd_files = [f for f in os.listdir(input_subfolder_path) if f.endswith('.nrrd')]
        for nrrd_file in nrrd_files:
            input_file_path = os.path.join(input_subfolder_path, nrrd_file)
            # 为避免文件名冲突，使用子文件夹名作为前缀
            output_file_name = f"{subfolder}_{nrrd_file}"
            output_file_path = os.path.join(processed_output_folder, output_file_name)

            image_array, original_image = load_and_convert_nrrd(input_file_path)
            if image_array is None:
                continue
            # save
            save_nrrd(image_array, output_file_path, original_image)

    print("Finished processing all nrrd files.")

def main(processed_output_folder, inference_output_file):
    # 获取处理后的nrrd
    processed_files = [os.path.join(processed_output_folder, f) for f in os.listdir(processed_output_folder) if f.endswith('.nrrd')]

    if len(processed_files) == 0:
        print("No processed nrrd files found. Exiting.")
        return

    print("Creating DataLoader for inference...")
    dataloader = create_dataloader(processed_files, batch_size=16, shuffle=False)

    print("Loading pre-trained ResNet-50 model...")
    model = load_pretrained_resnet50()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Starting inference...")
    results = inference(model, dataloader, device)

    save_results(results, inference_output_file)

    print("Finished.")

