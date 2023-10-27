import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import v2 as TT
from torchvision import transforms as T
import numpy as np
from PIL import Image
from tqdm.autonotebook import tqdm
from pathlib import Path, PurePath
import sys
sys.path.append(str(Path().absolute().parent))


class FixShape(TT.Transform):
    def __init__(self):
        """Forces input to have dimensions divisble by 32"""
        super().__init__()

    def __call__(self, img):
        M, N = img.shape[-2:]
        pad_M = (32 - M%32) % 32
        pad_N = (32 - N%32) % 32
        return TF.pad(img, padding=(0, 0, pad_N, pad_M)), (M, N)

    def __repr__(self):
        return self.__class__.__name__


def get_default_img_transform():
    """Tensor, dimension and normalisation default augs"""
    return T.Compose([T.ToTensor(), T.Normalize((0.1,), (0.2,)), FixShape()])
    

class ImgListDataset(Dataset):
    """Torch Dataset from img list"""
    def __init__(self, img_list):
        self.img_list = img_list
        if isinstance(img_list[0], (str, PurePath)):
            self.is_arr = False
        elif isinstance(img_list[0], np.ndarray):
            self.is_arr = True
        self.transform = get_default_img_transform()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if self.is_arr:
            img = (255*self.img_list[idx]/self.img_list[idx].max()).astype(np.uint8)
        else:
            img = Image.open(self.img_list[idx])
        img, shape = self.transform(img)
        return {'img': img, "crop":shape}


def get_img_list_dataloader(img_list, batch_size=16, num_workers=0, pin_memory=False):
    """Wrapper of Dataset into DataLoader"""
    dataset = ImgListDataset(img_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=pin_memory)
    return loader


class InferenceModel:
    def __init__(self, threshold=0.5):
        """Core inference class for DeepGPET"""
        
        self.transform = get_default_img_transform()
        self.threshold = threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weight_path = list(Path(sys.path[-1]).rglob("*.pth"))[0]
        self.model = torch.load(weight_path, map_location=self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_img(self, img, soft_pred=False):
        """Inference on a single image"""
        if isinstance(img, (str, PurePath)):
            # assume it's a path to an image
            img = Image.open(img)
        elif isinstance(img, np.ndarray):
            # assume it's a numpy array
            # NOTE: we assume that the image has not been normalized yet
            img = Image.fromarray(img)

        with torch.no_grad():
            img, (M, N) = self.transform(img)
            img = img.unsqueeze(0).to(self.device)
            pred = self.model(img).squeeze(0).sigmoid()
            if not soft_pred:
                pred = (pred > self.threshold).int()
            return pred.cpu().numpy()[0, :M, :N]

    def predict_list(self, img_list, soft_pred=False):
        """Inference on a list of images without batching"""
        preds = []
        with torch.no_grad():
            for img in tqdm(img_list):
                pred = self.predict_img(img, soft_pred=soft_pred)
                preds.append(pred)
        return preds

    def _predict_loader(self, loader, soft_pred=False):
        """Inference from a DataLoader"""
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc='Predicting', leave=False):
                img = batch['img'].to(self.device)
                batch_M, batch_N = batch['crop']
                pred = self.model(img).sigmoid().squeeze().cpu().numpy()
                if not soft_pred:
                    pred = (pred > self.threshold).astype(np.int64)
                pred = [p[:M,:N] for (p, M, N) in zip(pred, batch_M, batch_N)]
                preds.append(pred)
        return preds

    def batch_predict(self, img_list, soft_pred=False, batch_size=16, num_workers=0, pin_memory=False):
        """Wrapper for DataLoader inference"""
        loader = get_img_list_dataloader(img_list, batch_size=batch_size, num_workers=num_workers,
                                         pin_memory=pin_memory)
        preds = self._predict_loader(loader, soft_pred=soft_pred)
        return preds

    def __call__(self, x):
        """Direct call for inference on single  image"""
        return self.predict_img(x)

    def __repr__(self):
        return f'{self.__class__.__name__}(threshold={self.threshold})'
