import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from unet_328 import Model
from utils import AudioEncoder, AudDataset2, get_audio_features


load_dotenv()


class InferenceService:
    def __init__(self, asr: str = "ave", name: str = "Iris"):
        self.base_path = os.getenv("BASE_PATH")
        self.checkpoint_path = os.path.join(self.base_path, "checkpoint", name)
        self.checkpoint = os.path.join(
            self.checkpoint_path,
            sorted(
                os.listdir(self.checkpoint_path), key=lambda x: int(x.split(".")[0])
            )[-1],
        )
        self.dataset_dir = os.path.join(self.base_path, "dataset", name)
        self.mode = asr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioEncoder().to(self.device).eval()
        if self.device.type == "cuda":
            self.model = self.model
        ckpt_path = os.path.join(
            self.base_path, "model/checkpoints/audio_visual_encoder.pth"
        )
        self.ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(
            {f"audio_encoder.{k}": v for k, v in self.ckpt.items()}
        )
        self.net = Model(6, self.mode).cuda()
        self.net.load_state_dict(torch.load(self.checkpoint))
        self.net.eval()
        self.img_dir = os.path.join(self.dataset_dir, "full_body_img")
        self.lms_dir = os.path.join(self.dataset_dir, "landmarks")
        self.len_img = len(os.listdir(self.img_dir))
        self.images = [
            cv2.imread(f"{self.img_dir}/{i}.jpg") for i in range(self.len_img - 1)
        ]
        self.landmarks = []
        for i in range(self.len_img - 1):
            path = f"{self.lms_dir}/{i}.lms"
            lms_list = []
            with open(path, "r") as f:
                lines = f.read().splitlines()
                for line in lines:
                    arr = line.split(" ")
                    arr = np.array(arr, dtype=np.float32)
                    lms_list.append(arr)
            self.landmarks.append(np.array(lms_list, dtype=np.int32))

    def generate_frames(
        self,
        wav: np.ndarray,
        start_frame: int = 0,
    ):
        dataset = AudDataset2(wav)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        outputs = []
        for mel in data_loader:
            mel = mel.to(self.device)
            with torch.no_grad():
                out = self.model(mel)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=0).cpu()
        first_frame, last_frame = outputs[:1], outputs[-1:]
        audio_feats = torch.cat(
            [first_frame.repeat(1, 1), outputs, last_frame.repeat(1, 1)], dim=0
        ).numpy()
        exm_img = self.images[0]
        h, w = exm_img.shape[:2]
        step_stride = 0
        img_idx = 0
        for i in tqdm(range(audio_feats.shape[0])):
            if img_idx > self.len_img - 1:
                step_stride = -1
            if img_idx < 1:
                step_stride = 1
            img_idx += step_stride
            if img_idx >= self.len_img:
                img_idx = self.len_img - 1
            idx_img = min(img_idx + start_frame, self.len_img)
            img = self.images[idx_img].copy()
            lms = self.landmarks[idx_img].copy()
            xmin = lms[1][0]
            ymin = lms[52][1]
            xmax = lms[31][0]
            width = xmax - xmin
            ymax = ymin + width
            crop_img = img[ymin:ymax, xmin:xmax]
            h, w = crop_img.shape[:2]

            crop_img = cv2.resize(crop_img, (328, 328), interpolation=cv2.INTER_CUBIC)
            crop_img_ori = crop_img.copy()
            img_real_ex = crop_img[4:324, 4:324].copy()
            img_real_ex_ori = img_real_ex.copy()

            img_masked = cv2.rectangle(img_real_ex_ori, (5, 5, 310, 305), (0, 0, 0), -1)
            img_masked = img_masked.transpose(2, 0, 1).astype(np.float32)
            img_real_ex = img_real_ex.transpose(2, 0, 1).astype(np.float32)

            img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
            img_masked_T = torch.from_numpy(img_masked / 255.0)
            img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)[None]

            audio_feat = get_audio_features(audio_feats, i)
            if self.mode == "hubert":
                audio_feat = audio_feat.reshape(32, 32, 32)
            if self.mode == "wenet":
                audio_feat = audio_feat.reshape(256, 16, 32)
            if self.mode == "ave":
                audio_feat = audio_feat.reshape(32, 16, 16)
            audio_feat = audio_feat[None]
            audio_feat = audio_feat.cuda()
            img_concat_T = img_concat_T.cuda()

            with torch.no_grad():
                pred = self.net(img_concat_T, audio_feat)[0]

            pred = pred.cpu().numpy().transpose(1, 2, 0) * 255
            pred = np.array(pred, dtype=np.uint8)

            crop_img_ori[4:324, 4:324] = pred
            crop_img_ori = cv2.resize(
                crop_img_ori, (w, h), interpolation=cv2.INTER_CUBIC
            )
            img[ymin:ymax, xmin:xmax] = crop_img_ori

            yield img, idx_img


services = {}

services["Judy"] = InferenceService(name="Judy")


def get_inference_service():
    return services
