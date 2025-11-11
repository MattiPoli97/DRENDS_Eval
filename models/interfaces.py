import torch
import numpy as np
from PIL import Image
import os
import zipfile
import urllib.request
import hashlib
from torchvision import transforms

class BaseInterface:
    def __init__(self, img_size=None):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.height = None
        self.width = None
        if img_size is not None:
            self.height, self.width = img_size
    
    @torch.no_grad()
    def __call__(self, img : str):
        inputs = self._pre_processing(img)
        output = self._predict(inputs)
        depth = self._post_processing(output)
        return depth.astype(np.float32)

    def _pre_processing(self, img): # Mock funciton
        return {'input':img}

    def _predict(self, inputs): # Mock funciton
        return inputs
    
    def _post_processing(self, img): # Mock funciton
        return img

from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoConfig, ZoeDepthForDepthEstimation, ZoeDepthConfig

# =========================================================
# 1) Depth Anything v2 (Relative)
# =========================================================
class DAMv2(BaseInterface):
    def __init__(self, img_size, model_size="small"):
        super().__init__(img_size)
        model = f"depth-anything/Depth-Anything-V2-{model_size}-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(model)
        self.model = AutoModelForDepthEstimation.from_pretrained(model)
    
    def _pre_processing(self, img):
        img = Image.open(img)
        img = img.convert("RGB") if img.mode != "RGB" else img
        inputs = self.image_processor(images=img, return_tensors="pt")
        return inputs.to(self.device)
    
    def _predict(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
    
    def _post_processing(self, img):
        post_processed_output = self.image_processor.post_process_depth_estimation(
            img, target_sizes=[(self.height, self.width)])
    
        predicted_depth = post_processed_output[0]["predicted_depth"]
        depth = predicted_depth.squeeze().cpu().numpy()

        return depth

# =========================================================
# 2) Depth Anything v2 (Metric) 
# =========================================================

class DAMv2_Metric(BaseInterface):
    def __init__(self, img_size, model_size="Small", domain="Indoor", max_depth=None):
        super().__init__(img_size)
        repo = f"depth-anything/Depth-Anything-V2-Metric-{domain}-{model_size}-hf"

        config = AutoConfig.from_pretrained(repo)
        if max_depth is not None:
            config.max_depth = float(max_depth)

        self.image_processor = AutoImageProcessor.from_pretrained(repo)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo, config=config).to(self.device).eval()

    def _pre_processing(self, img):
        im = Image.open(img)
        if im.mode != "RGB":
            im = im.convert("RGB")
        return self.image_processor(images=im, return_tensors="pt").to(self.device)

    def _predict(self, inputs):
        with torch.no_grad():
            return self.model(**inputs)

    def _post_processing(self, outputs):
        post = self.image_processor.post_process_depth_estimation(
            outputs, target_sizes=[(self.height, self.width)]
        )[0]
        depth = post["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        return depth

# =========================================================
# 3) MiDaS (PyTorch Hub)
# =========================================================

class MiDaS(BaseInterface):
    """
    model_type: "DPT_Large" | "DPT_Hybrid" | "MiDaS_small"
    Returns relative depth. Good, fast baseline.
    """

    def __init__(self, img_size, model_type="DPT_Hybrid"):
        super().__init__(img_size)
        self.model_type = model_type

        self.midas = torch.hub.load("intel-isl/MiDaS", model_type).to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.dpt_transform if model_type in ("DPT_Large", "DPT_Hybrid") else transforms.small_transform
    
    def _pre_processing(self, img):
        img = Image.open(img).convert("RGB")
        img = self.transform(np.array(img)).to(self.device) 
        return img
    
    def _predict(self, inputs):
        with torch.no_grad():
            pred = self.midas(inputs)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False
            ).squeeze(1)
        return pred

    def _post_processing(self, pred):
        ## Midas returns relative inverse depth
        return pred.squeeze().detach().cpu().numpy()

# =========================================================
# 4) ZoeDepth
# =========================================================

class ZoeDepthInterface(BaseInterface):
    """
    HF ZoeDepth (metric-capable). Default weights: Intel/zoedepth-nyu-kitti
    """
    def __init__(self, img_size, repo="Intel/zoedepth-nyu-kitti", max_depth=None):
        super().__init__(img_size) 
        cfg = ZoeDepthConfig.from_pretrained(repo)
        if max_depth is not None:
            cfg.max_depth = float(max_depth)
            cfg.min_depth = 0.1  

        self.processor = AutoImageProcessor.from_pretrained(repo)
        self.model = ZoeDepthForDepthEstimation.from_pretrained(repo, config=cfg).to(self.device).eval()
        self._last_source_size = None  # (H, W)
        self.max_depth = max_depth if max_depth is not None else 80.0  # default max depth
        self.min_depth = 0.1

    def _pre_processing(self, img_path):
        from PIL import Image
        im = Image.open(img_path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        self._last_source_size = (im.height, im.width)
        return self.processor(images=im, return_tensors="pt").to(self.device)

    def _predict(self, inputs):
        return self.model(**inputs)

    def _post_processing(self, outputs):
        try:
            post = self.processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(self.height, self.width)],
                source_sizes=[self._last_source_size], 
            )[0]
        except TypeError:
            post = self.processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(self.height, self.width)],
                do_remove_padding=False,
            )[0]
        depth = post["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        depth = depth - depth.min()
        depth = depth / depth.max() * (self.max_depth - self.min_depth) + self.min_depth
        return depth

# =========================================================
# 5) Depth Pro
# =========================================================

class DepthProInterface(BaseInterface):
    """
    Apple Depth Pro (metric monocular).
    Requires: transformers >= 4.56 (has DepthPro* classes).
    Uses the -hf repo that includes the processor config.
    """
    def __init__(self, img_size, repo: str = "apple/DepthPro-hf", use_fov: bool = False, half: bool = False):
        super().__init__(img_size)

        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

        self.processor = DepthProImageProcessorFast.from_pretrained(repo)
        self.model = DepthProForDepthEstimation.from_pretrained(repo).to(self.device).eval()
        self.max_depth = 0.6
        if half and torch.cuda.is_available():
            self.model = self.model.half()

    def _pre_processing(self, img_path):
        im = Image.open(img_path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        inputs = self.processor(images=im, return_tensors="pt").to(self.device)
        if next(self.model.parameters()).dtype == torch.float16:
            # cast inputs for fp16 if needed
            for k in ["pixel_values"]:
                if k in inputs:
                    inputs[k] = inputs[k].half()
        return inputs

    def _predict(self, inputs):
        return self.model(**inputs)

    def _post_processing(self, outputs):
        post = self.processor.post_process_depth_estimation(
            outputs, target_sizes=[(self.height, self.width)]
        )[0]
        depth = post["predicted_depth"].squeeze().detach().cpu().numpy()
        
        return depth.astype(np.float32)

# =========================================================
# 6) MonoDepth2
# =========================================================

from networks import resnet_encoder, depth_decoder

class MonoDepth2(BaseInterface):
    def __init__(self, img_size, model_type="mono_640x192"):
        super().__init__(img_size)
        self.device = self.device 
        self.model_type = model_type
        self._download_model_if_needed(model_type)

        model_path = os.path.join("models", model_type)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        # ---- Load encoder ----
        self.encoder = resnet_encoder.ResnetEncoder(18, False)
        loaded_dict_enc = torch.load(encoder_path, map_location=self.device)
        self.feed_height = loaded_dict_enc["height"]
        self.feed_width  = loaded_dict_enc["width"]
        filtered = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered)
        self.encoder.to(self.device).eval()

        # ---- Load decoder ----
        self.decoder = depth_decoder.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dec = torch.load(depth_decoder_path, map_location=self.device)
        self.decoder.load_state_dict(loaded_dec)
        self.decoder.to(self.device).eval()

        # simple ToTensor (monodepth2 trains with 0..1 range)
        self.to_tensor = transforms.ToTensor()

    # -------- API methods --------
    def _pre_processing(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.feed_width, self.feed_height), Image.LANCZOS)
        tens = self.to_tensor(img).unsqueeze(0).to(self.device)           # (1,3,Hf,Wf)
        feats = self.encoder(tens)                                        # list of features
        return feats

    def _predict(self, feats):
        pred = self.decoder(feats)                                        # dict with ("disp", scale)
        return pred

    def _post_processing(self, pred_dict):
        disp = pred_dict["disp", 0]                                       # (B,1,Hf,Wf)
        disp = torch.nn.functional.interpolate(
            disp, size=(self.height, self.width), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0).detach().cpu().numpy()                    # (H,W)

        depth = self._disp_to_depth(disp, min_depth=0.05, max_depth=0.40)[1]
        return depth.astype(np.float32)

    # -------- helpers --------
    @staticmethod
    def _disp_to_depth(disp, min_depth, max_depth):
        min_disp = 1.0 / max_depth
        max_disp = 1.0 / min_depth
        scaled_disp = min_disp + (max_disp - min_disp) * disp
        depth = 1.0 / scaled_disp
        return scaled_disp, depth

    def _download_model_if_needed(self, model_name):
        download_paths = {
            "mono_640x192": ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
                             "a964b8356e08a02d009609d9e3928f7c"),
            "stereo_640x192": ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
                               "3dfb76bcff0786e4ec07ac00f658dd07"),
            "mono+stereo_640x192": ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
                                    "c024d69012485ed05d7eaa9617a96b81"),
            "mono_no_pt_640x192": ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
                                   "9c2f071e35027c895a4728358ffc913a"),
            "stereo_no_pt_640x192": ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
                                     "41ec2de112905f85541ac33a854742d1"),
            "mono+stereo_no_pt_640x192": ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
                                          "46c3b824f541d143a45c37df65fbab0a"),
            "mono_1024x320": ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
                              "0ab0766efdfeea89a0d9ea8ba90e1e63"),
            "stereo_1024x320": ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
                                "afc2f2126d70cf3fdf26b550898b501a"),
            "mono+stereo_1024x320": ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
                                     "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", model_name)
        enc_path = os.path.join(model_path, "encoder.pth")
        if os.path.exists(enc_path):
            return

        url, md5 = download_paths[model_name]
        zip_path = model_path + ".zip"

        def ok_md5(checksum, fpath):
            if not os.path.exists(fpath): return False
            with open(fpath, "rb") as f: 
                return hashlib.md5(f.read()).hexdigest() == checksum

        if not ok_md5(md5, zip_path):
            print(f"-> Downloading pretrained model to {zip_path}")
            urllib.request.urlretrieve(url, zip_path)
        if not ok_md5(md5, zip_path):
            raise RuntimeError("Failed to download correct monodepth2 weights.")

        print("   Unzipping model...")
        os.makedirs(model_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as f:
            f.extractall(model_path)
        print(f"   Model unzipped to {model_path}")
