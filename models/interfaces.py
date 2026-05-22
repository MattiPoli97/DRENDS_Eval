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
        #load a pth model 
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
        print("Post-processing DAMv2-Metric output...")
        print("Range of raw output:", outputs.predicted_depth.min().item(), "to", outputs.predicted_depth.max().item())
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
    Apple Depth Pro — metric monocular depth estimation.
    Uses the official apple/ml-depth-pro package directly, avoiding the
    transformers DepthProImageProcessorFast which requires PyTorch >= 2.1.
 
    Installation
    ------------
    pip install huggingface_hub
    huggingface-cli download --local-dir checkpoints apple/DepthPro
 
    Or let this class auto-download on first use (requires huggingface_hub).
 
    Parameters
    ----------
    img_size : tuple(int, int)
        Output (height, width) in pixels.
    ckpt_dir : str
        Directory containing the downloaded DepthPro checkpoint
        (must contain ). Defaults to "./checkpoints".
    depth_pro_root : str
        Path to the cloned apple/ml-depth-pro repo so that 
        is importable. Defaults to "./ml-depth-pro".
        If the package is already installed () you can
        leave this as None and set use_installed_pkg=True.
    use_installed_pkg : bool
        If True, import depth_pro from the installed package instead of
        the local repo. Requires: pip install git+https://github.com/apple/ml-depth-pro
    half : bool
        Use FP16 on CUDA (faster, less VRAM).
    """
 
    def __init__(
        self,
        img_size,
        ckpt_dir: str = "./checkpoints",
        depth_pro_root: str = "./ml-depth-pro",
        use_installed_pkg: bool = False,
        half: bool = False,
    ):
        super().__init__(img_size)
 
        # ---- make depth_pro importable ----
        import sys
        if not use_installed_pkg:
            src_path = os.path.join(os.path.abspath(depth_pro_root), "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
 
        import depth_pro
 
        # ---- auto-download checkpoint if missing ----
        ckpt_path = os.path.join(ckpt_dir, "depth_pro.pt")
        if not os.path.exists(ckpt_path):
            ckpt_path = self._download_checkpoint(ckpt_dir)
 
        # ---- load model ----
        # create_model_and_transforms builds the ViT + decoder and loads weights
        self.model, self.transform = depth_pro.create_model_and_transforms(
            device=self.device,
            precision=torch.float16 if (half and torch.cuda.is_available()) else torch.float32,
        )
        # override the default checkpoint path if user supplied a custom dir
        state = torch.load(ckpt_path, map_location=self.device)
        # checkpoint may be wrapped in a dict
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
 
        self._depth_pro = depth_pro   # keep reference for load_rgb helper
 
    # ------------------------------------------------------------------ API
    # single-image model — keeps the standard BaseInterface __call__ signature
    def _pre_processing(self, img_path):
        image, _, f_px = self._depth_pro.load_rgb(img_path)
        if f_px is None:
            f_px = torch.tensor(1967.78)  # default focal length in pixels if not provided
        image_t = self.transform(image).to(self.device)   # (3, H, W) normalised
        return image_t, f_px
 
    def _predict(self, inputs):
        image_t, f_px = inputs
        with torch.no_grad():
            prediction = self.model.infer(image_t, f_px=f_px)
        return prediction
 
    def _post_processing(self, prediction):
        """
        prediction["depth"]  : torch.Tensor (H, W) — metric depth in metres
        prediction["focallength_px"] : estimated focal length (float)
        """
        depth = prediction["depth"]          # (H, W) tensor, metric metres
        depth_np = depth.squeeze().detach().cpu().float().numpy()
 
        # resize to requested output resolution
        depth_t = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        depth_t = torch.nn.functional.interpolate(
            depth_t,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        return depth_t.squeeze().numpy().astype(np.float32)
 
    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _download_checkpoint(ckpt_dir: str) -> str:
        """Download the DepthPro checkpoint from HuggingFace Hub."""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Install huggingface_hub to auto-download DepthPro:"
                "  pip install huggingface_hub"
                "Or download manually:"
                "  huggingface-cli download --local-dir checkpoints apple/DepthPro"
            )
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"-> Downloading DepthPro checkpoint to {ckpt_dir}/depth_pro.pt")
        path = hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            local_dir=ckpt_dir,
        )
        return path
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

### =========================================================
# 7) RAFT (stereo)
# ===========================================================
class RAFTStereo(BaseInterface):
    """
    RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching.
    (https://github.com/princeton-vl/RAFT-Stereo, 3DV 2021 Best Student Paper)

    Takes a *stereo pair* (left + right images) and returns a disparity map for
    the left image.  Optionally converts disparity to metric depth when camera
    baseline and focal length are supplied.

    Parameters
    ----------
    img_size : tuple(int, int)
        Output (height, width) in pixels.
    ckpt_path : str
        Path to a pretrained .pth checkpoint (e.g. raftstereo-middlebury.pth).
        Download with: bash download_models.sh  inside the RAFT-Stereo repo.
    raft_stereo_root : str
        Path to the RAFT-Stereo repository root so that `core/` is importable.
        Defaults to "./RAFT-Stereo".
    valid_iters : int
        Number of recurrent update iterations at inference time (default 32).
    mixed_precision : bool
        Use FP16 feature maps (faster, less VRAM). Requires CUDA.
    shared_backbone : bool
        Use the lightweight shared-backbone variant (realtime model).
    n_downsample : int
        Feature-map downsampling factor: 2 (default) or 3 (less memory).
    n_gru_layers : int
        Number of GRU update layers (1, 2, or 3).
    slow_fast_gru : bool
        Enable slow-fast GRU scheduling (used with n_gru_layers=2 realtime).
    corr_implementation : str
        'reg' (default pure-Python), 'reg_cuda' (fast CUDA), or 'alt' (memory
        efficient for high-res images).
    baseline_m : float or None
        Camera baseline in metres.  Required for disparity → depth conversion.
    focal_px : float or None
        Focal length in *pixels*.  Required for disparity → depth conversion.
    cx_diff : float
        (cx1 - cx0): x-difference of principal points between right and left
        cameras.  Usually 0 for rectified pairs (default 0.0).

    Usage
    -----
    >>> model = RAFTStereo(img_size=(480, 640), ckpt_path="models/raftstereo-middlebury.pth")
    >>> depth  = model(left_img_path, right_img_path)   # np.float32 (H, W)

    If baseline_m and focal_px are not provided the output is the *disparity*
    map (in pixels, float32) rather than metric depth.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        img_size,
        ckpt_path: str,
        raft_stereo_root: str = "./RAFT-Stereo",
        valid_iters: int = 32,
        mixed_precision: bool = False,
        shared_backbone: bool = False,
        n_downsample: int = 2,
        n_gru_layers: int = 3,
        slow_fast_gru: bool = False,
        corr_implementation: str = "reg",
        baseline_m: float = None,
        focal_px: float = None,
        cx_diff: float = 0.0,
        hiera: bool = False,
    ):
        super().__init__(img_size)

        # ---- make RAFT-Stereo core importable ----
        import sys
        core_path = os.path.join(raft_stereo_root, "core")
        if core_path not in sys.path:
            sys.path.insert(0, core_path)
        if raft_stereo_root not in sys.path:
            sys.path.insert(0, raft_stereo_root)

        from raft_stereo import RAFTStereo as _RAFTStereoModel

        # ---- build args namespace expected by RAFTStereo ----
        import argparse
        args = argparse.Namespace(
            corr_implementation=corr_implementation,
            mixed_precision=mixed_precision,
            shared_backbone=shared_backbone,
            n_downsample=n_downsample,
            n_gru_layers=n_gru_layers,
            slow_fast_gru=slow_fast_gru,
            valid_iters=valid_iters,
            hidden_dims=[128] * 3,          
            context_norm="batch",           
            corr_levels=4,                  
            corr_radius=4,    
        )

        # ---- load model ----
        self._net = _RAFTStereoModel(args)
        state = torch.load(ckpt_path, map_location="cpu")
        # checkpoints may be saved with DataParallel wrapper
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self._net.load_state_dict(state, strict=True)
        self._net.to(self.device).eval()

        self.valid_iters = valid_iters
        self.mixed_precision = mixed_precision

        # metric conversion params
        self.baseline_m = baseline_m
        self.focal_px = focal_px
        self.cx_diff = cx_diff

        # preprocessing transform: ImageNet normalisation used by RAFT-Stereo
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.hiera = hiera
    # ------------------------------------------------------------------ API
    def __call__(self, left_img: str, right_img: str) -> np.ndarray:
        """
        Parameters
        ----------
        left_img  : str – path to the left rectified image.
        right_img : str – path to the right rectified image.

        Returns
        -------
        np.ndarray float32 (H, W)
            Disparity (px) if no camera params given; metric depth (m) otherwise.
        """
        inputs = self._pre_processing((left_img, right_img))
        output = self._predict(inputs)
        result = self._post_processing(output)
        return result.astype(np.float32)

    # ------------------------------------------------------------------ steps
    def _pre_processing(self, img_pair):
        left_path, right_path = img_pair

        left  = Image.open(left_path).convert("RGB")
        right = Image.open(right_path).convert("RGB")

        # RAFT-Stereo expects image dimensions divisible by 32
        orig_h, orig_w = left.height, left.width
        pad_h = (32 - orig_h % 32) % 32
        pad_w = (32 - orig_w % 32) % 32
        self._orig_size = (orig_h, orig_w)
        self._padded_size = (orig_h + pad_h, orig_w + pad_w)

        def to_tensor_padded(img):
            t = self._transform(img)                      # (3, H, W)  float32
            # pad bottom-right with reflection so network sees valid pixels
            t = torch.nn.functional.pad(
                t.unsqueeze(0), [0, pad_w, 0, pad_h], mode="reflect"
            ).squeeze(0)
            return t.unsqueeze(0).to(self.device)         # (1, 3, H', W')

        left_t  = to_tensor_padded(left)
        right_t = to_tensor_padded(right)

        if self.mixed_precision and torch.cuda.is_available():
            left_t  = left_t.half()
            right_t = right_t.half()

        return left_t, right_t

    def _predict(self, inputs):
        left_t, right_t = inputs
        with torch.no_grad():
            if self.hiera:
                disp = self._hierarchical_predict(left_t, right_t)
            else:
                result = self._net(
                    left_t, right_t,
                    iters=self.valid_iters,
                    test_mode=True,
                )
                # handle both return conventions:
                # - tuple (flow_up, disp_preds) like RAFT-Stereo
                # - single tensor (FoundationStereo test_mode)
                if isinstance(result, (tuple, list)):
                    disp = result[-1]
                    if isinstance(disp, (tuple, list)):
                        disp = disp[-1]   # (flow_up, [pred0, pred1, ...]) -> last pred
                else:
                    disp = result
        return disp

    def _hierarchical_predict(self, left_t, right_t):
        _H, _W = left_t.shape[-2], left_t.shape[-1]

        # -- coarse pass --
        left_small  = torch.nn.functional.interpolate(left_t,  scale_factor=0.5, mode="bilinear", align_corners=False)
        right_small = torch.nn.functional.interpolate(right_t, scale_factor=0.5, mode="bilinear", align_corners=False)

        result_small = self._net(left_small, right_small, iters=self.valid_iters, test_mode=True)
        if isinstance(result_small, (tuple, list)):
            disp_small = result_small[-1]
            if isinstance(disp_small, (tuple, list)):
                disp_small = disp_small[-1]
        else:
            disp_small = result_small

        if disp_small.dim() == 3:
            disp_small = disp_small.unsqueeze(1)

        disp_init = torch.nn.functional.interpolate(
            disp_small, size=(_H, _W), mode="bilinear", align_corners=False
        ) * 2.0

        # -- fine pass --
        result_full = self._net(
            left_t, right_t,
            iters=self.valid_iters,
            test_mode=True,
            flow_init=disp_init,
        )
        if isinstance(result_full, (tuple, list)):
            disp = result_full[-1]
            if isinstance(disp, (tuple, list)):
                disp = disp[-1]
        else:
            disp = result_full

        return disp

    def _post_processing(self, disp_tensor):
        orig_h, orig_w = self._orig_size

        # normalise to (1, 1, H, W) regardless of what the model returned
        if disp_tensor.dim() == 3:          # (1, H, W)  -> (1, 1, H, W)
            disp_tensor = disp_tensor.unsqueeze(1)
        elif disp_tensor.dim() == 2:        # (H, W)     -> (1, 1, H, W)
            disp_tensor = disp_tensor.unsqueeze(0).unsqueeze(0)

        # crop padding and resize to requested output resolution
        disp = disp_tensor[:, :, :orig_h, :orig_w]
        disp = torch.nn.functional.interpolate(
            disp,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        disp_np = disp.squeeze().detach().cpu().float().numpy()   # (H, W)

        scale = self.width / orig_w
        disp_np = disp_np * scale

        print("Disparity range (px):", disp_np.min(), "to", disp_np.max())
        if self.baseline_m is not None and self.focal_px is not None:
            denom = np.abs(disp_np + self.cx_diff)
            denom = np.where(denom > 1e-3, denom, 1e-3)
            depth = (self.baseline_m * self.focal_px) / denom
            print("Depth range (m):", depth.min(), "to", depth.max())
            return depth.astype(np.float32)

        return np.abs(disp_np).astype(np.float32)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def disp_to_depth(disp_px, baseline_m, focal_px, cx_diff=0.0):
        """Utility: convert a disparity map (px) to metric depth (m)."""
        denom = disp_px + cx_diff
        denom = np.where(denom > 1e-3, denom, 1e-3)
        return (baseline_m * focal_px) / denom

# =========================================================
# 8) FoundationStereo (CVPR 2025, NVLabs)
# =========================================================
 
class FoundationStereo(BaseInterface):
    """
    FoundationStereo: Zero-Shot Stereo Matching (CVPR 2025, Best Paper Nomination).
    https://github.com/NVlabs/FoundationStereo
 
    Takes a rectified stereo pair (left + right) and returns a disparity map for
    the left view. Optionally converts to metric depth when camera intrinsics
    and baseline are supplied.
 
    Parameters
    ----------
    img_size : tuple(int, int)
        Output (height, width) in pixels.
    ckpt_path : str
        Path to the pretrained .pth checkpoint, e.g.:
        "./pretrained_models/23-51-11/model_best_bp2.pth"
    foundation_stereo_root : str
        Path to the cloned FoundationStereo repo root so that `core/` is
        importable. Defaults to "./FoundationStereo".
    valid_iters : int
        Number of recurrent refinement iterations (default 32; use 16 for speed).
    mixed_precision : bool
        Use FP16 (faster, less VRAM). Requires CUDA.
    scale : float
        Rescale input images before inference, e.g. 0.5 to halve resolution.
        Output is always resized back to img_size.
    hiera : bool
        Enable hierarchical inference for high-resolution images (>1000 px).
        Slower but preserves full-resolution detail.
    baseline_m : float or None
        Camera baseline in metres. Required for disparity → depth conversion.
    focal_px : float or None
        Focal length in pixels (fx from intrinsic matrix K[0,0]).
        Required for disparity → depth conversion.
    cx_diff : float
        (cx_right - cx_left): principal-point x-offset between cameras.
        Usually 0.0 for well-rectified pairs (default 0.0).
 
    Usage
    -----
    >>> model = FoundationStereo(
    ...     img_size=(480, 640),
    ...     ckpt_path="pretrained_models/23-51-11/model_best_bp2.pth",
    ... )
    >>> disp  = model(left_img_path, right_img_path)   # np.float32 (H, W)
 
    Without baseline_m / focal_px the output is *disparity* in pixels.
    With both supplied the output is metric *depth* in metres.
    """
 
    # ------------------------------------------------------------------ init
    def __init__(
        self,
        img_size,
        ckpt_path: str,
        foundation_stereo_root: str = "./FoundationStereo",
        valid_iters: int = 32,
        mixed_precision: bool = False,
        scale: float = 1.0,
        hiera: bool = False,
        baseline_m: float = None,
        focal_px: float = None,
        cx_diff: float = 0.0,
    ):
        super().__init__(img_size)
 
        # ---- make FoundationStereo importable ----
        import sys
        core_path = os.path.join(os.path.abspath(foundation_stereo_root), "core")
        repo_path = os.path.abspath(foundation_stereo_root)
        for p in (core_path, repo_path):
            if p not in sys.path:
                sys.path.insert(0, p)
 
        from foundation_stereo import FoundationStereo as _FSModel
 
        # ---- build the cfg/args Namespace expected by FoundationStereo ----
        # open the cfg.yaml file stored in the same directory as the checkpoint and read the training config
        ckpt_dir = os.path.dirname(ckpt_path)
        cfg_path = os.path.join(ckpt_dir, "cfg.yaml")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                import yaml
                cfg_dict = yaml.safe_load(f)
                # add any missing keys like vit_size or corr_implementation with defaults
                cfg_dict.setdefault("vit_size", "vitl")

        else:
            cfg_dict = dict(
                mixed_precision=mixed_precision,
                valid_iters=valid_iters,
                hidden_dims=[128, 128, 128],
                corr_levels=4,
                corr_radius=4,
                n_downsample=2,
                n_gru_layers=3,
                slow_fast_gru=False,
                corr_implementation="reg",
                vit_size="vits",
                context_norm="layer",
                decoder_norm="layer",
                train_iters=22,
                max_disp=768,
                shared_backbone=False,
            )

        # FoundationStereo's top-level __init__ uses both dot-access (args.hidden_dims)
        # and subscript access (cfg['max_disp']), so wrap in a class that supports both
        class DotDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        cfg = DotDict(cfg_dict)
        self._net = _FSModel(cfg)
 
        self._net = _FSModel(cfg)
        state = torch.load(ckpt_path, map_location="cpu")
        # strip DataParallel / compiled-model prefixes if present
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self._net.load_state_dict(state, strict=False)   # strict=False: some
        # checkpoints carry extra keys (e.g. loss scalers) that are not part
        # of the model graph — safe to ignore.
        self._net.to(self.device).eval()
 
        self.valid_iters = valid_iters
        self.mixed_precision = mixed_precision
        self.scale = scale
        self.hiera = hiera
 
        # metric conversion
        self.baseline_m = baseline_m
        self.focal_px = focal_px
        self.cx_diff = cx_diff
 
        # FoundationStereo uses ImageNet normalisation (same as RAFT-Stereo)
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
 
    # ------------------------------------------------------------------ API
    def __call__(self, left_img: str, right_img: str) -> np.ndarray:
        """
        Parameters
        ----------
        left_img  : str – path to the left rectified image.
        right_img : str – path to the right rectified image.
 
        Returns
        -------
        np.ndarray float32 (H, W)
            Disparity (px) without camera params; metric depth (m) with them.
        """
        inputs = self._pre_processing((left_img, right_img))
        output = self._predict(inputs)
        result = self._post_processing(output)
        return result.astype(np.float32)
 
    # ------------------------------------------------------------------ steps
    def _pre_processing(self, img_pair):
        left_path, right_path = img_pair
 
        left  = Image.open(left_path).convert("RGB")
        right = Image.open(right_path).convert("RGB")
 
        self._orig_size = (left.height, left.width)
 
        # optional downscale for speed / memory
        if self.scale != 1.0:
            new_w = int(round(left.width  * self.scale))
            new_h = int(round(left.height * self.scale))
            left  = left.resize((new_w, new_h),  Image.LANCZOS)
            right = right.resize((new_w, new_h), Image.LANCZOS)
 
        # pad to multiples of 32 (required by the feature pyramid)
        h, w = left.height, left.width
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        self._inference_size = (h, w)           # unpadded inference size
 
        def prep(img):
            t = self._to_tensor(img)            # (3, H, W)
            t = torch.nn.functional.pad(
                t.unsqueeze(0), [0, pad_w, 0, pad_h], mode="reflect"
            ).squeeze(0)
            return t.unsqueeze(0).to(self.device)   # (1, 3, H', W')
 
        left_t  = prep(left)
        right_t = prep(right)
 
        if self.mixed_precision and torch.cuda.is_available():
            left_t  = left_t.half()
            right_t = right_t.half()
 
        return left_t, right_t
 
    @staticmethod
    def _unpack_disp(result):
        """Handles all return conventions from FoundationStereo test_mode=True:
        - bare Tensor
        - tuple/list whose last element is a Tensor
        - tuple/list whose last element is itself a list of Tensors
        """
        if isinstance(result, (tuple, list)):
            last = result[-1]
            if isinstance(last, (tuple, list)):
                return last[-1]
            return last
        return result

    def _predict(self, inputs):
        left_t, right_t = inputs
        with torch.no_grad():
            if self.hiera:
                disp = self._hierarchical_predict(left_t, right_t)
            else:
                result = self._net(
                    left_t, right_t,
                    iters=self.valid_iters,
                    test_mode=True,
                )
                disp = self._unpack_disp(result)
        return disp

    def _hierarchical_predict(self, left_t, right_t):
        """Two-pass inference: coarse at 0.5x then full-res refinement."""
        _H, _W = left_t.shape[-2], left_t.shape[-1]

        # -- coarse pass --
        left_small  = torch.nn.functional.interpolate(left_t,  scale_factor=0.5, mode="bilinear", align_corners=False)
        right_small = torch.nn.functional.interpolate(right_t, scale_factor=0.5, mode="bilinear", align_corners=False)
        result_small = self._net(left_small, right_small, iters=self.valid_iters, test_mode=True)
        disp_small = self._unpack_disp(result_small)

        if disp_small.dim() == 3:
            disp_small = disp_small.unsqueeze(1)

        disp_init = torch.nn.functional.interpolate(
            disp_small, size=(_H, _W), mode="bilinear", align_corners=False
        ) * 2.0

        # -- fine pass with warm start --
        result_full = self._net(
            left_t, right_t,
            iters=self.valid_iters,
            test_mode=True,
            flow_init=disp_init,
        )
        return self._unpack_disp(result_full)
 
    def _post_processing(self, disp_tensor):
        inf_h, inf_w = self._inference_size
        orig_h, orig_w = self._orig_size
 
        # normalise shape to (1, 1, H, W)
        if disp_tensor.dim() == 3:
            disp_tensor = disp_tensor.unsqueeze(1)
        elif disp_tensor.dim() == 2:
            disp_tensor = disp_tensor.unsqueeze(0).unsqueeze(0)
 
        # crop padding then resize to requested output size
        disp = disp_tensor[:, :, :inf_h, :inf_w]
        disp = torch.nn.functional.interpolate(
            disp,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        disp_np = disp.squeeze().detach().cpu().float().numpy()     # (H, W)
        disp_np = np.abs(disp_np)   # ensure positive
 
        # rescale disparity if scale or output-size differs from capture size
        scale_x = self.width / (orig_w * self.scale)
        disp_np = disp_np * scale_x
 
        if self.baseline_m is not None and self.focal_px is not None:
            denom = disp_np + self.cx_diff
            denom = np.where(denom > 1e-3, denom, 1e-3)
            depth = (self.baseline_m * self.focal_px) / denom
            return depth.astype(np.float32)
 
        return disp_np.astype(np.float32)
 
    # ------------------------------------------------------------------ utils
    @staticmethod
    def disp_to_depth(disp_px, baseline_m, focal_px, cx_diff=0.0):
        """Utility: convert a disparity map (px) to metric depth (m)."""
        denom = disp_px + cx_diff
        denom = np.where(denom > 1e-3, denom, 1e-3)
        return (baseline_m * focal_px) / denom

# =========================================================
# 9) STTR — STereo TRansformer (ICCV 2021 Oral)
# =========================================================
 
class STTRInterface(BaseInterface):
    """
    STereo TRansformer (STTR): Revisiting Stereo Depth Estimation From a
    Sequence-to-Sequence Perspective with Transformers (ICCV 2021 Oral).
    https://github.com/mli0603/stereo-transformer
 
    Key properties vs. other stereo models:
    - No fixed disparity range: the range scales automatically with resolution.
    - Explicit occlusion detection: occluded pixels are flagged and set to 0.
    - Uniqueness constraint via optimal-transport-style attention.
 
    Takes a rectified stereo pair (left + right) and returns a disparity map
    for the left view. Optionally converts to metric depth when camera
    intrinsics and baseline are provided.
 
    Parameters
    ----------
    img_size : tuple(int, int)
        Output (height, width) in pixels.
    ckpt_path : str
        Path to a pretrained checkpoint (.pth.tar), e.g.:
        "./stereo-transformer/sceneflow_pretrained_model.pth.tar"
    sttr_root : str
        Path to the cloned stereo-transformer repo root so that `module/`
        and `utilities/` are importable. Defaults to "./stereo-transformer".
    downsample : int
        Attention stride — controls the resolution of the attention map.
        1 = full resolution (most accurate, most memory);
        3 = default in the original repo (good balance).
        Increase to reduce memory on large images.
    channel_dim : int
        Feature channel dimension. 128 for STTR, 32 for STTR-light.
    position_encoding : str
        Positional encoding type: "sine1d_rel" (default) or "learned1d".
    num_attn_layers : int
        Number of transformer attention layers. 6 for STTR, 3 for STTR-light.
    baseline_m : float or None
        Camera baseline in metres. Required for disparity → depth conversion.
    focal_px : float or None
        Focal length in pixels (K[0,0]). Required for depth conversion.
    cx_diff : float
        (cx_right - cx_left). Usually 0.0 for well-rectified pairs.
 
    Usage
    -----
    >>> model = STTRInterface(
    ...     img_size=(480, 640),
    ...     ckpt_path="stereo-transformer/sceneflow_pretrained_model.pth.tar",
    ... )
    >>> disp = model(left_img_path, right_img_path)   # np.float32 (H, W)
 
    Without baseline_m / focal_px the output is disparity in pixels.
    Occluded pixels (detected by STTR) are set to 0 in the output.
    """
 
    # ------------------------------------------------------------------ init
    def __init__(
        self,
        img_size,
        ckpt_path: str,
        sttr_root: str = "./stereo-transformer",
        downsample: int = 3,
        channel_dim: int = 128,
        position_encoding: str = "sine1d_rel",
        num_attn_layers: int = 6,
        baseline_m: float = None,
        focal_px: float = None,
        cx_diff: float = 0.0,
    ):
        super().__init__(img_size)
 
        # ---- make STTR importable ----
        import sys
        for p in (os.path.abspath(sttr_root),):
            if p not in sys.path:
                sys.path.insert(0, p)
 
        from module.sttr import STTR
        from utilities.misc import NestedTensor
 
        self._NestedTensor = NestedTensor
 
        # ---- build model args ----
        import argparse
        args = argparse.Namespace(
            channel_dim=channel_dim,
            position_encoding=position_encoding,
            num_attn_layers=num_attn_layers,
            downsample=downsample,
            # regression head defaults
            regression_head="ot",         # optimal-transport head
            context_adjustment_layer="cal",
            cal_num_blocks=8,
            cal_feat_dim=16,
            cal_expansion_ratio=4,
        )
 
        self._net = STTR(args).to(self.device).eval()
 
        # ---- load checkpoint ----
        ckpt_path = self._download_ckpt_if_needed(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # checkpoints saved as {"state_dict": ..., "epoch": ...}
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self._net.load_state_dict(state, strict=False)
 
        self.downsample = downsample
        self.baseline_m = baseline_m
        self.focal_px = focal_px
        self.cx_diff = cx_diff
 
        # STTR uses ImageNet normalisation
        self._normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self._to_tensor = transforms.ToTensor()
 

    def _download_ckpt_if_needed(self, ckpt_path: str) -> str:
        """
        If ckpt_path is a known shorthand ('sceneflow', 'kitti', 'sceneflow_light'),
        download from Google Drive using gdown and return the local path.
        Otherwise return ckpt_path unchanged.
        """
        self._CHECKPOINTS = {
            "sceneflow": (
                "https://drive.google.com/uc?id=1R0YUpFzDRTKvjRfngF8SPj2JR2M1mMTF",
                "sceneflow_pretrained_model.pth.tar",
            ),
            "kitti": (
                "https://drive.google.com/uc?id=1UUESCCnOsb7TqzwYMkVV3d23k8shxNcE",
                "kitti_finetuned_model.pth.tar",
            ),
            "sceneflow_light": (
                "https://drive.google.com/uc?id=1MW5g1LQ1RaYbqeDS2AlHPZ96wAmkFG_O",
                "sttr_light_pretrained_model.pth.tar",
            ),
        }

        if ckpt_path not in self._CHECKPOINTS:
            return ckpt_path   # assume it's already a local path

        url, filename = self._CHECKPOINTS[ckpt_path]
        local_path = os.path.join("models", "sttr", filename)

        if os.path.exists(local_path):
            return local_path

        try:
            import gdown
        except ImportError:
            raise ImportError(
                "Install gdown to auto-download STTR weights: pip install gdown"
            )

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"-> Downloading STTR checkpoint to {local_path}")
        gdown.download(url, local_path, quiet=False, fuzzy=True)
        return local_path
    # ------------------------------------------------------------------ API
    def __call__(self, left_img: str, right_img: str) -> np.ndarray:
        inputs = self._pre_processing((left_img, right_img))
        output = self._predict(inputs)
        result = self._post_processing(output)
        return result.astype(np.float32)
 
    # ------------------------------------------------------------------ steps
    def _pre_processing(self, img_pair):
        left_path, right_path = img_pair
 
        left  = Image.open(left_path).convert("RGB")
        right = Image.open(right_path).convert("RGB")
 
        self._orig_size = (left.height, left.width)
 
        # STTR requires dimensions divisible by (downsample * 4) — pad safely
        # to the nearest multiple of 64 which covers all downsample values
        h, w = left.height, left.width
        pad_h = (64 - h % 64) % 64
        pad_w = (64 - w % 64) % 64
        self._inference_size = (h, w)
 
        def to_tensor_padded(img):
            t = self._normalize(self._to_tensor(img))   # (3, H, W)
            t = torch.nn.functional.pad(
                t.unsqueeze(0), [0, pad_w, 0, pad_h], mode="reflect"
            ).squeeze(0)
            return t   # (3, H', W')
 
        left_t  = to_tensor_padded(left).unsqueeze(0).to(self.device)   # (1,3,H',W')
        right_t = to_tensor_padded(right).unsqueeze(0).to(self.device)
 
        # STTR expects NestedTensor (tensor + mask); mask=None means all valid
        left_nt  = self._NestedTensor(left_t,  None)
        right_nt = self._NestedTensor(right_t, None)
 
        return left_nt, right_nt
 
    def _predict(self, inputs):
        left_nt, right_nt = inputs
        with torch.no_grad():
            output = self._net(left_nt, right_nt)
        return output
 
    def _post_processing(self, output):
        """
        STTR output dict contains:
          - "disp_pred"   : (1, H', W') float — raw disparity (px)
          - "occ_pred"    : (1, H', W') float — occlusion probability [0,1]
          - "disp_pred_low_res" : lower-res intermediate prediction (ignored)
        """
        inf_h, inf_w = self._inference_size
 
        disp = output["disp_pred"]                   # (1, H', W') or (H', W')
        occ  = output.get("occ_pred", None)
 
        if disp.dim() == 2:
            disp = disp.unsqueeze(0)
        disp = disp.unsqueeze(1)                     # (1, 1, H', W')
 
        # crop padding
        disp = disp[:, :, :inf_h, :inf_w]
 
        # resize to requested output resolution
        disp = torch.nn.functional.interpolate(
            disp,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        disp_np = disp.squeeze().detach().cpu().float().numpy()  # (H, W)
 
        # apply occlusion mask: set occluded pixels to 0 (matches STTR convention)
        if occ is not None:
            if occ.dim() == 2:
                occ = occ.unsqueeze(0)
            occ = occ[:, :inf_h, :inf_w]
            occ = torch.nn.functional.interpolate(
                occ.unsqueeze(1).float(),
                size=(self.height, self.width),
                mode="nearest",
            ).squeeze()
            occ_mask = occ.detach().cpu().numpy() > 0.5
            disp_np[occ_mask] = 0.0
 
        # scale disparity proportionally if output size differs from capture size
        scale_x = self.width / inf_w
        disp_np = disp_np * scale_x
 
        if self.baseline_m is not None and self.focal_px is not None:
            # only convert non-occluded pixels (disp > 0)
            valid = disp_np > 1e-3
            depth = np.zeros_like(disp_np)
            denom = disp_np[valid] + self.cx_diff
            denom = np.where(denom > 1e-3, denom, 1e-3)
            depth[valid] = (self.baseline_m * self.focal_px) / denom
            return depth.astype(np.float32)
 
        return disp_np.astype(np.float32)
 
    # ------------------------------------------------------------------ utils
    @staticmethod
    def disp_to_depth(disp_px, baseline_m, focal_px, cx_diff=0.0):
        """Utility: convert a disparity map (px) to metric depth (m).
        Pixels with disp_px <= 0 (occluded) are returned as 0."""
        valid = disp_px > 1e-3
        depth = np.zeros_like(disp_px, dtype=np.float32)
        denom = disp_px[valid] + cx_diff
        denom = np.where(denom > 1e-3, denom, 1e-3)
        depth[valid] = (baseline_m * focal_px) / denom
        return depth