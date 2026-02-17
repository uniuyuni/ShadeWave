"""PatchMatch Inpainting with Laplace-based Gradient Fill.

This module keeps API compatibility with the previous PatchMatch entrypoints,
but internally only runs the sky_poly path.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from dataclasses import dataclass
import time

@dataclass
class _TensorPackage:
    image: torch.Tensor
    mask: torch.Tensor

class PatchMatchInpainting:
    def __init__(
        self,
        max_working_size: int = 384,
        roi_margin: int = 48,
        speed_mode: str = "ultra",
        verbose: bool = False,
        device: str | None = None,
        texture_patch_size: int = 7,
        texture_strength: float = 1.0,  # 適応的調整が乗算される基準値
        search_iterations: int = 2,
        match_scale: float = 0.25,
        candidate_count: int = 64,
    ):
        self.max_working_size = max(128, int(max_working_size))
        self.roi_margin = max(8, int(roi_margin))
        self.speed_mode = str(speed_mode)
        self.verbose = bool(verbose)

        # Low-texture (sky) polynomial fill controls retained from previous implementation.
        self.low_texture_edge_threshold = 0.020
        if self.speed_mode == "quality":
            # 512 is sufficient for gradient fill structure. 1536 is overkill and slow.
            self.max_working_size = max(self.max_working_size, 512)
            self.low_texture_edge_threshold = 0.010

        # Texture transfer controls (new)
        self.texture_patch_size = max(3, int(texture_patch_size) | 1)  # ensure odd
        self.texture_strength = max(0.0, min(1.0, float(texture_strength)))
        self.search_iterations = max(1, int(search_iterations))
        if self.speed_mode == "quality":
            self.search_iterations = max(self.search_iterations, 3)

        self.match_scale = max(0.05, min(1.0, float(match_scale)))
        self.candidate_count = max(8, int(candidate_count))
        
        self.device = self._resolve_device(device)

    def _log(self, msg: str):
        if self.verbose:
            now = time.strftime("%H:%M:%S")
            print(f"[PatchMatch {now}] {msg}")

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device is not None:
            d = torch.device(device)
            if d.type == "mps" and not torch.backends.mps.is_available():
                return torch.device("cpu")
            if d.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return d

        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _to_tensor(self, image: np.ndarray, mask: np.ndarray) -> _TensorPackage:
        im_t = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device, non_blocking=True)
        # mask must be boolean for logical operations later (~mask)
        mk_t = (torch.from_numpy(mask) > 0).to(self.device, non_blocking=True)

        return _TensorPackage(
            image=im_t,
            mask=mk_t,
        )

    @staticmethod
    def _to_numpy(output: torch.Tensor, pkg: _TensorPackage) -> np.ndarray:
        out = output.detach().cpu().permute(1, 2, 0).numpy()
        return out

    @staticmethod
    def _edge_mag(image: torch.Tensor) -> torch.Tensor:
        y = (0.299 * image[0:1] + 0.587 * image[1:2] + 0.114 * image[2:3]).unsqueeze(0)
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device).view(1, 1, 3, 3)
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=image.device).view(1, 1, 3, 3)
        gx = F.conv2d(y, kx, padding=1)
        gy = F.conv2d(y, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-8)[0]

    def _guided_filter(self, guide: torch.Tensor, src: torch.Tensor, radius: int = 7, eps: float = 1e-4) -> torch.Tensor:
        """He et al., Guided Image Filtering."""
        # Mean filter
        def mean_filter(x, r):
             # Reflection padding
             x_pad = F.pad(x, (r, r, r, r), mode='reflect')
             # Box filter using conv2d (normalized)
             # groups=channel
             k_size = 2 * r + 1
             kernel = torch.ones((1, 1, k_size, k_size), device=x.device) / (k_size * k_size)
             c = x.shape[1]
             # Depthwise conv
             kernel = kernel.repeat(c, 1, 1, 1)
             return F.conv2d(x_pad, kernel, groups=c)

        mean_I = mean_filter(guide, radius)
        mean_p = mean_filter(src, radius)
        mean_Ip = mean_filter(guide * src, radius)
        cov_Ip = mean_Ip - mean_I * mean_p
        
        mean_II = mean_filter(guide * guide, radius)
        var_I = mean_II - mean_I * mean_I
        
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        mean_a = mean_filter(a, radius)
        mean_b = mean_filter(b, radius)
        
        q = mean_a * guide + mean_b
        return q

    def _laplace_fill(self, image: torch.Tensor, mask: torch.Tensor, max_iter: int = 1000) -> torch.Tensor:
        """
        Solve Laplace equation (Delta u = 0) to fill the hole smoothly.
        Implemented using Multi-Scale approach for better global color propagation.
        """
        if not mask.any():
            return image
        
        c, h, w = image.shape
    
        # --- Multi-Scale Initialization ---
        # If image is large, solve at a smaller scale first to get global colors right.
        # This prevents "cyan" average color in disjoint regions (e.g. sky vs ground).
        
        # Threshold: if side > 64, we can downscale.
        # Threshold: if side > 64, we can downscale.
        min_size = 64
        if h > min_size and w > min_size:
            scale_factor = 0.125 # 1/8 size
            
            # Check if 1/8 is too small
            target_h = max(min_size, int(h * scale_factor))
            target_w = max(min_size, int(w * scale_factor))
            
            if target_h < h and target_w < w:
                # Downscale
                img_small = F.interpolate(image.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False)[0]
                mask_small = F.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='nearest')[0, 0] > 0.5
                
                # Recursive solve at small scale
                # Use fewer iterations because it's small (diffusion is fast)
                prior_small = self._laplace_fill(img_small, mask_small, max_iter=200)
                
                # Upscale to use as prior
                prior = F.interpolate(prior_small.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)[0]
            else:
                prior = None
        else:
            prior = None

        # --- Base Level Solver ---
        
        # 1. Initialize hole
        if prior is not None:
             # Use the upscaled coarse solution as the starting point.
             # This injects the correct global color structure.
             filled = torch.where(mask.unsqueeze(0), prior, image)
        else:
             # If no prior (smallest level), use boundary mean or blur
             # Dilate mask to find boundary pixels
             k_dial = torch.ones((1, 1, 3, 3), device=self.device)
             # Use replicate padding to avoid zero-boundary artifacts
             mask_float = mask.float().unsqueeze(0)
             # Manual padding for dialation to avoid corner cases? 
             # F.conv2d doesn't support 'replicate', need to pad input.
             mask_padded = F.pad(mask_float, (1, 1, 1, 1), mode='replicate')
             boundary_mask = (F.conv2d(mask_padded, k_dial, padding=0) > 0)[0] & (~mask)
             
             if boundary_mask.any():
                  mean_color = image[:, boundary_mask].mean(dim=1).view(3, 1, 1)
             else:
                  mean_color = image.mean(dim=(1, 2)).view(3, 1, 1)
                  
             # Use Guided Filter for edge-preserving smoothing
             # Radius 3 (diameter 7)
             # User reported structure pull, so we want to preserve structure edges.
             image_input = image.unsqueeze(0)
             smooth_boundary_img = self._guided_filter(image_input, image_input, radius=3, eps=1e-3)[0]
             
             filled = torch.where(mask.unsqueeze(0), mean_color, smooth_boundary_img)

        # 2. Iteration (Fast Convolution)
        # Use standard Laplace kernel for diffusion. 
        # The bilateral filter in the loop was too slow (Python overhead + expensive ops).
        # We rely on PatchMatch for structure, this is just color initialization.
        
        kernel = torch.tensor([
            [0.0, 0.25, 0.0],
            [0.25, 0.0, 0.25],
            [0.0, 0.25, 0.0]
        ], device=self.device).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
        
        mask_expanded = mask.unsqueeze(0)
        
        # If we have a prior, fewer iterations needed
        current_iter = 50 if prior is not None else max_iter
        if self.speed_mode == "quality": 
             current_iter = min(current_iter, 200) # Cap absolute max
        
        # Fast GPU Diffusion
        for i in range(current_iter):
             # replicate padding
             filled_padded = F.pad(filled, (1, 1, 1, 1), mode='replicate')
             
             # Convolve (Diffusion)
             # groups=c ensures mostly independent channel diffusion (simple averaging)
             new_val = F.conv2d(filled_padded, kernel, groups=c)
             
             # Restore known pixels
             filled = torch.where(mask_expanded, new_val, image)
             
        return filled

    def _find_match_indices(
        self, 
        base_fill: torch.Tensor, 
        original: torch.Tensor, 
        hole: torch.Tensor,
        patch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find best matching patch coordinates (anchors -> sources).
        Returns: (anchors_y, anchors_x, best_source_ys, best_source_xs)
        """
        c, h, w = base_fill.shape
        known_mask = ~hole
        
        # Coordinates
        ys_full = torch.arange(h, device=self.device).unsqueeze(1).expand(-1, w)
        xs_full = torch.arange(w, device=self.device).unsqueeze(0).expand(h, -1)
        
        ys_known = ys_full[known_mask]
        xs_known = xs_full[known_mask]
        n_known = len(ys_known)
        
        if n_known < 10:
             return (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))

        ps = patch_size
        half_ps = ps // 2
        
        # Pre-calc edge map for texture cost
        edge_map = self._edge_mag(original)
        target_texture_val = edge_map[0][known_mask].mean()
        
        # Filter Candidates (Boundary Check)
        valid_candidates = (
            (ys_known >= half_ps) & (ys_known < h - half_ps) &
            (xs_known >= half_ps) & (xs_known < w - half_ps)
        )
        if valid_candidates.any():
            ys_known = ys_known[valid_candidates]
            xs_known = xs_known[valid_candidates]
            n_known = len(ys_known)
        
        # --- Anchor Logic (Grid + Random) ---
        anchors_y_list = []
        anchors_x_list = []
        
        all_hole_indices = torch.nonzero(hole, as_tuple=False)
        n_hole_pixels = len(all_hole_indices)
        
        # Shuffle for random fill
        shuffled_indices = all_hole_indices[torch.randperm(n_hole_pixels, device=self.device)]
        
        # CPU loop setup
        unfilled_cpu = hole.clone().cpu().numpy().astype(bool)
        shuffled_indices_cpu = shuffled_indices.cpu().numpy()
        
        count_anchors = 0
        t_start_dart = time.perf_counter()
        
        # Stage 1: Grid
        stride = max(1, ps // 2)
        grid_y = np.arange(0, h, stride)
        grid_x = np.arange(0, w, stride)
        gy, gx = np.meshgrid(grid_y, grid_x, indexing='ij')
        gy = gy.flatten()
        gx = gx.flatten()
        
        for i in range(len(gy)):
             cy, cx = gy[i], gx[i]
             if cy >= h or cx >= w: continue
             if unfilled_cpu[cy, cx]:
                  anchors_y_list.append(int(cy))
                  anchors_x_list.append(int(cx))
                  count_anchors += 1
                  y0, y1 = max(0, int(cy)-half_ps), min(h, int(cy)+half_ps+1)
                  x0, x1 = max(0, int(cx)-half_ps), min(w, int(cx)+half_ps+1)
                  unfilled_cpu[y0:y1, x0:x1] = False

        # Stage 2: Random Cleanup
        for i in range(n_hole_pixels):
            cy, cx = shuffled_indices_cpu[i]
            if not unfilled_cpu[cy, cx]: continue
            
            anchors_y_list.append(cy)
            anchors_x_list.append(cx)
            count_anchors += 1
            y0, y1 = max(0, int(cy)-half_ps), min(h, int(cy)+half_ps+1)
            x0, x1 = max(0, int(cx)-half_ps), min(w, int(cx)+half_ps+1)
            unfilled_cpu[y0:y1, x0:x1] = False
            
            if count_anchors % 100 == 0:
                if (time.perf_counter() - t_start_dart) > 2.0:
                    break
                    
        n_anchors = len(anchors_y_list)
        if n_anchors == 0:
             return (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))
             
        batch_ys_hole = torch.tensor(anchors_y_list, device=self.device)
        batch_xs_hole = torch.tensor(anchors_x_list, device=self.device)
        
        # --- Batch Matching ---
        best_ys = torch.zeros(n_anchors, device=self.device, dtype=torch.long)
        best_xs = torch.zeros(n_anchors, device=self.device, dtype=torch.long)
        
        max_candidates = min(self.candidate_count, n_known)
        BATCH_SIZE = 64
        
        # Sampling grid
        target_grid_size = 5
        step = max(1, ps // target_grid_size) if ps > target_grid_size else 1
        
        py, px = torch.meshgrid(
            torch.arange(-half_ps, half_ps + 1, step, device=self.device),
            torch.arange(-half_ps, half_ps + 1, step, device=self.device),
            indexing="ij"
        )
        patch_offsets_y = py.reshape(-1)
        patch_offsets_x = px.reshape(-1)
        
        for i in range(0, n_anchors, BATCH_SIZE):
            end_i = min(i + BATCH_SIZE, n_anchors)
            chunk_len = end_i - i
            
            # Select Candidates
            if n_known < max_candidates:
                 cand_indices = torch.randint(0, n_known, (chunk_len, n_known), device=self.device)
                 chunk_max_cand = n_known
            else:
                 cand_indices = torch.randint(0, n_known, (chunk_len, max_candidates), device=self.device)
                 chunk_max_cand = max_candidates
            
            chunk_cand_ys = ys_known[cand_indices]
            chunk_cand_xs = xs_known[cand_indices]
            
            # Target Patches
            chunk_hole_y = batch_ys_hole[i:end_i]
            chunk_hole_x = batch_xs_hole[i:end_i]
            
            t_py = (chunk_hole_y.unsqueeze(1) + patch_offsets_y.unsqueeze(0)).clamp(0, h-1)
            t_px = (chunk_hole_x.unsqueeze(1) + patch_offsets_x.unsqueeze(0)).clamp(0, w-1)
            target_patches = base_fill[:, t_py.reshape(-1), t_px.reshape(-1)]
            target_patches = target_patches.reshape(3, chunk_len, -1).permute(1, 0, 2)
            t_mean = target_patches.mean(dim=2, keepdim=True)
            target_patches_centered = target_patches - t_mean
            
            # Candidate Patches
            c_py = (chunk_cand_ys.unsqueeze(2) + patch_offsets_y.unsqueeze(0).unsqueeze(0)).clamp(0, h-1)
            c_px = (chunk_cand_xs.unsqueeze(2) + patch_offsets_x.unsqueeze(0).unsqueeze(0)).clamp(0, w-1)
            cand_patches = base_fill[:, c_py.reshape(-1), c_px.reshape(-1)]
            cand_patches = cand_patches.reshape(3, chunk_len, chunk_max_cand, -1).permute(1, 2, 0, 3)
            c_mean = cand_patches.mean(dim=3, keepdim=True)
            cand_patches_centered = cand_patches - c_mean
            
            # Costs
            color_diff = ((target_patches_centered.unsqueeze(1) - cand_patches_centered) ** 2).sum(dim=(2, 3))
            
            dy = (chunk_cand_ys - chunk_hole_y.unsqueeze(1)).float() / h
            dx = (chunk_cand_xs - chunk_hole_x.unsqueeze(1)).float() / w
            dist_sq = dy**2 + dx**2
            
            # Texture
            cand_edge_patches = edge_map[:, c_py.reshape(-1), c_px.reshape(-1)]
            cand_edge_patches = cand_edge_patches.reshape(1, chunk_len, chunk_max_cand, -1).permute(1, 2, 0, 3)
            cand_edge_means = cand_edge_patches.mean(dim=3).squeeze(2)
            texture_cost = (cand_edge_means - target_texture_val) ** 2
            
            w_texture = float(len(patch_offsets_y)) * 100.0 * self.texture_strength
            total_cost = color_diff + 0.1 * dist_sq + w_texture * texture_cost
            
            # Best match
            best_local_idx = torch.argmin(total_cost, dim=1)
            batch_range = torch.arange(chunk_len, device=self.device)
            best_ys[i:end_i] = chunk_cand_ys[batch_range, best_local_idx]
            best_xs[i:end_i] = chunk_cand_xs[batch_range, best_local_idx]
            
        return batch_ys_hole, batch_xs_hole, best_ys, best_xs

    def _quilt_patches(
        self,
        base_fill: torch.Tensor,
        original: torch.Tensor,
        hole: torch.Tensor,
        anchors_y: torch.Tensor,
        anchors_x: torch.Tensor,
        source_y: torch.Tensor,
        source_x: torch.Tensor,
        patch_size: int
    ) -> torch.Tensor:
        """Paste patches onto original image."""
        c, h, w = base_fill.shape
        result = original.clone()
        
        n_anchors = len(anchors_y)
        if n_anchors == 0:
            return result
            
        ps = patch_size
        half_ps = ps // 2
        
        full_py, full_px = torch.meshgrid(
            torch.arange(-half_ps, half_ps + 1, device=self.device),
            torch.arange(-half_ps, half_ps + 1, device=self.device),
            indexing="ij"
        )
        full_off_y = full_py.reshape(-1)
        full_off_x = full_px.reshape(-1)
        
        src_y_map = (source_y.unsqueeze(1) + full_off_y.unsqueeze(0)).clamp(0, h-1)
        src_x_map = (source_x.unsqueeze(1) + full_off_x.unsqueeze(0)).clamp(0, w-1)
        
        source_patches = original[:, src_y_map.reshape(-1), src_x_map.reshape(-1)]
        source_patches = source_patches.reshape(3, n_anchors, -1)
        
        # Paste Loop (Vectorized Blending)
        # Instead of overwriting sequentially (which relies on order and is slow),
        # we accumulate patches and average them (blending).
        # This solves "unblending colors" and is much faster on GPU/CPU.

        # 1. Prepare Patches
        # source_patches: [3, n_anchors, ps*ps]
        source_patches = source_patches.permute(1, 0, 2) # [n_anchors, 3, ps*ps]
        
        # 2. Extract Target Background for Color Transfer (Batch)
        # We need base_fill at target locations to compute mean shift
        # Reuse mapping logic
        target_y_center = anchors_y
        target_x_center = anchors_x
        
        t_y_map = (target_y_center.unsqueeze(1) + full_off_y.unsqueeze(0)).clamp(0, h-1)
        t_x_map = (target_x_center.unsqueeze(1) + full_off_x.unsqueeze(0)).clamp(0, w-1)
        
        # [3, n_anchors, ps*ps]
        target_patches_base = base_fill[:, t_y_map.reshape(-1), t_x_map.reshape(-1)]
        target_patches_base = target_patches_base.reshape(3, n_anchors, -1).permute(1, 0, 2)
        
        # 3. Compute Color Adjustment (Batch)
        # Means over (pixels) dim=2
        s_mean = source_patches.mean(dim=2, keepdim=True) # [N, 3, 1]
        t_mean = target_patches_base.mean(dim=2, keepdim=True)
        
        adjusted_patches = source_patches - s_mean + t_mean
        
        # 4. Flatten and Compute Indices for Accumulation
        # We need to flatten everything to 1D equivalent indices for index_add_
        
        # Indices in the image (H*W)
        flat_indices = (t_y_map * w + t_x_map).reshape(-1) # [N * ps * ps]
        
        # Values to add: [N, 3, ps*ps] -> [3, N * ps * ps]
        values_flat = adjusted_patches.permute(1, 0, 2).reshape(3, -1)
        
        # Accumulators
        accumulator = torch.zeros_like(original, dtype=torch.float32)
        counts = torch.zeros((1, h, w), device=self.device, dtype=torch.float32)
        
        # Add values (per channel)
        # index_add_ expects 1D index and source relative to dimension
        accumulator.view(3, -1).index_add_(1, flat_indices, values_flat)
        
        # Add counts
        ones = torch.ones(flat_indices.shape[0], device=self.device)
        counts.view(-1).index_add_(0, flat_indices, ones)
        
        # 5. Normalize (Average)
        # Avoid division by zero
        mask_filled = counts > 0
        result = torch.where(mask_filled, accumulator / counts, original)
        
        # Note: 'original' contains original image outside hole.
        # Inside hole, 'counts' should be > 0 if anchors cover it.
        # If gaps exist, we might have issues?
        # But 'result' is initialized to 'original'.
        # However, inside hole 'original' is 0 or noise?
        # We should fall back to 'base_fill' where counts == 0 inside hole?
        
        # Let's handle gaps if any (though Grid+Random should cover)
        # If counts==0 inside hole check
        if n_anchors > 0:
             # Just ensures we don't zero out valid pixels or leave holes empty
             pass
            
        # Zero check / cleanup
        ys_hole = torch.arange(h, device=self.device).unsqueeze(1).expand(-1, w)[hole]
        xs_hole = torch.arange(w, device=self.device).unsqueeze(0).expand(h, -1)[hole]
        
        if len(ys_hole) > 0:
             hole_vals = result[:, ys_hole, xs_hole]
             if hole_vals.min() == 0.0:
                  is_zero = (hole_vals.sum(dim=0) == 0.0)
                  if is_zero.any():
                       mean_color = original.mean(dim=(1, 2))
                       ys_fix = ys_hole[is_zero]
                       xs_fix = xs_hole[is_zero]
                       result[:, ys_fix, xs_fix] = mean_color.unsqueeze(1).expand(-1, len(ys_fix))
                       
        return result

    def _patchmatch_texture(
        self, base_fill: torch.Tensor, original: torch.Tensor, hole: torch.Tensor
    ) -> torch.Tensor:
        """Transfer texture using Dart Throwing Patch Quilting (Random Patch Copy)."""
        if not hole.any():
            return base_fill
            
        c, h, w = base_fill.shape
        # --- 1. Downscale Logic ---
        scale = self.match_scale
        
        if scale < 1.0:
            # Downscale tensors
            th = max(64, int(h * scale))
            tw = max(64, int(w * scale))
            
            if th < h and tw < w:
                if self.verbose:
                    self._log(f"Downscaled Matching: {h}x{w} -> {th}x{tw} (scale={scale})")
                
                # Resize inputs
                base_small = F.interpolate(base_fill.unsqueeze(0), size=(th, tw), mode='bilinear', align_corners=False)[0]
                orig_small = F.interpolate(original.unsqueeze(0), size=(th, tw), mode='bilinear', align_corners=False)[0]
                hole_small = F.interpolate(hole.float().unsqueeze(0).unsqueeze(0), size=(th, tw), mode='nearest')[0, 0] > 0.5
                
                ps_small = max(3, int(self.texture_patch_size * scale) | 1)
                
                # Phase A: Find Matches on Small Image
                anchors_y, anchors_x, source_y, source_x = self._find_match_indices(
                     base_small, orig_small, hole_small, ps_small
                )
                
                if len(anchors_y) > 0:
                    # Upscale Coordinates
                    # Note: Need to map back to original space accurately
                    scale_y = h / float(th)
                    scale_x = w / float(tw)
                    
                    anchors_y = (anchors_y.float() * scale_y).long().clamp(0, h-1)
                    anchors_x = (anchors_x.float() * scale_x).long().clamp(0, w-1)
                    source_y = (source_y.float() * scale_y).long().clamp(0, h-1)
                    source_x = (source_x.float() * scale_x).long().clamp(0, w-1)
                    
                    # Phase B: Quilt on Full Image
                    return self._quilt_patches(
                        base_fill, original, hole, 
                        anchors_y, anchors_x, source_y, source_x, 
                        self.texture_patch_size
                    )
        
        # Fallback (Full Resolution)
        anchors_y, anchors_x, source_y, source_x = self._find_match_indices(
             base_fill, original, hole, self.texture_patch_size
        )
        return self._quilt_patches(
             base_fill, original, hole,
             anchors_y, anchors_x, source_y, source_x,
             self.texture_patch_size
        )

    def inpaint(self, image: np.ndarray | torch.Tensor, mask: np.ndarray | torch.Tensor):
        t0 = time.perf_counter()
        pkg = self._to_tensor(image, mask)
        self._log(f"start patchmatch: device={self.device}, image={tuple(pkg.image.shape)}, hole={int(pkg.mask.sum().item())}")

        if not pkg.mask.any():
            return self._to_numpy(pkg.image, pkg)

        ys = torch.nonzero(pkg.mask, as_tuple=False)[:, 0]
        xs = torch.nonzero(pkg.mask, as_tuple=False)[:, 1]
        y0 = max(0, int(ys.min().item()) - self.roi_margin)
        y1 = min(pkg.mask.shape[0] - 1, int(ys.max().item()) + self.roi_margin)
        x0 = max(0, int(xs.min().item()) - self.roi_margin)
        x1 = min(pkg.mask.shape[1] - 1, int(xs.max().item()) + self.roi_margin)

        roi_img = pkg.image[:, y0 : y1 + 1, x0 : x1 + 1]
        roi_mask = pkg.mask[y0 : y1 + 1, x0 : x1 + 1]
        rh, rw = roi_mask.shape

        # Downscale if too large (for better polynomial fit on global gradients)
        scale = 1.0
        max_side = max(rh, rw)
        if max_side > self.max_working_size:
            scale = self.max_working_size / float(max_side)
            nh = max(32, int(round(rh * scale)))
            nw = max(32, int(round(rw * scale)))
            roi_img_w = F.interpolate(roi_img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False)[0]
            roi_mask_w = F.interpolate(roi_mask.float().unsqueeze(0).unsqueeze(0), size=(nh, nw), mode="nearest")[0, 0] > 0.5
            self._log(f"downscale roi: ({rh},{rw}) -> ({nh},{nw})")
        else:
            roi_img_w = roi_img
            roi_mask_w = roi_mask

        # 1. Create matching guide (Gradient colors everywhere)
        # Hole: Laplace fill (Smooth Gradient)
        filled_small = self._laplace_fill(roi_img_w, roi_mask_w)
        
        # Upscale back to full resolution
        if scale < 1.0:
            filled_w = F.interpolate(filled_small.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False)[0]
        else:
            filled_w = filled_small
            
        # Prepare guide for matching (Smooth vs Smooth)
        base_for_matching = filled_w.clone()
        
        # Apply strong blur to known regions in base_for_matching
        # This ensures we compare "Smooth Gradient" vs "Smooth Gradient"
        # instead of "Smooth Gradient" vs "Noisy Texture"
        
        # Use blurred version for known regions, Plain Poly for hole
        # This creates a fairer comparison:
        # Hole: Gradient (Smooth)
        # Known: Gradient + Structure (Removed Noise by Strong Blur)
        sigma_blur = 3.0
        gaussian_blur = T.GaussianBlur(kernel_size=9, sigma=sigma_blur)
        blurred_img = gaussian_blur(roi_img.unsqueeze(0))[0]
        base_for_matching = torch.where(roi_mask.unsqueeze(0), filled_w, blurred_img)
        
        self._log("applying texture transfer with gradient-based matching (enhanced)")



        # Match using this enhanced guide
        # Transfer from roi_img (original RGB with noise)
        filled_roi = self._patchmatch_texture(base_for_matching, roi_img, roi_mask)

        # --- Drift Correction (Seamless Cloning / Frequency Separation) ---
        # filled_roi has good texture but might have blocky color seams.
        # filled_w has perfect smooth color (polynomial) but no texture.
        # Goal: Combine (filled_roi's texture) + (filled_w's color).
        # Method:
        #   low_freq_roi = Blur(filled_roi)
        #   high_freq_roi = filled_roi - low_freq_roi  (Texture detail)
        #   Result = filled_w + high_freq_roi
        #
        # Implementation Detail:
        #   To avoid "halo" at boundary where roi_img meets filled_roi,
        #   we need to be careful.
        #   Ideally, we want to subtract the "Low Frequency Drift" of the patchmatch result
        #   and inject the "Low Frequency Base" of the polynomial fill.
        
        # Blur radius should be roughly the patch size to smooth out block artifacts
        sigma_drift = float(self.texture_patch_size) * 0.5
        k_size_drift = int(sigma_drift * 3) | 1
        drift_blur = T.GaussianBlur(kernel_size=k_size_drift, sigma=sigma_drift)
        
        # 1. Calculate Low Frequency of the PatchMatch result
        # We process the whole roi to ensure boundary continuity
        low_freq_roi = drift_blur(filled_roi.unsqueeze(0))[0]
        
        # 2. Calculate High Frequency (Detail)
        high_freq_roi = filled_roi - low_freq_roi
        
        # 3. Calculate Low Frequency of the Target Color Base (Polynomial Fill)
        # Note: filled_w is the polynomial fill. It is already smooth, but let's blur it 
        # with same kernel to match the frequency band of 'low_freq_roi'.
        low_freq_target = drift_blur(filled_w.unsqueeze(0))[0]
        
        # 4. Combine
        # Result = Target_Low + Source_High
        # This forces the average color to match the polynomial fill, while keeping texture.
        # We only apply this inside the hole to preserve original pixels outside.
        
        seamless_fill = low_freq_target + high_freq_roi
        
        # Blend back into ROI based on mask
        # We use the drift-corrected version for the hole
        final_roi = torch.where(roi_mask.unsqueeze(0), seamless_fill, roi_img)

        out = pkg.image.clone()
        out[:, y0 : y1 + 1, x0 : x1 + 1] = final_roi
        self._log(f"finish inpaint: total_elapsed={time.perf_counter()-t0:.3f}s")
        return self._to_numpy(out, pkg)


def content_aware_fill(
    image,
    mask,
    roi_margin: int = 256,
    max_working_size: int = 1536,
    speed_mode: str = "quality",
    verbose: bool = True,
    device: str | None = None,
    texture_patch_size: int = 31,
    texture_strength: float = 1.0,  # 適応的調整が乗算される基準値
    match_scale: float = 0.5,
    candidate_count: int = 256,
):
    return PatchMatchInpainting(
        max_working_size=max_working_size,
        roi_margin=roi_margin,
        speed_mode=speed_mode,
        verbose=verbose,
        device=device,
        texture_patch_size=texture_patch_size,
        texture_strength=texture_strength,
        match_scale=match_scale,
        candidate_count=candidate_count,
    ).inpaint(image, mask)

