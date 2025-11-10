import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image

from rin_pytorch.utils.diffusion_utils import Scheduler


def load_image_as_tensor(image_path: str, device: torch.device, center_crop_square: bool = False) -> torch.Tensor:
	# Load image -> tensor in [-1, 1], shape [1, C, H, W]
	image = Image.open(image_path).convert("RGB")
	if center_crop_square:
		w, h = image.size
		crop_size = min(w, h)
		image = T.CenterCrop(crop_size)(image)
	to_tensor = T.ToTensor()
	x0 = to_tensor(image).unsqueeze(0).to(device)  # [0, 1]
	x0 = x0 * 2.0 - 1.0  # [-1, 1]
	return x0


@torch.no_grad()
def ddpm_forward_chain(
	x0: torch.Tensor,
	steps: int,
	save_every: int,
	scheduler: Scheduler,
	output_dir: Path,
	seed: int | None = None,
):
	"""
	Generate a forward DDPM Markov chain x_0 -> x_1 -> ... -> x_T and
	save intermediate noised images every `save_every` steps.
	We use the repository's time schedule gamma(t) as alpha_bar(t).
	"""
	device = x0.device
	gen = torch.Generator(device=device)
	if seed is not None:
		gen = gen.manual_seed(seed)

	# Save original
	save_image((x0.clamp(-1, 1) + 1.0) * 0.5, output_dir / f"noised_step_{0:04d}.png")

	# Precompute gamma(t_i) for i = 0..steps
	# t in [0, 1], uniformly spaced
	t_vals = torch.linspace(0.0, 1.0, steps + 1, device=device)
	# Shape as broadcastable to image: [steps+1, 1, 1, 1]
	t_vals_b = t_vals.view(steps + 1, 1, 1, 1)
	gamma_vals = scheduler.time_transform(t_vals_b)  # alpha_bar(t)
	gamma_vals = torch.clamp(gamma_vals, 1e-9, 1.0)

	x_t = x0
	for i in range(1, steps + 1):
		gamma_prev = gamma_vals[i - 1]
		gamma_now = gamma_vals[i]
		# alpha_t = alpha_bar(t_i) / alpha_bar(t_{i-1})
		alpha_t = torch.clamp(gamma_now / gamma_prev, 1e-9, 1.0)
		eps = torch.randn(x_t.shape, device=device, generator=gen)
		x_t = torch.sqrt(alpha_t) * x_t + torch.sqrt(1.0 - alpha_t) * eps

		if (i % save_every == 0) or (i == steps):
			save_image(
				(x_t.clamp(-1, 1) + 1.0) * 0.5,
				output_dir / f"noised_step_{i:04d}.png",
			)


def main():
	parser = argparse.ArgumentParser(description="Add DDPM forward noise to an image and save intermediate steps.")
	parser.add_argument("--image", type=str, required=True, help="Path to input image")
	parser.add_argument("--output_dir", type=str, default="noised_outputs", help="Directory to save frames")
	parser.add_argument("--steps", type=int, default=100, help="Total number of forward steps (T)")
	parser.add_argument("--save_every", type=int, default=10, help="Save every N steps")
	parser.add_argument(
		"--schedule",
		type=str,
		default="sigmoid@-3,3,0.9",
		help="Time schedule string used by Scheduler (e.g., 'sigmoid@-3,3,0.9', 'cosine', 'cosine@0.0,0.5,1.0')",
	)
	parser.add_argument("--center_crop_square", action="store_true", help="Center-crop input image to square before noising")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for noise")
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
	args = parser.parse_args()

	device = torch.device(args.device)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Load image
	x0 = load_image_as_tensor(args.image, device, center_crop_square=args.center_crop_square)

	# Build scheduler
	scheduler = Scheduler(args.schedule)

	# Run forward chain
	ddpm_forward_chain(
		x0=x0,
		steps=args.steps,
		save_every=args.save_every,
		scheduler=scheduler,
		output_dir=output_dir,
		seed=args.seed,
	)

	print(f"Saved frames to: {str(output_dir)}")


if __name__ == "__main__":
	main()


