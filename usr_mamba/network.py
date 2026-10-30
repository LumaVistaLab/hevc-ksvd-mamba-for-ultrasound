"""USR-Mamba: convolutional encoding, dual-domain SSM and residual reconstruction."""
import torch
from torch import nn


def scan_indices(height, width, order, device):
    if order == "raster":
        return torch.arange(height * width, device=device)
    if order not in ("outside_in", "inside_out"):
        raise ValueError("Unknown scan order")
    grid = torch.arange(height * width).reshape(height, width)
    rings = []
    while grid.numel():
        rings.extend(grid[0].tolist())
        if grid.shape[0] > 1:
            rings.extend(grid[1:, -1].tolist())
        if min(grid.shape) > 1:
            rings.extend(grid[-1, :-1].flip(0).tolist())
        if grid.shape[1] > 1 and grid.shape[0] > 2:
            rings.extend(grid[1:-1, 0].flip(0).tolist())
        grid = grid[1:-1, 1:-1]
    if order == "inside_out":
        rings.reverse()
    return torch.tensor(rings, dtype=torch.long, device=device)


def batch_norm(channels, config):
    return nn.BatchNorm2d(channels, eps=config["norm_epsilon"], momentum=config["batch_momentum"])


class ResidualConvolution(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        radius = config["feature_kernel"] // 2
        def convolution():
            return [nn.ReflectionPad2d(radius), nn.Conv2d(channels, channels,
                config["feature_kernel"], bias=config["feature_bias"]), batch_norm(channels, config)]
        self.layers = nn.Sequential(*convolution(), nn.ReLU(True), *convolution())

    def forward(self, features):
        return features + self.layers(features)


class FeatureFusion(nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=config["feature_bias"]), batch_norm(channels, config)
        )
        radius = config["feature_kernel"] // 2
        self.layers = nn.Sequential(
            nn.Conv2d(2 * channels, channels, config["feature_kernel"], padding=radius, bias=config["feature_bias"]),
            batch_norm(channels, config), nn.ReLU(),
            nn.Conv2d(channels, channels, config["feature_kernel"], padding=radius, bias=config["feature_bias"]), batch_norm(channels, config))

    def forward(self, features):
        return torch.relu(self.layers(features) + self.shortcut(features))


class DualDomainState(nn.Module):
    def __init__(self, channels, config, frequency_branch, order):
        super().__init__()
        from mamba_ssm import Mamba
        state = dict(d_state=config["state_size"], d_conv=config["state_conv"], expand=config["state_expansion"])
        self.spatial_norm = nn.LayerNorm(channels, eps=config["norm_epsilon"])
        self.spatial_ssm = Mamba(d_model=channels, **state)
        self.frequency_branch = frequency_branch
        if frequency_branch:
            self.frequency_norm = nn.LayerNorm(2 * channels, eps=config["norm_epsilon"])
            self.frequency_ssm = Mamba(d_model=2 * channels, **state)
        self.merge = nn.Conv2d(2 * channels, channels, 1) if frequency_branch else nn.Identity()
        self.order = order
        self._indices = None
        self._shape = None

    def forward(self, features):
        batch, channels, height, width = features.shape
        cache_key = (height, width, features.device)
        if cache_key != self._shape:
            indices = scan_indices(height, width, self.order, features.device)
            self._indices = (indices, indices.argsort())
            self._shape = cache_key
        indices, inverse = self._indices
        def sequence(image):
            return image.flatten(2)[:, :, indices].transpose(1, 2)
        def image(tokens):
            return tokens.transpose(1, 2)[:, :, inverse].reshape(batch, channels, height, width)
        spatial = image(self.spatial_ssm(self.spatial_norm(sequence(features))))
        if not self.frequency_branch:
            return spatial
        spectrum = torch.fft.fftshift(torch.fft.fft2(features), dim=(-2, -1))
        tokens = torch.cat((sequence(spectrum.real), sequence(spectrum.imag)), dim=-1)
        real, imaginary = self.frequency_ssm(self.frequency_norm(tokens)).chunk(2, dim=-1)
        transformed = torch.complex(image(real), image(imaginary))
        frequency = torch.fft.ifft2(torch.fft.ifftshift(transformed, dim=(-2, -1))).real
        return self.merge(torch.cat((spatial, frequency), dim=1))


class StateRefinement(nn.Module):
    def __init__(self, channels, config, frequency_branch, refinement, order):
        super().__init__()
        self.state = DualDomainState(channels, config, frequency_branch, order)
        self.fusion = FeatureFusion(channels, config)
        self.refinement = ResidualConvolution(channels, config) if refinement else nn.Identity()

    def forward(self, features):
        return self.refinement(self.fusion(torch.cat((features, self.state(features)), dim=1)))


VARIANTS = ("full", "conv_bottleneck", "no_fourier", "no_refinement", "raster_scan")


class USRMamba(nn.Module):
    def __init__(self, config, variant):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError("Unknown USR-Mamba variant")
        widths = config["channels"]
        if len(widths) < 2 or any(c < 1 for c in widths):
            raise ValueError("Specify positive encoder widths")
        kernel, feature_kernel = config["image_kernel"], config["feature_kernel"]
        if kernel % 2 != 1 or feature_kernel % 2 != 1:
            raise ValueError("Convolution kernels must be odd")
        self.divisor = 2 ** (len(widths) - 1)
        self.encoders = nn.ModuleList([nn.Sequential(nn.ReflectionPad2d(kernel // 2),
            nn.Conv2d(config["input_channels"], widths[0], kernel, bias=config["feature_bias"]),
            batch_norm(widths[0], config), nn.ReLU(True))])
        for before, after in zip(widths, widths[1:]):
            self.encoders.append(nn.Sequential(nn.Conv2d(before, after, feature_kernel, stride=2,
                padding=feature_kernel // 2, bias=config["feature_bias"]), batch_norm(after, config), nn.ReLU(True)))
        positions = set(config["ssm_positions"]) if variant != "conv_bottleneck" else set()
        order = "raster" if variant == "raster_scan" else config["scan_order"]
        self.bottleneck = nn.Sequential(*(StateRefinement(widths[-1], config,
            variant != "no_fourier", variant != "no_refinement", order)
            if index in positions else ResidualConvolution(widths[-1], config)
            for index in range(1, config["bottleneck_depth"] + 1)))
        self.decoders = nn.ModuleList()
        for before, after in zip(reversed(widths[1:]), reversed(widths[:-1])):
            self.decoders.append(nn.Sequential(nn.ConvTranspose2d(2 * before, after, feature_kernel,
                stride=2, padding=feature_kernel // 2, output_padding=1, bias=config["feature_bias"]),
                batch_norm(after, config), nn.ReLU(True)))
        self.output = nn.Sequential(nn.ReflectionPad2d(kernel // 2),
            nn.Conv2d(2 * widths[0], config["output_channels"], kernel, bias=config["output_bias"]), nn.Tanh())

    def forward(self, image):
        if image.shape[-1] % self.divisor or image.shape[-2] % self.divisor:
            raise ValueError("Image dimensions must divide evenly through the encoder")
        features = []
        current = image
        for encoder in self.encoders:
            current = encoder(current)
            features.append(current)
        current = self.bottleneck(current)
        for decoder, skip in zip(self.decoders, reversed(features[1:])):
            current = decoder(torch.cat((current, skip), dim=1))
        return self.output(torch.cat((current, features[0]), dim=1))
