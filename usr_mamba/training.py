"""Conditional adversarial training and independent validation for USR-Mamba."""
import csv
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from common.config import save_config
from common.files import new_output
from .dataset import read_manifest, PairedImages
from .network import USRMamba


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels, config):
        super().__init__()
        width, depth, kernel = (config[key] for key in ("discriminator_width", "discriminator_depth", "discriminator_kernel"))
        padding = (kernel - 1) // 2
        maximum = config["discriminator_max_multiplier"]
        layers = [nn.Conv2d(input_channels, width, kernel, 2, padding), nn.LeakyReLU(config["discriminator_slope"], True)]
        for level in range(1, depth + 1):
            before, after = width * min(2 ** (level - 1), maximum), width * min(2 ** level, maximum)
            layers.extend((nn.Conv2d(before, after, kernel, 2 if level < depth else 1, padding),
                           nn.InstanceNorm2d(after, eps=config["discriminator_norm_epsilon"]),
                           nn.LeakyReLU(config["discriminator_slope"], True)))
        layers.append(nn.Conv2d(after, 1, kernel, padding=padding))
        self.layers = nn.Sequential(*layers)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, 0, config["initialization_sigma"])
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, image_pair):
        return self.layers(image_pair)


class ReplayBuffer:
    def __init__(self, capacity, probability):
        self.capacity, self.probability = capacity, probability
        self.samples = []

    def query(self, batch):
        if self.capacity == 0:
            return batch
        output = []
        for image in batch.detach().split(1):
            if len(self.samples) < self.capacity:
                self.samples.append(image.clone())
                output.append(image)
            elif random.random() > 1 - self.probability:
                index = random.randrange(self.capacity)
                output.append(self.samples[index].clone())
                self.samples[index] = image.clone()
            else:
                output.append(image)
        return torch.cat(output)


def validate(config):
    positive = ("batch_size", "constant_epochs", "learning_rate", "save_every", "discriminator_width", "discriminator_depth")
    if any(config[key] <= 0 for key in positive):
        raise ValueError("Training counts and learning rate must be positive")
    nonnegative = ("decay_epochs", "workers", "l1_weight", "adversarial_weight", "replay_capacity")
    if any(config[key] < 0 for key in nonnegative) or not 0 <= config["replay_probability"] <= 1:
        raise ValueError("Invalid training parameter")


def train(settings, manifest, data_root, output_root, variant, device):
    config = settings["training"]
    validate(config)
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    pairs = read_manifest(manifest, data_root)
    model = USRMamba(settings["usr_mamba"], variant).to(device)
    datasets = {split: PairedImages(pairs, split, config, settings["seed"], model.divisor) for split in ("train", "val")}
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise ValueError("Training and validation must both be nonempty")
    loaders = {split: DataLoader(dataset, batch_size=config["batch_size"], shuffle=split == "train",
        num_workers=config["workers"]) for split, dataset in datasets.items()}
    paired_channels = settings["usr_mamba"]["input_channels"] + settings["usr_mamba"]["output_channels"]
    discriminator = PatchDiscriminator(paired_channels, config).to(device)
    optimizers = [torch.optim.Adam(network.parameters(), lr=config["learning_rate"], betas=tuple(config["adam_betas"]))
                  for network in (model, discriminator)]
    replay = ReplayBuffer(config["replay_capacity"], config["replay_probability"])
    absolute, squared = nn.L1Loss(), nn.MSELoss()
    output = new_output(output_root)
    save_config(settings, output / "parameters.json")
    total = config["constant_epochs"] + config["decay_epochs"]
    with (output / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "training_l1", "validation_l1", "learning_rate"))
        for epoch in range(1, total + 1):
            fraction = 1 - max(0, epoch - config["constant_epochs"]) / (config["decay_epochs"] + 1)
            rate = config["learning_rate"] * fraction
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] = rate
            model.train()
            discriminator.train()
            train_sum, train_count = 0.0, 0
            for low, high, _ in loaders["train"]:
                low, high = low.to(device), high.to(device)
                prediction = model(low)
                if config["adversarial_weight"]:
                    optimizers[1].zero_grad(set_to_none=True)
                    fake = discriminator(replay.query(torch.cat((low, prediction.detach()), 1)))
                    real = discriminator(torch.cat((low, high), 1))
                    loss_d = (squared(fake, torch.zeros_like(fake)) + squared(real, torch.ones_like(real))) / 2
                    (loss_d * config["adversarial_weight"]).backward()
                    optimizers[1].step()
                discriminator.requires_grad_(False)
                optimizers[0].zero_grad(set_to_none=True)
                reconstruction = absolute(prediction, high)
                loss = reconstruction * config["l1_weight"]
                if config["adversarial_weight"]:
                    score = discriminator(torch.cat((low, prediction), 1))
                    loss = loss + squared(score, torch.ones_like(score)) * config["adversarial_weight"]
                loss.backward()
                optimizers[0].step()
                discriminator.requires_grad_(True)
                train_sum += reconstruction.item() * len(low)
                train_count += len(low)
            model.eval()
            val_sum, val_count = 0.0, 0
            with torch.no_grad():
                for low, high, _ in loaders["val"]:
                    val_sum += absolute(model(low.to(device)), high.to(device)).item() * len(low)
                    val_count += len(low)
            writer.writerow((epoch, train_sum / train_count, val_sum / val_count, rate))
            stream.flush()
            print(f"Epoch {epoch}/{total}: validation L1 {val_sum / val_count:.6f}", flush=True)
            if epoch % config["save_every"] == 0 or epoch == total:
                torch.save({"model_config": settings["usr_mamba"], "variant": variant, "epoch": epoch,
                    "generator": model.state_dict(), "discriminator": discriminator.state_dict(),
                    "optimizer_g": optimizers[0].state_dict(), "optimizer_d": optimizers[1].state_dict()},
                    output / f"usr_mamba_{epoch:03d}.pt")
    return output
