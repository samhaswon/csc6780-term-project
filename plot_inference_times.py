"""
Plot model inference timings per resolution.
"""
import matplotlib.pyplot as plt

models = {
    "BiRefNet_lite": [0.6290, 0.9046, 2.6359, 12.5315, 15.6293, 27.5822, 62.9691],
    "U²-Net": [0.3108, 0.4828, 1.2603, 5.5562, 9.0918, 14.7011, 21.2722],
    "U²-NetP": [0.1559, 0.2331, 0.7607, 3.1737, 4.7568, 8.6204, 12.4256],
    "StraightU²Net": [0.0741, 0.1203, 0.3765, 1.9290, 3.0581, 5.6680, 7.7672],
    "DeepLabV3MobileNetV3": [0.0255, 0.0374, 0.0829, 0.4163, 0.6279, 1.1827, 1.9184],
}

resolutions = [256, 320, 512, 1024, 1280, 1728, 2048]

plt.figure(figsize=(10, 6))
plt.style.use('ggplot')
for name, times in models.items():
    plt.plot(resolutions, times, marker='o', label=name)

plt.xlabel("Resolution (square, px)")
plt.ylabel("Time (s)")
plt.title("PyTorch model inference times vs resolution")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("plots/inference_times.png", dpi=300)
plt.show()
