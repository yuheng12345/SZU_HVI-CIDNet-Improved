import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

input_dir = Path("datasets/DICM")
comparison_dir = Path("comparison_2x2")
comparison_dir.mkdir(exist_ok=True)

scripts = [
    ("eval_hf.py", Path("output_hf")),
    ("scripts/eval_hf_adaptive_linear.py", Path("output_hf_adaptive")),
    ("scripts/eval_hf_adaptive.py", Path("output_hf_adaptive")),
]

for script, _ in scripts:
    if not Path(script).is_file():
        raise SystemExit("找不到脚本：{}".format(script))

images = sorted(
    p for p in input_dir.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

if not images:
    raise SystemExit("没有找到图片，请检查 datasets/DICM。")

print("找到 {} 张图片。".format(len(images)), flush=True)

env = os.environ.copy()
env["HF_ENDPOINT"] = "https://hf-mirror.com"
env["CUDA_VISIBLE_DEVICES"] = "2"

for index, image_path in enumerate(images, 1):
    destination = comparison_dir / (image_path.stem + "_comparison.png")

    if destination.exists():
        print("[{}/{}] 已存在，跳过：{}".format(index, len(images), destination), flush=True)
        continue

    print("[{}/{}] 处理 {}".format(index, len(images), image_path.name), flush=True)

    with Image.open(str(image_path)) as img:
        panels = [img.convert("RGB")]

    # 每次推理后立即读入结果，避免两个改进脚本的同名输出互相覆盖。
    for script, output_dir in scripts:
        command = [
            sys.executable, "-u", script,
            "--path", "Fediory/HVI-CIDNet-LOLv1-wperc",
            "--input_img", str(image_path),
            "--alpha_s", "1.0",
            "--alpha_i", "1.0",
            "--gamma", "1.0",
        ]
        subprocess.run(command, env=env, check=True)

        result_path = output_dir / image_path.name
        with Image.open(str(result_path)) as img:
            panels.append(img.convert("RGB"))

    width, height = panels[0].size
    if any(panel.size != (width, height) for panel in panels):
        raise SystemExit("{} 的结果尺寸不一致。".format(image_path.name))

    label_height = 28
    canvas = Image.new(
        "RGB", (width * 2, (height + label_height) * 2), "white"
    )
    draw = ImageDraw.Draw(canvas)

    labels = ["Original", "CIDNet", "Linear fusion", "Sigmoid fusion"]
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x = (i % 2) * width
        y = (i // 2) * (height + label_height)
        draw.text((x + 8, y + 7), label, fill="black")
        canvas.paste(panel, (x, y + label_height))

    canvas.save(str(destination))
    print("已保存：{}".format(destination), flush=True)

print("处理完成。", flush=True)
