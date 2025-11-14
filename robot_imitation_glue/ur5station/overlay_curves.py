from PIL import Image
import os
import glob

def overlay_curves(curve_dir, opacity=0.5):
    # Get all curve png files
    files = sorted(glob.glob(os.path.join(curve_dir, "curve_episode*.png")))
    if not files:
        print(f"No curve PNGs found in {curve_dir}")
        return

    # Open the first one to get size
    base = Image.open(files[0]).convert("RGBA")
    # Apply opacity to first
    if opacity < 1.0:
        alpha = base.split()[3].point(lambda p: int(p * opacity))
        base.putalpha(alpha)

    # Overlay all others
    for f in files[1:]:
        img = Image.open(f).convert("RGBA")
        if opacity < 1.0:
            alpha = img.split()[3].point(lambda p: int(p * opacity))
            img.putalpha(alpha)
        base = Image.alpha_composite(base, img)

    # --- Save transparent version ---
    out_transparent = os.path.join(curve_dir, "all_curves_overlay.png")
    base.save(out_transparent, "PNG")
    print(f"Saved transparent overlay -> {out_transparent}")

    # --- Save white background version ---
    white_bg = Image.new("RGBA", base.size, (255, 255, 255, 255))  # white background
    white_bg = Image.alpha_composite(white_bg, base)
    out_white = os.path.join(curve_dir, "all_curves_overlay_white.png")
    white_bg.convert("RGB").save(out_white, "PNG")
    print(f"Saved white background overlay -> {out_white}")

if __name__ == "__main__":
    curve_dir = "datasets/spoofed_binary_clothes_hanger_EVAL_curves"
    overlay_curves(curve_dir)
