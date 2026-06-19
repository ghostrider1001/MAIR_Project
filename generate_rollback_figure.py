import urllib.request
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import os

def download_sample_image():
    filepath = "base_image.jpg"
    
    if not os.path.exists(filepath):
        print("ERROR: Please place an image named 'base_image.jpg' in your MAIR_Project folder.")
        # Fallback synthetic image just so it doesn't crash
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.circle(img, (200, 200), 120, (200, 150, 100), -1)
    else:
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize for consistency
    img = cv2.resize(img, (300, 300))
    return img

def create_images(img):
    # 1. Dark Input (Simulate extreme low-light)
    input_img = (img * 0.15).astype(np.uint8)
    
    # 2. Hallucinated output (Simulate a broken curve estimator)
    # Boost brightness insanely, add neon noise, clip
    broken_img = (img * 2.5).clip(0, 255).astype(np.uint8)
    # Add severe chromatic noise
    noise = np.random.normal(50, 80, broken_img.shape).astype(np.int16)
    noise[:, :, 0] = noise[:, :, 0] * 1.5 # heavy red noise
    broken_img = np.clip(broken_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return input_img, broken_img

def build_flowchart():
    img = download_sample_image()
    input_img, broken_img = create_images(img)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor('white')
    
    # --- Box 1: Input ---
    axes[0].imshow(input_img)
    axes[0].set_title("1. Stage Input\n(Severe Low-Light)", fontsize=14, fontweight='bold', pad=15)
    axes[0].axis('off')
    
    # --- Box 2: Failed Expert ---
    axes[1].imshow(broken_img)
    axes[1].set_title("2. Expert Output\n(Catastrophic Hallucination)", fontsize=14, fontweight='bold', pad=15)
    axes[1].axis('off')
    
    # Draw Big Red X on middle image
    axes[1].plot([0, 300], [0, 300], color='red', linewidth=8, alpha=0.8)
    axes[1].plot([300, 0], [0, 300], color='red', linewidth=8, alpha=0.8)
    axes[1].text(150, 150, "Quality Gate Rejected\nSSIM < 0.50", 
                 color='white', fontsize=14, fontweight='bold', ha='center', va='center',
                 bbox=dict(facecolor='red', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))
    
    # --- Box 3: Safe Rollback ---
    axes[2].imshow(input_img)
    axes[2].set_title("3. Preserved Output\n(Deterministic Rollback)", fontsize=14, fontweight='bold', color='green', pad=15)
    axes[2].axis('off')
    
    # Add border around preserved output
    rect = patches.Rectangle((0,0), 299, 299, linewidth=6, edgecolor='green', facecolor='none')
    axes[2].add_patch(rect)
    
    # Add Arrows
    # Arrow 1 -> 2
    fig.add_artist(patches.ConnectionPatch(xyA=(1.0, 0.5), xyB=(0.0, 0.5), 
                                           coordsA='axes fraction', coordsB='axes fraction', 
                                           axesA=axes[0], axesB=axes[1],
                                           color="black", arrowstyle="-|>,head_width=0.4,head_length=0.6", linewidth=3))
    
    # Arrow 2 -> 3 (Rejected Path)
    fig.add_artist(patches.ConnectionPatch(xyA=(1.0, 0.5), xyB=(0.0, 0.5), 
                                           coordsA='axes fraction', coordsB='axes fraction', 
                                           axesA=axes[1], axesB=axes[2],
                                           color="red", linestyle="--", arrowstyle="-|>,head_width=0.4,head_length=0.6", linewidth=3))
    
    # Arrow 1 -> 3 (Safe Rollback Path)
    fig.add_artist(patches.ConnectionPatch(xyA=(0.5, 0.0), xyB=(0.5, 0.0), 
                                           coordsA='axes fraction', coordsB='axes fraction', 
                                           axesA=axes[0], axesB=axes[2],
                                           connectionstyle="bar,fraction=-0.15",
                                           color="green", arrowstyle="-|>,head_width=0.4,head_length=0.6", linewidth=4))
    
    plt.suptitle("MAIR+ v2: Quality Gate Rollback Mechanism", fontsize=18, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    output_path = "rollback_figure.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Flowchart successfully generated and saved to {output_path}")

if __name__ == "__main__":
    build_flowchart()
