from ultralytics import YOLO

def main():
    """
    This script fine-tunes the YOLOv11n model from the initial training run
    using stronger data augmentation to improve accuracy and robustness.
    """
    print("="*60)
    print("✨ STARTING FINE-TUNING with Heavy Augmentation (YOLOv11n)")
    print("="*60)

    # Path to the 'best.pt' file from your initial training run
    model_to_finetune = r"runs/train/yolo11n_fast_training/weights/best.pt"

    # Load the model
    model = YOLO(model_to_finetune)

    # Continue training with stronger augmentations
    model.train(
        # --- Core Settings ---
        data="config/data.yaml",   # your dataset yaml
        epochs=15,                         # set max epochs (early stop may cut earlier)
        patience=10,                       # 🔑 early stopping
        imgsz=416,                         # resolution
        batch=16,                           # safe for 4GB VRAM
        device=0,                          # use GPU
        project="runs",
        name="yolo11n_finetuned",

        # --- Performance & Augmentation Settings ---
        workers=2,                         # reduce workers to avoid CPU overload
        amp=True,
        cache=False,

        # --- Data Augmentations ---
        degrees=10.0,                      # rotation
        translate=0.1,                     # translation
        scale=0.2,                         # scaling
        shear=5.0,                         # less aggressive shear
        perspective=0.0005,                # mild perspective
        flipud=0.3,                        # vertical flip
        fliplr=0.5,                        # horizontal flip
        mosaic=1.0,                        # keep mosaic
        copy_paste=0.1                     # slight copy-paste augmentation
    )

    print("\n" + "="*60)
    print("✅✅✅ Fine-Tuning Complete! ✅✅✅")
    print("Your final, most robust model is in:")
    print("runs/train/yolo11n_finetuned/weights/best.pt")
    print("="*60)

if __name__ == "__main__":
    main()
