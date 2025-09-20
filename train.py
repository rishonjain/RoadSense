from ultralytics import YOLO
import os

def main():
    # This script replaces the old fine-tuning process with a more powerful
    # two-stage training regimen for the medium (yolov8m) model.

    # --- STAGE 1: Initial Training of the Medium Model ---
    print("="*50)
    print("🚀 STARTING STAGE 1: Initial Training (YOLOv8-Medium)")
    print("="*50)

    # Load the medium-sized YOLOv8 model.
    # The 'yolov8m.pt' weights will be downloaded automatically on the first run.
    model_stage1 = YOLO("yolov8m.pt")

    # Train the base model. The results will be saved in 'runs/yolov8m_initial_training'.
    results_stage1 = model_stage1.train(
        data="config/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,  # IMPORTANT: Lower this to 8 or 4 if you get GPU "Out of Memory" errors.
        device=0,
        project="runs",
        name="yolov8m_initial_training"
    )

    # --- STAGE 2: Fine-Tuning with Enhanced Augmentation ---
    print("\n" + "="*50)
    print("✨ STARTING STAGE 2: Fine-Tuning with Enhanced Augmentation")
    print("="*50)

    # Automatically get the path to the best model from Stage 1
    best_model_path_stage1 = results_stage1.save_dir / 'weights' / 'best.pt'
    
    # Check if the best model from Stage 1 was created successfully
    if not os.path.exists(best_model_path_stage1):
        print(f"Error: Could not find the best model from Stage 1 at {best_model_path_stage1}")
        print("Aborting Stage 2 fine-tuning.")
        return

    # Load the newly trained model from Stage 1 to continue training
    model_stage2 = YOLO(best_model_path_stage1)
    
    # Fine-tune the model with heavy augmentation for fewer epochs
    model_stage2.train(
        data="config/data.yaml",
        epochs=25,
        imgsz=640,
        batch=16,
        device=0,
        project="runs",
        name="yolov8m_finetuned", # The final, best model will be saved here

        # --- Enhanced Data Augmentation Settings ---
        degrees=15.0,
        translate=0.1,
        scale=0.2,
        shear=10.0,
        perspective=0.001,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        copy_paste=0.1
    )
    
    print("\n" + "="*50)
    print("✅✅✅ Training and Fine-Tuning Complete! ✅✅✅")
    print(f"Your final, best model is in: runs/yolov8m_finetuned/weights/")
    print("You can now safely delete the old 'runs/train5' folder to save space.")
    print("="*50)

if __name__ == "__main__":
    main()