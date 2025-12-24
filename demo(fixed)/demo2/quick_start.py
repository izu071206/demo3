"""
Quick Start Script
Chạy toàn bộ pipeline: Dataset Generation -> Training -> Evaluation
File: quick_start.py (đặt ở root folder demo2/)
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def run_command(cmd, description):
    """Run shell command"""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with error code {e.returncode}")
        return False


def check_environment():
    """Check if environment is ready"""
    print_section("STEP 1: CHECKING ENVIRONMENT")
    
    # Check Python version
    logger.info(f"Python version: {sys.version}")
    
    # Check if required directories exist
    required_dirs = ['data/benign', 'data/obfuscated', 'models', 'config']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            logger.warning(f"Directory not found: {dir_path}")
    
    # Count samples
    benign_count = len(list(Path('data/benign').rglob('*.exe'))) if os.path.exists('data/benign') else 0
    obfuscated_count = len(list(Path('data/obfuscated').rglob('*.exe'))) if os.path.exists('data/obfuscated') else 0
    
    logger.info(f"Benign samples found: {benign_count}")
    logger.info(f"Obfuscated samples found: {obfuscated_count}")
    
    if benign_count < 5 or obfuscated_count < 5:
        logger.warning("⚠️  WARNING: You need at least 5 samples of each type for good training!")
        logger.warning("   Please add more samples to data/benign/ and data/obfuscated/")
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    return True


def generate_dataset():
    """Generate dataset with improved features"""
    print_section("STEP 2: GENERATING DATASET WITH ADVANCED FEATURES")
    
    cmd = [
        sys.executable,
        'src/dataset/generate_dataset_improved.py',
        '--config', 'config/dataset_config.yaml'
    ]
    
    return run_command(cmd, "Dataset Generation")


def train_models():
    """Train all models"""
    print_section("STEP 3: TRAINING MODELS (Base + Ensemble + Family Classifier)")
    
    cmd = [sys.executable, 'src/models/train_improved.py']
    
    return run_command(cmd, "Model Training")


def test_models():
    """Quick test of trained models"""
    print_section("STEP 4: TESTING MODELS")
    
    # Check if test data exists
    test_data = 'data/processed/test_features.pkl'
    if not os.path.exists(test_data):
        logger.warning(f"Test data not found: {test_data}")
        return False
    
    # Check which models are available
    models_to_test = []
    if os.path.exists('models/ensemble_model_metadata.pkl'):
        models_to_test.append(('ensemble', 'Ensemble Model'))
    if os.path.exists('models/combined_model.pkl'):
        models_to_test.append(('combined', 'Combined Model'))
    
    if not models_to_test:
        logger.warning("No trained models found!")
        return False
    
    logger.info(f"Found {len(models_to_test)} model(s) to test")
    
    # Quick evaluation
    try:
        import pickle
        import numpy as np
        
        with open(test_data, 'rb') as f:
            X_test, y_test = pickle.load(f)
        
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"  Benign: {np.sum(y_test == 0)}")
        logger.info(f"  Obfuscated: {np.sum(y_test == 1)}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error testing models: {e}")
        return False


def show_results():
    """Show training results"""
    print_section("STEP 5: RESULTS SUMMARY")
    
    # Check for evaluation results
    eval_dir = 'data/evaluation_results'
    if os.path.exists(eval_dir):
        results = list(Path(eval_dir).glob('*_metrics.json'))
        logger.info(f"Found {len(results)} evaluation results")
        
        if results:
            import json
            
            print("\nModel Performance:")
            print("-" * 60)
            
            for result_file in results:
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    model_name = data.get('model_name', result_file.stem)
                    metrics = data.get('metrics', {})
                    
                    print(f"\n{model_name.upper()}:")
                    print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
                    print(f"  Precision: {metrics.get('precision', 0):.4f}")
                    print(f"  Recall:    {metrics.get('recall', 0):.4f}")
                    print(f"  F1-Score:  {metrics.get('f1_score', 0):.4f}")
                
                except Exception as e:
                    logger.warning(f"Error reading {result_file}: {e}")
        
        print("\n" + "-" * 60)
    
    # Show trained models
    print("\n📦 Trained Models:")
    models_dir = Path('models')
    if models_dir.exists():
        models = list(models_dir.glob('*.pkl')) + list(models_dir.glob('*.json')) + list(models_dir.glob('*.pt'))
        for model in sorted(models):
            size = model.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✓ {model.name} ({size:.2f} MB)")
    
    # Show next steps
    print("\n📝 Next Steps:")
    print("  1. Test with your own samples:")
    print("     python test_single_file.py path/to/your/sample.exe")
    print("  2. Run batch testing:")
    print("     python batch_test.py path/to/test/directory/")
    print("  3. Start dashboard:")
    print("     python main.py dashboard")
    print("  4. View detailed results:")
    print("     ls data/evaluation_results/")


def main():
    """Main quick start pipeline"""
    print("\n")
    print("="*70)
    print("  🚀 QUICK START - IMPROVED OBFUSCATION DETECTION SYSTEM")
    print("="*70)
    print("\nThis script will:")
    print("  1. Check your environment")
    print("  2. Generate dataset with advanced features")
    print("  3. Train all models (RF, XGB, NN, Ensemble, Family Classifier)")
    print("  4. Evaluate models")
    print("  5. Show results")
    print("\nEstimated time: 15-30 minutes (depending on dataset size)")
    print("\n" + "="*70 + "\n")
    
    response = input("Ready to start? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Run pipeline
    success = True
    
    # Step 1: Check environment
    if not check_environment():
        logger.error("Environment check failed. Please fix issues and try again.")
        return
    
    # Step 2: Generate dataset
    if success:
        success = generate_dataset()
        if not success:
            logger.error("Dataset generation failed. Check logs above.")
            return
    
    # Step 3: Train models
    if success:
        success = train_models()
        if not success:
            logger.error("Model training failed. Check logs above.")
            return
    
    # Step 4: Test models
    if success:
        success = test_models()
    
    # Step 5: Show results
    show_results()
    
    # Final message
    print("\n" + "="*70)
    if success:
        print("  ✅ QUICK START COMPLETED SUCCESSFULLY!")
    else:
        print("  ⚠️  COMPLETED WITH WARNINGS - Check logs above")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()