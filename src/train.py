"""
Training script for Rock-Paper-Scissors classification model.
Can be run standalone to train the initial model.
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import ImagePreprocessor
from model import RPSModel


def main():
    parser = argparse.ArgumentParser(description='Train Rock-Paper-Scissors classification model')
    parser.add_argument('--train-dir', type=str, default='data/raw/train',
                        help='Directory containing training images')
    parser.add_argument('--test-dir', type=str, default='data/raw/test',
                        help='Directory containing test images')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=150,
                        help='Image size (width and height)')
    parser.add_argument('--output', type=str, default='models/rps_model.h5',
                        help='Output path for saved model')
    parser.add_argument('--no-transfer-learning', action='store_true',
                        help='Disable transfer learning (use custom CNN)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Rock-Paper-Scissors Model Training")
    print("="*60)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize preprocessor
    print("\n1. Initializing preprocessor...")
    preprocessor = ImagePreprocessor(img_size=(args.img_size, args.img_size), normalize=True)
    
    # Load data
    print("\n2. Loading data...")
    X_train, y_train, X_test, y_test = preprocessor.prepare_training_data(
        args.train_dir, args.test_dir
    )
    
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Image shape: {X_train.shape[1:]}")
    
    # Create model
    print("\n3. Creating model...")
    rps_model = RPSModel(img_size=(args.img_size, args.img_size), num_classes=3)
    rps_model.build_model(use_transfer_learning=not args.no_transfer_learning)
    
    if not args.no_transfer_learning:
        print("   Using MobileNetV2 (Transfer Learning)")
    else:
        print("   Using custom CNN architecture")
    
    # Train model
    print("\n4. Training model...")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    
    history = rps_model.train(
        X_train, y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_early_stopping=True,
        use_reduce_lr=True
    )
    
    # Evaluate model
    print("\n5. Evaluating model...")
    metrics = rps_model.evaluate(X_test, y_test)
    
    print("\n" + "="*60)
    print("TRAINING RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"Loss:      {metrics['loss']:.4f}")
    print("="*60)
    
    # Save model
    print(f"\n6. Saving model to {args.output}...")
    rps_model.save_model(args.output)
    
    print("\nTraining completed successfully!")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()

