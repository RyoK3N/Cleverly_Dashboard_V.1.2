#!/usr/bin/env python3
"""
Enhanced Lead Prediction System with Dynamic Path Management
Integrates with LeadNet v4 and Flask web interface
"""

from enhanced_lead_prediction import LeadNetPipeline, DynamicPathManager
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='LeadNet Prediction System')
    parser.add_argument('action', choices=['train', 'predict', 'status'], 
                       help='Action to perform')
    parser.add_argument('--sessions-dir', default='./sessions/sessions',
                       help='Path to sessions directory')
    
    args = parser.parse_args()
    
    pipeline = LeadNetPipeline(sessions_dir=args.sessions_dir)
    
    if args.action == 'train':
        print("Starting model training...")
        result = pipeline.train_model()
        if result["status"] == "success":
            print(f"✅ Training completed successfully!")
            print(f"Model saved to: {result['model_path']}")
            print(f"Training samples: {result['data_samples']}")
            print(f"Metrics: {result['metrics']}")
        else:
            print(f"❌ Training failed: {result['message']}")
            sys.exit(1)
    
    elif args.action == 'predict':
        print("Generating predictions...")
        try:
            predictions = pipeline.run_predictions()
            print(f"✅ Generated {len(predictions)} predictions")
            
            successful = len([p for p in predictions if "error" not in p])
            print(f"Successful predictions: {successful}/{len(predictions)}")
            
            # Display first few predictions
            for i, pred in enumerate(predictions[:5]):
                if "error" not in pred:
                    print(f"Lead {i+1}: {pred['prediction']} (confidence: {pred['confidence']:.3f})")
                else:
                    print(f"Lead {i+1}: Error - {pred.get('error', 'Unknown error')}")
                    
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            sys.exit(1)
    
    elif args.action == 'status':
        # Check file availability
        path_manager = DynamicPathManager(args.sessions_dir)
        available_files = path_manager.get_latest_files()
        
        print("📊 LeadNet System Status")
        print("=" * 40)
        print("Available Data Files:")
        for file_type, file_path in available_files.items():
            if file_path:
                print(f"  ✅ {file_type}: {file_path}")
            else:
                print(f"  ❌ {file_type}: Not found")
        
        # Check model status
        import os
        model_path = "./models_weights/leadnet_best.keras"
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / 1024 / 1024
            print(f"  ✅ Trained Model: {model_path} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ No trained model found")
        
        # Check preprocessed data
        data_path = "./prepped_data/preprocessed_data.csv"
        if os.path.exists(data_path):
            import pandas as pd
            df = pd.read_csv(data_path)
            print(f"  ✅ Preprocessed Data: {data_path} ({len(df)} samples)")
        else:
            print(f"  ❌ No preprocessed data found")

if __name__ == "__main__":
    main()
