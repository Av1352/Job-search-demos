"""
Run Multiple Experiments with Different Hyperparameters
This demonstrates ClearML's experiment comparison capabilities
"""
import subprocess
import time

experiments = [
    {
        'name': 'Baseline',
        'params': '--batch_size 64 --learning_rate 0.001 --hidden_size 128 --dropout 0.25 --epochs 5'
    },
    {
        'name': 'Higher LR',
        'params': '--batch_size 64 --learning_rate 0.01 --hidden_size 128 --dropout 0.25 --epochs 5'
    },
    {
        'name': 'Larger Hidden',
        'params': '--batch_size 64 --learning_rate 0.001 --hidden_size 256 --dropout 0.25 --epochs 5'
    },
    {
        'name': 'Less Dropout',
        'params': '--batch_size 64 --learning_rate 0.001 --hidden_size 128 --dropout 0.1 --epochs 5'
    }
]

print("="*60)
print("Running Multiple Experiments for ClearML Comparison")
print("="*60)
print(f"\nTotal experiments: {len(experiments)}\n")

for i, exp in enumerate(experiments, 1):
    print(f"\n{'='*60}")
    print(f"Experiment {i}/{len(experiments)}: {exp['name']}")
    print(f"{'='*60}")
    print(f"Parameters: {exp['params']}\n")
    
    cmd = f"python train_mnist.py {exp['params']}"
    subprocess.run(cmd, shell=True)
    
    print(f"\n✅ Completed: {exp['name']}")
    
    if i < len(experiments):
        print("\nWaiting 5 seconds before next experiment...")
        time.sleep(5)

print(f"\n{'='*60}")
print("All Experiments Complete!")
print(f"{'='*60}")
print("\n🎉 View all experiments in ClearML Web UI:")
print("   https://app.clear.ml")
print("\n📊 You can now:")
print("   - Compare experiments side-by-side")
print("   - See which hyperparameters performed best")
print("   - Clone the best experiment")
print("   - Download models")
print(f"{'='*60}\n")