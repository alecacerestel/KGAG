# Source - https://stackoverflow.com/a
# Posted by MaxNoe
# Retrieved 2026-01-19, License - CC BY-SA 3.0

# To use : python KGAG/read_loss_history.py

import numpy as np
import os

def main():
    # Path to the loss history file
    loss_file = os.path.join('checkpoint', 'KGAG', 'loss_history.npy')
    
    # Check if file exists
    if not os.path.exists(loss_file):
        print(f"Error: File not found at {loss_file}")
        return
    
    # Load the loss history
    data = np.load(loss_file)
    
    print(f"Loss History ({len(data)} epochs):")
    for epoch, loss in enumerate(data, start=1):
        print(f"Epoch {epoch}: {loss:.6f}")
    print(f"Initial loss: {data[0]:.6f}")
    print(f"Final loss:   {data[-1]:.6f}")
    print(f"Total improvement: {data[0] - data[-1]:.6f}")

if __name__ == '__main__':
    main()
