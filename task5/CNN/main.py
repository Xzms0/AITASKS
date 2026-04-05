from pathlib import Path

import numpy as np
import pickle

from model import CNN
from utils import loader, optimizer, solver, manager

ROOT_DIR = Path(__file__).absolute().parent
TRAIN_SAMPLE = 50000

def main():
    print("Loading CIFAR-10 data...")
    
    X_train, y_train, X_test, y_test, label_names = loader.load_cifar10(
        path=ROOT_DIR / 'cifar_10',
        batch=5,
        normalize=True
    )
    
    model = CNN()
    #model = manager.load_model(ROOT_DIR / 'cifar_10' / 'model.pkl')

    print("\nModel created with layers:")
    for i, layer in enumerate(model.model):
        print(f" {i}: {layer.__class__.__name__}")
    
    optimizer_sgd = optimizer.SGD(lr=0.01)
    
    print("\nStart training...")
    
    solver_obj = solver.Solver(
        model=model,
        loss_fn=model.loss_fn,
        optimizer=optimizer_sgd,
        X=X_train[:TRAIN_SAMPLE, :, :, :],
        y=y_train[:],
        epochs=100,
        batch_size=500,
        verbose=True,
        print_every=10,
        decay_every=50
    )

    losses = solver_obj.train()

    print("\nEvaluating...")
    acc = solver_obj.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc*100:.2f}%")

    print("\nSaving...")
    manager.save_model(solver_obj.model, ROOT_DIR / 'cifar_10' / 'model.pkl')


if __name__ == "__main__":
    main()