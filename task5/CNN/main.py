import numpy as np
from model import CNN
from utils import loader, optimizer, solver

def main():
    print("Loading CIFAR-10 data...")
    
    X_train, y_train, X_test, y_test, label_names = loader.load_cifar10(
        path='./cifar-10-batches-py',
        batch=5,
        normalize=True
    )
    
    model = CNN()
    print("\nModel created with layers:")
    for i, layer in enumerate(model.model):
        print(f" {i}: {layer.__class__.__name__}")
    
    optimizer_sgd = optimizer.SGD(lr=0.01)
    
    print("\nStart training...")
    
    solver_obj = solver.Solver(
        model=model,
        loss_fn=model.loss_fn,
        optimizer=optimizer_sgd,
        X_train=X_train,
        y_train=y_train,
        epochs=50,
        batch_size=64,
        verbose=True,
        print_every=5,
    )

    losses = solver_obj.train()
    acc = solver_obj.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc:.4f}")

    solver_obj.save_model('model.pkl')


if __name__ == "__main__":
    main()