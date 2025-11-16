import matplotlib.pyplot as plt

def plot_history(hist, out_png):
    """hist: dict with keys 'round', 'train_acc', 'val_acc', 'train_loss', 'val_loss'"""
    r = hist['round']
    plt.figure()
    plt.plot(r, hist['train_acc'], label='train_acc')
    plt.plot(r, hist['val_acc'], label='val_acc')
    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.replace('.png','_acc.png'))
    plt.close()

    plt.figure()
    plt.plot(r, hist['train_loss'], label='train_loss')
    plt.plot(r, hist['val_loss'], label='val_loss')
    plt.xlabel('Communication Round')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.replace('.png','_loss.png'))
    plt.close()
