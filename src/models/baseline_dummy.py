import torch

def generate_dummy_face(n=1, size=64):
    """
    Génère n images factices (bruit aléatoire).
    """
    # Du bruit blanc normalisé [-1,1] pour simuler une image, juste pour brancher l'UI
    imgs = torch.rand(n, 3, size, size) * 2 - 1
    return imgs