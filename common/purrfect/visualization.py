import matplotlib.pyplot as plt
def plot_epsilon(epsilon) -> None:
    e1 = epsilon[0, :]
    e2 = epsilon[1, :]
    delta_e = epsilon[2, :]
    # Crear una figura con tres subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    vmin = min(e1.min(), e2.min(), delta_e.min())
    vmax = max(e1.max(), e2.max(), delta_e.max())

    # Mostrar cada imagen en su subplot correspondiente
    cmap = 'viridis'  # Colormap azul-blanco-rojo

    im1 = axs[0].imshow(e1, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title(r'$\epsilon_{1}$')

    im2 = axs[1].imshow(e2, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title(r'$\epsilon_{2}$')

    im3 = axs[2].imshow(delta_e, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].set_title(r'$\Delta\epsilon$')

    # Añadir una barra de color
    cbar = fig.colorbar(im1, ax=axs, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar.set_label(r'$\epsilon$')

    # Añadir un título global
    fig.suptitle('Imágenes de entrada para el problema', fontsize=14)

    # Mostrar la figura
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_kappa(kappa) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    cmap = 'viridis'
    im = ax.imshow(kappa, cmap=cmap, vmin=kappa.min(), vmax=kappa.max())
    ax.set_title('Imágen de salida esperada para el problema', fontsize=14)
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar.set_label(r'$\kappa$')
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
def plot_kappas(kappa_true,kappa_pred) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    cmap = 'viridis'
    vmin = min(kappa_true.min(), kappa_pred.min())
    vmax = max(kappa_true.max(), kappa_pred.max())
    im_true = axs[0].imshow(kappa_true, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title('True')
    im_true = axs[1].imshow(kappa_pred, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_title('pred')

    cbar = fig.colorbar(im_true, ax=axs, orientation='horizontal', fraction=0.02, pad=0.1)
    cbar.set_label(r'$\kappa$')
    fig.suptitle('Imágenes de salida para el problema', fontsize=14)
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
