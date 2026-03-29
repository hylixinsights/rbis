import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def specificity_climb(adata, cluster=None, title="RBIS Specificity Climb", save=None):
    """
    Plota a curva de especificidade (climb) para um ou todos os clusters.
    """
    if 'rbis' not in adata.uns or 'climb_curves' not in adata.uns['rbis']:
        print("Erro: Resultados do RBIS não encontrados em adata.uns['rbis']. Rode rbis.tl.find_markers_sc primeiro.")
        return

    climb_data = adata.uns['rbis']['climb_curves']
    
    plt.figure(figsize=(8, 5))
    if cluster:
        sns.lineplot(data=climb_data[climb_data['cluster'] == cluster], x='rank', y='specificity')
    else:
        sns.lineplot(data=climb_data, x='rank', y='specificity', hue='cluster')
    
    plt.title(title)
    plt.xlabel("Gene Rank")
    plt.ylabel("Specificity Score")
    sns.despine()
    
    if save:
        plt.savefig(save)
    plt.show()
