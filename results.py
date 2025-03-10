import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_results_dirs():
    """
    Crée les dossiers pour les figures et les tables s'ils n'existent pas.
    """
    figures_dir = os.path.join("results", "figures")
    tables_dir = os.path.join("results", "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    return figures_dir, tables_dir

def save_metrics_csv(metrics_dict, filename):
    """
    Sauvegarde les métriques dans un fichier CSV.
    
    :param metrics_dict: Dictionnaire contenant les métriques.
    :param filename: Chemin complet du fichier CSV.
    """
    df = pd.DataFrame(metrics_dict)
    df.to_csv(filename, index=False)
    print(f"Les métriques ont été sauvegardées dans {filename}")

def plot_metrics(metric_name, models, values, filename):
    """
    Génère un graphique en barres pour comparer une métrique entre différents modèles.
    
    :param metric_name: Nom de la métrique (ex: Accuracy, Precision).
    :param models: Liste des noms de modèles.
    :param values: Liste des valeurs correspondantes.
    :param filename: Chemin complet pour sauvegarder le graphique.
    """
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=values, palette="viridis")
    plt.title(f"Comparaison de {metric_name}")
    plt.xlabel("Modèles")
    plt.ylabel(metric_name)
    plt.savefig(filename)
    plt.close()
    print(f"Graphique pour {metric_name} sauvegardé dans {filename}")

def main():
    # Création des dossiers de résultats
    figures_dir, tables_dir = ensure_results_dirs()
    

    
    # Sauvegarde des métriques dans un fichier CSV
    metrics_csv = os.path.join(tables_dir, "model_metrics.csv")
    save_metrics_csv(metrics, metrics_csv)
    
    # Génération des graphiques pour Accuracy et F1-Score
    plot_metrics("Accuracy", metrics["Model"], metrics["Accuracy"],
                 os.path.join(figures_dir, "accuracy_comparison.png"))
    plot_metrics("F1-Score", metrics["Model"], metrics["F1-Score"],
                 os.path.join(figures_dir, "f1_score_comparison.png"))

if __name__ == "__main__":
    main()
