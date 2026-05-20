import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pandas as pd

class Visualizer:
    def __init__(self, output_dir="reports/figures"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        os.makedirs(os.path.join(output_dir, "static"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "dynamic"), exist_ok=True)

    def plot_static_correlations(self, correlations):
        self.logger.info("Generando gráfico de correlaciones estáticas...")
        plt.figure(figsize=(10, 6))
        if hasattr(correlations, 'columns'):
            data = correlations['LowTemp'].drop('LowTemp', errors='ignore').sort_values()
            sns.barplot(x=data.values, y=data.index, palette='viridis')
        plt.title("Correlación Estática con LowTemp")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "static/static_corr_comparison.png"))
        plt.close()

    def plot_dynamic_results(self, dynamic_results):
        """
        Genera la grilla de subplots (Pearson, Spearman, Kendall) para cada variable.
        """
        self.logger.info("Generando grilla de evolución de correlaciones (Lags)...")
        
        # Extraemos los DataFrames del diccionario
        df_p = dynamic_results['Pearson']
        df_s = dynamic_results['Spearman']
        df_k = dynamic_results['Kendall']
        
        # Variables a graficar (todas menos 'Hora')
        variables = [c for c in df_p.columns if c != 'Hora']
        n_vars = len(variables)
        
        # Configuración de la grilla (3 columnas)
        cols = 3
        rows = (n_vars + cols - 1) // cols
        
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 4 * rows))
        axs = axs.flatten()

        for i, var in enumerate(variables):
            ax = axs[i]
            ax.plot(df_p['Hora'], df_p[var], label='Pearson', color='#005088', marker='o', markersize=3)
            ax.plot(df_s['Hora'], df_s[var], label='Spearman', color='#11caa0', marker='s', markersize=3)
            ax.plot(df_k['Hora'], df_k[var], label='Kendall', color='#ef4444', linestyle='--')

            ax.set_title(f"Variable: {var}", fontsize=13, fontweight='bold')
            ax.set_xlabel("Lag (Horas)")
            ax.set_ylabel("Coeficiente")
            ax.set_xticks(range(1, 25, 2)) # Marcas cada 2 horas para no saturar
            ax.grid(True, alpha=0.3)
            
            # El índice (1), (2), etc.
            ax.text(0.95, 0.95, f'({i+1})', transform=ax.transAxes, 
                    ha='right', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))

        # Ocultar cuadros vacíos si sobran
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        # Leyenda única abajo
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.01))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1) # Espacio para la leyenda
        
        path = os.path.join(self.output_dir, "dynamic/comparacion_correlacion.png")
        plt.savefig(path, dpi=300)
        plt.close()
        self.logger.info(f"Gráfico dinámico profesional guardado en: {path}")
    
    def plot_model_comparison(self, results_map, metric='F1'):
        """
        Grafica la comparativa de una métrica para distintas arquitecturas.
        """
        import matplotlib.pyplot as plt
        import os

        plt.figure(figsize=(12, 6))
        
        for arch_name, df_arch in results_map.items():
            # Agrupamos por Lead Time para sacar el promedio de los Folds
            # Si el model.py ahora usa 'Lead Time', aquí usamos lo mismo
            stats = df_arch.groupby('Lead Time')[metric].agg(['mean', 'std']).reset_index()
            
            plt.plot(stats['Lead Time'], stats['mean'], marker='o', label=f"Arch: {arch_name}")
            plt.fill_between(
                stats['Lead Time'], 
                stats['mean'] - stats['std'], 
                stats['mean'] + stats['std'], 
                alpha=0.2
            )

        plt.title(f'Comparativa de Rendimiento: {metric}')
        plt.xlabel('Horas de Antelación (Lead Time)')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # EL TRUCO: Asegurar que la carpeta exista antes de guardar
        output_path = os.path.join(self.output_dir, 'models', f'comparativa_{metric.lower()}.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            plt.savefig(output_path)
            self.logger.info(f"Gráfico de {metric} guardado en: {output_path}")
        except Exception as e:
            self.logger.error(f"Error al guardar gráfico {metric}: {str(e)}")
        finally:
            plt.close()