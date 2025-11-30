import pandas as pd
import numpy as np
import warnings
from itertools import combinations
import pmdarima as pm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

warnings.filterwarnings("ignore")

class SimpleARIMAXOptimizer:
    """
    Optimisation simple ARIMAX avec tests de validation
    """
    
    def __init__(self, max_vars_combination=3, max_arima_order=2):
        self.max_vars_combination = max_vars_combination
        self.max_arima_order = max_arima_order
        self.valid_models = []
        
    def generate_combinations(self, X):
        """Génère les combinaisons de variables"""
        all_combos = []
        for r in range(1, min(self.max_vars_combination + 1, len(X.columns) + 1)):
            all_combos.extend(combinations(X.columns, r))
        return all_combos
    
    def check_residuals(self, residuals, alpha=0.05):
        """Effectue tous les tests sur les résidus"""
        tests_results = {}
        
        # Test de blancheur (Ljung-Box)
        lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        tests_results['blancheur'] = lb_test['lb_pvalue'].iloc[0] > alpha
        
        # Test de normalité (Jarque-Bera)
        jb_stat, jb_pvalue = jarque_bera(residuals)
        tests_results['normalite'] = jb_pvalue > alpha
        
        # Test d'hétéroscédasticité (ARCH)
        try:
            arch_test = het_arch(residuals)
            tests_results['homoscedasticite'] = arch_test[1] > alpha
        except:
            tests_results['homoscedasticite'] = False
        
        # Test d'autocorrélation (Durbin-Watson)
        dw_stat = sm.stats.stattools.durbin_watson(residuals)
        tests_results['autocorrelation'] = 1.5 < dw_stat < 2.5
        
        return tests_results
    
    def calculate_vif(self, X):
        """Calcule le VIF pour les variables explicatives"""
        if X.shape[1] < 2:
            return {'VIF_ok': True, 'max_vif': 0}
        
        vif_values = {}
        for i, col in enumerate(X.columns):
            vif = variance_inflation_factor(X.values, i)
            vif_values[col] = vif
        
        max_vif = max(vif_values.values())
        return {'VIF_ok': max_vif < 5, 'max_vif': max_vif, 'vif_values': vif_values}
    
    def optimize_models(self, y, X):
        """Optimisation principale"""
        self.valid_models = []
        
        combinations_list = self.generate_combinations(X)
        print(f"Testing {len(combinations_list)} combinations...")
        
        for i, combo in enumerate(combinations_list, 1):
            combo_vars = list(combo)
            X_combo = X[combo_vars]
            
            print(f"Testing combination {i}/{len(combinations_list)}: {combo_vars}")
            
            try:
                # Auto-ARIMA avec recherche des meilleurs retards
                model = pm.auto_arima(
                    y=y,
                    X=X_combo,
                    start_p=0, max_p=self.max_arima_order,
                    start_q=0, max_q=self.max_arima_order, 
                    max_d=self.max_arima_order,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                
                # Récupération des résidus
                residuals = model.resid()
                
                # Tests statistiques
                residuals_tests = self.check_residuals(residuals)
                vif_test = self.calculate_vif(X_combo)
                
                # Vérification si tous les tests passent
                all_tests_passed = all(residuals_tests.values()) and vif_test['VIF_ok']
                
                if all_tests_passed:
                    model_info = {
                        'combo_name': '+'.join(combo_vars),
                        'variables': combo_vars,
                        'arima_order': model.order,
                        'aic': model.aic(),
                        'bic': model.bic(),
                        'residuals_tests': residuals_tests,
                        'vif_test': vif_test,
                        'model_object': model
                    }
                    self.valid_models.append(model_info)
                    print(f"  ✅ Model validated (AIC: {model.aic():.2f})")
                else:
                    print(f"  ❌ Model rejected - Tests failed")
                    
            except Exception as e:
                print(f"  ❌ Model failed: {str(e)}")
                continue
        
        # Tri par AIC
        self.valid_models.sort(key=lambda x: x['aic'])
        return self.valid_models
    
    def get_best_model(self):
        """Retourne le meilleur modèle"""
        return self.valid_models[0] if self.valid_models else None
    
    def summary(self):
        """Affiche un résumé des modèles validés"""
        if not self.valid_models:
            print("No valid models found.")
            return
        
        summary_data = []
        for i, model in enumerate(self.valid_models, 1):
            summary_data.append({
                'Rank': i,
                'Variables': model['combo_name'],
                'ARIMA_Order': model['arima_order'],
                'AIC': f"{model['aic']:.2f}",
                'BIC': f"{model['bic']:.2f}",
                'Blancheur': '✅' if model['residuals_tests']['blancheur'] else '❌',
                'Normalité': '✅' if model['residuals_tests']['normalite'] else '❌',
                'Homoscédasticité': '✅' if model['residuals_tests']['homoscedasticite'] else '❌',
                'Autocorrelation': '✅' if model['residuals_tests']['autocorrelation'] else '❌',
                'VIF_OK': '✅' if model['vif_test']['VIF_ok'] else '❌',
                'Max_VIF': f"{model['vif_test']['max_vif']:.2f}"
            })
        
        return pd.DataFrame(summary_data)

# Exemple d'utilisation
if __name__ == "__main__":
    # Données d'exemple
    np.random.seed(42)
    n = 200
    
    # Série cible
    y = pd.Series(np.cumsum(np.random.normal(0, 1, n)) + 0.1 * np.arange(n))
    
    # Variables explicatives
    X = pd.DataFrame({
        'var1': 0.3 * y + np.random.normal(0, 1, n),
        'var2': 0.2 * y + np.random.normal(0, 1, n),
        'var3': 0.4 * y + np.random.normal(0, 1, n),
        'var4': np.random.normal(0, 1, n),  # Variable indépendante
    })
    
    # Optimisation
    optimizer = SimpleARIMAXOptimizer(max_vars_combination=2, max_arima_order=2)
    valid_models = optimizer.optimize_models(y, X)
    
    # Résultats
    print("\n" + "="*80)
    print("SUMMARY OF VALID MODELS")
    print("="*80)
    
    summary_df = optimizer.summary()
    if summary_df is not None:
        print(summary_df.to_string(index=False))
        
        # Détails du meilleur modèle
        best_model = optimizer.get_best_model()
        print(f"\nBEST MODEL:")
        print(f"Variables: {best_model['combo_name']}")
        print(f"ARIMA Order: {best_model['arima_order']}")
        print(f"AIC: {best_model['aic']:.2f}")
        
        # Prévisions avec le meilleur modèle
        n_forecast = 10
        forecast, conf_int = best_model['model_object'].predict(
            n_periods=n_forecast, 
            X=X[best_model['variables']].iloc[-n_forecast:],
            return_conf_int=True
        )
        
        print(f"\nForecast for next {n_forecast} periods:")
        for i, (f, (lower, upper)) in enumerate(zip(forecast, conf_int)):
            print(f"Period {i+1}: {f:.2f} ({lower:.2f}, {upper:.2f})")