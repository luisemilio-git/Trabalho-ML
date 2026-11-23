import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def treinar_mlp (X, Y, hidden=(64,32), lr=0.001, weight_decay=0.0, batch_size=32, epochs=200, paciencia=15, seed=42, verbose=False):
    torch.manual_seed(seed)

    X_treino, X_val, Y_treino, Y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)

    
    preproce = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

    X_treino_t = torch.tensor(preproce.fit_transform(X_treino), dtype=torch.float32)
    X_val_t = torch.tensor(preproce.transform(X_val), dtype=torch.float32)

    y_media = Y_treino.mean()
    y_desvio = Y_treino.std()
    Y_treino_t = torch.tensor((Y_treino - y_media) / y_desvio, dtype=torch.float32).unsqueeze(1)
    Y_val_t = torch.tensor((Y_val - y_media) / y_desvio, dtype=torch.float32).unsqueeze(1)

    dataset_treino = TensorDataset(X_treino_t, Y_treino_t)
    loader = DataLoader(dataset_treino, batch_size=batch_size, shuffle=True)

    model = nn.Sequential(
        nn.Linear(X_treino_t.shape[1], hidden[0]), nn.ReLU(), 
        nn.Linear(hidden[0], hidden[1]), nn.ReLU(),
        nn.Linear(hidden[1], 1)
    )

    loss_func = nn.MSELoss()
    otimizador = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    melhor_val = float('inf')
    melhor_estado = None
    sem_melhora = 0
    historico_loss = []
    historico_val = []

    for epoch in range(epochs):
        model.train()
        loss_epoca = 0.0

        for xb, yb in loader:
            pred = model(xb)
            loss = loss_func(pred, yb)

            otimizador.zero_grad()
            loss.backward()
            otimizador.step()

            loss_epoca += loss.item() * xb.size(0)

        historico_loss.append(loss_epoca / len(dataset_treino))

        model.eval()
        with torch.no_grad():
            loss_val = loss_func(model(X_val_t), Y_val_t).item()
        historico_val.append(loss_val)

        if loss_val < melhor_val - 1e-4:
            melhor_val = loss_val
            sem_melhora = 0
            melhor_estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(melhor_estado)
    model.eval()
    return model, preproce, y_media, y_desvio, historico_loss, historico_val


def prever_mlp (model, preproce, y_media, y_desvio, X):
    X_t = torch.tensor(preproce.transform(X), dtype=torch.float32)
    with torch.no_grad():
        return model(X_t).squeeze(1).numpy() * y_desvio + y_media

# cross-validation
def cv_mlp (X, Y, kf, **params):
    metricas = {'rmse': [], 'mae': [], 'r2': []}

    for id_treino, id_aval in kf.split(X):
        X_treino = X[id_treino]
        X_aval = X[id_aval]
        Y_treino = Y[id_treino]
        Y_aval = Y[id_aval]

        model_fold, prep_fold, mu, sigma, _, _ = treinar_mlp(X_treino, Y_treino, **params)
        y_pred = prever_mlp(model_fold, prep_fold, mu, sigma, X_aval)

        metricas['rmse'].append(np.sqrt(mean_squared_error(Y_aval, y_pred)))
        metricas['mae'].append(mean_absolute_error(Y_aval, y_pred))
        metricas['r2'].append(r2_score(Y_aval, y_pred))
    
    return metricas

def grafico_previsto_real(Y_real, previsoes, nome_arquivo='previsto_real.png'):
    """previsoes: dict {nome_do_modelo: array de previsões}"""
    n = len(previsoes)
    fig, eixos = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharex=True, sharey=True)
    if n == 1:
        eixos = [eixos]

    lim_min = min(Y_real.min(), *(p.min() for p in previsoes.values()))
    lim_max = max(Y_real.max(), *(p.max() for p in previsoes.values()))

    for ax, (nome, y_pred) in zip(eixos, previsoes.items()):
        ax.scatter(Y_real, y_pred, s=12, alpha=0.5)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1, label='Perfect prediction')
        ax.set_title(f"{nome} (R² = {r2_score(Y_real, y_pred):.4f})")
        ax.set_xlabel('Real calories')
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect('equal', adjustable='box')

    eixos[0].set_ylabel('Predicted calories')
    eixos[0].legend()
    plt.tight_layout()
    plt.savefig(nome_arquivo, bbox_inches='tight', dpi=200)
    plt.close()


if __name__ == "__main__":
    df = pd.read_csv('gym.csv')

    # separar campos de entrada e saída
    colunas_features = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
    'Session_Duration (hours)', 'Fat_Percentage', 'Water_Intake (liters)',
    'Workout_Frequency (days/week)', 'Experience_Level', 'BMI']
    one_hot = pd.get_dummies(df[['Gender', 'Workout_Type']], dtype=float, dummy_na=True)
    X_temp = pd.concat ([df[colunas_features], one_hot], axis=1)
    X = X_temp.values

    Y = df['Calories_Burned'].values.astype(float)

#--------------------------------------------------------------

    # divisao dos dados (80% - Cross Validation e 20% - teste)
    X_treino_val, X_teste, Y_treino_val, Y_teste = train_test_split(X, Y, test_size=0.20, random_state=42)


#--------------------------------------------------------------

    # Ridge e Lasso

    ridge = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(cv=5))
    ridge.fit(X_treino_val, Y_treino_val)

    lasso = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), LassoCV(cv=5, random_state=42))
    lasso.fit(X_treino_val, Y_treino_val)

    previsoes = {}

    # printa métricas e features
    for nome, modelo in [('Ridge', ridge), ('Lasso', lasso)]:
        y_pred = modelo.predict(X_teste)
        previsoes[nome] = y_pred
        rmse = np.sqrt(mean_squared_error(Y_teste, y_pred))
        mae = mean_absolute_error(Y_teste, y_pred)
        r2 = r2_score(Y_teste, y_pred)
        print(f"-{nome}")
        print(f"Best alpha: {modelo[-1].alpha_}")
        print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        print("Features:")
        for feat, coef in zip(X_temp.columns, modelo[-1].coef_):
            print(f"  {feat}: {coef:.4f}")
        print()

#--------------------------------------------------------------

    # MLP com Cross Validation

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    grade = {
        'hidden': [(32, 16), (64, 32), (128, 64)],
        'lr': [0.001, 0.003, 0.01, 0.03],
        'weight_decay': [0.0, 1e-4]
    }

    resultados = []

    for valores in product(*grade.values()):
        params = dict(zip(grade.keys(), valores))
        m = cv_mlp(X_treino_val, Y_treino_val, kf, **params)
        rmse_medio = np.mean(m['rmse'])
        rmse_std = np.std(m['rmse'])
        resultados.append((params, rmse_medio))
        print(f"{params} -> RMSE {rmse_medio:.4f} ± {rmse_std:.4f} | R2 {np.mean(m['r2']):.4f}")
    
    melhor_params = min(resultados, key=lambda r: r[1])[0]
    print("\nBest Hyperparameters:", melhor_params)

    model, preproce, y_media, y_desvio, historico_loss, historico_val = treinar_mlp(X_treino_val, Y_treino_val, **melhor_params, verbose=True)
    y_pred_mlp = prever_mlp(model, preproce, y_media, y_desvio, X_teste)
    previsoes['MLP'] = y_pred_mlp
    grafico_previsto_real(Y_teste, {'MLP': previsoes['MLP'],
                                'Ridge': previsoes['Ridge'],
                                'Lasso': previsoes['Lasso']})

    # Curvas de perda (treino x validação)
    plt.plot(historico_loss, label='Training')
    plt.plot(historico_val, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MSE (standardised target)')
    plt.legend()
    plt.savefig('curva_loss.png', bbox_inches='tight')
    plt.close()

    rmse_mlp = np.sqrt(mean_squared_error(Y_teste, y_pred_mlp))
    mae_mlp = mean_absolute_error(Y_teste, y_pred_mlp)
    r2_mlp = r2_score(Y_teste, y_pred_mlp)

    print("- MLP (PyTorch)")
    print(f"RMSE: {rmse_mlp:.4f} | MAE: {mae_mlp:.4f} | R2: {r2_mlp:.4f}")

    nomes = np.array(X_temp.columns)
    coef_ridge = ridge[-1].coef_
    coef_lasso = lasso[-1].coef_

    # remove colunas nan e ordena pela magnitude do Ridge
    mask = np.array([not n.endswith('_nan') for n in nomes])
    nomes, coef_ridge, coef_lasso = nomes[mask], coef_ridge[mask], coef_lasso[mask]
    ordem = np.argsort(np.abs(coef_ridge))

    y = np.arange(len(nomes))
    plt.figure(figsize=(7, 6))
    plt.barh(y - 0.2, coef_ridge[ordem], height=0.4, label='Ridge')
    plt.barh(y + 0.2, coef_lasso[ordem], height=0.4, label='Lasso')
    plt.yticks(y, nomes[ordem])
    plt.axvline(0, color='k', linewidth=0.8)
    plt.xlabel('Coefficient (standardised variables)')
    plt.legend()
    plt.savefig('coeficientes.png', bbox_inches='tight', dpi=200)
    plt.close()
